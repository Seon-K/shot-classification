from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from tune_clip_dinov2_ensemble_hyperparams import (
    CLIP_DIR,
    DEVICE,
    SEED,
    SHOT_TYPES,
    TrainConfig,
    combine_global,
    predict_multitask,
    predict_shot,
    train_clip,
    train_dino,
)


ROOT = Path(__file__).resolve().parents[1]
DINOv3_EMB_PATH = ROOT / "outputs" / "model_experiments" / "dinov3_backbones" / "dinov3_vit_small_patch16_embeddings.npz"
SIGLIP2_EMB_PATH = ROOT / "outputs" / "model_experiments" / "siglip2_backbones" / "siglip2_vit_b16_256_webli_embeddings.npz"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "dinov3_siglip2_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tx(arr):
    return torch.tensor(arr, dtype=torch.float32)


def ty(arr):
    return torch.tensor(arr, dtype=torch.long)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def top2_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    top2 = np.argsort(prob, axis=1)[:, -2:]
    return float(np.mean([truth in row for truth, row in zip(y_true, top2)]))


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    correct = (y_true == y_pred).astype(float)
    total = len(y_true)
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (confidence >= start) & (confidence <= end) if end == 1.0 else (confidence >= start) & (confidence < end)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_start": start, "bin_end": end, "count": 0, "accuracy": np.nan, "confidence": np.nan, "gap": np.nan})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        ece += (count / total) * gap
        mce = max(mce, gap)
        rows.append({"bin_start": start, "bin_end": end, "count": count, "accuracy": acc, "confidence": conf, "gap": gap})
    return float(ece), float(mce), pd.DataFrame(rows)


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None, prefix: str, label_count: int) -> dict[str, float]:
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_macro_precision": precision,
        f"{prefix}_macro_recall": recall,
    }
    if prob is not None and label_count > 2:
        out[f"{prefix}_top2_accuracy"] = top2_accuracy(y_true, prob)
    return out


def search_global_alpha(shot_true: np.ndarray, siglip2_prob: np.ndarray, dinov3_prob: np.ndarray) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for alpha_siglip2 in np.round(np.arange(0.0, 1.001, 0.01), 2):
        prob = combine_global(siglip2_prob, dinov3_prob, float(alpha_siglip2))
        pred = prob.argmax(axis=1)
        row = {
            "alpha_siglip2": float(alpha_siglip2),
            "alpha_dinov3": float(1.0 - alpha_siglip2),
            "val_shot_acc": accuracy_score(shot_true, pred),
            "val_shot_macro_f1": f1_score(shot_true, pred, average="macro", zero_division=0),
        }
        rows.append(row)
        if best is None or row["val_shot_macro_f1"] > best["val_shot_macro_f1"]:
            best = row
    return best, pd.DataFrame(rows)


def search_text_threshold(shot_true: np.ndarray, shot_prob: np.ndarray, text_true: np.ndarray, text_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    shot_pred = shot_prob.argmax(axis=1)
    rows = []
    best = None
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        text_pred = (text_prob[:, 1] >= float(threshold)).astype(np.int64)
        row = {
            "text_threshold": float(threshold),
            "val_joint_acc": float(np.mean((shot_true == shot_pred) & (text_true == text_pred))),
            "val_text_acc": accuracy_score(text_true, text_pred),
            "val_text_macro_f1": f1_score(text_true, text_pred, average="macro", zero_division=0),
        }
        rows.append(row)
        if best is None or row["val_joint_acc"] > best["val_joint_acc"] or (
            row["val_joint_acc"] == best["val_joint_acc"] and row["val_text_macro_f1"] > best["val_text_macro_f1"]
        ):
            best = row
    return float(best["text_threshold"]), pd.DataFrame(rows)


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, labels: list[str], label_col: str) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))), zero_division=0)
    rows = []
    for idx, label in enumerate(labels):
        pred_mask = y_pred == idx
        true_mask = y_true == idx
        rows.append({
            label_col: label,
            "precision": precision[idx],
            "recall_per_class_accuracy": recall[idx],
            "f1": f1[idx],
            "support": int(support[idx]),
            "false_positive_count": int(np.sum(pred_mask & ~true_mask)),
            "false_negative_count": int(np.sum(true_mask & ~pred_mask)),
            "avg_confidence_when_predicted": float(prob[pred_mask, idx].mean()) if pred_mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    seed_everything()
    started = time.perf_counter()
    print("device:", DEVICE)
    meta = pd.read_csv(CLIP_DIR / "clip_embedding_metadata.csv")
    dinov3_emb = np.load(DINOv3_EMB_PATH)["embeddings"].astype("float32")
    siglip2_emb = np.load(SIGLIP2_EMB_PATH)["embeddings"].astype("float32")

    labels = meta["shot_type"].map({name: i for i, name in enumerate(SHOT_TYPES)}).to_numpy(dtype=np.int64)
    text = meta["has_text"].to_numpy(dtype=np.int64)
    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}

    config = TrainConfig("dinov3_siglip2_focal_ensemble", loss_type="focal")
    print("training SigLIP2 focal multitask head")
    siglip2_model = train_clip(
        config,
        tx(siglip2_emb[masks["train"]]), ty(labels[masks["train"]]), ty(text[masks["train"]]),
        tx(siglip2_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]]),
    )
    print("training DINOv3 focal shot head")
    dinov3_model = train_dino(
        config,
        tx(dinov3_emb[masks["train"]]), ty(labels[masks["train"]]),
        tx(dinov3_emb[masks["val"]]), ty(labels[masks["val"]]),
    )

    siglip2_val_loader = DataLoader(TensorDataset(tx(siglip2_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]])), batch_size=256)
    siglip2_test_loader = DataLoader(TensorDataset(tx(siglip2_emb[masks["test"]]), ty(labels[masks["test"]]), ty(text[masks["test"]])), batch_size=256)
    dinov3_val_loader = DataLoader(TensorDataset(tx(dinov3_emb[masks["val"]]), ty(labels[masks["val"]])), batch_size=256)
    dinov3_test_loader = DataLoader(TensorDataset(tx(dinov3_emb[masks["test"]]), ty(labels[masks["test"]])), batch_size=256)

    shot_val_true, text_val_true, siglip2_val_shot_prob, siglip2_val_text_prob = predict_multitask(siglip2_model, siglip2_val_loader)
    shot_test_true, text_test_true, siglip2_test_shot_prob, siglip2_test_text_prob = predict_multitask(siglip2_model, siglip2_test_loader)
    _, dinov3_val_shot_prob = predict_shot(dinov3_model, dinov3_val_loader)
    _, dinov3_test_shot_prob = predict_shot(dinov3_model, dinov3_test_loader)

    alpha_best, alpha_rows = search_global_alpha(shot_val_true, siglip2_val_shot_prob, dinov3_val_shot_prob)
    alpha_siglip2 = float(alpha_best["alpha_siglip2"])
    val_shot_prob = combine_global(siglip2_val_shot_prob, dinov3_val_shot_prob, alpha_siglip2)
    test_shot_prob = combine_global(siglip2_test_shot_prob, dinov3_test_shot_prob, alpha_siglip2)
    text_threshold, threshold_rows = search_text_threshold(shot_val_true, val_shot_prob, text_val_true, siglip2_val_text_prob)

    shot_pred = test_shot_prob.argmax(axis=1)
    text_pred = (siglip2_test_text_prob[:, 1] >= text_threshold).astype(np.int64)
    shot_conf = test_shot_prob.max(axis=1)
    text_conf = np.maximum(siglip2_test_text_prob[:, 1], 1.0 - siglip2_test_text_prob[:, 1])
    joint_conf = shot_conf * text_conf
    joint_correct = (shot_test_true == shot_pred) & (text_test_true == text_pred)

    pred_df = meta[masks["test"]].copy().reset_index(drop=True)
    pred_df["shot_true_idx"] = shot_test_true
    pred_df["shot_pred_idx"] = shot_pred
    pred_df["shot_pred"] = [SHOT_TYPES[i] for i in shot_pred]
    pred_df["text_true"] = text_test_true
    pred_df["text_pred"] = text_pred
    pred_df["text_probability_siglip2"] = siglip2_test_text_prob[:, 1]
    pred_df["alpha_siglip2"] = alpha_siglip2
    pred_df["alpha_dinov3"] = 1.0 - alpha_siglip2
    pred_df["text_threshold"] = text_threshold
    pred_df["shot_confidence"] = shot_conf
    pred_df["text_confidence"] = text_conf
    pred_df["joint_confidence"] = joint_conf
    pred_df["shot_correct"] = shot_test_true == shot_pred
    pred_df["text_correct"] = text_test_true == text_pred
    pred_df["joint_correct"] = joint_correct
    for idx, label in enumerate(SHOT_TYPES):
        pred_df[f"prob_shot_{label}"] = test_shot_prob[:, idx]
    pred_df.to_csv(OUTPUT_DIR / "test_predictions_with_confidence.csv", index=False, encoding="utf-8-sig")

    alpha_rows.to_csv(OUTPUT_DIR / "alpha_search.csv", index=False, encoding="utf-8-sig")
    threshold_rows.to_csv(OUTPUT_DIR / "text_threshold_joint_search.csv", index=False, encoding="utf-8-sig")
    per_class_table(shot_test_true, shot_pred, test_shot_prob, SHOT_TYPES, "shot_type").to_csv(
        OUTPUT_DIR / "shot_type_per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )
    per_class_table(text_test_true, text_pred, siglip2_test_text_prob, ["notext", "text"], "text_class").to_csv(
        OUTPUT_DIR / "text_per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(confusion_matrix(shot_test_true, shot_pred, labels=list(range(len(SHOT_TYPES)))), index=SHOT_TYPES, columns=SHOT_TYPES).to_csv(
        OUTPUT_DIR / "shot_type_confusion_matrix.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(confusion_matrix(text_test_true, text_pred, labels=[0, 1]), index=["notext", "text"], columns=["notext", "text"]).to_csv(
        OUTPUT_DIR / "text_confusion_matrix.csv", encoding="utf-8-sig"
    )

    threshold_tradeoff = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        covered = joint_conf >= threshold
        threshold_tradeoff.append({
            "threshold": threshold,
            "coverage": safe_divide(int(covered.sum()), len(covered)),
            "reliability_joint_accuracy_when_covered": float(joint_correct[covered].mean()) if covered.any() else np.nan,
            "reliable_guide_ratio_correct_and_covered": safe_divide(int((covered & joint_correct).sum()), len(covered)),
            "covered_count": int(covered.sum()),
        })
    pd.DataFrame(threshold_tradeoff).to_csv(OUTPUT_DIR / "confidence_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "dinov3_siglip2_focal_ensemble",
        "shot_source": "SigLIP2 ViT-B/16-256 + DINOv3 ViT-S/16 focal ensemble",
        "text_source": "SigLIP2 ViT-B/16-256 text head",
        "alpha_siglip2": alpha_siglip2,
        "alpha_dinov3": 1.0 - alpha_siglip2,
        "text_threshold": text_threshold,
        "runtime_sec": time.perf_counter() - started,
        "joint_accuracy": float(joint_correct.mean()),
        "guide_reliability_score": float(joint_correct.mean()),
        "conditional_text_given_shot": safe_divide(int((pred_df["shot_correct"] & pred_df["text_correct"]).sum()), int(pred_df["shot_correct"].sum())),
        "conditional_shot_given_text": safe_divide(int((pred_df["shot_correct"] & pred_df["text_correct"]).sum()), int(pred_df["text_correct"].sum())),
        "both_correct_ratio": float(joint_correct.mean()),
        "shot_only_error_ratio": safe_divide(int((~pred_df["shot_correct"] & pred_df["text_correct"]).sum()), len(pred_df)),
        "text_only_error_ratio": safe_divide(int((pred_df["shot_correct"] & ~pred_df["text_correct"]).sum()), len(pred_df)),
        "both_wrong_ratio": safe_divide(int((~pred_df["shot_correct"] & ~pred_df["text_correct"]).sum()), len(pred_df)),
    }
    summary.update(classifier_metrics(shot_test_true, shot_pred, test_shot_prob, "shot", len(SHOT_TYPES)))
    summary.update(classifier_metrics(text_test_true, text_pred, siglip2_test_text_prob, "text", 2))

    for target, y_true, y_pred, conf in [
        ("shot", shot_test_true, shot_pred, shot_conf),
        ("text", text_test_true, text_pred, text_conf),
        ("joint", joint_correct.astype(int), np.ones_like(joint_correct, dtype=int), joint_conf),
    ]:
        ece, mce, hist = expected_calibration_error(y_true, y_pred, conf)
        summary[f"{target}_ECE"] = ece
        summary[f"{target}_MCE"] = mce
        hist.to_csv(OUTPUT_DIR / f"{target}_confidence_histogram.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
