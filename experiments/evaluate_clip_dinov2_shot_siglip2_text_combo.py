from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from tune_clip_dinov2_ensemble_hyperparams import (
    CLIP_DIR,
    DINO_DIR,
    DEVICE,
    SEED,
    SHOT_TYPES,
    TrainConfig,
    combine_global,
    predict_multitask,
    predict_shot,
    search_global_alpha,
    train_clip,
    train_dino,
)


ROOT = Path(__file__).resolve().parents[1]
SIGLIP2_EMB_PATH = ROOT / "outputs" / "model_experiments" / "siglip2_backbones" / "siglip2_vit_b16_256_webli_embeddings.npz"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "clip_dinov2_shot_siglip2_text_combo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class TextHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (512, 256), dropout: float = 0.25):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tx(arr):
    return torch.tensor(arr, dtype=torch.float32)


def ty(arr):
    return torch.tensor(arr, dtype=torch.long)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    return total / (num_classes * torch.clamp(counts, min=1))


def train_siglip2_text_head(x_train, y_train, x_val, y_val) -> TextHead:
    seed_everything()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256)
    model = TextHead(x_train.shape[1]).to(DEVICE)
    loss_fn = FocalLoss(weight=class_weights(y_train, 2).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state = None
    best_macro = -1.0
    wait = PATIENCE

    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()

        true, prob = predict_text(model, val_loader)
        macro = f1_score(true, prob.argmax(axis=1), average="macro", zero_division=0)
        if macro > best_macro:
            best_macro = float(macro)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break

    model.load_state_dict(best_state)
    return model


def predict_text(model: TextHead, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            true.extend(y.numpy().tolist())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.array(true), np.vstack(probs)


def search_combo_text_threshold(shot_true: np.ndarray, shot_prob: np.ndarray, text_true: np.ndarray, text_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
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
    clip_emb = np.load(CLIP_DIR / "clip_vit_b32_openai_embeddings.npz")["embeddings"].astype("float32")
    dino_emb = np.load(DINO_DIR / "vit_small_patch14_dinov2_embeddings.npz")["embeddings"].astype("float32")
    siglip2_emb = np.load(SIGLIP2_EMB_PATH)["embeddings"].astype("float32")

    labels = meta["shot_type"].map({name: i for i, name in enumerate(SHOT_TYPES)}).to_numpy(dtype=np.int64)
    text = meta["has_text"].to_numpy(dtype=np.int64)
    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}

    config = TrainConfig("clip_dinov2_focal_shot_siglip2_text", loss_type="focal")
    print("training CLIP focal multitask head")
    clip_model = train_clip(
        config,
        tx(clip_emb[masks["train"]]), ty(labels[masks["train"]]), ty(text[masks["train"]]),
        tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]]),
    )
    print("training DINOv2 focal shot head")
    dino_model = train_dino(
        config,
        tx(dino_emb[masks["train"]]), ty(labels[masks["train"]]),
        tx(dino_emb[masks["val"]]), ty(labels[masks["val"]]),
    )
    print("training SigLIP2 text head")
    siglip2_text_model = train_siglip2_text_head(
        tx(siglip2_emb[masks["train"]]), ty(text[masks["train"]]),
        tx(siglip2_emb[masks["val"]]), ty(text[masks["val"]]),
    )

    clip_val_loader = DataLoader(TensorDataset(tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]])), batch_size=256)
    clip_test_loader = DataLoader(TensorDataset(tx(clip_emb[masks["test"]]), ty(labels[masks["test"]]), ty(text[masks["test"]])), batch_size=256)
    dino_val_loader = DataLoader(TensorDataset(tx(dino_emb[masks["val"]]), ty(labels[masks["val"]])), batch_size=256)
    dino_test_loader = DataLoader(TensorDataset(tx(dino_emb[masks["test"]]), ty(labels[masks["test"]])), batch_size=256)
    siglip2_val_loader = DataLoader(TensorDataset(tx(siglip2_emb[masks["val"]]), ty(text[masks["val"]])), batch_size=256)
    siglip2_test_loader = DataLoader(TensorDataset(tx(siglip2_emb[masks["test"]]), ty(text[masks["test"]])), batch_size=256)

    shot_val_true, _clip_val_text_true, clip_val_shot_prob, _clip_val_text_prob = predict_multitask(clip_model, clip_val_loader)
    shot_test_true, _clip_test_text_true, clip_test_shot_prob, _clip_test_text_prob = predict_multitask(clip_model, clip_test_loader)
    _, dino_val_shot_prob = predict_shot(dino_model, dino_val_loader)
    _, dino_test_shot_prob = predict_shot(dino_model, dino_test_loader)
    text_val_true, siglip2_val_text_prob = predict_text(siglip2_text_model, siglip2_val_loader)
    text_test_true, siglip2_test_text_prob = predict_text(siglip2_text_model, siglip2_test_loader)

    alpha_best, alpha_rows = search_global_alpha(shot_val_true, clip_val_shot_prob, dino_val_shot_prob, lo=0.4, hi=0.6, step=0.01)
    alpha = float(alpha_best["alpha_clip"])
    val_shot_prob = combine_global(clip_val_shot_prob, dino_val_shot_prob, alpha)
    test_shot_prob = combine_global(clip_test_shot_prob, dino_test_shot_prob, alpha)
    text_threshold, threshold_rows = search_combo_text_threshold(shot_val_true, val_shot_prob, text_val_true, siglip2_val_text_prob)

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
    pred_df["alpha_clip"] = alpha
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
        "experiment": "clip_dinov2_focal_shot_siglip2_text",
        "shot_source": "CLIP ViT-B/32 + DINOv2 ViT-S/14 focal ensemble",
        "text_source": "SigLIP2 ViT-B/16-256 text head",
        "alpha_clip": alpha,
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
