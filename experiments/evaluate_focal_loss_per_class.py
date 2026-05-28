from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from tune_clip_dinov2_ensemble_hyperparams import (
    CLIP_DIR,
    DINO_DIR,
    DEVICE,
    OUTPUT_DIR,
    SEED,
    SHOT_TYPES,
    TrainConfig,
    combine_global,
    search_global_alpha,
    search_text_threshold,
    train_clip,
    train_dino,
    predict_multitask,
    predict_shot,
)


PER_CLASS_DIR = OUTPUT_DIR / "focal_loss_per_class_metrics"
PER_CLASS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def top2_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    top2 = np.argsort(prob, axis=1)[:, -2:]
    return float(np.mean([truth in row for truth, row in zip(y_true, top2)]))


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(SHOT_TYPES))),
        zero_division=0,
    )
    rows = []
    for idx, label in enumerate(SHOT_TYPES):
        pred_mask = y_pred == idx
        true_mask = y_true == idx
        rows.append(
            {
                "shot_type": label,
                "precision": precision[idx],
                "recall_per_class_accuracy": recall[idx],
                "f1": f1[idx],
                "support": int(support[idx]),
                "false_positive_count": int(np.sum(pred_mask & ~true_mask)),
                "false_negative_count": int(np.sum(true_mask & ~pred_mask)),
                "avg_confidence_when_predicted": float(prob[pred_mask, idx].mean()) if pred_mask.any() else np.nan,
                "top2_recall": float(np.mean([idx in row for row in np.argsort(prob[true_mask], axis=1)[:, -2:]])) if true_mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def top_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SHOT_TYPES))))
    rows = []
    for true_idx, true_label in enumerate(SHOT_TYPES):
        total = cm[true_idx].sum()
        for pred_idx, pred_label in enumerate(SHOT_TYPES):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            rows.append(
                {
                    "true_class": true_label,
                    "pred_class": pred_label,
                    "count": count,
                    "ratio_within_true": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["count", "ratio_within_true"], ascending=False)


def main() -> None:
    seed_everything()
    meta = pd.read_csv(CLIP_DIR / "clip_embedding_metadata.csv")
    clip_emb = np.load(CLIP_DIR / "clip_vit_b32_openai_embeddings.npz")["embeddings"].astype("float32")
    dino_emb = np.load(DINO_DIR / "vit_small_patch14_dinov2_embeddings.npz")["embeddings"].astype("float32")
    labels = meta["shot_type"].map({name: i for i, name in enumerate(SHOT_TYPES)}).to_numpy(dtype=np.int64)
    text = meta["has_text"].to_numpy(dtype=np.int64)
    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}

    def tx(arr):
        return torch.tensor(arr, dtype=torch.float32)

    def ty(arr):
        return torch.tensor(arr, dtype=torch.long)

    config = TrainConfig("focal_loss", loss_type="focal")
    clip_model = train_clip(
        config,
        tx(clip_emb[masks["train"]]), ty(labels[masks["train"]]), ty(text[masks["train"]]),
        tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]]),
    )
    dino_model = train_dino(
        config,
        tx(dino_emb[masks["train"]]), ty(labels[masks["train"]]),
        tx(dino_emb[masks["val"]]), ty(labels[masks["val"]]),
    )

    clip_val_loader = DataLoader(TensorDataset(tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]])), batch_size=256)
    clip_test_loader = DataLoader(TensorDataset(tx(clip_emb[masks["test"]]), ty(labels[masks["test"]]), ty(text[masks["test"]])), batch_size=256)
    dino_val_loader = DataLoader(TensorDataset(tx(dino_emb[masks["val"]]), ty(labels[masks["val"]])), batch_size=256)
    dino_test_loader = DataLoader(TensorDataset(tx(dino_emb[masks["test"]]), ty(labels[masks["test"]])), batch_size=256)

    shot_val_true, text_val_true, clip_val_shot_prob, clip_val_text_prob = predict_multitask(clip_model, clip_val_loader)
    shot_test_true, text_test_true, clip_test_shot_prob, clip_test_text_prob = predict_multitask(clip_model, clip_test_loader)
    _, dino_val_shot_prob = predict_shot(dino_model, dino_val_loader)
    _, dino_test_shot_prob = predict_shot(dino_model, dino_test_loader)

    alpha_best, alpha_rows = search_global_alpha(shot_val_true, clip_val_shot_prob, dino_val_shot_prob)
    val_shot_prob = combine_global(clip_val_shot_prob, dino_val_shot_prob, alpha_best["alpha_clip"])
    text_best, text_rows = search_text_threshold(shot_val_true, val_shot_prob, text_val_true, clip_val_text_prob, objective="joint")
    test_shot_prob = combine_global(clip_test_shot_prob, dino_test_shot_prob, alpha_best["alpha_clip"])
    test_shot_pred = test_shot_prob.argmax(axis=1)
    test_text_pred = (clip_test_text_prob[:, 1] >= text_best["text_threshold"]).astype(np.int64)

    per_class = per_class_table(shot_test_true, test_shot_pred, test_shot_prob)
    per_class.to_csv(PER_CLASS_DIR / "shot_type_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(shot_test_true, test_shot_pred, labels=list(range(len(SHOT_TYPES))))
    pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES).to_csv(PER_CLASS_DIR / "shot_type_confusion_matrix.csv", encoding="utf-8-sig")
    top_confusion_pairs(shot_test_true, test_shot_pred).to_csv(PER_CLASS_DIR / "shot_type_top_confusion_pairs.csv", index=False, encoding="utf-8-sig")

    pred_df = meta[masks["test"]].copy().reset_index(drop=True)
    pred_df["shot_pred"] = [SHOT_TYPES[i] for i in test_shot_pred]
    pred_df["shot_correct"] = shot_test_true == test_shot_pred
    pred_df["text_pred"] = test_text_pred
    pred_df["text_correct"] = text_test_true == test_text_pred
    pred_df["joint_correct"] = pred_df["shot_correct"] & pred_df["text_correct"]
    pred_df["shot_confidence"] = test_shot_prob.max(axis=1)
    for i, label in enumerate(SHOT_TYPES):
        pred_df[f"prob_shot_{label}"] = test_shot_prob[:, i]
    pred_df.to_csv(PER_CLASS_DIR / "focal_loss_test_predictions_with_shot_probs.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "focal_loss",
        "alpha_clip": alpha_best["alpha_clip"],
        "text_threshold": text_best["text_threshold"],
        "shot_accuracy": accuracy_score(shot_test_true, test_shot_pred),
        "shot_macro_f1": f1_score(shot_test_true, test_shot_pred, average="macro", zero_division=0),
        "shot_balanced_accuracy": balanced_accuracy_score(shot_test_true, test_shot_pred),
        "shot_weighted_f1": f1_score(shot_test_true, test_shot_pred, average="weighted", zero_division=0),
        "shot_top2_accuracy": top2_accuracy(shot_test_true, test_shot_prob),
        "text_accuracy": accuracy_score(text_test_true, test_text_pred),
        "text_macro_f1": f1_score(text_test_true, test_text_pred, average="macro", zero_division=0),
        "joint_accuracy": float(np.mean((shot_test_true == test_shot_pred) & (text_test_true == test_text_pred))),
    }
    with open(PER_CLASS_DIR / "focal_loss_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
