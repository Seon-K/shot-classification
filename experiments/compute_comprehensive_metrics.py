from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "model_experiments" / "comprehensive_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_DIR = ROOT / "outputs" / "clip_embeddings"
BASELINE_DIR = ROOT / "outputs" / "baseline"
EXPERIMENT_DIR = ROOT / "outputs" / "model_experiments"
CHECKPOINT_DIR = ROOT / "checkpoints"

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
TEXT_TYPES = ["notext", "text"]
JOINT_LABELS = [f"{shot}_{text}" for shot in SHOT_TYPES for text in TEXT_TYPES]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipEmbeddingMultiTaskHead(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dims=None, hidden_dim=256, num_shot_classes=5, dropout=0.20):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [hidden_dim]
        layers = []
        prev = embedding_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        self.shared = nn.Sequential(*layers)
        self.shot_head = nn.Linear(prev, num_shot_classes)
        self.text_head = nn.Linear(prev, 2)

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_final_predictions():
    meta = pd.read_csv(CLIP_DIR / "clip_embedding_metadata.csv")
    embeddings = np.load(CLIP_DIR / "clip_vit_b32_openai_embeddings.npz")["embeddings"].astype("float32")
    checkpoint = torch.load(CHECKPOINT_DIR / "clip_vit_b32_multitask_head.pt", map_location=DEVICE, weights_only=False)
    hidden_dims = checkpoint.get("hidden_dims") or [checkpoint.get("hidden_dim", 256)]

    head = ClipEmbeddingMultiTaskHead(
        embedding_dim=checkpoint.get("embedding_dim", 512),
        hidden_dims=hidden_dims,
        num_shot_classes=len(checkpoint["shot_to_idx"]),
        dropout=checkpoint.get("dropout", 0.20),
    ).to(DEVICE)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()

    test_mask = meta["split"].to_numpy() == "test"
    x_test = torch.tensor(embeddings[test_mask], dtype=torch.float32).to(DEVICE)
    test_df = meta.loc[test_mask].copy().reset_index(drop=True)

    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    idx_to_shot = {i: name for name, i in shot_to_idx.items()}
    shot_true = test_df["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text_true = test_df["has_text"].to_numpy(dtype=np.int64)

    start = time.perf_counter()
    with torch.no_grad():
        shot_logits, text_logits = head(x_test)
        shot_prob = torch.softmax(shot_logits, dim=1).cpu().numpy()
        text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()
    inference_time = time.perf_counter() - start

    shot_pred = shot_prob.argmax(axis=1)
    text_pred = text_prob.argmax(axis=1)
    shot_conf = shot_prob.max(axis=1)
    text_conf = text_prob.max(axis=1)
    text_positive_prob = text_prob[:, 1]

    pred_df = test_df.copy()
    pred_df["shot_true"] = [idx_to_shot[i] for i in shot_true]
    pred_df["shot_pred"] = [idx_to_shot[i] for i in shot_pred]
    pred_df["text_true"] = text_true
    pred_df["text_pred"] = text_pred
    pred_df["shot_confidence"] = shot_conf
    pred_df["text_confidence"] = text_conf
    pred_df["text_probability"] = text_positive_prob
    pred_df["joint_true"] = [
        f"{idx_to_shot[s]}_{'text' if t == 1 else 'notext'}" for s, t in zip(shot_true, text_true)
    ]
    pred_df["joint_pred"] = [
        f"{idx_to_shot[s]}_{'text' if t == 1 else 'notext'}" for s, t in zip(shot_pred, text_pred)
    ]
    pred_df["joint_confidence"] = pred_df["shot_confidence"] * pred_df["text_confidence"]
    pred_df["shot_correct"] = shot_true == shot_pred
    pred_df["text_correct"] = text_true == text_pred
    pred_df["joint_correct"] = pred_df["shot_correct"] & pred_df["text_correct"]

    return {
        "meta": meta,
        "test_df": pred_df,
        "embeddings": embeddings,
        "test_mask": test_mask,
        "shot_true": shot_true,
        "shot_pred": shot_pred,
        "shot_prob": shot_prob,
        "text_true": text_true,
        "text_pred": text_pred,
        "text_prob": text_prob,
        "inference_time": inference_time,
        "checkpoint": checkpoint,
    }


def per_class_table(y_true, y_pred, labels, names, probs=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, name in enumerate(names):
        true_mask = y_true == labels[i]
        pred_mask = y_pred == labels[i]
        avg_conf = np.nan
        if probs is not None and len(probs):
            avg_conf = float(np.mean(np.max(probs[pred_mask], axis=1))) if np.any(pred_mask) else np.nan
        rows.append(
            {
                "class": name,
                "precision": precision[i],
                "recall_per_class_accuracy": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
                "false_positive_count": int(pred_mask.sum() - cm[i, i]),
                "false_negative_count": int(true_mask.sum() - cm[i, i]),
                "avg_prediction_confidence_for_predicted_class": avg_conf,
            }
        )
    return pd.DataFrame(rows)


def ece_mce(correct, confidence, n_bins=10):
    correct = np.asarray(correct).astype(float)
    confidence = np.asarray(confidence).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            acc = np.nan
            conf = np.nan
            gap = 0.0
        else:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            gap = abs(acc - conf)
            ece += (count / len(correct)) * gap
            mce = max(mce, gap)
        rows.append(
            {
                "bin": i + 1,
                "lower": lo,
                "upper": hi,
                "count": count,
                "accuracy": acc,
                "avg_confidence": conf,
                "gap": gap,
            }
        )
    return float(ece), float(mce), pd.DataFrame(rows)


def calibration_summary(name, correct, confidence):
    ece, mce, bins = ece_mce(correct, confidence)
    high = confidence >= 0.8
    low = confidence < 0.5
    return {
        "target": name,
        "ECE": ece,
        "MCE": mce,
        "high_confidence_accuracy": float(np.mean(correct[high])) if np.any(high) else np.nan,
        "low_confidence_accuracy": float(np.mean(correct[low])) if np.any(low) else np.nan,
        "high_confidence_count": int(high.sum()),
        "low_confidence_count": int(low.sum()),
        "overconfidence_ratio_wrong_conf_ge_0_8": float(np.mean((~correct) & high)),
        "underconfidence_ratio_correct_conf_lt_0_5": float(np.mean(correct & low)),
    }, bins


def threshold_tradeoff(pred_df):
    rows = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = pred_df["joint_confidence"].to_numpy() >= threshold
        coverage = float(mask.mean())
        reliability = float(pred_df.loc[mask, "joint_correct"].mean()) if mask.any() else np.nan
        rows.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "reliability_joint_accuracy_when_covered": reliability,
                "reliable_guide_ratio_correct_and_covered": float((mask & pred_df["joint_correct"].to_numpy()).mean()),
                "covered_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def zero_shot_predictions(data):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    shot_prompts = [
        "a close-up shot of the main subject",
        "a medium shot showing the subject and surrounding context",
        "a wide shot showing a broad space and subject placement",
        "a product or object focused shot",
        "a space or background focused shot",
    ]
    # Prompt order must match SHOT_TYPES.
    shot_prompt_order = ["close-up", "medium", "wide", "object", "space"]
    reorder = [shot_prompt_order.index(name) for name in SHOT_TYPES]

    text_prompts = ["an image without visible text", "an image with visible text or captions"]

    with torch.no_grad():
        shot_tokens = tokenizer(shot_prompts).to(DEVICE)
        shot_text_features = model.encode_text(shot_tokens)
        shot_text_features = shot_text_features / shot_text_features.norm(dim=-1, keepdim=True)
        shot_text_features = shot_text_features[reorder]

        text_tokens = tokenizer(text_prompts).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_features = torch.tensor(
        data["embeddings"][data["test_mask"]], dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        shot_logits = 100.0 * image_features @ shot_text_features.T
        text_logits = 100.0 * image_features @ text_features.T
        shot_prob = torch.softmax(shot_logits, dim=1).cpu().numpy()
        text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()

    return {
        "zero_shot_shot_pred": shot_prob.argmax(axis=1),
        "zero_shot_text_pred": text_prob.argmax(axis=1),
        "zero_shot_shot_prob": shot_prob,
        "zero_shot_text_prob": text_prob,
    }


def bootstrap_improvement(y_true, base_pred, improved_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    base_correct = np.asarray(base_pred) == np.asarray(y_true)
    improved_correct = np.asarray(improved_pred) == np.asarray(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(float(improved_correct[idx].mean() - base_correct[idx].mean()))
    diffs = np.array(diffs)
    return {
        "bootstrap_improvement_mean": float(diffs.mean()),
        "ci95_lower": float(np.quantile(diffs, 0.025)),
        "ci95_upper": float(np.quantile(diffs, 0.975)),
    }


class ExperimentHead(nn.Module):
    def __init__(self, in_dim, hidden_dims=(512, 256), dropout=0.25, out_text=True):
        super().__init__()
        layers = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        self.shared = nn.Sequential(*layers)
        self.shot_head = nn.Linear(prev, len(SHOT_TYPES))
        self.text_head = nn.Linear(prev, 2) if out_text else None

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


def class_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    return total / (num_classes * torch.clamp(counts, min=1))


def train_seed_once(seed, meta, embeddings):
    seed_everything(seed)
    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    labels = meta["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text = meta["has_text"].to_numpy(dtype=np.int64)
    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}

    def x(split):
        return torch.tensor(embeddings[masks[split]], dtype=torch.float32)

    def yshot(split):
        return torch.tensor(labels[masks[split]], dtype=torch.long)

    def ytext(split):
        return torch.tensor(text[masks[split]], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x("train"), yshot("train"), ytext("train")), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x("val"), yshot("val"), ytext("val")), batch_size=256, shuffle=False)
    test_x, test_shot, test_text = x("test").to(DEVICE), yshot("test").numpy(), ytext("test").numpy()

    model = ExperimentHead(embeddings.shape[1]).to(DEVICE)
    shot_loss = nn.CrossEntropyLoss(weight=class_weights(yshot("train"), len(SHOT_TYPES)).to(DEVICE))
    text_loss = nn.CrossEntropyLoss(weight=class_weights(ytext("train"), 2).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state = None
    best_joint = -1
    patience = 12
    for _epoch in range(1, 81):
        model.train()
        for xb, ys, yt in train_loader:
            optimizer.zero_grad()
            slog, tlog = model(xb.to(DEVICE))
            loss = shot_loss(slog, ys.to(DEVICE)) + text_loss(tlog, yt.to(DEVICE))
            loss.backward()
            optimizer.step()

        model.eval()
        val_true_s, val_pred_s, val_true_t, val_pred_t = [], [], [], []
        with torch.no_grad():
            for xb, ys, yt in val_loader:
                slog, tlog = model(xb.to(DEVICE))
                val_true_s.extend(ys.numpy().tolist())
                val_true_t.extend(yt.numpy().tolist())
                val_pred_s.extend(slog.argmax(1).cpu().numpy().tolist())
                val_pred_t.extend(tlog.argmax(1).cpu().numpy().tolist())
        joint = float(np.mean((np.array(val_true_s) == np.array(val_pred_s)) & (np.array(val_true_t) == np.array(val_pred_t))))
        if joint > best_joint:
            best_joint = joint
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 12
        else:
            patience -= 1
            if patience <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        slog, tlog = model(test_x)
        pred_s = slog.argmax(1).cpu().numpy()
        pred_t = tlog.argmax(1).cpu().numpy()
    return {
        "seed": seed,
        "shot_accuracy": accuracy_score(test_shot, pred_s),
        "shot_macro_f1": f1_score(test_shot, pred_s, average="macro", zero_division=0),
        "shot_balanced_accuracy": balanced_accuracy_score(test_shot, pred_s),
        "shot_weighted_f1": f1_score(test_shot, pred_s, average="weighted", zero_division=0),
        "text_accuracy": accuracy_score(test_text, pred_t),
        "text_macro_f1": f1_score(test_text, pred_t, average="macro", zero_division=0),
        "joint_accuracy": float(np.mean((test_shot == pred_s) & (test_text == pred_t))),
    }


def scene_output_statistics():
    roots = [ROOT / "outputs" / "instagram_overlay", ROOT / "outputs" / "final_instagram_pipeline"]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/scene_predictions.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            shot_seq = df["pred_shot_type"].astype(str).tolist()
            durations = df["duration_sec"].astype(float).to_numpy() if "duration_sec" in df else np.array([])
            text_ratio = float(df["pred_has_text"].mean()) if "pred_has_text" in df else np.nan
            transition_count = int(np.sum(np.array(shot_seq[1:]) != np.array(shot_seq[:-1]))) if len(shot_seq) > 1 else 0
            repeated_ratio = 1.0 - transition_count / max(len(shot_seq) - 1, 1) if len(shot_seq) > 1 else np.nan
            avg_duration = float(np.mean(durations)) if len(durations) else np.nan
            fast_cut_ratio = float(np.mean(durations < 1.0)) if len(durations) else np.nan
            rows.append(
                {
                    "source_root": root.name,
                    "video_id": path.parent.name,
                    "scene_count": int(len(df)),
                    "opening_shot_type": shot_seq[0],
                    "ending_shot_type": shot_seq[-1],
                    "dominant_shot_type": df["pred_shot_type"].value_counts().idxmax(),
                    "avg_scene_duration": avg_duration,
                    "fast_cut_ratio_duration_lt_1s": fast_cut_ratio,
                    "text_cut_ratio": text_ratio,
                    "repeated_shot_ratio": repeated_ratio,
                    "transition_count": transition_count,
                    "text_density_level": "high" if text_ratio >= 0.7 else ("medium" if text_ratio >= 0.3 else "low"),
                    "cut_speed_category": "fast" if avg_duration < 1.0 else ("medium" if avg_duration < 2.0 else "slow"),
                }
            )
    return pd.DataFrame(rows)


def main():
    print("loading final model predictions...")
    data = load_final_predictions()
    pred_df = data["test_df"]
    pred_df.to_csv(OUT_DIR / "final_model_test_predictions_with_confidence.csv", index=False, encoding="utf-8-sig")

    shot_true = data["shot_true"]
    shot_pred = data["shot_pred"]
    shot_prob = data["shot_prob"]
    text_true = data["text_true"]
    text_pred = data["text_pred"]
    text_prob = data["text_prob"]

    shot_correct = shot_true == shot_pred
    text_correct = text_true == text_pred
    joint_correct = shot_correct & text_correct

    basic_rows = [
        {
            "classifier": "shot",
            "accuracy": accuracy_score(shot_true, shot_pred),
            "macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(shot_true, shot_pred),
            "weighted_f1": f1_score(shot_true, shot_pred, average="weighted", zero_division=0),
            "top_2_accuracy": top_k_accuracy_score(shot_true, shot_prob, k=2, labels=list(range(len(SHOT_TYPES)))),
            "macro_precision": precision_recall_fscore_support(shot_true, shot_pred, average="macro", zero_division=0)[0],
            "macro_recall": precision_recall_fscore_support(shot_true, shot_pred, average="macro", zero_division=0)[1],
        },
        {
            "classifier": "text",
            "accuracy": accuracy_score(text_true, text_pred),
            "macro_f1": f1_score(text_true, text_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(text_true, text_pred),
            "weighted_f1": f1_score(text_true, text_pred, average="weighted", zero_division=0),
            "top_2_accuracy": np.nan,
            "macro_precision": precision_recall_fscore_support(text_true, text_pred, average="macro", zero_division=0)[0],
            "macro_recall": precision_recall_fscore_support(text_true, text_pred, average="macro", zero_division=0)[1],
        },
    ]
    pd.DataFrame(basic_rows).to_csv(OUT_DIR / "01_basic_classifier_metrics.csv", index=False, encoding="utf-8-sig")

    per_class_table(shot_true, shot_pred, list(range(len(SHOT_TYPES))), SHOT_TYPES, shot_prob).to_csv(
        OUT_DIR / "01_shot_per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )
    per_class_table(text_true, text_pred, [0, 1], TEXT_TYPES, text_prob).to_csv(
        OUT_DIR / "01_text_per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )

    print("computing zero-shot comparison...")
    zs = zero_shot_predictions(data)
    zs_rows = []
    for target, y_true, zs_pred, ft_pred in [
        ("shot", shot_true, zs["zero_shot_shot_pred"], shot_pred),
        ("text", text_true, zs["zero_shot_text_pred"], text_pred),
    ]:
        boot = bootstrap_improvement(y_true, zs_pred, ft_pred)
        zs_rows.append(
            {
                "target": target,
                "zero_shot_accuracy": accuracy_score(y_true, zs_pred),
                "fine_tuned_accuracy": accuracy_score(y_true, ft_pred),
                "improvement_pp": (accuracy_score(y_true, ft_pred) - accuracy_score(y_true, zs_pred)) * 100,
                **boot,
            }
        )
    pd.DataFrame(zs_rows).to_csv(OUT_DIR / "02_zero_shot_vs_finetuned.csv", index=False, encoding="utf-8-sig")

    joint_micro = f1_score(pred_df["joint_true"], pred_df["joint_pred"], labels=JOINT_LABELS, average="micro", zero_division=0)
    joint_macro = f1_score(pred_df["joint_true"], pred_df["joint_pred"], labels=JOINT_LABELS, average="macro", zero_division=0)
    joint_weighted = f1_score(pred_df["joint_true"], pred_df["joint_pred"], labels=JOINT_LABELS, average="weighted", zero_division=0)
    joint_rows = [
        {
            "joint_accuracy": accuracy_score(pred_df["joint_true"], pred_df["joint_pred"]),
            "joint_micro_f1": joint_micro,
            "joint_macro_f1": joint_macro,
            "joint_weighted_f1": joint_weighted,
            "guide_reliability_score": float(joint_correct.mean()),
            "conditional_text_given_shot": float(text_correct[shot_correct].mean()),
            "conditional_shot_given_text": float(shot_correct[text_correct].mean()),
        }
    ]
    pd.DataFrame(joint_rows).to_csv(OUT_DIR / "03_joint_reliability.csv", index=False, encoding="utf-8-sig")

    error_rows = [
        {"error_type": "both_correct", "ratio": float((shot_correct & text_correct).mean()), "count": int((shot_correct & text_correct).sum())},
        {"error_type": "shot_only_error", "ratio": float((~shot_correct & text_correct).mean()), "count": int((~shot_correct & text_correct).sum())},
        {"error_type": "text_only_error", "ratio": float((shot_correct & ~text_correct).mean()), "count": int((shot_correct & ~text_correct).sum())},
        {"error_type": "both_wrong", "ratio": float((~shot_correct & ~text_correct).mean()), "count": int((~shot_correct & ~text_correct).sum())},
    ]
    pd.DataFrame(error_rows).to_csv(OUT_DIR / "04_error_propagation.csv", index=False, encoding="utf-8-sig")

    shot_cal, shot_bins = calibration_summary("shot", shot_correct, shot_prob.max(axis=1))
    text_cal, text_bins = calibration_summary("text", text_correct, text_prob.max(axis=1))
    joint_cal, joint_bins = calibration_summary("joint", np.asarray(joint_correct), pred_df["joint_confidence"].to_numpy())
    pd.DataFrame([shot_cal, text_cal, joint_cal]).to_csv(OUT_DIR / "05_calibration_summary.csv", index=False, encoding="utf-8-sig")
    shot_bins.to_csv(OUT_DIR / "05_shot_confidence_histogram.csv", index=False, encoding="utf-8-sig")
    text_bins.to_csv(OUT_DIR / "05_text_confidence_histogram.csv", index=False, encoding="utf-8-sig")
    joint_bins.to_csv(OUT_DIR / "05_joint_confidence_histogram.csv", index=False, encoding="utf-8-sig")

    class_ece_rows = []
    for i, name in enumerate(SHOT_TYPES):
        mask = shot_pred == i
        if mask.any():
            ece, mce, _ = ece_mce(shot_correct[mask], shot_prob.max(axis=1)[mask])
            class_ece_rows.append({"class": name, "ECE": ece, "MCE": mce, "predicted_count": int(mask.sum())})
    pd.DataFrame(class_ece_rows).to_csv(OUT_DIR / "05_class_wise_ece.csv", index=False, encoding="utf-8-sig")

    threshold_tradeoff(pred_df).to_csv(OUT_DIR / "06_confidence_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")

    # Per-class, confusion, and minority analysis.
    shot_pc = per_class_table(shot_true, shot_pred, list(range(len(SHOT_TYPES))), SHOT_TYPES, shot_prob)
    minority = shot_pc.sort_values("support").head(2).copy()
    minority["minority_class_recall"] = minority["recall_per_class_accuracy"]
    minority.to_csv(OUT_DIR / "07_minority_class_analysis.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(shot_true, shot_pred, labels=list(range(len(SHOT_TYPES))))
    cm_df = pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES)
    cm_df.to_csv(OUT_DIR / "08_shot_confusion_matrix.csv", encoding="utf-8-sig")
    pairs = []
    for i, true_name in enumerate(SHOT_TYPES):
        row_total = cm[i].sum()
        for j, pred_name in enumerate(SHOT_TYPES):
            if i == j:
                continue
            pairs.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(cm[i, j]),
                    "ratio_within_true": float(cm[i, j] / row_total) if row_total else np.nan,
                }
            )
    pd.DataFrame(pairs).sort_values(["count", "ratio_within_true"], ascending=False).to_csv(
        OUT_DIR / "08_top_confusion_pairs.csv", index=False, encoding="utf-8-sig"
    )
    pred_df.sort_values("joint_confidence", ascending=False).query("joint_correct == False").head(30).to_csv(
        OUT_DIR / "08_high_confidence_errors.csv", index=False, encoding="utf-8-sig"
    )
    pred_df.sort_values("joint_confidence", ascending=True).head(30).to_csv(
        OUT_DIR / "08_low_confidence_samples.csv", index=False, encoding="utf-8-sig"
    )

    print("running multi-seed stability on cached embeddings...")
    seeds = [1, 2, 3, 4, 5]
    seed_rows = [train_seed_once(seed, data["meta"], data["embeddings"]) for seed in seeds]
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "09_multi_seed_results.csv", index=False, encoding="utf-8-sig")
    seed_summary = seed_df.drop(columns=["seed"]).agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    seed_summary.to_csv(OUT_DIR / "09_multi_seed_stability_summary.csv", index=False, encoding="utf-8-sig")

    meta = data["meta"]
    dataset_stats = {
        "total_labeled_cuts_used": int(len(meta)),
        "number_of_videos": int(meta["video_id"].nunique()),
        "train_count": int((meta["split"] == "train").sum()),
        "val_count": int((meta["split"] == "val").sum()),
        "test_count": int((meta["split"] == "test").sum()),
    }
    with open(OUT_DIR / "10_dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
    pd.crosstab(meta["shot_type"], meta["split"]).to_csv(OUT_DIR / "10_shot_class_distribution_by_split.csv", encoding="utf-8-sig")
    pd.crosstab(meta["joint_label"], meta["split"]).to_csv(OUT_DIR / "10_joint_class_distribution_by_split.csv", encoding="utf-8-sig")

    scene_stats = scene_output_statistics()
    scene_stats.to_csv(OUT_DIR / "18_temporal_structure_summary.csv", index=False, encoding="utf-8-sig")
    if not scene_stats.empty:
        scene_stats[["video_id", "scene_count", "avg_scene_duration", "fast_cut_ratio_duration_lt_1s"]].to_csv(
            OUT_DIR / "11_scene_detection_available_statistics.csv", index=False, encoding="utf-8-sig"
        )

    runtime_rows = [
        {
            "metric": "classifier_inference_time_on_test_embeddings_sec",
            "value": data["inference_time"],
            "note": "final head inference only, excludes CLIP embedding extraction and overlay rendering",
        },
        {
            "metric": "classifier_throughput_samples_per_sec",
            "value": len(pred_df) / data["inference_time"] if data["inference_time"] > 0 else np.nan,
            "note": "test samples / classifier inference time",
        },
    ]
    pd.DataFrame(runtime_rows).to_csv(OUT_DIR / "12_runtime_efficiency_partial.csv", index=False, encoding="utf-8-sig")

    if (EXPERIMENT_DIR / "model_experiment_overall_summary.csv").exists():
        exp = pd.read_csv(EXPERIMENT_DIR / "model_experiment_overall_summary.csv")
        exp.to_csv(OUT_DIR / "13_backbone_ablation_summary.csv", index=False, encoding="utf-8-sig")

    per_video = pred_df.groupby("video_id").apply(
        lambda g: pd.Series(
            {
                "sample_count": len(g),
                "per_video_shot_accuracy": float(g["shot_correct"].mean()),
                "per_video_joint_accuracy": float(g["joint_correct"].mean()),
                "per_video_avg_joint_confidence": float(g["joint_confidence"].mean()),
            }
        ),
        include_groups=False,
    ).reset_index()
    per_video.to_csv(OUT_DIR / "15_per_video_variance.csv", index=False, encoding="utf-8-sig")
    per_video.drop(columns=["video_id"]).agg(["mean", "std", "min", "max"]).to_csv(
        OUT_DIR / "15_per_video_variance_summary.csv", encoding="utf-8-sig"
    )
    per_video.sort_values("per_video_joint_accuracy").head(10).to_csv(
        OUT_DIR / "15_hardest_videos.csv", index=False, encoding="utf-8-sig"
    )
    per_video.sort_values("per_video_joint_accuracy", ascending=False).head(10).to_csv(
        OUT_DIR / "15_easiest_videos.csv", index=False, encoding="utf-8-sig"
    )

    deployment_rows = [
        {"metric": "end_to_end_reliability", "value": float(joint_correct.mean()), "note": "same as joint accuracy"},
        {
            "metric": "confidence_filtered_reliability_at_0_7",
            "value": float(pred_df.loc[pred_df["joint_confidence"] >= 0.7, "joint_correct"].mean()),
            "note": "joint accuracy where joint confidence >= 0.7",
        },
        {
            "metric": "guide_reliability",
            "value": float(joint_correct.mean()),
            "note": "guide is considered reliable when both shot and text are correct",
        },
        {
            "metric": "error_propagation_gap_shot_acc_minus_joint_acc",
            "value": accuracy_score(shot_true, shot_pred) - float(joint_correct.mean()),
            "note": "drop from shot-only correctness to end-to-end correctness",
        },
    ]
    pd.DataFrame(deployment_rows).to_csv(OUT_DIR / "19_reliability_deployment_metrics.csv", index=False, encoding="utf-8-sig")

    reproducibility = {
        "random_seed_main": 42,
        "multi_seed_values": seeds,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(DEVICE),
        "pipeline_execution_status": "metrics computed successfully",
    }
    with open(OUT_DIR / "20_reproducibility_system_info.json", "w", encoding="utf-8") as f:
        json.dump(reproducibility, f, ensure_ascii=False, indent=2)

    unavailable = [
        {"section": "11_scene_detection_metrics", "metric": "mismatch_ratio", "reason": "manual scene boundary ground truth is not available"},
        {"section": "11_scene_detection_metrics", "metric": "avg_absolute_count_difference", "reason": "manual scene count ground truth is not available"},
        {"section": "11_scene_detection_metrics", "metric": "over_detected_count/under_detected_count", "reason": "manual scene count ground truth is not available"},
        {"section": "12_runtime_efficiency", "metric": "embedding_extraction_time/overlay_rendering_time/total_runtime", "reason": "runtime logging was not captured during pipeline execution"},
        {"section": "14_data_size_experiment", "metric": "performance_vs_train_size", "reason": "subsample training experiments were not run"},
        {"section": "16_annotation_label_quality", "metric": "raw_agreement/Cohen_Kappa", "reason": "second annotator labels are not available"},
        {"section": "17_human_evaluation", "metric": "overlay_usefulness/guide_helpfulness", "reason": "human survey results are not available"},
    ]
    pd.DataFrame(unavailable).to_csv(OUT_DIR / "00_unavailable_or_placeholder_metrics.csv", index=False, encoding="utf-8-sig")

    final_summary = {
        "shot_accuracy": basic_rows[0]["accuracy"],
        "shot_macro_f1": basic_rows[0]["macro_f1"],
        "shot_balanced_accuracy": basic_rows[0]["balanced_accuracy"],
        "shot_weighted_f1": basic_rows[0]["weighted_f1"],
        "shot_top2_accuracy": basic_rows[0]["top_2_accuracy"],
        "text_accuracy": basic_rows[1]["accuracy"],
        "text_macro_f1": basic_rows[1]["macro_f1"],
        "text_weighted_f1": basic_rows[1]["weighted_f1"],
        "joint_accuracy": joint_rows[0]["joint_accuracy"],
        "joint_macro_f1": joint_macro,
        "joint_weighted_f1": joint_weighted,
        "joint_micro_f1": joint_micro,
    }
    with open(OUT_DIR / "final_metric_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    md = [
        "# Comprehensive Metrics Summary",
        "",
        "## Final Model Core Metrics",
        "",
    ]
    for key, value in final_summary.items():
        md.append(f"- {key}: {value:.4f}")
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- Zero-shot metrics use prompt-based CLIP text similarity on the same test split.",
            "- Scene mismatch, annotation agreement, human evaluation, and full runtime metrics require extra ground truth or logs and are listed in `00_unavailable_or_placeholder_metrics.csv`.",
            "- All generated CSV files are saved under `outputs/model_experiments/comprehensive_metrics/`.",
        ]
    )
    (OUT_DIR / "comprehensive_metrics_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("saved comprehensive metrics to:", OUT_DIR)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
