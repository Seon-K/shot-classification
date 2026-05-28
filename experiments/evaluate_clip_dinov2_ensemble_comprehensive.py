from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from run_clip_dinov2_ensemble_experiment import (
    DEVICE,
    SEED,
    SHOT_TYPES,
    DINO_DIR,
    CLIP_DIR,
    OUTPUT_DIR,
    metric_row,
    multitask_predict,
    shot_predict,
    train_clip,
    train_dino,
)


METRICS_DIR = OUTPUT_DIR / "comprehensive_metrics" / "clip_dinov2_ensemble_best_alpha"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def top2_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(prob, axis=1)[:, -2:]
    return float(np.mean([truth in row for truth, row in zip(y_true, order)]))


def binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def tune_text_threshold(text_true: np.ndarray, text_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (text_prob[:, 1] >= float(threshold)).astype(np.int64)
        score = binary_macro_f1(text_true, pred)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    total = len(y_true)
    correct = (y_true == y_pred).astype(float)
    for start, end in zip(edges[:-1], edges[1:]):
        if end == 1.0:
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)
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


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None, labels: list[str], prefix: str) -> dict[str, float]:
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_macro_precision": precision,
        f"{prefix}_macro_recall": recall,
    }
    if prob is not None and len(labels) > 2:
        out[f"{prefix}_top2_accuracy"] = top2_accuracy(y_true, prob)
    return out


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None, labels: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    rows = []
    median_support = np.median(support) if len(support) else 0
    for idx, label in enumerate(labels):
        pred_mask = y_pred == idx
        true_mask = y_true == idx
        confidence = float(prob[pred_mask, idx].mean()) if prob is not None and pred_mask.any() else np.nan
        rows.append(
            {
                "class": label,
                "precision": precision[idx],
                "recall_per_class_accuracy": recall[idx],
                "f1": f1[idx],
                "support": int(support[idx]),
                "false_positive_count": int(np.sum(pred_mask & ~true_mask)),
                "false_negative_count": int(np.sum(true_mask & ~pred_mask)),
                "avg_prediction_confidence_for_predicted_class": confidence,
                "minority_class_recall": recall[idx] if support[idx] <= median_support else np.nan,
            }
        )
    return pd.DataFrame(rows)


def top_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    rows = []
    for true_idx, true_label in enumerate(labels):
        total = int(cm[true_idx].sum())
        for pred_idx, pred_label in enumerate(labels):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            rows.append(
                {
                    "true_class": true_label,
                    "pred_class": pred_label,
                    "count": count,
                    "ratio_within_true": safe_divide(count, total),
                }
            )
    return pd.DataFrame(rows).sort_values(["count", "ratio_within_true"], ascending=False)


def threshold_tradeoff(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        covered = pred_df["joint_confidence"] >= threshold
        covered_count = int(covered.sum())
        reliability = float(pred_df.loc[covered, "joint_correct"].mean()) if covered_count else np.nan
        rows.append(
            {
                "threshold": threshold,
                "coverage": safe_divide(covered_count, len(pred_df)),
                "reliability_joint_accuracy_when_covered": reliability,
                "reliable_guide_ratio_correct_and_covered": safe_divide(int((covered & pred_df["joint_correct"]).sum()), len(pred_df)),
                "covered_count": covered_count,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_frame(meta: pd.DataFrame, test_mask: np.ndarray, pred: dict[str, np.ndarray]) -> pd.DataFrame:
    pred_df = meta[test_mask].copy().reset_index(drop=True)
    pred_df["shot_true_idx"] = pred["shot_true"]
    pred_df["shot_pred_idx"] = pred["shot_pred"]
    pred_df["shot_pred"] = [SHOT_TYPES[i] for i in pred["shot_pred"]]
    pred_df["text_true"] = pred["text_true"]
    pred_df["text_pred"] = pred["text_pred"]
    pred_df["text_probability"] = pred["text_prob"][:, 1]
    pred_df["shot_confidence"] = pred["shot_prob"].max(axis=1)
    pred_df["text_confidence"] = np.maximum(pred["text_prob"][:, 1], 1.0 - pred["text_prob"][:, 1])
    pred_df["joint_confidence"] = pred_df["shot_confidence"] * pred_df["text_confidence"]
    pred_df["shot_correct"] = pred_df["shot_true_idx"] == pred_df["shot_pred_idx"]
    pred_df["text_correct"] = pred_df["text_true"] == pred_df["text_pred"]
    pred_df["joint_correct"] = pred_df["shot_correct"] & pred_df["text_correct"]
    for i, label in enumerate(SHOT_TYPES):
        pred_df[f"prob_shot_{label}"] = pred["shot_prob"][:, i]
    pred_df.to_csv(OUTPUT_DIR / "ensemble_alpha_clip_0.45_test_predictions_with_confidence.csv", index=False, encoding="utf-8-sig")
    return pred_df


def error_propagation(pred_df: pd.DataFrame) -> pd.DataFrame:
    shot_correct = pred_df["shot_correct"].to_numpy(dtype=bool)
    text_correct = pred_df["text_correct"].to_numpy(dtype=bool)
    total = len(pred_df)
    return pd.DataFrame(
        [
            {
                "both_correct_ratio": safe_divide(int(np.sum(shot_correct & text_correct)), total),
                "shot_only_error_ratio": safe_divide(int(np.sum(~shot_correct & text_correct)), total),
                "text_only_error_ratio": safe_divide(int(np.sum(shot_correct & ~text_correct)), total),
                "both_wrong_ratio": safe_divide(int(np.sum(~shot_correct & ~text_correct)), total),
            }
        ]
    )


def joint_reliability(pred_df: pd.DataFrame) -> pd.DataFrame:
    shot_correct = pred_df["shot_correct"].to_numpy(dtype=bool)
    text_correct = pred_df["text_correct"].to_numpy(dtype=bool)
    joint_correct = shot_correct & text_correct
    return pd.DataFrame(
        [
            {
                "joint_accuracy": float(joint_correct.mean()),
                "guide_reliability_score": float(joint_correct.mean()),
                "conditional_text_given_shot": safe_divide(int(np.sum(shot_correct & text_correct)), int(np.sum(shot_correct))),
                "conditional_shot_given_text": safe_divide(int(np.sum(shot_correct & text_correct)), int(np.sum(text_correct))),
            }
        ]
    )


def calibration_tables(pred_df: pd.DataFrame, pred: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    shot_true = pred["shot_true"]
    shot_pred = pred["shot_pred"]
    shot_conf = pred["shot_prob"].max(axis=1)
    text_true = pred["text_true"]
    text_pred = pred["text_pred"]
    text_conf = np.maximum(pred["text_prob"][:, 1], 1.0 - pred["text_prob"][:, 1])
    joint_true = pred_df["joint_correct"].astype(int).to_numpy()
    joint_pred = np.ones_like(joint_true)
    joint_conf = pred_df["joint_confidence"].to_numpy()

    summary_rows = []
    histograms = {}
    for name, y_true, y_pred, conf in [
        ("shot", shot_true, shot_pred, shot_conf),
        ("text", text_true, text_pred, text_conf),
        ("joint", joint_true, joint_pred, joint_conf),
    ]:
        ece, mce, hist = expected_calibration_error(y_true, y_pred, conf)
        correct = y_true == y_pred
        high = conf >= 0.8
        low = conf < 0.5
        summary_rows.append(
            {
                "target": name,
                "ECE": ece,
                "MCE": mce,
                "high_confidence_accuracy": float(correct[high].mean()) if high.any() else np.nan,
                "low_confidence_accuracy": float(correct[low].mean()) if low.any() else np.nan,
                "high_confidence_count": int(high.sum()),
                "low_confidence_count": int(low.sum()),
                "overconfidence_ratio_wrong_conf_ge_0_8": safe_divide(int(np.sum(~correct & high)), len(correct)),
                "underconfidence_ratio_correct_conf_lt_0_5": safe_divide(int(np.sum(correct & low)), len(correct)),
            }
        )
        histograms[name] = hist
    return pd.DataFrame(summary_rows), histograms


def per_video_variance(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = pred_df.groupby("video_id", dropna=False).agg(
        sample_count=("filename", "count"),
        per_video_shot_accuracy=("shot_correct", "mean"),
        per_video_joint_accuracy=("joint_correct", "mean"),
        per_video_confidence=("joint_confidence", "mean"),
    ).reset_index()
    summary = pd.DataFrame(
        [
            {
                "video_count": int(grouped["video_id"].nunique()),
                "video_variance_std": float(grouped["per_video_joint_accuracy"].std(ddof=0)) if len(grouped) else np.nan,
                "mean_per_video_joint_accuracy": float(grouped["per_video_joint_accuracy"].mean()) if len(grouped) else np.nan,
            }
        ]
    )
    hardest = grouped.sort_values(["per_video_joint_accuracy", "sample_count"], ascending=[True, False]).head(10)
    easiest = grouped.sort_values(["per_video_joint_accuracy", "sample_count"], ascending=[False, False]).head(10)
    return grouped, summary, hardest, easiest


def dataset_statistics(meta: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stats = {
        "total_labeled_cuts": int(len(meta)),
        "number_of_videos": int(meta["video_id"].nunique()),
        "scene_count": int(len(meta)),
        "avg_scenes_per_video": float(len(meta) / max(meta["video_id"].nunique(), 1)),
        "avg_scene_duration": None,
    }
    split_dist = meta["split"].value_counts().rename_axis("split").reset_index(name="count")
    class_dist = meta.groupby(["shot_type", "has_text"]).size().reset_index(name="count")
    return stats, split_dist, class_dist


def temporal_structure(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for video_id, group in pred_df.sort_values(["video_id", "filename"]).groupby("video_id", dropna=False):
        shots = group["shot_pred"].tolist()
        rows.append(
            {
                "video_id": video_id,
                "opening_shot_type": shots[0] if shots else "",
                "ending_shot_type": shots[-1] if shots else "",
                "dominant_shot_type": group["shot_pred"].mode().iloc[0] if len(group) else "",
                "fast_cut_ratio": np.nan,
                "text_cut_ratio": float(group["text_pred"].mean()) if len(group) else np.nan,
                "repeated_shot_ratio": safe_divide(sum(a == b for a, b in zip(shots, shots[1:])), max(len(shots) - 1, 0)),
                "transition_count": max(len(shots) - 1, 0),
                "text_density_level": "high" if float(group["text_pred"].mean()) >= 0.5 else "low",
                "cut_speed_category": "not_available_from_labeled_frames",
            }
        )
    return pd.DataFrame(rows)


def save_placeholder_metrics() -> None:
    placeholders = [
        ("02_zero_shot_vs_finetuned.csv", "zero-shot CLIP text prompt baseline is not computed in the ensemble script"),
        ("11_scene_detection_metrics.csv", "scene detection requires video-level manual cut labels"),
        ("14_data_size_experiment.csv", "train-size scaling is not run in this script"),
        ("16_annotation_label_quality.csv", "annotator2 labels are not available"),
        ("17_human_evaluation.csv", "human survey results are not available"),
    ]
    for filename, reason in placeholders:
        pd.DataFrame([{"metric_status": "placeholder", "reason": reason}]).to_csv(METRICS_DIR / filename, index=False, encoding="utf-8-sig")


def save_comprehensive_metrics(meta: pd.DataFrame, pred_df: pd.DataFrame, pred: dict[str, np.ndarray], runtime: dict[str, float], alpha_rows: list[dict], rows: list[dict]) -> dict[str, float]:
    shot_metrics = classifier_metrics(pred["shot_true"], pred["shot_pred"], pred["shot_prob"], SHOT_TYPES, "shot")
    text_metrics = classifier_metrics(pred["text_true"], pred["text_pred"], pred["text_prob"], ["notext", "text"], "text")
    basic = {**shot_metrics, **text_metrics}
    basic.update(
        {
            "joint_accuracy": float(pred_df["joint_correct"].mean()),
            "joint_macro_f1": f1_score(pred_df["joint_label"], pred_df["joint_pred_label"], average="macro", zero_division=0),
            "joint_weighted_f1": f1_score(pred_df["joint_label"], pred_df["joint_pred_label"], average="weighted", zero_division=0),
            "best_alpha_clip": runtime["best_alpha_clip"],
        }
    )
    pd.DataFrame([basic]).to_csv(METRICS_DIR / "01_basic_classifier_metrics.csv", index=False, encoding="utf-8-sig")
    per_class_metrics(pred["shot_true"], pred["shot_pred"], pred["shot_prob"], SHOT_TYPES).to_csv(METRICS_DIR / "01_shot_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    per_class_metrics(pred["text_true"], pred["text_pred"], pred["text_prob"], ["notext", "text"]).to_csv(METRICS_DIR / "01_text_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    joint_reliability(pred_df).to_csv(METRICS_DIR / "03_joint_reliability.csv", index=False, encoding="utf-8-sig")
    error_propagation(pred_df).to_csv(METRICS_DIR / "04_error_propagation.csv", index=False, encoding="utf-8-sig")

    calibration_summary, histograms = calibration_tables(pred_df, pred)
    calibration_summary.to_csv(METRICS_DIR / "05_calibration_summary.csv", index=False, encoding="utf-8-sig")
    for name, table in histograms.items():
        table.to_csv(METRICS_DIR / f"05_{name}_confidence_histogram.csv", index=False, encoding="utf-8-sig")

    threshold_tradeoff(pred_df).to_csv(METRICS_DIR / "06_confidence_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")
    minority = per_class_metrics(pred["shot_true"], pred["shot_pred"], pred["shot_prob"], SHOT_TYPES)
    minority.sort_values("support").head(2).to_csv(METRICS_DIR / "07_minority_class_analysis.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(pred["shot_true"], pred["shot_pred"], labels=list(range(len(SHOT_TYPES))))
    pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES).to_csv(METRICS_DIR / "08_shot_confusion_matrix.csv", encoding="utf-8-sig")
    top_confusion_pairs(pred["shot_true"], pred["shot_pred"], SHOT_TYPES).to_csv(METRICS_DIR / "08_top_confusion_pairs.csv", index=False, encoding="utf-8-sig")
    pred_df[~pred_df["joint_correct"]].sort_values("joint_confidence", ascending=False).head(50).to_csv(METRICS_DIR / "08_high_confidence_errors.csv", index=False, encoding="utf-8-sig")
    pred_df.sort_values("joint_confidence", ascending=True).head(50).to_csv(METRICS_DIR / "08_low_confidence_samples.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(pred_df["joint_label"], pred_df["joint_pred_label"]).to_csv(METRICS_DIR / "08_joint_confusion.csv", encoding="utf-8-sig")

    stability = pd.DataFrame([{f"{key}_mean": value for key, value in basic.items() if isinstance(value, (int, float, np.floating))}])
    for key in ["shot_accuracy", "shot_macro_f1", "shot_balanced_accuracy", "shot_weighted_f1"]:
        if f"{key}_mean" in stability.columns:
            stability[f"{key}_std"] = 0.0
    stability["seed_count"] = 1
    stability.to_csv(METRICS_DIR / "09_multi_seed_stability_summary.csv", index=False, encoding="utf-8-sig")

    stats, split_dist, class_dist = dataset_statistics(meta)
    with open(METRICS_DIR / "10_dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    split_dist.to_csv(METRICS_DIR / "10_train_val_test_split.csv", index=False, encoding="utf-8-sig")
    class_dist.to_csv(METRICS_DIR / "10_class_distribution.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([runtime]).to_csv(METRICS_DIR / "12_runtime_efficiency.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(METRICS_DIR / "13_backbone_ablation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(alpha_rows).to_csv(METRICS_DIR / "13_alpha_ablation.csv", index=False, encoding="utf-8-sig")

    video_rows, video_summary, hardest, easiest = per_video_variance(pred_df)
    video_rows.to_csv(METRICS_DIR / "15_per_video_variance.csv", index=False, encoding="utf-8-sig")
    video_summary.to_csv(METRICS_DIR / "15_per_video_variance_summary.csv", index=False, encoding="utf-8-sig")
    hardest.to_csv(METRICS_DIR / "15_hardest_videos.csv", index=False, encoding="utf-8-sig")
    easiest.to_csv(METRICS_DIR / "15_easiest_videos.csv", index=False, encoding="utf-8-sig")

    temporal_structure(pred_df).to_csv(METRICS_DIR / "18_temporal_structure_summary.csv", index=False, encoding="utf-8-sig")
    tradeoff = threshold_tradeoff(pred_df)
    pd.DataFrame(
        [
            {
                "end_to_end_reliability": basic["joint_accuracy"],
                "confidence_filtered_reliability_at_0_8": tradeoff.loc[tradeoff["threshold"] == 0.8, "reliability_joint_accuracy_when_covered"].iloc[0],
                "guide_reliability": basic["joint_accuracy"],
                "pipeline_consistency": "classifier_only_evaluated",
                "error_propagation_gap": 1.0 - basic["joint_accuracy"],
            }
        ]
    ).to_csv(METRICS_DIR / "19_reliability_deployment_metrics.csv", index=False, encoding="utf-8-sig")

    with open(METRICS_DIR / "20_reproducibility_system_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": str(DEVICE),
                "pipeline_execution_status": "ensemble comprehensive metrics generated",
            },
            f,
            indent=2,
        )

    save_placeholder_metrics()
    with open(METRICS_DIR / "final_metric_summary.json", "w", encoding="utf-8") as f:
        json.dump(basic, f, indent=2)
    return basic


def main() -> None:
    seed_everything()
    start_time = time.perf_counter()
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

    fit_start = time.perf_counter()
    clip_model = train_clip(
        tx(clip_emb[masks["train"]]),
        ty(labels[masks["train"]]),
        ty(text[masks["train"]]),
        tx(clip_emb[masks["val"]]),
        ty(labels[masks["val"]]),
        ty(text[masks["val"]]),
    )
    dino_model = train_dino(
        tx(dino_emb[masks["train"]]),
        ty(labels[masks["train"]]),
        tx(dino_emb[masks["val"]]),
        ty(labels[masks["val"]]),
    )
    fit_time_sec = time.perf_counter() - fit_start

    clip_val_loader = DataLoader(TensorDataset(tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]])), batch_size=256, shuffle=False)
    clip_test_loader = DataLoader(TensorDataset(tx(clip_emb[masks["test"]]), ty(labels[masks["test"]]), ty(text[masks["test"]])), batch_size=256, shuffle=False)
    dino_val_loader = DataLoader(TensorDataset(tx(dino_emb[masks["val"]]), ty(labels[masks["val"]])), batch_size=256)
    dino_test_loader = DataLoader(TensorDataset(tx(dino_emb[masks["test"]]), ty(labels[masks["test"]])), batch_size=256)

    infer_start = time.perf_counter()
    shot_val_true, text_val_true, clip_val_shot_prob, clip_val_text_prob = multitask_predict(clip_model, clip_val_loader)
    shot_test_true, text_test_true, clip_test_shot_prob, clip_test_text_prob = multitask_predict(clip_model, clip_test_loader)
    _, dino_val_shot_prob = shot_predict(dino_model, dino_val_loader)
    _, dino_test_shot_prob = shot_predict(dino_model, dino_test_loader)
    infer_time_sec = time.perf_counter() - infer_start

    alpha_rows = []
    best_alpha = 0.0
    best_macro = -1.0
    for alpha in np.linspace(0, 1, 21):
        val_prob = alpha * clip_val_shot_prob + (1 - alpha) * dino_val_shot_prob
        val_pred = val_prob.argmax(1)
        macro = f1_score(shot_val_true, val_pred, average="macro", zero_division=0)
        alpha_rows.append({"alpha_clip": float(alpha), "val_shot_macro_f1": macro})
        if macro > best_macro:
            best_macro = macro
            best_alpha = float(alpha)

    text_threshold, val_text_macro_f1 = tune_text_threshold(text_val_true, clip_val_text_prob)
    test_prob = best_alpha * clip_test_shot_prob + (1 - best_alpha) * dino_test_shot_prob
    test_shot_pred = test_prob.argmax(1)
    test_text_pred = (clip_test_text_prob[:, 1] >= text_threshold).astype(np.int64)

    pred = {
        "shot_true": shot_test_true,
        "shot_pred": test_shot_pred,
        "text_true": text_test_true,
        "text_pred": test_text_pred,
        "shot_prob": test_prob,
        "text_prob": clip_test_text_prob,
    }
    pred_df = build_prediction_frame(meta, masks["test"], pred)
    pred_df["joint_pred_label"] = pred_df["shot_pred"] + "_" + pred_df["text_pred"].map({0: "notext", 1: "text"})
    pred_df["text_threshold"] = text_threshold

    rows = [
        metric_row("clip_stronger_retrained", shot_test_true, clip_test_shot_prob.argmax(1), text_test_true, clip_test_text_prob.argmax(1)),
        metric_row("dinov2_shot_retrained", shot_test_true, dino_test_shot_prob.argmax(1)),
        metric_row(f"ensemble_alpha_clip_{best_alpha:.2f}", shot_test_true, test_shot_pred, text_test_true, test_text_pred),
    ]
    rows[-1]["text_threshold"] = text_threshold
    rows[-1]["val_text_macro_f1_at_threshold"] = val_text_macro_f1

    pd.DataFrame(alpha_rows).to_csv(OUTPUT_DIR / "ensemble_alpha_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ensemble_experiment_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "ensemble_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_alpha_clip": best_alpha, "rows": rows}, f, indent=2)

    runtime = {
        "embedding_extraction_time_sec": 0.0,
        "classifier_fit_time_sec": fit_time_sec,
        "classifier_inference_time_sec": infer_time_sec,
        "overlay_rendering_time_sec": np.nan,
        "scene_detection_time_sec": np.nan,
        "total_runtime_sec": time.perf_counter() - start_time,
        "throughput_samples_per_sec": safe_divide(len(meta), time.perf_counter() - start_time),
        "best_alpha_clip": best_alpha,
        "note": "uses cached CLIP and DINOv2 embeddings; video overlay is not run here",
    }
    basic = save_comprehensive_metrics(meta, pred_df, pred, runtime, alpha_rows, rows)
    print("best_alpha_clip:", best_alpha)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(basic, indent=2))


if __name__ == "__main__":
    main()
