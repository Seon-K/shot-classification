from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "outputs" / "clip_embeddings" / "clip_embedding_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "siglip_backbones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 15
DEFAULT_TEXT_THRESHOLD = 0.5


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    model_name: str
    batch_size: int = 32


@dataclass(frozen=True)
class HeadConfig:
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float = 0.25
    closeup_factor: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-3


class ImagePathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.loc[idx]
        path = resolve_image_path(row)
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        self.shared = nn.Sequential(*layers)
        self.shot_head = nn.Linear(prev, len(SHOT_TYPES))
        self.text_head = nn.Linear(prev, 2)

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


def resolve_image_path(row: pd.Series) -> Path:
    path = Path(str(row["filepath"]))
    if path.exists():
        return path
    fallback = ROOT / "labeled_dataset" / str(row["original_class"]) / str(row["filename"])
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot find image for row: {row.to_dict()}")


def class_weights(labels: torch.Tensor, num_classes: int, closeup_factor: float = 1.0) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * torch.clamp(counts, min=1))
    if num_classes == len(SHOT_TYPES):
        weights[0] *= closeup_factor
    return weights


def normalize_features(features: torch.Tensor | dict | tuple | list) -> torch.Tensor:
    if isinstance(features, dict):
        for key in ["x_norm_clstoken", "pooled", "features"]:
            if key in features:
                features = features[key]
                break
        else:
            features = next(iter(features.values()))
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim == 3:
        features = features[:, 0]
    return features / features.norm(dim=-1, keepdim=True)


def create_siglip_model(config: BackboneConfig):
    if not timm.list_models(config.model_name):
        return None, None
    model = timm.create_model(config.model_name, pretrained=True, num_classes=0).to(DEVICE)
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def extract_embeddings(df: pd.DataFrame, config: BackboneConfig) -> np.ndarray | None:
    emb_path = OUTPUT_DIR / f"{config.name}_embeddings.npz"
    if emb_path.exists():
        print("loading cached embeddings:", emb_path)
        return np.load(emb_path)["embeddings"].astype("float32")

    model, transform = create_siglip_model(config)
    if model is None:
        print(f"skipping unavailable timm model: {config.model_name}")
        return None

    loader = DataLoader(ImagePathDataset(df, transform), batch_size=config.batch_size, shuffle=False, num_workers=0)
    chunks = []
    order = []
    with torch.no_grad():
        for images, indices in tqdm(loader, desc=f"{config.name} embeddings"):
            images = images.to(DEVICE)
            features = normalize_features(model(images))
            chunks.append(features.cpu().float().numpy())
            order.extend(indices.numpy().tolist())

    order = np.array(order)
    assert np.all(order == np.arange(len(df)))
    embeddings = np.concatenate(chunks, axis=0).astype("float32")
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        filepath=df["filepath"].to_numpy(),
        model_name=config.model_name,
    )
    return embeddings


def predict(model, loader, text_threshold=DEFAULT_TEXT_THRESHOLD):
    model.eval()
    shot_true, shot_pred, text_true, text_pred = [], [], [], []
    shot_prob_rows, text_prob_rows = [], []
    with torch.no_grad():
        for x, y_shot, y_text in loader:
            shot_logits, text_logits = model(x.to(DEVICE))
            shot_prob = torch.softmax(shot_logits, dim=1).cpu().numpy()
            text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()
            shot_true.extend(y_shot.numpy().tolist())
            text_true.extend(y_text.numpy().tolist())
            shot_pred.extend(shot_prob.argmax(axis=1).tolist())
            text_pred.extend((text_prob[:, 1] >= float(text_threshold)).astype(np.int64).tolist())
            shot_prob_rows.append(shot_prob)
            text_prob_rows.append(text_prob)
    return {
        "shot_true": np.array(shot_true),
        "shot_pred": np.array(shot_pred),
        "text_true": np.array(text_true),
        "text_pred": np.array(text_pred),
        "shot_prob": np.vstack(shot_prob_rows),
        "text_prob": np.vstack(text_prob_rows),
    }


def tune_text_threshold(text_true: np.ndarray, text_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = DEFAULT_TEXT_THRESHOLD
    best_score = -1.0
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (text_prob[:, 1] >= float(threshold)).astype(np.int64)
        score = f1_score(text_true, pred, average="macro", zero_division=0)
        if score > best_score or (score == best_score and abs(threshold - DEFAULT_TEXT_THRESHOLD) < abs(best_threshold - DEFAULT_TEXT_THRESHOLD)):
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def metrics(pred: dict[str, np.ndarray]) -> dict[str, float]:
    shot_true = pred["shot_true"]
    shot_pred = pred["shot_pred"]
    text_true = pred["text_true"]
    text_pred = pred["text_pred"]
    return {
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
        "text_acc": accuracy_score(text_true, text_pred),
        "text_macro_f1": f1_score(text_true, text_pred, average="macro", zero_division=0),
        "text_f1": f1_score(text_true, text_pred, average="binary", zero_division=0),
        "joint_acc": float(np.mean((shot_true == shot_pred) & (text_true == text_pred))),
    }


def train(config: HeadConfig, data) -> dict:
    seed_everything()
    x_train, y_shot_train, y_text_train, x_val, y_shot_val, y_text_val, x_test, y_shot_test, y_text_test = data
    train_loader = DataLoader(TensorDataset(x_train, y_shot_train, y_text_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_shot_val, y_text_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_shot_test, y_text_test), batch_size=256, shuffle=False)

    model = MultiTaskHead(x_train.shape[1], config.hidden_dims, config.dropout).to(DEVICE)
    shot_loss = nn.CrossEntropyLoss(weight=class_weights(y_shot_train, len(SHOT_TYPES), config.closeup_factor).to(DEVICE))
    text_loss = nn.CrossEntropyLoss(weight=class_weights(y_text_train, 2).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_joint = -1.0
    wait = PATIENCE
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y_shot, y_text in train_loader:
            optimizer.zero_grad()
            shot_logits, text_logits = model(x.to(DEVICE))
            loss = shot_loss(shot_logits, y_shot.to(DEVICE)) + text_loss(text_logits, y_text.to(DEVICE))
            loss.backward()
            optimizer.step()

        val_pred = predict(model, val_loader)
        val_metrics = metrics(val_pred)
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["joint_acc"] > best_joint:
            best_joint = val_metrics["joint_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break

    model.load_state_dict(best_state)
    val_pred = predict(model, val_loader)
    text_threshold, val_text_macro_f1 = tune_text_threshold(val_pred["text_true"], val_pred["text_prob"])
    test_pred = predict(model, test_loader, text_threshold=text_threshold)
    test_metrics = metrics(test_pred)
    test_metrics["text_threshold"] = text_threshold
    test_metrics["val_text_macro_f1_at_threshold"] = val_text_macro_f1
    return {"config": config.__dict__, "history": history, "test_metrics": test_metrics, "test_pred": test_pred}


def build_data(df: pd.DataFrame, embeddings: np.ndarray):
    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    shot_labels = df["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text_labels = df["has_text"].to_numpy(dtype=np.int64)
    masks = {split: df["split"].to_numpy() == split for split in ["train", "val", "test"]}

    def x(split: str):
        return torch.tensor(embeddings[masks[split]], dtype=torch.float32)

    def y(arr):
        return torch.tensor(arr, dtype=torch.long)

    return (
        x("train"),
        y(shot_labels[masks["train"]]),
        y(text_labels[masks["train"]]),
        x("val"),
        y(shot_labels[masks["val"]]),
        y(text_labels[masks["val"]]),
        x("test"),
        y(shot_labels[masks["test"]]),
        y(text_labels[masks["test"]]),
    ), masks


def save_predictions(df: pd.DataFrame, mask: np.ndarray, result: dict, output_path: Path) -> pd.DataFrame:
    pred = result["test_pred"]
    pred_df = df[mask].copy().reset_index(drop=True)
    pred_df["shot_true_idx"] = pred["shot_true"]
    pred_df["shot_pred_idx"] = pred["shot_pred"]
    pred_df["shot_pred"] = [SHOT_TYPES[i] for i in pred["shot_pred"]]
    pred_df["text_true"] = pred["text_true"]
    pred_df["text_pred"] = pred["text_pred"]
    pred_df["text_threshold"] = result["test_metrics"]["text_threshold"]
    pred_df["text_probability"] = pred["text_prob"][:, 1]
    pred_df["shot_confidence"] = pred["shot_prob"].max(axis=1)
    pred_df["joint_confidence"] = pred_df["shot_confidence"] * np.maximum(pred_df["text_probability"], 1.0 - pred_df["text_probability"])
    pred_df["shot_correct"] = pred_df["shot_true_idx"] == pred_df["shot_pred_idx"]
    pred_df["text_correct"] = pred_df["text_true"] == pred_df["text_pred"]
    pred_df["joint_correct"] = pred_df["shot_correct"] & pred_df["text_correct"]
    for i, label in enumerate(SHOT_TYPES):
        pred_df[f"prob_shot_{label}"] = pred["shot_prob"][:, i]
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return pred_df


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def top2_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(prob, axis=1)[:, -2:]
    return float(np.mean([truth in row for truth, row in zip(y_true, order)]))


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
    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_macro_precision": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0],
        f"{prefix}_macro_recall": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1],
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
                "minority_class_recall": recall[idx] if support[idx] <= np.median(support) else np.nan,
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


def high_low_confidence_samples(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_errors = pred_df[~pred_df["joint_correct"]].sort_values("joint_confidence", ascending=False).head(50)
    low_conf = pred_df.sort_values("joint_confidence", ascending=True).head(50)
    return high_errors, low_conf


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


def dataset_statistics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stats = {
        "total_labeled_cuts": int(len(df)),
        "number_of_videos": int(df["video_id"].nunique()),
        "scene_count": int(len(df)),
        "avg_scenes_per_video": float(len(df) / max(df["video_id"].nunique(), 1)),
        "avg_scene_duration": None,
    }
    split_dist = df["split"].value_counts().rename_axis("split").reset_index(name="count")
    class_dist = df.groupby(["shot_type", "has_text"]).size().reset_index(name="count")
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


def save_placeholder_metrics(base_dir: Path) -> None:
    placeholders = [
        ("02_zero_shot_vs_finetuned.csv", "zero-shot SigLIP text prompt baseline is not implemented in this script"),
        ("11_scene_detection_metrics.csv", "scene detection requires video-level manual cut labels"),
        ("14_data_size_experiment.csv", "train-size scaling is not run in this script"),
        ("16_annotation_label_quality.csv", "annotator2 labels are not available"),
        ("17_human_evaluation.csv", "human survey results are not available"),
    ]
    for filename, reason in placeholders:
        pd.DataFrame([{"metric_status": "placeholder", "reason": reason}]).to_csv(base_dir / filename, index=False, encoding="utf-8-sig")


def save_comprehensive_metrics(
    experiment_name: str,
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    result: dict,
    backbone: BackboneConfig,
    head_config: HeadConfig,
    embedding_dim: int,
    embedding_time_sec: float,
    fit_time_sec: float,
) -> None:
    metrics_dir = OUTPUT_DIR / "comprehensive_metrics" / experiment_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred = result["test_pred"]

    shot_metrics = classifier_metrics(pred["shot_true"], pred["shot_pred"], pred["shot_prob"], SHOT_TYPES, "shot")
    text_metrics = classifier_metrics(pred["text_true"], pred["text_pred"], pred["text_prob"], ["notext", "text"], "text")
    basic = {**shot_metrics, **text_metrics, **result["test_metrics"]}
    pd.DataFrame([basic]).to_csv(metrics_dir / "01_basic_classifier_metrics.csv", index=False, encoding="utf-8-sig")
    per_class_metrics(pred["shot_true"], pred["shot_pred"], pred["shot_prob"], SHOT_TYPES).to_csv(metrics_dir / "01_shot_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    per_class_metrics(pred["text_true"], pred["text_pred"], pred["text_prob"], ["notext", "text"]).to_csv(metrics_dir / "01_text_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    joint_reliability(pred_df).to_csv(metrics_dir / "03_joint_reliability.csv", index=False, encoding="utf-8-sig")
    error_propagation(pred_df).to_csv(metrics_dir / "04_error_propagation.csv", index=False, encoding="utf-8-sig")

    calibration_summary, histograms = calibration_tables(pred_df, pred)
    calibration_summary.to_csv(metrics_dir / "05_calibration_summary.csv", index=False, encoding="utf-8-sig")
    for name, table in histograms.items():
        table.to_csv(metrics_dir / f"05_{name}_confidence_histogram.csv", index=False, encoding="utf-8-sig")

    threshold_tradeoff(pred_df).to_csv(metrics_dir / "06_confidence_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")
    minority = per_class_metrics(pred["shot_true"], pred["shot_pred"], pred["shot_prob"], SHOT_TYPES)
    minority.sort_values("support").head(2).to_csv(metrics_dir / "07_minority_class_analysis.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(pred["shot_true"], pred["shot_pred"], labels=list(range(len(SHOT_TYPES))))
    pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES).to_csv(metrics_dir / "08_shot_confusion_matrix.csv", encoding="utf-8-sig")
    top_confusion_pairs(pred["shot_true"], pred["shot_pred"], SHOT_TYPES).to_csv(metrics_dir / "08_top_confusion_pairs.csv", index=False, encoding="utf-8-sig")
    high_errors, low_conf = high_low_confidence_samples(pred_df)
    high_errors.to_csv(metrics_dir / "08_high_confidence_errors.csv", index=False, encoding="utf-8-sig")
    low_conf.to_csv(metrics_dir / "08_low_confidence_samples.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(pred_df["joint_label"], pred_df["shot_pred"] + "_" + pred_df["text_pred"].map({0: "notext", 1: "text"})).to_csv(metrics_dir / "08_joint_confusion.csv", encoding="utf-8-sig")

    stability = pd.DataFrame([{f"{key}_mean": value for key, value in basic.items() if isinstance(value, (int, float, np.floating))}])
    for key in ["shot_accuracy", "shot_macro_f1", "shot_balanced_accuracy", "shot_weighted_f1"]:
        if f"{key}_mean" in stability.columns:
            stability[f"{key}_std"] = 0.0
    stability["seed_count"] = 1
    stability.to_csv(metrics_dir / "09_multi_seed_stability_summary.csv", index=False, encoding="utf-8-sig")

    stats, split_dist, class_dist = dataset_statistics(df)
    with open(metrics_dir / "10_dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    split_dist.to_csv(metrics_dir / "10_train_val_test_split.csv", index=False, encoding="utf-8-sig")
    class_dist.to_csv(metrics_dir / "10_class_distribution.csv", index=False, encoding="utf-8-sig")

    runtime = pd.DataFrame(
        [
            {
                "embedding_extraction_time_sec": embedding_time_sec,
                "classifier_fit_time_sec": fit_time_sec,
                "classifier_inference_time_sec": np.nan,
                "overlay_rendering_time_sec": np.nan,
                "scene_detection_time_sec": np.nan,
                "total_runtime_sec": embedding_time_sec + fit_time_sec,
                "throughput_samples_per_sec": safe_divide(len(df), embedding_time_sec + fit_time_sec),
                "note": "image embedding cache + classifier training only; video overlay is not run here",
            }
        ]
    )
    runtime.to_csv(metrics_dir / "12_runtime_efficiency.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "backbone": backbone.name,
                "model_name": backbone.model_name,
                "embedding_dimension": embedding_dim,
                "embedding_time_sec": embedding_time_sec,
                "fit_time_sec": fit_time_sec,
                "backbone_macro_f1": basic["shot_macro_f1"],
                "backbone_accuracy": basic["shot_accuracy"],
                "head_hidden_dims": str(head_config.hidden_dims),
                "dropout": head_config.dropout,
                "class_weight_closeup_factor": head_config.closeup_factor,
            }
        ]
    ).to_csv(metrics_dir / "13_backbone_ablation_summary.csv", index=False, encoding="utf-8-sig")

    video_rows, video_summary, hardest, easiest = per_video_variance(pred_df)
    video_rows.to_csv(metrics_dir / "15_per_video_variance.csv", index=False, encoding="utf-8-sig")
    video_summary.to_csv(metrics_dir / "15_per_video_variance_summary.csv", index=False, encoding="utf-8-sig")
    hardest.to_csv(metrics_dir / "15_hardest_videos.csv", index=False, encoding="utf-8-sig")
    easiest.to_csv(metrics_dir / "15_easiest_videos.csv", index=False, encoding="utf-8-sig")

    temporal_structure(pred_df).to_csv(metrics_dir / "18_temporal_structure_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "end_to_end_reliability": basic["joint_acc"],
                "confidence_filtered_reliability_at_0_8": threshold_tradeoff(pred_df).loc[lambda x: x["threshold"] == 0.8, "reliability_joint_accuracy_when_covered"].iloc[0],
                "guide_reliability": basic["joint_acc"],
                "pipeline_consistency": "classifier_only_evaluated",
                "error_propagation_gap": 1.0 - basic["joint_acc"],
            }
        ]
    ).to_csv(metrics_dir / "19_reliability_deployment_metrics.csv", index=False, encoding="utf-8-sig")

    with open(metrics_dir / "20_reproducibility_system_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": str(DEVICE),
                "pipeline_execution_status": "script_generated_metrics_when_experiment_runs",
            },
            f,
            indent=2,
        )

    save_placeholder_metrics(metrics_dir)
    with open(metrics_dir / "final_metric_summary.json", "w", encoding="utf-8") as f:
        json.dump(basic, f, indent=2)



def main() -> None:
    print("device:", DEVICE)
    df = pd.read_csv(META_CSV)
    backbones = [
        BackboneConfig("siglip_base_patch16_384", "vit_base_patch16_siglip_384", batch_size=32),
        BackboneConfig("siglip_so400m_patch14_384", "vit_so400m_patch14_siglip_384", batch_size=16),
        BackboneConfig("siglip2_base_patch16_256", "vit_base_patch16_siglip2_256", batch_size=32),
        BackboneConfig("siglip2_so400m_patch14_384", "vit_so400m_patch14_siglip2_384", batch_size=16),
    ]
    head_configs = [
        HeadConfig("multitask_512_256", (512, 256), 0.25, 1.0),
        HeadConfig("multitask_768_384", (768, 384), 0.30, 1.0),
        HeadConfig("multitask_512_256_closeup_1_5", (512, 256), 0.25, 1.5),
    ]

    rows = []
    unavailable_rows = []
    for backbone in backbones:
        embedding_start = time.perf_counter()
        embeddings = extract_embeddings(df, backbone)
        embedding_time_sec = time.perf_counter() - embedding_start
        if embeddings is None:
            unavailable_rows.append({"backbone": backbone.name, "model_name": backbone.model_name, "reason": "not_available_in_timm"})
            continue

        data, masks = build_data(df, embeddings)
        for head_config in head_configs:
            experiment_name = f"{backbone.name}_{head_config.name}"
            print("running", experiment_name)
            fit_start = time.perf_counter()
            result = train(head_config, data)
            fit_time_sec = time.perf_counter() - fit_start
            row = {
                "experiment": experiment_name,
                "backbone": backbone.name,
                "model_name": backbone.model_name,
                "embedding_dim": int(embeddings.shape[1]),
                "embedding_time_sec": embedding_time_sec,
                "fit_time_sec": fit_time_sec,
                **result["test_metrics"],
            }
            rows.append(row)
            print(row)
            pred_df = save_predictions(
                df,
                masks["test"],
                result,
                OUTPUT_DIR / f"{experiment_name}_test_predictions.csv",
            )
            save_comprehensive_metrics(
                experiment_name,
                df,
                pred_df,
                result,
                backbone,
                head_config,
                int(embeddings.shape[1]),
                embedding_time_sec,
                fit_time_sec,
            )

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(OUTPUT_DIR / "siglip_backbone_experiment_summary.csv", index=False, encoding="utf-8-sig")
        with open(OUTPUT_DIR / "siglip_backbone_experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        best = summary.sort_values(["shot_macro_f1", "joint_acc"], ascending=False).iloc[0].to_dict()
        with open(OUTPUT_DIR / "best_siglip_backbone_experiment.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print("best:", best)

    if unavailable_rows:
        pd.DataFrame(unavailable_rows).to_csv(OUTPUT_DIR / "unavailable_siglip_models.csv", index=False, encoding="utf-8-sig")
        print("unavailable models:", unavailable_rows)


if __name__ == "__main__":
    main()
