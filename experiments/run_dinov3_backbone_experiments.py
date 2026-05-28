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
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "dinov3_backbones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
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
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-3
    loss_type: str = "focal"
    label_smoothing: float = 0.0


class ImagePathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.loc[idx]
        image = Image.open(resolve_image_path(row)).convert("RGB")
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


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def resolve_image_path(row: pd.Series) -> Path:
    path = Path(str(row["filepath"]))
    if path.exists():
        return path
    fallback = ROOT / "labeled_dataset" / str(row["original_class"]) / str(row["filename"])
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot find image for row: {row.to_dict()}")


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    return total / (num_classes * torch.clamp(counts, min=1))


def make_loss(labels: torch.Tensor, num_classes: int, config: HeadConfig):
    weights = class_weights(labels, num_classes).to(DEVICE)
    if config.loss_type == "focal":
        return FocalLoss(weight=weights, gamma=2.0, label_smoothing=config.label_smoothing)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)


def normalize_features(features: torch.Tensor | dict | tuple | list) -> torch.Tensor:
    if isinstance(features, dict):
        for key in ["x_norm_clstoken", "pooled", "features", "last_hidden_state"]:
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
    return features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-8)


def create_dinov3_model(config: BackboneConfig):
    if config.model_name not in timm.list_models("*dinov3*"):
        print(f"skipping unavailable timm model: {config.model_name}")
        return None, None
    model = timm.create_model(config.model_name, pretrained=True, num_classes=0).to(DEVICE)
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def extract_embeddings(df: pd.DataFrame, config: BackboneConfig) -> tuple[np.ndarray | None, float]:
    emb_path = OUTPUT_DIR / f"{config.name}_embeddings.npz"
    if emb_path.exists():
        print("loading cached embeddings:", emb_path)
        started = time.perf_counter()
        return np.load(emb_path)["embeddings"].astype("float32"), time.perf_counter() - started

    model, transform = create_dinov3_model(config)
    if model is None:
        return None, 0.0

    loader = DataLoader(ImagePathDataset(df, transform), batch_size=config.batch_size, shuffle=False, num_workers=0)
    chunks = []
    order = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, indices in tqdm(loader, desc=f"{config.name} embeddings"):
            features = normalize_features(model(images.to(DEVICE)))
            chunks.append(features.cpu().float().numpy())
            order.extend(indices.numpy().tolist())

    order_arr = np.array(order)
    assert np.all(order_arr == np.arange(len(df)))
    embeddings = np.concatenate(chunks, axis=0).astype("float32")
    elapsed = time.perf_counter() - started
    np.savez_compressed(emb_path, embeddings=embeddings, filepath=df["filepath"].to_numpy(), model_name=config.model_name)
    return embeddings, elapsed


def build_data(df: pd.DataFrame, embeddings: np.ndarray):
    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    shot_labels = df["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text_labels = df["has_text"].to_numpy(dtype=np.int64)
    split = df["split"].to_numpy()
    masks = {name: split == name for name in ["train", "val", "test"]}

    def tx(arr):
        return torch.tensor(arr, dtype=torch.float32)

    def ty(arr):
        return torch.tensor(arr, dtype=torch.long)

    return (
        tx(embeddings[masks["train"]]), ty(shot_labels[masks["train"]]), ty(text_labels[masks["train"]]),
        tx(embeddings[masks["val"]]), ty(shot_labels[masks["val"]]), ty(text_labels[masks["val"]]),
        tx(embeddings[masks["test"]]), ty(shot_labels[masks["test"]]), ty(text_labels[masks["test"]]),
    ), masks


def predict(model, loader, text_threshold: float = DEFAULT_TEXT_THRESHOLD) -> dict[str, np.ndarray]:
    model.eval()
    shot_true, shot_pred, text_true, text_pred = [], [], [], []
    shot_probs, text_probs = [], []
    with torch.no_grad():
        for x, y_shot, y_text in loader:
            shot_logits, text_logits = model(x.to(DEVICE))
            shot_prob = torch.softmax(shot_logits, dim=1).cpu().numpy()
            text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()
            shot_true.extend(y_shot.numpy().tolist())
            text_true.extend(y_text.numpy().tolist())
            shot_pred.extend(shot_prob.argmax(axis=1).tolist())
            text_pred.extend((text_prob[:, 1] >= float(text_threshold)).astype(np.int64).tolist())
            shot_probs.append(shot_prob)
            text_probs.append(text_prob)
    return {
        "shot_true": np.array(shot_true),
        "shot_pred": np.array(shot_pred),
        "text_true": np.array(text_true),
        "text_pred": np.array(text_pred),
        "shot_prob": np.vstack(shot_probs),
        "text_prob": np.vstack(text_probs),
    }


def tune_text_threshold(text_true: np.ndarray, text_prob: np.ndarray, shot_true: np.ndarray, shot_pred: np.ndarray) -> tuple[float, float, float]:
    best = (DEFAULT_TEXT_THRESHOLD, -1.0, -1.0)
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        text_pred = (text_prob[:, 1] >= float(threshold)).astype(np.int64)
        joint = float(np.mean((shot_true == shot_pred) & (text_true == text_pred)))
        macro = f1_score(text_true, text_pred, average="macro", zero_division=0)
        if joint > best[1] or (joint == best[1] and macro > best[2]):
            best = (float(threshold), joint, float(macro))
    return best


def train_head(config: HeadConfig, data) -> dict:
    seed_everything()
    x_train, y_shot_train, y_text_train, x_val, y_shot_val, y_text_val, x_test, y_shot_test, y_text_test = data
    train_loader = DataLoader(TensorDataset(x_train, y_shot_train, y_text_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_shot_val, y_text_val), batch_size=256)
    test_loader = DataLoader(TensorDataset(x_test, y_shot_test, y_text_test), batch_size=256)

    model = MultiTaskHead(x_train.shape[1], config.hidden_dims, config.dropout).to(DEVICE)
    shot_loss = make_loss(y_shot_train, len(SHOT_TYPES), config)
    text_loss = make_loss(y_text_train, 2, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_joint = -1.0
    wait = PATIENCE
    fit_started = time.perf_counter()
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y_shot, y_text in train_loader:
            optimizer.zero_grad()
            shot_logits, text_logits = model(x.to(DEVICE))
            loss = shot_loss(shot_logits, y_shot.to(DEVICE)) + text_loss(text_logits, y_text.to(DEVICE))
            loss.backward()
            optimizer.step()
        val_pred = predict(model, val_loader)
        joint = float(np.mean((val_pred["shot_true"] == val_pred["shot_pred"]) & (val_pred["text_true"] == val_pred["text_pred"])))
        if joint > best_joint:
            best_joint = joint
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break
    fit_time = time.perf_counter() - fit_started
    model.load_state_dict(best_state)

    val_pred = predict(model, val_loader)
    threshold, val_joint, val_text_macro = tune_text_threshold(
        val_pred["text_true"], val_pred["text_prob"], val_pred["shot_true"], val_pred["shot_pred"]
    )
    inference_started = time.perf_counter()
    test_pred = predict(model, test_loader, text_threshold=threshold)
    inference_time = time.perf_counter() - inference_started
    return {
        "config": config.__dict__,
        "fit_time_sec": fit_time,
        "classifier_inference_time_sec": inference_time,
        "text_threshold": threshold,
        "val_joint_at_threshold": val_joint,
        "val_text_macro_f1_at_threshold": val_text_macro,
        "test_pred": test_pred,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def top2_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(prob, axis=1)[:, -2:]
    return float(np.mean([truth in row for truth, row in zip(y_true, order)]))


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
        ece += count / total * gap
        mce = max(mce, gap)
        rows.append({"bin_start": start, "bin_end": end, "count": count, "accuracy": acc, "confidence": conf, "gap": gap})
    return float(ece), float(mce), pd.DataFrame(rows)


def build_prediction_frame(df: pd.DataFrame, mask: np.ndarray, pred: dict[str, np.ndarray], threshold: float) -> pd.DataFrame:
    pred_df = df[mask].copy().reset_index(drop=True)
    pred_df["shot_true_idx"] = pred["shot_true"]
    pred_df["shot_pred_idx"] = pred["shot_pred"]
    pred_df["shot_pred"] = [SHOT_TYPES[i] for i in pred["shot_pred"]]
    pred_df["text_true"] = pred["text_true"]
    pred_df["text_pred"] = pred["text_pred"]
    pred_df["text_threshold"] = threshold
    pred_df["text_probability"] = pred["text_prob"][:, 1]
    pred_df["shot_confidence"] = pred["shot_prob"].max(axis=1)
    pred_df["text_confidence"] = np.maximum(pred["text_prob"][:, 1], 1.0 - pred["text_prob"][:, 1])
    pred_df["joint_confidence"] = pred_df["shot_confidence"] * pred_df["text_confidence"]
    pred_df["shot_correct"] = pred_df["shot_true_idx"] == pred_df["shot_pred_idx"]
    pred_df["text_correct"] = pred_df["text_true"] == pred_df["text_pred"]
    pred_df["joint_correct"] = pred_df["shot_correct"] & pred_df["text_correct"]
    for i, label in enumerate(SHOT_TYPES):
        pred_df[f"prob_shot_{label}"] = pred["shot_prob"][:, i]
    return pred_df


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None, prefix: str, labels: list[str]) -> dict[str, float]:
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


def save_metric_tables(name: str, df: pd.DataFrame, masks: dict[str, np.ndarray], result: dict, extraction_time: float, embedding_dim: int) -> dict:
    pred = result["test_pred"]
    pred_df = build_prediction_frame(df, masks["test"], pred, result["text_threshold"])
    run_dir = OUTPUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(run_dir / "test_predictions_with_confidence.csv", index=False, encoding="utf-8-sig")

    shot_true, shot_pred, shot_prob = pred["shot_true"], pred["shot_pred"], pred["shot_prob"]
    text_true, text_pred, text_prob = pred["text_true"], pred["text_pred"], pred["text_prob"]
    text_labels = ["notext", "text"]
    summary = {
        "experiment": name,
        "embedding_dimension": int(embedding_dim),
        "embedding_extraction_time_sec": extraction_time,
        "fit_time_sec": result["fit_time_sec"],
        "classifier_inference_time_sec": result["classifier_inference_time_sec"],
        "total_runtime_sec": extraction_time + result["fit_time_sec"] + result["classifier_inference_time_sec"],
        "throughput_samples_per_sec": safe_divide(len(df), extraction_time + result["classifier_inference_time_sec"]),
        "text_threshold": result["text_threshold"],
        "val_joint_at_threshold": result["val_joint_at_threshold"],
        "val_text_macro_f1_at_threshold": result["val_text_macro_f1_at_threshold"],
        "joint_accuracy": float(pred_df["joint_correct"].mean()),
        "guide_reliability_score": float(pred_df["joint_correct"].mean()),
        "conditional_text_given_shot": safe_divide(int((pred_df["shot_correct"] & pred_df["text_correct"]).sum()), int(pred_df["shot_correct"].sum())),
        "conditional_shot_given_text": safe_divide(int((pred_df["shot_correct"] & pred_df["text_correct"]).sum()), int(pred_df["text_correct"].sum())),
        "both_correct_ratio": float(pred_df["joint_correct"].mean()),
        "shot_only_error_ratio": safe_divide(int((~pred_df["shot_correct"] & pred_df["text_correct"]).sum()), len(pred_df)),
        "text_only_error_ratio": safe_divide(int((pred_df["shot_correct"] & ~pred_df["text_correct"]).sum()), len(pred_df)),
        "both_wrong_ratio": safe_divide(int((~pred_df["shot_correct"] & ~pred_df["text_correct"]).sum()), len(pred_df)),
    }
    summary.update(classifier_metrics(shot_true, shot_pred, shot_prob, "shot", SHOT_TYPES))
    summary.update(classifier_metrics(text_true, text_pred, text_prob, "text", text_labels))

    for target, y_true, y_pred, conf in [
        ("shot", shot_true, shot_pred, shot_prob.max(axis=1)),
        ("text", text_true, text_pred, np.maximum(text_prob[:, 1], 1.0 - text_prob[:, 1])),
        ("joint", pred_df["joint_correct"].astype(int).to_numpy(), np.ones(len(pred_df), dtype=int), pred_df["joint_confidence"].to_numpy()),
    ]:
        ece, mce, hist = expected_calibration_error(y_true, y_pred, conf)
        summary[f"{target}_ECE"] = ece
        summary[f"{target}_MCE"] = mce
        hist.to_csv(run_dir / f"{target}_confidence_histogram.csv", index=False, encoding="utf-8-sig")

    precision, recall, f1, support = precision_recall_fscore_support(shot_true, shot_pred, labels=list(range(len(SHOT_TYPES))), zero_division=0)
    per_class_rows = []
    for idx, label in enumerate(SHOT_TYPES):
        pred_mask = shot_pred == idx
        true_mask = shot_true == idx
        per_class_rows.append({
            "shot_type": label,
            "precision": precision[idx],
            "recall_per_class_accuracy": recall[idx],
            "f1": f1[idx],
            "support": int(support[idx]),
            "false_positive_count": int(np.sum(pred_mask & ~true_mask)),
            "false_negative_count": int(np.sum(true_mask & ~pred_mask)),
            "avg_confidence_when_predicted": float(shot_prob[pred_mask, idx].mean()) if pred_mask.any() else np.nan,
        })
    pd.DataFrame(per_class_rows).to_csv(run_dir / "shot_type_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm = confusion_matrix(shot_true, shot_pred, labels=list(range(len(SHOT_TYPES))))
    pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES).to_csv(run_dir / "shot_type_confusion_matrix.csv", encoding="utf-8-sig")

    threshold_rows = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        covered = pred_df["joint_confidence"] >= threshold
        threshold_rows.append({
            "threshold": threshold,
            "coverage": safe_divide(int(covered.sum()), len(pred_df)),
            "reliability_joint_accuracy_when_covered": float(pred_df.loc[covered, "joint_correct"].mean()) if covered.any() else np.nan,
            "reliable_guide_ratio_correct_and_covered": safe_divide(int((covered & pred_df["joint_correct"]).sum()), len(pred_df)),
        })
    pd.DataFrame(threshold_rows).to_csv(run_dir / "confidence_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")

    dataset_stats = {
        "total_labeled_cuts": int(len(df)),
        "train_count": int(masks["train"].sum()),
        "val_count": int(masks["val"].sum()),
        "test_count": int(masks["test"].sum()),
        "number_of_videos": int(df["video_id"].nunique()) if "video_id" in df else np.nan,
    }
    with open(run_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    seed_everything()
    print("device:", DEVICE)
    print("system:", platform.platform())
    df = pd.read_csv(META_CSV)
    backbones = [
        BackboneConfig("dinov3_vit_small_patch16", "vit_small_patch16_dinov3", batch_size=32),
    ]
    head_configs = [
        HeadConfig("focal_multitask_512_256", loss_type="focal"),
    ]
    rows = []
    for backbone in backbones:
        print("extracting", backbone)
        embeddings, extraction_time = extract_embeddings(df, backbone)
        if embeddings is None:
            continue
        data, masks = build_data(df, embeddings)
        for head_config in head_configs:
            experiment_name = f"{backbone.name}_{head_config.name}"
            print("training", experiment_name)
            result = train_head(head_config, data)
            summary = save_metric_tables(experiment_name, df, masks, result, extraction_time, embeddings.shape[1])
            summary.update({"backbone": backbone.model_name, **head_config.__dict__})
            rows.append(summary)
            print(json.dumps(summary, indent=2))

    summary_df = pd.DataFrame(rows).sort_values(["joint_accuracy", "shot_macro_f1"], ascending=False) if rows else pd.DataFrame()
    summary_df.to_csv(OUTPUT_DIR / "dinov3_backbone_experiment_summary.csv", index=False, encoding="utf-8-sig")
    if rows:
        best = summary_df.iloc[0].to_dict()
        with open(OUTPUT_DIR / "best_dinov3_backbone_experiment.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print("best:", best)


if __name__ == "__main__":
    main()
