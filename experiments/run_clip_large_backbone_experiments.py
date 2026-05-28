from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "outputs" / "clip_embeddings" / "clip_embedding_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "clip_large_backbones"
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
    pretrained: str
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


def extract_embeddings(df: pd.DataFrame, config: BackboneConfig) -> np.ndarray:
    emb_path = OUTPUT_DIR / f"{config.name}_embeddings.npz"
    if emb_path.exists():
        print("loading cached embeddings:", emb_path)
        return np.load(emb_path)["embeddings"].astype("float32")

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=config.pretrained,
        device=DEVICE,
    )
    model.eval()
    loader = DataLoader(ImagePathDataset(df, preprocess), batch_size=config.batch_size, shuffle=False, num_workers=0)

    chunks = []
    order = []
    with torch.no_grad():
        for images, indices in tqdm(loader, desc=f"{config.name} embeddings"):
            images = images.to(DEVICE)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
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
        pretrained=config.pretrained,
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


def save_predictions(df: pd.DataFrame, mask: np.ndarray, result: dict, output_path: Path) -> None:
    pred = result["test_pred"]
    pred_df = df[mask].copy().reset_index(drop=True)
    pred_df["shot_pred"] = [SHOT_TYPES[i] for i in pred["shot_pred"]]
    pred_df["text_pred"] = pred["text_pred"]
    pred_df["text_threshold"] = result["test_metrics"]["text_threshold"]
    pred_df["text_probability"] = pred["text_prob"][:, 1]
    for i, label in enumerate(SHOT_TYPES):
        pred_df[f"prob_shot_{label}"] = pred["shot_prob"][:, i]
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    print("device:", DEVICE)
    df = pd.read_csv(META_CSV)
    backbones = [
        BackboneConfig("clip_vit_l14_openai", "ViT-L-14", "openai", batch_size=24),
        BackboneConfig("clip_vit_h14_laion2b", "ViT-H-14", "laion2b_s32b_b79k", batch_size=16),
    ]
    head_configs = [
        HeadConfig("multitask_512_256", (512, 256), 0.25, 1.0),
        HeadConfig("multitask_768_384", (768, 384), 0.30, 1.0),
        HeadConfig("multitask_512_256_closeup_1_5", (512, 256), 0.25, 1.5),
    ]

    rows = []
    for backbone in backbones:
        embeddings = extract_embeddings(df, backbone)
        data, masks = build_data(df, embeddings)
        for head_config in head_configs:
            experiment_name = f"{backbone.name}_{head_config.name}"
            print("running", experiment_name)
            result = train(head_config, data)
            row = {
                "experiment": experiment_name,
                "backbone": backbone.name,
                "model_name": backbone.model_name,
                "pretrained": backbone.pretrained,
                **result["test_metrics"],
            }
            rows.append(row)
            print(row)
            save_predictions(
                df,
                masks["test"],
                result,
                OUTPUT_DIR / f"{experiment_name}_test_predictions.csv",
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "clip_large_backbone_experiment_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "clip_large_backbone_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    best = summary.sort_values(["shot_macro_f1", "joint_acc"], ascending=False).iloc[0].to_dict()
    with open(OUTPUT_DIR / "best_clip_large_backbone_experiment.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("best:", best)


if __name__ == "__main__":
    main()
