from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "outputs" / "clip_embeddings" / "clip_embedding_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "dinov2_shot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "vit_small_patch14_dinov2"
SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 15


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ImagePathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        path = self.frame.loc[idx, "filepath"]
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


class Head(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float, out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass(frozen=True)
class Config:
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float = 0.25
    closeup_factor: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-3


def class_weights(labels: torch.Tensor, closeup_factor: float) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=len(SHOT_TYPES)).float()
    total = counts.sum()
    weights = total / (len(SHOT_TYPES) * torch.clamp(counts, min=1))
    weights[0] *= closeup_factor
    return weights


def extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_path = OUTPUT_DIR / f"{MODEL_NAME}_embeddings.npz"
    if emb_path.exists():
        print("loading cached embeddings:", emb_path)
        return np.load(emb_path)["embeddings"].astype("float32")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(DEVICE)
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    loader = DataLoader(ImagePathDataset(df, transform), batch_size=32, shuffle=False, num_workers=0)

    chunks = []
    order = []
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="DINOv2 embeddings"):
            images = images.to(DEVICE)
            features = model(images)
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().float().numpy())
            order.extend(indices.numpy().tolist())

    order = np.array(order)
    assert np.all(order == np.arange(len(df)))
    embeddings = np.concatenate(chunks, axis=0).astype("float32")
    np.savez_compressed(emb_path, embeddings=embeddings, filepath=df["filepath"].to_numpy())
    return embeddings


def predict(model, loader):
    model.eval()
    true, pred = [], []
    prob_rows = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            true.extend(y.numpy().tolist())
            pred.extend(prob.argmax(axis=1).tolist())
            prob_rows.append(prob)
    return np.array(true), np.array(pred), np.vstack(prob_rows)


def train(config: Config, data):
    seed_everything()
    x_train, y_train, x_val, y_val, x_test, y_test = data
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256, shuffle=False)

    model = Head(x_train.shape[1], config.hidden_dims, config.dropout, len(SHOT_TYPES)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, config.closeup_factor).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_score = -1
    best_state = None
    wait = PATIENCE
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x.to(DEVICE))
            loss = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            optimizer.step()

        val_true, val_pred, _ = predict(model, val_loader)
        val_acc = accuracy_score(val_true, val_pred)
        val_macro = f1_score(val_true, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "val_shot_acc": val_acc, "val_shot_macro_f1": val_macro})

        if val_macro > best_score:
            best_score = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break

    model.load_state_dict(best_state)
    test_true, test_pred, test_prob = predict(model, test_loader)
    return {
        "config": config.__dict__,
        "history": history,
        "shot_acc": accuracy_score(test_true, test_pred),
        "shot_macro_f1": f1_score(test_true, test_pred, average="macro", zero_division=0),
        "true": test_true,
        "pred": test_pred,
        "prob": test_prob,
        "state": model.state_dict(),
    }


def main() -> None:
    print("device:", DEVICE)
    df = pd.read_csv(META_CSV)
    embeddings = extract_embeddings(df)
    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    labels = df["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    masks = {split: df["split"].to_numpy() == split for split in ["train", "val", "test"]}

    data = (
        torch.tensor(embeddings[masks["train"]], dtype=torch.float32),
        torch.tensor(labels[masks["train"]], dtype=torch.long),
        torch.tensor(embeddings[masks["val"]], dtype=torch.float32),
        torch.tensor(labels[masks["val"]], dtype=torch.long),
        torch.tensor(embeddings[masks["test"]], dtype=torch.float32),
        torch.tensor(labels[masks["test"]], dtype=torch.long),
    )

    configs = [
        Config("dinov2_shot_512_256", (512, 256), 0.25, 1.0),
        Config("dinov2_shot_512_256_closeup_1_5", (512, 256), 0.25, 1.5),
        Config("dinov2_shot_768_384", (768, 384), 0.30, 1.0),
        Config("dinov2_shot_linear", (), 0.0, 1.0),
    ]

    rows = []
    best_result = None
    for config in configs:
        print("running", config.name)
        result = train(config, data)
        row = {
            "experiment": config.name,
            "shot_acc": result["shot_acc"],
            "shot_macro_f1": result["shot_macro_f1"],
        }
        rows.append(row)
        print(row)

        pred_df = df[masks["test"]].copy().reset_index(drop=True)
        pred_df["shot_pred"] = [SHOT_TYPES[i] for i in result["pred"]]
        pred_df.to_csv(OUTPUT_DIR / f"{config.name}_test_predictions.csv", index=False, encoding="utf-8-sig")

        if best_result is None or result["shot_macro_f1"] > best_result["shot_macro_f1"]:
            best_result = {**result, "name": config.name}

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "dinov2_shot_experiment_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "dinov2_shot_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(OUTPUT_DIR / "best_dinov2_shot_experiment.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": best_result["name"],
                "shot_acc": best_result["shot_acc"],
                "shot_macro_f1": best_result["shot_macro_f1"],
            },
            f,
            indent=2,
        )
    print("best:", best_result["name"], best_result["shot_acc"], best_result["shot_macro_f1"])


if __name__ == "__main__":
    main()
