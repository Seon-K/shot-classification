from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "clip_embeddings"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_PATH = CHECKPOINT_DIR / "clip_vit_b32_multitask_head.pt"
METRICS_PATH = INPUT_DIR / "clip_stronger_head_test_metrics.json"

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 15
HIDDEN_DIMS = [512, 256]
DROPOUT = 0.25
DEFAULT_TEXT_THRESHOLD = 0.5


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ClipEmbeddingMultiTaskHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = embedding_dim
        for next_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, next_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = next_dim
        self.shared = nn.Sequential(*layers)
        self.shot_head = nn.Linear(prev_dim, len(SHOT_TYPES))
        self.text_head = nn.Linear(prev_dim, 2)

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    return total / (num_classes * torch.clamp(counts, min=1))


def predict_outputs(model, loader):
    model.eval()
    shot_true, shot_pred, text_true = [], [], []
    text_prob_rows = []
    with torch.no_grad():
        for x, y_shot, y_text in loader:
            shot_logits, text_logits = model(x.to(DEVICE))
            text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()
            shot_true.extend(y_shot.numpy().tolist())
            text_true.extend(y_text.numpy().tolist())
            shot_pred.extend(shot_logits.argmax(1).cpu().numpy().tolist())
            text_prob_rows.append(text_prob)
    return {
        "shot_true": np.array(shot_true),
        "shot_pred": np.array(shot_pred),
        "text_true": np.array(text_true),
        "text_prob": np.vstack(text_prob_rows),
    }


def predict(model, loader, text_threshold=DEFAULT_TEXT_THRESHOLD):
    out = predict_outputs(model, loader)
    text_pred = (out["text_prob"][:, 1] >= float(text_threshold)).astype(np.int64)
    return out["shot_true"], out["shot_pred"], out["text_true"], text_pred


def tune_text_threshold(text_true, text_prob, thresholds=None):
    # text/notext 불균형을 고려해 validation macro F1 기준으로 threshold를 선택합니다.
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best_threshold = DEFAULT_TEXT_THRESHOLD
    best_score = -1.0
    for threshold in thresholds:
        pred = (text_prob[:, 1] >= float(threshold)).astype(np.int64)
        current = f1_score(text_true, pred, average="macro", zero_division=0)
        if current > best_score or (current == best_score and abs(threshold - DEFAULT_TEXT_THRESHOLD) < abs(best_threshold - DEFAULT_TEXT_THRESHOLD)):
            best_threshold = float(threshold)
            best_score = float(current)
    return best_threshold, best_score


def score(shot_true, shot_pred, text_true, text_pred):
    return {
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
        "text_acc": accuracy_score(text_true, text_pred),
        "text_f1": f1_score(text_true, text_pred, average="binary", zero_division=0),
        "joint_acc": float(np.mean((shot_true == shot_pred) & (text_true == text_pred))),
    }


def main() -> None:
    seed_everything()
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    meta = pd.read_csv(INPUT_DIR / "clip_embedding_metadata.csv")
    embeddings = np.load(INPUT_DIR / "clip_vit_b32_openai_embeddings.npz")["embeddings"].astype("float32")

    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    labels = meta["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text = meta["has_text"].to_numpy(dtype=np.int64)
    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}

    def x(split: str):
        return torch.tensor(embeddings[masks[split]], dtype=torch.float32)

    def y_shot(split: str):
        return torch.tensor(labels[masks[split]], dtype=torch.long)

    def y_text(split: str):
        return torch.tensor(text[masks[split]], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x("train"), y_shot("train"), y_text("train")), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x("val"), y_shot("val"), y_text("val")), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(x("test"), y_shot("test"), y_text("test")), batch_size=256, shuffle=False)

    model = ClipEmbeddingMultiTaskHead(embeddings.shape[1], HIDDEN_DIMS, DROPOUT).to(DEVICE)
    shot_loss = nn.CrossEntropyLoss(weight=class_weights(y_shot("train"), len(SHOT_TYPES)).to(DEVICE))
    text_loss = nn.CrossEntropyLoss(weight=class_weights(y_text("train"), 2).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_state = None
    best_joint = -1
    wait = PATIENCE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_x, batch_shot, batch_text in train_loader:
            optimizer.zero_grad()
            shot_logits, text_logits = model(batch_x.to(DEVICE))
            loss = shot_loss(shot_logits, batch_shot.to(DEVICE)) + text_loss(text_logits, batch_text.to(DEVICE))
            loss.backward()
            optimizer.step()

        val_score = score(*predict(model, val_loader, text_threshold=DEFAULT_TEXT_THRESHOLD))
        if val_score["joint_acc"] > best_joint:
            best_joint = val_score["joint_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break

    model.load_state_dict(best_state)
    val_outputs = predict_outputs(model, val_loader)
    text_threshold, val_text_macro_f1 = tune_text_threshold(val_outputs["text_true"], val_outputs["text_prob"])
    test_score = score(*predict(model, test_loader, text_threshold=text_threshold))
    test_score["text_threshold"] = text_threshold
    test_score["val_text_macro_f1_at_threshold"] = val_text_macro_f1
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "shot_types": SHOT_TYPES,
        "shot_to_idx": shot_to_idx,
        "idx_to_shot": {i: name for name, i in shot_to_idx.items()},
        "embedding_dim": embeddings.shape[1],
        "hidden_dims": HIDDEN_DIMS,
        "hidden_dim": HIDDEN_DIMS[-1],
        "dropout": DROPOUT,
        "clip_model": "ViT-B-32",
        "clip_model_name": "ViT-B-32",
        "clip_pretrained": "openai",
        "text_threshold": text_threshold,
        "text_threshold_metric": "validation_text_macro_f1",
        "val_text_macro_f1_at_threshold": val_text_macro_f1,
        "seed": SEED,
        "dataset_rows": len(meta),
        "test_metrics": test_score,
    }
    torch.save(checkpoint, OUTPUT_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_score, f, indent=2)
    print(json.dumps(test_score, indent=2))
    print("saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
