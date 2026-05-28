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
CLIP_DIR = ROOT / "outputs" / "clip_embeddings"
DINO_DIR = ROOT / "outputs" / "model_experiments" / "dinov2_shot"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 15


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(512, 256), dropout=0.25):
        super().__init__()
        layers = []
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


class ShotHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(512, 256), dropout=0.25):
        super().__init__()
        layers = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, len(SHOT_TYPES)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    return total / (num_classes * torch.clamp(counts, min=1))


def multitask_predict(model, loader):
    model.eval()
    shot_true, text_true = [], []
    shot_probs, text_probs = [], []
    with torch.no_grad():
        for x, y_shot, y_text in loader:
            shot_logits, text_logits = model(x.to(DEVICE))
            shot_true.extend(y_shot.numpy().tolist())
            text_true.extend(y_text.numpy().tolist())
            shot_probs.append(torch.softmax(shot_logits, dim=1).cpu().numpy())
            text_probs.append(torch.softmax(text_logits, dim=1).cpu().numpy())
    return np.array(shot_true), np.array(text_true), np.vstack(shot_probs), np.vstack(text_probs)


def shot_predict(model, loader):
    model.eval()
    true = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            true.extend(y.numpy().tolist())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.array(true), np.vstack(probs)


def train_clip(x_train, y_shot_train, y_text_train, x_val, y_shot_val, y_text_val):
    seed_everything()
    train_loader = DataLoader(TensorDataset(x_train, y_shot_train, y_text_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_shot_val, y_text_val), batch_size=256, shuffle=False)
    model = MultiTaskHead(x_train.shape[1]).to(DEVICE)
    shot_loss = nn.CrossEntropyLoss(weight=class_weights(y_shot_train, len(SHOT_TYPES)).to(DEVICE))
    text_loss = nn.CrossEntropyLoss(weight=class_weights(y_text_train, 2).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state = None
    best_joint = -1
    wait = PATIENCE
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y_shot, y_text in train_loader:
            optimizer.zero_grad()
            shot_logits, text_logits = model(x.to(DEVICE))
            loss = shot_loss(shot_logits, y_shot.to(DEVICE)) + text_loss(text_logits, y_text.to(DEVICE))
            loss.backward()
            optimizer.step()
        shot_true, text_true, shot_prob, text_prob = multitask_predict(model, val_loader)
        joint = np.mean((shot_true == shot_prob.argmax(1)) & (text_true == text_prob.argmax(1)))
        if joint > best_joint:
            best_joint = joint
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break
    model.load_state_dict(best_state)
    return model


def train_dino(x_train, y_train, x_val, y_val):
    seed_everything()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)
    model = ShotHead(x_train.shape[1]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, len(SHOT_TYPES)).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state = None
    best_macro = -1
    wait = PATIENCE
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()
        true, prob = shot_predict(model, val_loader)
        macro = f1_score(true, prob.argmax(1), average="macro", zero_division=0)
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break
    model.load_state_dict(best_state)
    return model


def metric_row(name, shot_true, shot_pred, text_true=None, text_pred=None):
    row = {
        "experiment": name,
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
    }
    if text_true is not None and text_pred is not None:
        row.update(
            {
                "text_acc": accuracy_score(text_true, text_pred),
                "text_f1": f1_score(text_true, text_pred, average="binary", zero_division=0),
                "joint_acc": float(np.mean((shot_true == shot_pred) & (text_true == text_pred))),
            }
        )
    return row


def main() -> None:
    print("device:", DEVICE)
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

    clip_val_loader = DataLoader(
        TensorDataset(tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]])),
        batch_size=256,
        shuffle=False,
    )
    clip_test_loader = DataLoader(
        TensorDataset(tx(clip_emb[masks["test"]]), ty(labels[masks["test"]]), ty(text[masks["test"]])),
        batch_size=256,
        shuffle=False,
    )
    dino_val_loader = DataLoader(TensorDataset(tx(dino_emb[masks["val"]]), ty(labels[masks["val"]])), batch_size=256)
    dino_test_loader = DataLoader(TensorDataset(tx(dino_emb[masks["test"]]), ty(labels[masks["test"]])), batch_size=256)

    shot_val_true, text_val_true, clip_val_shot_prob, clip_val_text_prob = multitask_predict(clip_model, clip_val_loader)
    shot_test_true, text_test_true, clip_test_shot_prob, clip_test_text_prob = multitask_predict(clip_model, clip_test_loader)
    _, dino_val_shot_prob = shot_predict(dino_model, dino_val_loader)
    _, dino_test_shot_prob = shot_predict(dino_model, dino_test_loader)

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

    test_prob = best_alpha * clip_test_shot_prob + (1 - best_alpha) * dino_test_shot_prob
    test_shot_pred = test_prob.argmax(1)
    test_text_pred = clip_test_text_prob.argmax(1)

    rows = [
        metric_row("clip_stronger_retrained", shot_test_true, clip_test_shot_prob.argmax(1), text_test_true, clip_test_text_prob.argmax(1)),
        metric_row("dinov2_shot_retrained", shot_test_true, dino_test_shot_prob.argmax(1)),
        metric_row(f"ensemble_alpha_clip_{best_alpha:.2f}", shot_test_true, test_shot_pred, text_test_true, test_text_pred),
    ]

    pd.DataFrame(alpha_rows).to_csv(OUTPUT_DIR / "ensemble_alpha_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ensemble_experiment_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "ensemble_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_alpha_clip": best_alpha, "rows": rows}, f, indent=2)
    print("best_alpha_clip:", best_alpha)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
