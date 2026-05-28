from __future__ import annotations

import json
import random
from dataclasses import dataclass
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
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "clip_head"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_LOSS_WEIGHT = 1.0
EPOCHS = 100
PATIENCE = 15


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class SingleTaskHead(nn.Module):
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
    mode: str
    hidden_dims: tuple[int, ...]
    dropout: float = 0.2
    closeup_factor: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-3


def class_weights(labels: torch.Tensor, num_classes: int, closeup_factor: float = 1.0) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * torch.clamp(counts, min=1))
    weights[0] *= closeup_factor
    return weights


def metrics(shot_true, shot_pred, text_true=None, text_pred=None) -> dict[str, float]:
    out = {
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
    }
    if text_true is not None and text_pred is not None:
        shot_true_np = np.array(shot_true)
        shot_pred_np = np.array(shot_pred)
        text_true_np = np.array(text_true)
        text_pred_np = np.array(text_pred)
        out.update(
            {
                "text_acc": accuracy_score(text_true, text_pred),
                "text_f1": f1_score(text_true, text_pred, average="binary", zero_division=0),
                "joint_acc": float(np.mean((shot_true_np == shot_pred_np) & (text_true_np == text_pred_np))),
            }
        )
    return out


def predict_multitask(model, loader):
    model.eval()
    shot_true, shot_pred, text_true, text_pred = [], [], [], []
    shot_prob_rows, text_prob_rows = [], []
    with torch.no_grad():
        for x, y_shot, y_text in loader:
            x = x.to(DEVICE)
            shot_logits, text_logits = model(x)
            shot_prob = torch.softmax(shot_logits, dim=1).cpu().numpy()
            text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()
            shot_true.extend(y_shot.numpy().tolist())
            text_true.extend(y_text.numpy().tolist())
            shot_pred.extend(shot_prob.argmax(axis=1).tolist())
            text_pred.extend(text_prob.argmax(axis=1).tolist())
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


def predict_single(model, loader):
    model.eval()
    y_true, y_pred = [], []
    prob_rows = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(prob.argmax(axis=1).tolist())
            prob_rows.append(prob)
    return np.array(y_true), np.array(y_pred), np.vstack(prob_rows)


def train_multitask(config: Config, data) -> dict:
    seed_everything()
    x_train, y_shot_train, y_text_train, x_val, y_shot_val, y_text_val, x_test, y_shot_test, y_text_test = data
    train_loader = DataLoader(TensorDataset(x_train, y_shot_train, y_text_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_shot_val, y_text_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_shot_test, y_text_test), batch_size=256, shuffle=False)

    model = MultiTaskHead(x_train.shape[1], config.hidden_dims, config.dropout).to(DEVICE)
    shot_loss = nn.CrossEntropyLoss(weight=class_weights(y_shot_train, len(SHOT_TYPES), config.closeup_factor).to(DEVICE))
    text_loss = nn.CrossEntropyLoss(weight=class_weights(y_text_train, 2).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_score = -1
    best_state = None
    wait = PATIENCE
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y_shot, y_text in train_loader:
            x = x.to(DEVICE)
            y_shot = y_shot.to(DEVICE)
            y_text = y_text.to(DEVICE)
            optimizer.zero_grad()
            shot_logits, text_logits = model(x)
            loss = shot_loss(shot_logits, y_shot) + TEXT_LOSS_WEIGHT * text_loss(text_logits, y_text)
            loss.backward()
            optimizer.step()

        val_pred = predict_multitask(model, val_loader)
        val_metrics = metrics(val_pred["shot_true"], val_pred["shot_pred"], val_pred["text_true"], val_pred["text_pred"])
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        score = val_metrics["joint_acc"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break

    model.load_state_dict(best_state)
    test_pred = predict_multitask(model, test_loader)
    test_metrics = metrics(test_pred["shot_true"], test_pred["shot_pred"], test_pred["text_true"], test_pred["text_pred"])
    return {"config": config.__dict__, "history": history, "test_metrics": test_metrics, "test_pred": test_pred, "state": model.state_dict()}


def train_single(config: Config, data, task: str) -> dict:
    seed_everything()
    x_train, y_shot_train, y_text_train, x_val, y_shot_val, y_text_val, x_test, y_shot_test, y_text_test = data
    y_train = y_shot_train if task == "shot" else y_text_train
    y_val = y_shot_val if task == "shot" else y_text_val
    y_test = y_shot_test if task == "shot" else y_text_test
    out_dim = len(SHOT_TYPES) if task == "shot" else 2

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256, shuffle=False)

    model = SingleTaskHead(x_train.shape[1], config.hidden_dims, config.dropout, out_dim).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(y_train, out_dim, config.closeup_factor if task == "shot" else 1.0).to(DEVICE)
    )
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

        val_true, val_pred, _ = predict_single(model, val_loader)
        if task == "shot":
            score = f1_score(val_true, val_pred, average="macro", zero_division=0)
            val_metrics = {
                "shot_acc": accuracy_score(val_true, val_pred),
                "shot_macro_f1": score,
            }
        else:
            score = f1_score(val_true, val_pred, average="binary", zero_division=0)
            val_metrics = {
                "text_acc": accuracy_score(val_true, val_pred),
                "text_f1": score,
            }
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break

    model.load_state_dict(best_state)
    test_true, test_pred, test_prob = predict_single(model, test_loader)
    return {"config": config.__dict__, "history": history, "true": test_true, "pred": test_pred, "prob": test_prob, "state": model.state_dict()}


def main() -> None:
    print("device:", DEVICE)
    meta = pd.read_csv(INPUT_DIR / "clip_embedding_metadata.csv")
    emb = np.load(INPUT_DIR / "clip_vit_b32_openai_embeddings.npz")["embeddings"].astype("float32")
    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    shot_labels = meta["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text_labels = meta["has_text"].to_numpy(dtype=np.int64)

    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}

    def t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    def y(arr):
        return torch.tensor(arr, dtype=torch.long)

    data = (
        t(emb[masks["train"]]),
        y(shot_labels[masks["train"]]),
        y(text_labels[masks["train"]]),
        t(emb[masks["val"]]),
        y(shot_labels[masks["val"]]),
        y(text_labels[masks["val"]]),
        t(emb[masks["test"]]),
        y(shot_labels[masks["test"]]),
        y(text_labels[masks["test"]]),
    )

    multitask_configs = [
        Config("multitask_baseline_256", "multitask", (256,), 0.20, 1.0),
        Config("multitask_stronger_512_256", "multitask", (512, 256), 0.25, 1.0),
        Config("multitask_closeup_weight_1_8", "multitask", (256,), 0.20, 1.8),
        Config("multitask_stronger_closeup_1_5", "multitask", (512, 256), 0.25, 1.5),
    ]
    single_configs = [
        Config("shot_only_512_256", "shot", (512, 256), 0.25, 1.0),
        Config("shot_only_closeup_weight_1_8", "shot", (512, 256), 0.25, 1.8),
        Config("text_only_256", "text", (256,), 0.20, 1.0),
    ]

    summary_rows = []
    saved_payloads = {}
    for config in multitask_configs:
        print("running", config.name)
        result = train_multitask(config, data)
        saved_payloads[config.name] = result
        row = {"experiment": config.name, **result["test_metrics"]}
        summary_rows.append(row)
        print(row)

        pred = result["test_pred"]
        pred_df = meta[masks["test"]].copy().reset_index(drop=True)
        pred_df["shot_pred"] = [SHOT_TYPES[i] for i in pred["shot_pred"]]
        pred_df["text_pred"] = pred["text_pred"]
        pred_df.to_csv(OUTPUT_DIR / f"{config.name}_test_predictions.csv", index=False, encoding="utf-8-sig")

    single_results = {}
    for config in single_configs:
        print("running", config.name)
        task = "shot" if config.mode == "shot" else "text"
        result = train_single(config, data, task)
        single_results[config.name] = result
        if task == "shot":
            row = {
                "experiment": config.name,
                "shot_acc": accuracy_score(result["true"], result["pred"]),
                "shot_macro_f1": f1_score(result["true"], result["pred"], average="macro", zero_division=0),
            }
        else:
            row = {
                "experiment": config.name,
                "text_acc": accuracy_score(result["true"], result["pred"]),
                "text_f1": f1_score(result["true"], result["pred"], average="binary", zero_division=0),
            }
        summary_rows.append(row)
        print(row)

    best_shot_name = max(
        [name for name in single_results if name.startswith("shot_only")],
        key=lambda name: f1_score(single_results[name]["true"], single_results[name]["pred"], average="macro", zero_division=0),
    )
    text_name = "text_only_256"
    shot_result = single_results[best_shot_name]
    text_result = single_results[text_name]
    combined = metrics(shot_result["true"], shot_result["pred"], text_result["true"], text_result["pred"])
    combined_row = {"experiment": f"separate_{best_shot_name}_plus_{text_name}", **combined}
    summary_rows.append(combined_row)
    print(combined_row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "clip_head_experiment_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "clip_head_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    best = summary.sort_values(["shot_macro_f1", "joint_acc"], ascending=False, na_position="last").iloc[0].to_dict()
    with open(OUTPUT_DIR / "best_clip_head_experiment.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("best_by_shot_macro_f1:", best)


if __name__ == "__main__":
    main()
