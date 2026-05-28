from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
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
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "ensemble_hyperparam_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class TrainConfig:
    name: str
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-3
    closeup_factor: float = 1.0
    layernorm: bool = False
    loss_type: str = "ce"
    label_smoothing: float = 0.0


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float, layernorm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        self.shared = nn.Sequential(*layers)
        self.shot_head = nn.Linear(prev, len(SHOT_TYPES))
        self.text_head = nn.Linear(prev, 2)

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


class ShotHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float, layernorm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, len(SHOT_TYPES)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
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


def class_weights(labels: torch.Tensor, num_classes: int, closeup_factor=1.0) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * torch.clamp(counts, min=1))
    if num_classes == len(SHOT_TYPES):
        weights[0] *= closeup_factor
    return weights


def make_loss(labels: torch.Tensor, num_classes: int, config: TrainConfig, closeup=True):
    factor = config.closeup_factor if closeup else 1.0
    weights = class_weights(labels, num_classes, factor).to(DEVICE)
    if config.loss_type == "focal":
        return FocalLoss(weight=weights, gamma=2.0, label_smoothing=config.label_smoothing)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)


def predict_multitask(model, loader):
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


def predict_shot(model, loader):
    model.eval()
    true = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            true.extend(y.numpy().tolist())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.array(true), np.vstack(probs)


def train_clip(config: TrainConfig, x_train, y_shot_train, y_text_train, x_val, y_shot_val, y_text_val):
    seed_everything()
    train_loader = DataLoader(TensorDataset(x_train, y_shot_train, y_text_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_shot_val, y_text_val), batch_size=256, shuffle=False)
    model = MultiTaskHead(x_train.shape[1], config.hidden_dims, config.dropout, config.layernorm).to(DEVICE)
    shot_loss = make_loss(y_shot_train, len(SHOT_TYPES), config, closeup=True)
    text_loss = make_loss(y_text_train, 2, config, closeup=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = None
    best_joint = -1.0
    wait = PATIENCE
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y_shot, y_text in train_loader:
            optimizer.zero_grad()
            shot_logits, text_logits = model(x.to(DEVICE))
            loss = shot_loss(shot_logits, y_shot.to(DEVICE)) + text_loss(text_logits, y_text.to(DEVICE))
            loss.backward()
            optimizer.step()
        shot_true, text_true, shot_prob, text_prob = predict_multitask(model, val_loader)
        joint = np.mean((shot_true == shot_prob.argmax(1)) & (text_true == text_prob.argmax(1)))
        if joint > best_joint:
            best_joint = float(joint)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break
    model.load_state_dict(best_state)
    return model


def train_dino(config: TrainConfig, x_train, y_train, x_val, y_val):
    seed_everything()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)
    model = ShotHead(x_train.shape[1], config.hidden_dims, config.dropout, config.layernorm).to(DEVICE)
    loss_fn = make_loss(y_train, len(SHOT_TYPES), config, closeup=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = None
    best_macro = -1.0
    wait = PATIENCE
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()
        true, prob = predict_shot(model, val_loader)
        macro = f1_score(true, prob.argmax(1), average="macro", zero_division=0)
        if macro > best_macro:
            best_macro = float(macro)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = PATIENCE
        else:
            wait -= 1
            if wait <= 0:
                break
    model.load_state_dict(best_state)
    return model


def combine_global(clip_prob, dino_prob, alpha):
    return alpha * clip_prob + (1.0 - alpha) * dino_prob


def combine_classwise(clip_prob, dino_prob, alpha_vec):
    alpha = np.asarray(alpha_vec, dtype=np.float32).reshape(1, -1)
    prob = alpha * clip_prob + (1.0 - alpha) * dino_prob
    row_sum = np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
    return prob / row_sum


def metric_dict(name, shot_true, shot_prob, text_true, text_prob, text_threshold):
    shot_pred = shot_prob.argmax(1)
    text_pred = (text_prob[:, 1] >= text_threshold).astype(np.int64)
    return {
        "experiment": name,
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
        "text_acc": accuracy_score(text_true, text_pred),
        "text_macro_f1": f1_score(text_true, text_pred, average="macro", zero_division=0),
        "joint_acc": float(np.mean((shot_true == shot_pred) & (text_true == text_pred))),
        "text_threshold": float(text_threshold),
    }


def search_global_alpha(shot_true, clip_prob, dino_prob, lo=0.4, hi=0.6, step=0.01):
    rows = []
    best = None
    for alpha in np.round(np.arange(lo, hi + 1e-9, step), 2):
        prob = combine_global(clip_prob, dino_prob, float(alpha))
        pred = prob.argmax(1)
        row = {
            "alpha_clip": float(alpha),
            "val_shot_acc": accuracy_score(shot_true, pred),
            "val_shot_macro_f1": f1_score(shot_true, pred, average="macro", zero_division=0),
        }
        rows.append(row)
        if best is None or row["val_shot_macro_f1"] > best["val_shot_macro_f1"]:
            best = row
    return best, pd.DataFrame(rows)


def search_text_threshold(shot_true, shot_prob, text_true, text_prob, objective="joint"):
    rows = []
    best = None
    shot_pred = shot_prob.argmax(1)
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        text_pred = (text_prob[:, 1] >= float(threshold)).astype(np.int64)
        row = {
            "text_threshold": float(threshold),
            "val_text_macro_f1": f1_score(text_true, text_pred, average="macro", zero_division=0),
            "val_text_acc": accuracy_score(text_true, text_pred),
            "val_joint_acc": float(np.mean((shot_true == shot_pred) & (text_true == text_pred))),
        }
        rows.append(row)
        score_key = "val_joint_acc" if objective == "joint" else "val_text_macro_f1"
        if best is None or row[score_key] > best[score_key]:
            best = row
    return best, pd.DataFrame(rows)


def search_classwise_alpha(shot_true, clip_prob, dino_prob, initial_alpha=0.5, grid=None, rounds=3):
    if grid is None:
        grid = np.round(np.arange(0.0, 1.001, 0.05), 2)
    alpha_vec = np.full(len(SHOT_TYPES), initial_alpha, dtype=np.float32)
    rows = []
    best_macro = f1_score(shot_true, combine_classwise(clip_prob, dino_prob, alpha_vec).argmax(1), average="macro", zero_division=0)
    for round_idx in range(rounds):
        for class_idx, class_name in enumerate(SHOT_TYPES):
            class_best_alpha = float(alpha_vec[class_idx])
            class_best_macro = best_macro
            for alpha in grid:
                trial = alpha_vec.copy()
                trial[class_idx] = float(alpha)
                prob = combine_classwise(clip_prob, dino_prob, trial)
                pred = prob.argmax(1)
                macro = f1_score(shot_true, pred, average="macro", zero_division=0)
                rows.append({
                    "round": round_idx + 1,
                    "class": class_name,
                    "candidate_alpha": float(alpha),
                    "val_shot_macro_f1": macro,
                    **{f"alpha_{name}": float(value) for name, value in zip(SHOT_TYPES, trial)},
                })
                if macro > class_best_macro:
                    class_best_macro = float(macro)
                    class_best_alpha = float(alpha)
            alpha_vec[class_idx] = class_best_alpha
            best_macro = class_best_macro
    final_prob = combine_classwise(clip_prob, dino_prob, alpha_vec)
    final_pred = final_prob.argmax(1)
    best = {
        "val_shot_acc": accuracy_score(shot_true, final_pred),
        "val_shot_macro_f1": f1_score(shot_true, final_pred, average="macro", zero_division=0),
        **{f"alpha_{name}": float(value) for name, value in zip(SHOT_TYPES, alpha_vec)},
    }
    return best, pd.DataFrame(rows)


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

    base_config = TrainConfig("baseline")
    clip_model = train_clip(
        base_config,
        tx(clip_emb[masks["train"]]), ty(labels[masks["train"]]), ty(text[masks["train"]]),
        tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]]),
    )
    dino_model = train_dino(
        base_config,
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

    # 1. global alpha fine search
    alpha_best, alpha_rows = search_global_alpha(shot_val_true, clip_val_shot_prob, dino_val_shot_prob)
    alpha_rows.to_csv(OUTPUT_DIR / "01_global_alpha_fine_search.csv", index=False, encoding="utf-8-sig")
    global_val_prob = combine_global(clip_val_shot_prob, dino_val_shot_prob, alpha_best["alpha_clip"])
    global_test_prob = combine_global(clip_test_shot_prob, dino_test_shot_prob, alpha_best["alpha_clip"])

    # 2. class-wise alpha
    classwise_best, classwise_rows = search_classwise_alpha(shot_val_true, clip_val_shot_prob, dino_val_shot_prob, initial_alpha=alpha_best["alpha_clip"])
    classwise_rows.to_csv(OUTPUT_DIR / "02_classwise_alpha_search.csv", index=False, encoding="utf-8-sig")
    alpha_vec = np.array([classwise_best[f"alpha_{name}"] for name in SHOT_TYPES], dtype=np.float32)
    classwise_val_prob = combine_classwise(clip_val_shot_prob, dino_val_shot_prob, alpha_vec)
    classwise_test_prob = combine_classwise(clip_test_shot_prob, dino_test_shot_prob, alpha_vec)

    # 3. text threshold joint search
    text_best, text_rows = search_text_threshold(shot_val_true, classwise_val_prob, text_val_true, clip_val_text_prob, objective="joint")
    text_rows.to_csv(OUTPUT_DIR / "03_text_threshold_joint_search.csv", index=False, encoding="utf-8-sig")

    rows = []
    rows.append(metric_dict("baseline_global_alpha_0_50_text_0_50", shot_test_true, combine_global(clip_test_shot_prob, dino_test_shot_prob, 0.5), text_test_true, clip_test_text_prob, 0.5))
    rows.append(metric_dict(f"fine_global_alpha_{alpha_best['alpha_clip']:.2f}", shot_test_true, global_test_prob, text_test_true, clip_test_text_prob, 0.5))
    rows.append(metric_dict("classwise_alpha_text_0_50", shot_test_true, classwise_test_prob, text_test_true, clip_test_text_prob, 0.5))
    rows.append(metric_dict("classwise_alpha_joint_text_threshold", shot_test_true, classwise_test_prob, text_test_true, clip_test_text_prob, text_best["text_threshold"]))

    # 4-6. retrain closeup/class weights, lr/dropout/layernorm, focal/l smoothing
    configs = [
        TrainConfig("closeup_factor_1_2", closeup_factor=1.2),
        TrainConfig("closeup_factor_1_5", closeup_factor=1.5),
        TrainConfig("closeup_factor_1_8", closeup_factor=1.8),
        TrainConfig("lr_5e_4", lr=5e-4),
        TrainConfig("lr_3e_4", lr=3e-4),
        TrainConfig("dropout_0_10", dropout=0.10),
        TrainConfig("dropout_0_30", dropout=0.30),
        TrainConfig("layernorm", layernorm=True),
        TrainConfig("hidden_768_384", hidden_dims=(768, 384), dropout=0.30),
        TrainConfig("focal_loss", loss_type="focal"),
        TrainConfig("label_smoothing_0_05", label_smoothing=0.05),
        TrainConfig("focal_label_smoothing_0_05", loss_type="focal", label_smoothing=0.05),
    ]
    train_rows = []
    for config in configs:
        print("training", config.name)
        clip_model_cfg = train_clip(
            config,
            tx(clip_emb[masks["train"]]), ty(labels[masks["train"]]), ty(text[masks["train"]]),
            tx(clip_emb[masks["val"]]), ty(labels[masks["val"]]), ty(text[masks["val"]]),
        )
        dino_model_cfg = train_dino(
            config,
            tx(dino_emb[masks["train"]]), ty(labels[masks["train"]]),
            tx(dino_emb[masks["val"]]), ty(labels[masks["val"]]),
        )
        sv, tv, cvsp, cvtp = predict_multitask(clip_model_cfg, clip_val_loader)
        st, tt, ctsp, cttp = predict_multitask(clip_model_cfg, clip_test_loader)
        _, dvsp = predict_shot(dino_model_cfg, dino_val_loader)
        _, dtsp = predict_shot(dino_model_cfg, dino_test_loader)
        cfg_alpha_best, _ = search_global_alpha(sv, cvsp, dvsp)
        cfg_val_prob = combine_global(cvsp, dvsp, cfg_alpha_best["alpha_clip"])
        cfg_text_best, _ = search_text_threshold(sv, cfg_val_prob, tv, cvtp, objective="joint")
        cfg_test_prob = combine_global(ctsp, dtsp, cfg_alpha_best["alpha_clip"])
        row = metric_dict(config.name, st, cfg_test_prob, tt, cttp, cfg_text_best["text_threshold"])
        row.update({"alpha_clip": cfg_alpha_best["alpha_clip"], **asdict(config)})
        train_rows.append(row)
        rows.append(row)

    pd.DataFrame(train_rows).to_csv(OUTPUT_DIR / "04_05_06_training_hyperparam_search.csv", index=False, encoding="utf-8-sig")
    result_df = pd.DataFrame(rows).sort_values(["joint_acc", "shot_macro_f1"], ascending=False)
    result_df.to_csv(OUTPUT_DIR / "ensemble_hyperparam_tuning_summary.csv", index=False, encoding="utf-8-sig")
    best = result_df.iloc[0].to_dict()
    with open(OUTPUT_DIR / "best_ensemble_hyperparam_tuning.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(OUTPUT_DIR / "classwise_alpha_best.json", "w", encoding="utf-8") as f:
        json.dump(classwise_best, f, indent=2)
    with open(OUTPUT_DIR / "text_threshold_best.json", "w", encoding="utf-8") as f:
        json.dump(text_best, f, indent=2)
    print(result_df.to_string(index=False))
    print("best", best)


if __name__ == "__main__":
    main()
