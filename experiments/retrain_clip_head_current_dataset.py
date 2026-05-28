from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "labeled_dataset"
BASELINE_DIR = ROOT / "outputs" / "baseline"
OUTPUT_DIR = ROOT / "outputs" / "clip_embeddings"
CHECKPOINT_DIR = ROOT / "checkpoints"

SHOT_TYPES = ["close-up", "medium", "object", "space", "wide"]
VALID_JOINT_LABELS = {f"{shot}_{text}" for shot in SHOT_TYPES for text in ["notext", "text"]}

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 80
PATIENCE = 12
TEXT_LOSS_WEIGHT = 1.0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def infer_video_id(filename: str) -> str:
    match = re.match(r"(\d{4})", filename)
    if match:
        return match.group(1)
    return Path(filename).stem.split("_")[0]


def build_dataset_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        original_class = class_dir.name
        image_paths = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if original_class not in VALID_JOINT_LABELS:
            for path in image_paths:
                skipped.append(
                    {
                        "filepath": str(path.resolve()),
                        "filename": path.name,
                        "original_class": original_class,
                        "reason": "unsupported_label",
                    }
                )
            continue

        shot_type, text_label = original_class.rsplit("_", 1)
        has_text = 1 if text_label == "text" else 0

        for path in image_paths:
            rows.append(
                {
                    "filepath": str(path.resolve()),
                    "filename": path.name,
                    "video_id": infer_video_id(path.name),
                    "original_class": original_class,
                    "shot_type": shot_type,
                    "has_text": has_text,
                    "joint_label": original_class,
                }
            )

    df = pd.DataFrame(rows).sort_values(["video_id", "original_class", "filename"]).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped)
    return df, skipped_df


def add_video_level_splits(df: pd.DataFrame) -> pd.DataFrame:
    groups = df["video_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_idx, temp_idx = next(gss.split(df, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df["video_id"].to_numpy()
    gss_temp = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    val_rel_idx, test_rel_idx = next(gss_temp.split(temp_df, groups=temp_groups))

    split = pd.Series(index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[temp_df.iloc[val_rel_idx].index] = "val"
    split.iloc[temp_df.iloc[test_rel_idx].index] = "test"

    out = df.copy()
    out["split"] = split
    return out


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


class ClipEmbeddingMultiTaskHead(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256, num_shot_classes=5, dropout=0.20):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shot_head = nn.Linear(hidden_dim, num_shot_classes)
        self.text_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


def make_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    return torch.tensor(
        [total / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )


def extract_embeddings(df: pd.DataFrame, preprocess, model) -> np.ndarray:
    loader = DataLoader(ImagePathDataset(df, preprocess), batch_size=64, shuffle=False, num_workers=0)
    chunks = []
    order = []

    with torch.no_grad():
        for images, indices in tqdm(loader, desc="CLIP embeddings"):
            images = images.to(DEVICE)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().float().numpy())
            order.extend(indices.numpy().tolist())

    order = np.array(order)
    assert np.all(order == np.arange(len(df)))
    return np.concatenate(chunks, axis=0).astype("float32")


def evaluate(head, loader, shot_criterion, text_criterion):
    head.eval()
    total_loss = 0.0
    shot_true, shot_pred = [], []
    text_true, text_pred = [], []

    with torch.no_grad():
        for x, y_shot, y_text in loader:
            x = x.to(DEVICE)
            y_shot = y_shot.to(DEVICE)
            y_text = y_text.to(DEVICE)
            shot_logits, text_logits = head(x)
            loss = shot_criterion(shot_logits, y_shot) + TEXT_LOSS_WEIGHT * text_criterion(text_logits, y_text)
            total_loss += loss.item() * x.size(0)
            shot_true.extend(y_shot.cpu().numpy().tolist())
            shot_pred.extend(shot_logits.argmax(1).cpu().numpy().tolist())
            text_true.extend(y_text.cpu().numpy().tolist())
            text_pred.extend(text_logits.argmax(1).cpu().numpy().tolist())

    shot_true_np = np.array(shot_true)
    shot_pred_np = np.array(shot_pred)
    text_true_np = np.array(text_true)
    text_pred_np = np.array(text_pred)

    return {
        "loss": total_loss / len(loader.dataset),
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
        "text_acc": accuracy_score(text_true, text_pred),
        "text_f1": f1_score(text_true, text_pred, average="binary", zero_division=0),
        "joint_acc": float(np.mean((shot_true_np == shot_pred_np) & (text_true_np == text_pred_np))),
    }


def main() -> None:
    seed_everything()
    BASELINE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    df, skipped_df = build_dataset_index()
    df = add_video_level_splits(df)

    split_csv = BASELINE_DIR / "dataset_index_with_splits.csv"
    skipped_csv = BASELINE_DIR / "dataset_index_skipped_rows.csv"
    df.to_csv(split_csv, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_csv, index=False, encoding="utf-8-sig")

    print("device:", DEVICE)
    print("rows:", len(df))
    print("videos:", df["video_id"].nunique())
    print("skipped:", len(skipped_df))
    print("split counts")
    print(df["split"].value_counts().sort_index())
    print("joint label split")
    print(pd.crosstab(df["joint_label"], df["split"]))

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    model.eval()
    embeddings = extract_embeddings(df, preprocess, model)

    np.savez_compressed(
        OUTPUT_DIR / "clip_vit_b32_openai_embeddings.npz",
        embeddings=embeddings,
        has_text=df["has_text"].to_numpy(dtype=np.int64),
        split=df["split"].to_numpy(),
        shot_type=df["shot_type"].to_numpy(),
        filepath=df["filepath"].to_numpy(),
    )
    df.assign(embedding_index=np.arange(len(df))).to_csv(
        OUTPUT_DIR / "clip_embedding_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shot_to_idx = {name: i for i, name in enumerate(SHOT_TYPES)}
    shot_labels = df["shot_type"].map(shot_to_idx).to_numpy(dtype=np.int64)
    text_labels = df["has_text"].to_numpy(dtype=np.int64)

    def split_arrays(split_name: str):
        mask = df["split"].to_numpy() == split_name
        x = torch.tensor(embeddings[mask], dtype=torch.float32)
        y_shot = torch.tensor(shot_labels[mask], dtype=torch.long)
        y_text = torch.tensor(text_labels[mask], dtype=torch.long)
        return x, y_shot, y_text, mask

    x_train, y_shot_train, y_text_train, train_mask = split_arrays("train")
    x_val, y_shot_val, y_text_val, val_mask = split_arrays("val")
    x_test, y_shot_test, y_text_test, test_mask = split_arrays("test")

    train_loader = DataLoader(TensorDataset(x_train, y_shot_train, y_text_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_shot_val, y_text_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_shot_test, y_text_test), batch_size=256, shuffle=False)

    head = ClipEmbeddingMultiTaskHead(embedding_dim=embeddings.shape[1], hidden_dim=256).to(DEVICE)
    shot_criterion = nn.CrossEntropyLoss(weight=make_class_weights(y_shot_train, len(SHOT_TYPES)).to(DEVICE))
    text_criterion = nn.CrossEntropyLoss(weight=make_class_weights(y_text_train, 2).to(DEVICE))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)

    history = []
    best_val_joint = -1.0
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        head.train()
        for x, y_shot, y_text in train_loader:
            x = x.to(DEVICE)
            y_shot = y_shot.to(DEVICE)
            y_text = y_text.to(DEVICE)
            optimizer.zero_grad()
            shot_logits, text_logits = head(x)
            loss = shot_criterion(shot_logits, y_shot) + TEXT_LOSS_WEIGHT * text_criterion(text_logits, y_text)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(head, train_loader, shot_criterion, text_criterion)
        val_metrics = evaluate(head, val_loader, shot_criterion, text_criterion)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        if epoch == 1 or epoch % 5 == 0:
            print(row)

        if val_metrics["joint_acc"] > best_val_joint:
            best_val_joint = val_metrics["joint_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping:", epoch)
                break

    assert best_state is not None
    head.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "shot_types": SHOT_TYPES,
            "shot_to_idx": shot_to_idx,
            "idx_to_shot": {idx: name for name, idx in shot_to_idx.items()},
            "embedding_dim": embeddings.shape[1],
            "hidden_dim": 256,
            "clip_model": MODEL_NAME,
            "clip_model_name": MODEL_NAME,
            "clip_pretrained": PRETRAINED,
            "text_loss_weight": TEXT_LOSS_WEIGHT,
            "seed": SEED,
            "dataset_rows": len(df),
            "skipped_rows": len(skipped_df),
        },
        CHECKPOINT_DIR / "clip_vit_b32_multitask_head.pt",
    )

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "clip_head_training_history.csv", index=False)

    head.eval()
    shot_true, shot_pred = [], []
    text_true, text_pred = [], []
    shot_conf, text_prob = [], []

    with torch.no_grad():
        for x, y_shot, y_text in test_loader:
            x = x.to(DEVICE)
            shot_logits, text_logits = head(x)
            shot_p = torch.softmax(shot_logits, dim=1)
            text_p = torch.softmax(text_logits, dim=1)
            shot_true.extend(y_shot.numpy().tolist())
            text_true.extend(y_text.numpy().tolist())
            shot_pred.extend(shot_p.argmax(1).cpu().numpy().tolist())
            text_pred.extend(text_p.argmax(1).cpu().numpy().tolist())
            shot_conf.extend(shot_p.max(1).values.cpu().numpy().tolist())
            text_prob.extend(text_p[:, 1].cpu().numpy().tolist())

    test_metrics = {
        "shot_acc": accuracy_score(shot_true, shot_pred),
        "shot_macro_f1": f1_score(shot_true, shot_pred, average="macro", zero_division=0),
        "text_acc": accuracy_score(text_true, text_pred),
        "text_f1": f1_score(text_true, text_pred, average="binary", zero_division=0),
        "joint_acc": float(
            np.mean((np.array(shot_true) == np.array(shot_pred)) & (np.array(text_true) == np.array(text_pred)))
        ),
    }

    with open(OUTPUT_DIR / "clip_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    test_df = df[test_mask].copy().reset_index(drop=True)
    test_df["shot_true"] = [SHOT_TYPES[i] for i in shot_true]
    test_df["shot_pred"] = [SHOT_TYPES[i] for i in shot_pred]
    test_df["text_true"] = text_true
    test_df["text_pred"] = text_pred
    test_df["shot_conf"] = shot_conf
    test_df["text_prob"] = text_prob
    test_df.to_csv(OUTPUT_DIR / "clip_test_predictions.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(test_metrics, indent=2))
    print("\nShot type report")
    print(classification_report(shot_true, shot_pred, target_names=SHOT_TYPES, zero_division=0))
    print("\nText report")
    print(classification_report(text_true, text_pred, target_names=["notext", "text"], zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    shot_cm = confusion_matrix(shot_true, shot_pred, labels=list(range(len(SHOT_TYPES))))
    sns.heatmap(shot_cm, annot=True, fmt="d", cmap="Blues", xticklabels=SHOT_TYPES, yticklabels=SHOT_TYPES, ax=axes[0])
    axes[0].set_title("CLIP Shot Type Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    text_cm = confusion_matrix(text_true, text_pred, labels=[0, 1])
    sns.heatmap(text_cm, annot=True, fmt="d", cmap="Greens", xticklabels=["notext", "text"], yticklabels=["notext", "text"], ax=axes[1])
    axes[1].set_title("CLIP Text Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clip_confusion_matrices.png", dpi=160)


if __name__ == "__main__":
    main()
