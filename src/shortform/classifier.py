from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ShotClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TextPresenceClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_embedding_dataset(
    path: str | Path,
    target: str = "shot_type",
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    embeddings: torch.Tensor = data["embeddings"].float()
    if target == "shot_type":
        labels = torch.tensor([int(record["label"]) for record in data["records"]], dtype=torch.long)
    elif target == "has_text":
        labels = torch.tensor(
            [1 if parse_bool(record["has_text"]) else 0 for record in data["records"]],
            dtype=torch.long,
        )
    else:
        raise ValueError(f"Unsupported target: {target}")
    return embeddings, labels


def build_embedding_loader(
    embedding_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = False,
    target: str = "shot_type",
) -> DataLoader:
    embeddings, labels = load_embedding_dataset(embedding_path, target=target)
    dataset = TensorDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def macro_f1_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    f1_scores: list[float] = []
    for class_idx in range(num_classes):
        pred_positive = predictions == class_idx
        true_positive = labels == class_idx
        tp = (pred_positive & true_positive).sum().item()
        fp = (pred_positive & ~true_positive).sum().item()
        fn = (~pred_positive & true_positive).sum().item()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    criterion = nn.CrossEntropyLoss()

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.shape[0]
        total_count += labels.shape[0]
        all_predictions.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    accuracy = float((predictions == labels).float().mean().item())
    macro_f1 = macro_f1_from_predictions(predictions, labels, num_classes=num_classes)

    return {
        "loss": total_loss / total_count if total_count else 0.0,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    embedding_dim: int = 512,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> tuple[ShotClassifier, list[dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShotClassifier(embedding_dim=embedding_dim, num_classes=num_classes).to(device)
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        device=device,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        total_correct = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.shape[0]
            total_count += labels.shape[0]
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_metrics = {
            "loss": total_loss / total_count,
            "accuracy": total_correct / total_count,
        }
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

    return model, history


def save_training_outputs(
    output_dir: str | Path,
    model: nn.Module,
    history: list[dict],
    test_metrics: dict[str, float],
    class_to_idx: dict[str, int],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "shot_classifier.pt")
    summary = {
        "class_to_idx": class_to_idx,
        "history": history,
        "best_val_accuracy": max(item["val_accuracy"] for item in history),
        "best_val_macro_f1": max(item["val_macro_f1"] for item in history),
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
