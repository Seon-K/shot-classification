from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class ShotClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        norm: str = "none",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(embedding_dim, hidden_dim)]
        if norm == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == "layer":
            layers.append(nn.LayerNorm(hidden_dim))
        elif norm != "none":
            raise ValueError(f"Unsupported norm: {norm}")
        layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        )
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x.float(), dim=-1)
        return self.fc(x)


class TextPresenceClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        norm: str = "none",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(embedding_dim, hidden_dim)]
        if norm == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == "layer":
            layers.append(nn.LayerNorm(hidden_dim))
        elif norm != "none":
            raise ValueError(f"Unsupported norm: {norm}")
        layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 2),
            ]
        )
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x.float(), dim=-1)
        return self.fc(x)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_embedding_data(path: str | Path, target: str = "shot_type") -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    data = torch.load(path, map_location="cpu")
    embeddings: torch.Tensor = data["embeddings"].float()
    records: list[dict] = data["records"]
    if target == "shot_type":
        labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.long)
    elif target == "has_text":
        labels = torch.tensor(
            [1 if parse_bool(record["has_text"]) else 0 for record in records],
            dtype=torch.long,
        )
    else:
        raise ValueError(f"Unsupported target: {target}")
    return embeddings, labels, records


def load_embedding_dataset(
    path: str | Path,
    target: str = "shot_type",
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, labels, _ = load_embedding_data(path, target=target)
    return embeddings, labels


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    total = counts.sum()
    weights = total / (counts.clamp_min(1.0) * num_classes)
    weights[counts == 0] = 0.0
    return weights


def build_embedding_loader(
    embedding_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = False,
    target: str = "shot_type",
    use_weighted_sampler: bool = False,
    num_classes: int | None = None,
) -> DataLoader:
    embeddings, labels = load_embedding_dataset(embedding_path, target=target)
    dataset = TensorDataset(embeddings, labels)
    sampler = None
    if use_weighted_sampler:
        class_count = num_classes or int(labels.max().item()) + 1
        class_weights = compute_class_weights(labels, class_count)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)


def build_loss(
    loss_type: str = "ce",
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    weight = class_weights.to(device) if class_weights is not None and device is not None else class_weights
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if loss_type == "focal":
        return FocalLoss(weight=weight)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def confusion_matrix_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_idx, pred_idx in zip(labels.view(-1), predictions.view(-1)):
        matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def classification_report_from_confusion(
    confusion: torch.Tensor,
    class_names: list[str] | None = None,
) -> list[dict]:
    num_classes = int(confusion.shape[0])
    names = class_names or [str(idx) for idx in range(num_classes)]
    rows: list[dict] = []

    for class_idx in range(num_classes):
        tp = int(confusion[class_idx, class_idx].item())
        fp = int(confusion[:, class_idx].sum().item()) - tp
        fn = int(confusion[class_idx, :].sum().item()) - tp
        support = int(confusion[class_idx, :].sum().item())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_accuracy = tp / support if support else 0.0
        rows.append(
            {
                "class": names[class_idx],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "per_class_accuracy": per_class_accuracy,
                "support": support,
            }
        )

    return rows


def summarize_metrics_from_confusion(confusion: torch.Tensor, report: list[dict]) -> dict:
    total = int(confusion.sum().item())
    correct = int(torch.diag(confusion).sum().item())
    accuracy = correct / total if total else 0.0
    macro_f1 = sum(row["f1"] for row in report) / len(report) if report else 0.0
    balanced_accuracy = sum(row["recall"] for row in report) / len(report) if report else 0.0
    weighted_f1 = (
        sum(row["f1"] * row["support"] for row in report) / total
        if total and report
        else 0.0
    )
    per_class_accuracy = {row["class"]: row["per_class_accuracy"] for row in report}
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": per_class_accuracy,
    }


def build_misclassification_analysis(confusion: torch.Tensor, class_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for true_idx, true_label in enumerate(class_names):
        true_total = int(confusion[true_idx, :].sum().item())
        if true_total == 0:
            continue
        for pred_idx, pred_label in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx].item())
            if count == 0:
                continue
            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                    "ratio_within_true": count / true_total,
                }
            )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def write_confusion_matrix_csv(path: str | Path, confusion: torch.Tensor, class_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, confusion.tolist()):
            writer.writerow([class_name, *row])


def write_classification_report_csv(path: str | Path, report: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class", "precision", "recall", "f1", "per_class_accuracy", "support"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)


def write_misclassification_analysis_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["true_label", "pred_label", "count", "ratio_within_true"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list[str] | None = None,
    criterion: nn.Module | None = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_top2: list[torch.Tensor] = []
    criterion = criterion or nn.CrossEntropyLoss()

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.shape[0]
        total_count += labels.shape[0]
        all_predictions.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        top_k = min(2, logits.shape[1])
        all_top2.append(logits.topk(k=top_k, dim=1).indices.cpu())

    if all_predictions:
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        top2 = torch.cat(all_top2)
    else:
        predictions = torch.empty(0, dtype=torch.long)
        labels = torch.empty(0, dtype=torch.long)
        top2 = torch.empty((0, min(2, num_classes)), dtype=torch.long)

    confusion = confusion_matrix_from_predictions(predictions, labels, num_classes=num_classes)
    report = classification_report_from_confusion(confusion, class_names=class_names)
    summary = summarize_metrics_from_confusion(confusion, report)
    top2_accuracy = float((top2 == labels.unsqueeze(1)).any(dim=1).float().mean().item()) if total_count else 0.0
    class_names = class_names or [str(idx) for idx in range(num_classes)]
    misclassification = build_misclassification_analysis(confusion, class_names)

    return {
        "loss": total_loss / total_count if total_count else 0.0,
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "balanced_accuracy": summary["balanced_accuracy"],
        "weighted_f1": summary["weighted_f1"],
        "top2_accuracy": top2_accuracy,
        "per_class_accuracy": summary["per_class_accuracy"],
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "misclassification_analysis": misclassification,
    }


@torch.no_grad()
def collect_prediction_rows(
    model: nn.Module,
    embedding_path: str | Path,
    device: torch.device,
    class_names: list[str],
    target: str = "shot_type",
    batch_size: int = 256,
) -> list[dict]:
    embeddings, labels, records = load_embedding_data(embedding_path, target=target)
    rows: list[dict] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        end = start + batch_size
        batch_embeddings = embeddings[start:end].to(device)
        batch_labels = labels[start:end]
        logits = model(batch_embeddings)
        probabilities = torch.softmax(logits, dim=1).cpu()
        confidence, predictions = probabilities.max(dim=1)
        for offset, record in enumerate(records[start:end]):
            true_idx = int(batch_labels[offset].item())
            pred_idx = int(predictions[offset].item())
            rows.append(
                {
                    "path": record.get("path", ""),
                    "video_id": record.get("video_id", ""),
                    "cut_id": record.get("cut_id", ""),
                    "true_label": class_names[true_idx],
                    "pred_label": class_names[pred_idx],
                    "confidence": float(confidence[offset].item()),
                }
            )
    return rows


def save_error_samples(
    output_dir: str | Path,
    prediction_rows: list[dict],
    high_count: int = 20,
    low_count: int = 20,
    copy_images: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = [row for row in prediction_rows if row["true_label"] != row["pred_label"]]
    high_errors = sorted(errors, key=lambda row: row["confidence"], reverse=True)[:high_count]
    low_errors = sorted(errors, key=lambda row: row["confidence"])[:low_count]

    def write_rows(path: Path, rows: list[dict]) -> None:
        fieldnames = ["path", "video_id", "cut_id", "true_label", "pred_label", "confidence"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(output_dir / "high_confidence_errors.csv", high_errors)
    write_rows(output_dir / "low_confidence_errors.csv", low_errors)

    if not copy_images:
        return
    for group_name, rows in {"high": high_errors, "low": low_errors}.items():
        image_dir = output_dir / group_name
        image_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            src = Path(row["path"])
            if not src.exists():
                continue
            filename = (
                f"true_{row['true_label']}_pred_{row['pred_label']}_"
                f"{row['video_id']}_{row['cut_id']}.jpg"
            ).replace("/", "_")
            shutil.copy2(src, image_dir / filename)


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    embedding_dim: int = 512,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    norm: str = "none",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_type: str = "ce",
    class_weights: torch.Tensor | None = None,
    patience: int = 0,
    scheduler: bool = False,
    device: torch.device | None = None,
) -> tuple[ShotClassifier, list[dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShotClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        norm=norm,
    ).to(device)
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        loss_type=loss_type,
        class_weights=class_weights,
        patience=patience,
        scheduler=scheduler,
        device=device,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_type: str = "ce",
    class_weights: torch.Tensor | None = None,
    patience: int = 0,
    scheduler: bool = False,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = build_loss(loss_type=loss_type, class_weights=class_weights, device=device)
    lr_scheduler = None
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, patience // 2 or 1),
        )
    history: list[dict] = []
    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_state_dict = deepcopy(model.state_dict())
    epochs_without_improvement = 0

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
            "loss": total_loss / total_count if total_count else 0.0,
            "accuracy": total_correct / total_count if total_count else 0.0,
        }
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
            criterion=criterion,
        )

        is_best = val_metrics["macro_f1"] > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": is_best,
            }
        )

        if patience and epochs_without_improvement >= patience:
            break

    model._best_state_dict = best_state_dict
    model._last_state_dict = deepcopy(model.state_dict())
    model._best_epoch = best_epoch
    return model, history


def save_training_outputs(
    output_dir: str | Path,
    model: nn.Module,
    history: list[dict],
    test_metrics: dict | None = None,
    class_to_idx: dict[str, int] | None = None,
    test_metrics_last: dict | None = None,
    test_metrics_best: dict | None = None,
    checkpoint_prefix: str = "shot_classifier",
    training_config: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_idx = class_to_idx or {}
    idx_to_class = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    last_state_dict = getattr(model, "_last_state_dict", model.state_dict())
    best_state_dict = getattr(model, "_best_state_dict", model.state_dict())
    torch.save(last_state_dict, output_dir / f"{checkpoint_prefix}_last.pt")
    torch.save(best_state_dict, output_dir / f"{checkpoint_prefix}_best.pt")
    if checkpoint_prefix == "shot_classifier":
        torch.save(best_state_dict, output_dir / "shot_classifier.pt")
    elif checkpoint_prefix == "text_classifier":
        torch.save(best_state_dict, output_dir / "text_classifier.pt")

    best_row = max(history, key=lambda item: item["val_macro_f1"]) if history else {}
    test_metrics_last = test_metrics_last or test_metrics or {}
    test_metrics_best = test_metrics_best or test_metrics or {}

    if test_metrics_best and idx_to_class:
        confusion = torch.tensor(test_metrics_best.get("confusion_matrix", []), dtype=torch.long)
        report = test_metrics_best.get("classification_report", [])
        misclassification = test_metrics_best.get("misclassification_analysis", [])
        if confusion.numel():
            write_confusion_matrix_csv(output_dir / "confusion_matrix.csv", confusion, idx_to_class)
        if report:
            write_classification_report_csv(output_dir / "classification_report.csv", report)
        write_misclassification_analysis_csv(output_dir / "misclassification_analysis.csv", misclassification)

    summary = {
        "class_to_idx": class_to_idx,
        "training_config": training_config or {},
        "history": history,
        "best_epoch": best_row.get("epoch", 0),
        "best_val_accuracy": best_row.get("val_accuracy", 0.0),
        "best_val_macro_f1": best_row.get("val_macro_f1", 0.0),
        "test_metrics_last": test_metrics_last,
        "test_metrics_best": test_metrics_best,
        "misclassification_analysis": test_metrics_best.get("misclassification_analysis", []),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
