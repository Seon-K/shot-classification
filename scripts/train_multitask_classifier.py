from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.classifier import (  # noqa: E402
    classification_report_from_confusion,
    confusion_matrix_from_predictions,
    parse_bool,
    summarize_metrics_from_confusion,
    write_classification_report_csv,
)
from shortform.utils.reproducibility import configure_reproducibility  # noqa: E402


SHOT_SCALE_CLASSES = ["close-up", "medium", "wide", "unknown"]
SHOT_SUBJECT_CLASSES = ["person", "object", "space", "text", "unknown"]
TEXT_CLASSES = ["notext", "text"]


class MultiTaskShotClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        shot_type_classes: int = 5,
        shot_scale_classes: int = 4,
        shot_subject_classes: int = 5,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shot_type_head = nn.Linear(hidden_dim, shot_type_classes)
        self.shot_scale_head = nn.Linear(hidden_dim, shot_scale_classes)
        self.shot_subject_head = nn.Linear(hidden_dim, shot_subject_classes)
        self.has_text_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(F.normalize(x.float(), dim=-1))
        return {
            "shot_type": self.shot_type_head(z),
            "shot_scale": self.shot_scale_head(z),
            "shot_subject": self.shot_subject_head(z),
            "has_text": self.has_text_head(z),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experimental multi-task shot classifier with weak auxiliary labels.")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/multitask_classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def infer_scale_subject(record: dict) -> tuple[str, str]:
    shot_type = str(record.get("shot_type", "unknown"))
    has_text = parse_bool(record.get("has_text", False))
    scale = str(record.get("shot_scale", "unknown"))
    subject = str(record.get("shot_subject", "unknown"))
    if scale == "unknown" and shot_type in {"close-up", "medium", "wide"}:
        scale = shot_type
    if subject == "unknown":
        if shot_type in {"close-up", "medium", "wide"}:
            subject = "person"
        elif shot_type in {"object", "space"}:
            subject = shot_type
        elif has_text:
            subject = "text"
    return scale, subject


def load_multitask_data(path: str | Path) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict]]:
    data = torch.load(path, map_location="cpu")
    embeddings: torch.Tensor = data["embeddings"].float()
    records: list[dict] = data["records"]
    scale_to_idx = {label: idx for idx, label in enumerate(SHOT_SCALE_CLASSES)}
    subject_to_idx = {label: idx for idx, label in enumerate(SHOT_SUBJECT_CLASSES)}

    labels = {
        "shot_type": torch.tensor([int(record["label"]) for record in records], dtype=torch.long),
        "shot_scale": [],
        "shot_subject": [],
        "has_text": torch.tensor([1 if parse_bool(record.get("has_text", False)) else 0 for record in records], dtype=torch.long),
    }
    for record in records:
        scale, subject = infer_scale_subject(record)
        labels["shot_scale"].append(scale_to_idx.get(scale, scale_to_idx["unknown"]))
        labels["shot_subject"].append(subject_to_idx.get(subject, subject_to_idx["unknown"]))
    labels["shot_scale"] = torch.tensor(labels["shot_scale"], dtype=torch.long)
    labels["shot_subject"] = torch.tensor(labels["shot_subject"], dtype=torch.long)
    return embeddings, labels, records


def build_loader(path: str | Path, batch_size: int, shuffle: bool) -> DataLoader:
    embeddings, labels, _ = load_multitask_data(path)
    dataset = TensorDataset(
        embeddings,
        labels["shot_type"],
        labels["shot_scale"],
        labels["shot_subject"],
        labels["has_text"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def labels_from_batch(batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    embeddings, shot_type, shot_scale, shot_subject, has_text = batch
    return embeddings, {
        "shot_type": shot_type,
        "shot_scale": shot_scale,
        "shot_subject": shot_subject,
        "has_text": has_text,
    }


def compute_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(F.cross_entropy(outputs[key], labels[key]) for key in outputs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: dict[str, list[str]]) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds: dict[str, list[torch.Tensor]] = {key: [] for key in class_names}
    golds: dict[str, list[torch.Tensor]] = {key: [] for key in class_names}
    for batch in loader:
        embeddings, labels = labels_from_batch(batch)
        embeddings = embeddings.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}
        outputs = model(embeddings)
        loss = compute_loss(outputs, labels)
        total_loss += float(loss.item()) * embeddings.shape[0]
        total_count += embeddings.shape[0]
        for key in class_names:
            preds[key].append(outputs[key].argmax(dim=1).cpu())
            golds[key].append(labels[key].cpu())

    metrics: dict[str, dict] = {}
    macro_scores: list[float] = []
    for key, names in class_names.items():
        pred = torch.cat(preds[key]) if preds[key] else torch.empty(0, dtype=torch.long)
        gold = torch.cat(golds[key]) if golds[key] else torch.empty(0, dtype=torch.long)
        confusion = confusion_matrix_from_predictions(pred, gold, len(names))
        report = classification_report_from_confusion(confusion, names)
        summary = summarize_metrics_from_confusion(confusion, report)
        metrics[key] = {
            **summary,
            "confusion_matrix": confusion.tolist(),
            "classification_report": report,
        }
        macro_scores.append(summary["macro_f1"])

    return {
        "loss": total_loss / total_count if total_count else 0.0,
        "avg_macro_f1": sum(macro_scores) / len(macro_scores) if macro_scores else 0.0,
        "tasks": metrics,
    }


def train(args: argparse.Namespace) -> tuple[MultiTaskShotClassifier, list[dict]]:
    embedding_dir = Path(args.embedding_dir)
    train_loader = build_loader(embedding_dir / "train_shot_embeddings.pt", args.batch_size, True)
    val_loader = build_loader(embedding_dir / "val_shot_embeddings.pt", args.batch_size, False)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shot_type_names = [label for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
    class_names = {
        "shot_type": shot_type_names,
        "shot_scale": SHOT_SCALE_CLASSES,
        "shot_subject": SHOT_SUBJECT_CLASSES,
        "has_text": TEXT_CLASSES,
    }

    device = torch.device(args.device)
    model = MultiTaskShotClassifier(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        shot_type_classes=len(shot_type_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_score = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            embeddings, labels = labels_from_batch(batch)
            embeddings = embeddings.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(embeddings)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * embeddings.shape[0]
            total_count += embeddings.shape[0]

        val_metrics = evaluate(model, val_loader, device, class_names)
        is_best = val_metrics["avg_macro_f1"] > best_score
        if is_best:
            best_score = val_metrics["avg_macro_f1"]
            best_state = deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_count if total_count else 0.0,
                "val_loss": val_metrics["loss"],
                "val_avg_macro_f1": val_metrics["avg_macro_f1"],
                "is_best": is_best,
            }
        )

    model._best_state_dict = best_state
    return model, history


def write_task_reports(output_dir: Path, metrics: dict) -> None:
    for task_name, filename in {
        "shot_type": "shot_type_report.csv",
        "shot_scale": "shot_scale_report.csv",
        "shot_subject": "shot_subject_report.csv",
        "has_text": "has_text_report.csv",
    }.items():
        write_classification_report_csv(output_dir / filename, metrics["tasks"][task_name]["classification_report"])


def main() -> None:
    args = parse_args()
    configure_reproducibility(seed=args.seed, project_root=PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, history = train(args)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shot_type_names = [label for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
    class_names = {
        "shot_type": shot_type_names,
        "shot_scale": SHOT_SCALE_CLASSES,
        "shot_subject": SHOT_SUBJECT_CLASSES,
        "has_text": TEXT_CLASSES,
    }
    device = torch.device(args.device)
    best_model = MultiTaskShotClassifier(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        shot_type_classes=len(shot_type_names),
    ).to(device)
    best_model.load_state_dict(model._best_state_dict)
    test_loader = build_loader(Path(args.embedding_dir) / "test_shot_embeddings.pt", args.batch_size, False)
    test_metrics = evaluate(best_model, test_loader, device, class_names)

    torch.save(model._best_state_dict, output_dir / "multitask_classifier_best.pt")
    write_task_reports(output_dir, test_metrics)
    summary = {
        "note": "shot_scale and shot_subject are weakly-derived auxiliary labels from folder names.",
        "history": history,
        "test_metrics_best": test_metrics,
        "training_config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {output_dir}")
    print(f"test avg macro F1: {test_metrics['avg_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
