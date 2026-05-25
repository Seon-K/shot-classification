from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path



DEFAULT_BACKBONES = ["ViT-B-32:openai", "ViT-L-14:openai"]
FIELDNAMES = [
    "backbone",
    "pretrained",
    "embedding_dim",
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_weighted_f1",
    "fit_time_sec",
    "embedding_time_sec",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CLIP backbones by extracting embeddings and training the same classifier head.")
    parser.add_argument("--manifest-dir", default="artifacts/dataset")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES, help="Items formatted as model_name:pretrained.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm", choices=["none", "batch", "layer"], default="none")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing embeddings/metrics when present.")
    return parser.parse_args()


def parse_backbone(value: str) -> tuple[str, str]:
    if ":" not in value:
        return value, "openai"
    model_name, pretrained = value.split(":", 1)
    return model_name, pretrained


def slugify(value: str) -> str:
    return value.replace("/", "-").replace(":", "-").replace(" ", "_")


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_embeddings(args: argparse.Namespace, model_name: str, pretrained: str, embedding_dir: Path) -> float:
    expected = embedding_dir / "train_shot_embeddings.pt"
    if args.skip_existing and expected.exists():
        return 0.0
    start = time.perf_counter()
    run_command(
        [
            sys.executable,
            "scripts/extract_clip_embeddings.py",
            "--manifest-dir",
            args.manifest_dir,
            "--output-dir",
            str(embedding_dir),
            "--model-name",
            model_name,
            "--pretrained",
            pretrained,
            "--batch-size",
            str(args.embedding_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
    )
    return time.perf_counter() - start


def embedding_dim(path: str | Path) -> int:
    import torch

    data = torch.load(path, map_location="cpu")
    return int(data["embeddings"].shape[1])


def train_classifier(args: argparse.Namespace, model_name: str, pretrained: str, experiment_dir: Path, embedding_dir: Path, dim: int) -> float:
    metrics_path = experiment_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return 0.0
    start = time.perf_counter()
    run_command(
        [
            sys.executable,
            "scripts/train_shot_classifier.py",
            "--embedding-dir",
            str(embedding_dir),
            "--manifest",
            args.manifest,
            "--output-dir",
            str(experiment_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--device",
            args.device,
            "--embedding-dim",
            str(dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--dropout",
            str(args.dropout),
            "--weight-decay",
            str(args.weight_decay),
            "--norm",
            args.norm,
            "--model-name",
            model_name,
            "--pretrained",
            pretrained,
        ]
    )
    return time.perf_counter() - start


def success_row(backbone: str, pretrained: str, dim: int, metrics: dict, fit_time: float, embedding_time: float) -> dict:
    test_metrics = metrics.get("test_metrics_best", {})
    return {
        "backbone": backbone,
        "pretrained": pretrained,
        "embedding_dim": dim,
        "best_val_macro_f1": metrics.get("best_val_macro_f1", 0.0),
        "test_accuracy": test_metrics.get("accuracy", 0.0),
        "test_macro_f1": test_metrics.get("macro_f1", 0.0),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
        "test_weighted_f1": test_metrics.get("weighted_f1", 0.0),
        "fit_time_sec": round(fit_time, 3),
        "embedding_time_sec": round(embedding_time, 3),
        "status": "success",
        "error_message": "",
    }


def failed_row(backbone: str, pretrained: str, error_message: str, embedding_time: float = 0.0, fit_time: float = 0.0) -> dict:
    return {
        "backbone": backbone,
        "pretrained": pretrained,
        "embedding_dim": 0,
        "best_val_macro_f1": 0.0,
        "test_accuracy": 0.0,
        "test_macro_f1": 0.0,
        "test_balanced_accuracy": 0.0,
        "test_weighted_f1": 0.0,
        "fit_time_sec": round(fit_time, 3),
        "embedding_time_sec": round(embedding_time, 3),
        "status": "failed",
        "error_message": error_message.replace("\n", " ")[:500],
    }


def write_results(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows: list[dict] = []

    for item in args.backbones:
        model_name, pretrained = parse_backbone(item)
        slug = slugify(model_name)
        experiment_dir = output_dir / f"backbone_{slug}"
        embedding_dir = experiment_dir / "embeddings"
        print(f"[backbone] running {model_name}/{pretrained}")
        embedding_time = 0.0
        fit_time = 0.0
        try:
            embedding_time = extract_embeddings(args, model_name, pretrained, embedding_dir)
            dim = embedding_dim(embedding_dir / "train_shot_embeddings.pt")
            fit_time = train_classifier(args, model_name, pretrained, experiment_dir, embedding_dir, dim)
            metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
            rows.append(success_row(model_name, pretrained, dim, metrics, fit_time, embedding_time))
        except Exception as exc:  # noqa: BLE001 - backbone failures should not abort the sweep.
            print(f"[backbone] failed {model_name}/{pretrained}: {exc}")
            rows.append(failed_row(model_name, pretrained, str(exc), embedding_time, fit_time))

    results_path = output_dir / "backbone_results.csv"
    write_results(results_path, rows)
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
