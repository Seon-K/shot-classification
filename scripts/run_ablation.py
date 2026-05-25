from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "experiment_name": "baseline",
        "loss_type": "ce",
        "use_class_weight": False,
        "use_weighted_sampler": False,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "norm": "none",
        "extra_args": [],
    },
    {
        "experiment_name": "class_weight",
        "loss_type": "ce",
        "use_class_weight": True,
        "use_weighted_sampler": False,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "norm": "none",
        "extra_args": ["--use-class-weight"],
    },
    {
        "experiment_name": "weighted_sampler",
        "loss_type": "ce",
        "use_class_weight": False,
        "use_weighted_sampler": True,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "norm": "none",
        "extra_args": ["--use-weighted-sampler"],
    },
    {
        "experiment_name": "focal",
        "loss_type": "focal",
        "use_class_weight": False,
        "use_weighted_sampler": False,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "norm": "none",
        "extra_args": ["--loss-type", "focal"],
    },
    {
        "experiment_name": "class_weight_focal",
        "loss_type": "focal",
        "use_class_weight": True,
        "use_weighted_sampler": False,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "norm": "none",
        "extra_args": ["--loss-type", "focal", "--use-class-weight"],
    },
    {
        "experiment_name": "regularized",
        "loss_type": "ce",
        "use_class_weight": False,
        "use_weighted_sampler": False,
        "dropout": 0.4,
        "weight_decay": 1e-3,
        "norm": "none",
        "extra_args": ["--dropout", "0.4", "--weight-decay", "0.001", "--patience", "10", "--scheduler"],
    },
    {
        "experiment_name": "layernorm",
        "loss_type": "ce",
        "use_class_weight": False,
        "use_weighted_sampler": False,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "norm": "layer",
        "extra_args": ["--norm", "layer"],
    },
]


FIELDNAMES = [
    "experiment_name",
    "loss_type",
    "use_class_weight",
    "use_weighted_sampler",
    "dropout",
    "weight_decay",
    "norm",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_weighted_f1",
    "test_top2_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shot classifier ablation experiments sequentially.")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing metrics.json when present.")
    return parser.parse_args()


def metrics_to_row(experiment: dict, metrics: dict) -> dict:
    test_metrics = metrics.get("test_metrics_best", {})
    return {
        "experiment_name": experiment["experiment_name"],
        "loss_type": experiment["loss_type"],
        "use_class_weight": experiment["use_class_weight"],
        "use_weighted_sampler": experiment["use_weighted_sampler"],
        "dropout": experiment["dropout"],
        "weight_decay": experiment["weight_decay"],
        "norm": experiment["norm"],
        "best_epoch": metrics.get("best_epoch", 0),
        "best_val_accuracy": metrics.get("best_val_accuracy", 0.0),
        "best_val_macro_f1": metrics.get("best_val_macro_f1", 0.0),
        "test_accuracy": test_metrics.get("accuracy", 0.0),
        "test_macro_f1": test_metrics.get("macro_f1", 0.0),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
        "test_weighted_f1": test_metrics.get("weighted_f1", 0.0),
        "test_top2_accuracy": test_metrics.get("top2_accuracy", 0.0),
    }


def run_experiment(args: argparse.Namespace, experiment: dict) -> dict:
    experiment_dir = Path(args.output_dir) / experiment["experiment_name"]
    metrics_path = experiment_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "scripts/train_shot_classifier.py",
        "--embedding-dir",
        args.embedding_dir,
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
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(experiment["dropout"]),
        "--weight-decay",
        str(experiment["weight_decay"]),
        "--norm",
        experiment["norm"],
        "--model-name",
        args.model_name,
        "--pretrained",
        args.pretrained,
        *experiment["extra_args"],
    ]
    subprocess.run(cmd, check=True)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_results(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def select_best_experiment(results_csv: str | Path, output_dir: str | Path) -> dict:
    results_csv = Path(results_csv)
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {results_csv}")
    best = max(rows, key=lambda row: float(row["best_val_macro_f1"]))
    src_dir = results_csv.parent / best["experiment_name"]
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "shot_classifier_best.pt",
        "shot_classifier_last.pt",
        "shot_classifier.pt",
        "training_config.json",
        "metrics.json",
        "confusion_matrix.csv",
        "classification_report.csv",
        "misclassification_analysis.csv",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    return best


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows: list[dict] = []
    for experiment in EXPERIMENTS:
        print(f"[ablation] running {experiment['experiment_name']}")
        metrics = run_experiment(args, experiment)
        rows.append(metrics_to_row(experiment, metrics))

    results_path = output_dir / "ablation_results.csv"
    write_results(results_path, rows)
    best = select_best_experiment(results_path, "artifacts/classifier_best_overall")
    print(f"results: {results_path}")
    print(
        "best experiment: "
        f"{best['experiment_name']} val_macro_f1={float(best['best_val_macro_f1']):.4f}"
    )
    print("best copied to: artifacts/classifier_best_overall")


if __name__ == "__main__":
    main()
