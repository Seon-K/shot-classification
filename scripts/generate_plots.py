from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style matplotlib plots from experiment artifacts.")
    parser.add_argument("--classifier-metrics", default="artifacts/classifier/metrics.json")
    parser.add_argument("--ablation-results", default="artifacts/experiments/ablation_results.csv")
    parser.add_argument("--data-size-results", default="artifacts/experiments/data_size_results.csv")
    parser.add_argument("--backbone-results", default="artifacts/experiments/backbone_results.csv")
    parser.add_argument("--output-dir", default="artifacts/plots")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save_empty_plot(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ablation(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row.get("status", "success") != "failed"]
    if not rows:
        save_empty_plot(path, "Ablation Comparison", "No ablation results")
        return
    labels = [row.get("experiment_name", "") for row in rows]
    values = [to_float(row.get("test_macro_f1")) for row in rows]
    fig = plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test Macro F1")
    plt.title("Ablation Comparison")
    plt.ylim(0, max(values + [1.0]))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_data_size(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row.get("status", "success") != "failed"]
    if not rows:
        save_empty_plot(path, "Data Size vs Macro F1", "No data size results")
        return
    labels = [row.get("train_size", "") for row in rows]
    values = [to_float(row.get("test_macro_f1")) for row in rows]
    fig = plt.figure(figsize=(8, 5))
    plt.plot(labels, values, marker="o")
    plt.xlabel("Train Size")
    plt.ylabel("Test Macro F1")
    plt.title("Data Size vs Macro F1")
    plt.ylim(0, max(values + [1.0]))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_backbone(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row.get("status") == "success"]
    if not rows:
        save_empty_plot(path, "Backbone Comparison", "No successful backbone results")
        return
    labels = [f"{row.get('backbone')}\n{row.get('pretrained')}" for row in rows]
    values = [to_float(row.get("test_macro_f1")) for row in rows]
    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Test Macro F1")
    plt.title("Backbone Comparison")
    plt.ylim(0, max(values + [1.0]))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(metrics: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = (metrics.get("test_metrics_best") or metrics.get("test_metrics", {})).get("confusion_matrix", [])
    class_to_idx = metrics.get("class_to_idx", {})
    if not matrix:
        save_empty_plot(path, "Confusion Matrix", "No confusion matrix")
        return
    labels = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])] or [str(i) for i in range(len(matrix))]
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_curve(metrics: dict[str, Any], path: Path, key_train: str, key_val: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    history = metrics.get("history", [])
    if not history:
        save_empty_plot(path, title, "No training history")
        return
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]
    train_values = [to_float(row.get(key_train)) for row in history]
    val_values = [to_float(row.get(key_val)) for row in history]
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, marker="o", label="train")
    plt.plot(epochs, val_values, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def write_status(output_dir: Path, rows: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "plot_generation_status.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["plot", "status", "error_message"])
        writer.writeheader()
        writer.writerows(rows)


def run_plot(name: str, func, output_path: Path) -> dict[str, str]:
    try:
        func(output_path)
        return {"plot": name, "status": "success", "error_message": ""}
    except Exception as exc:  # noqa: BLE001 - plot generation should not abort reports.
        return {"plot": name, "status": "failed", "error_message": str(exc).replace("\n", " ")[:500]}

def main() -> None:
    # Check if matplotlib is available
    matplotlib_available = True
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        matplotlib_available = False

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not matplotlib_available:
        # Write matplotlib unavailable status
        status_rows = [
            {"plot": "ablation_comparison", "status": "failed", "error_message": "matplotlib import failed"},
            {"plot": "data_size_curve", "status": "failed", "error_message": "matplotlib import failed"},
            {"plot": "backbone_comparison", "status": "failed", "error_message": "matplotlib import failed"},
            {"plot": "confusion_matrix_heatmap", "status": "failed", "error_message": "matplotlib import failed"},
            {"plot": "training_accuracy_curve", "status": "failed", "error_message": "matplotlib import failed"},
            {"plot": "training_loss_curve", "status": "failed", "error_message": "matplotlib import failed"},
        ]
        write_status(output_dir, status_rows)
        print(f"plots: {output_dir}")
        print("matplotlib import failed - plots not generated")
        return

    metrics = read_json(args.classifier_metrics)
    ablation_rows = read_csv(args.ablation_results)
    data_size_rows = read_csv(args.data_size_results)
    backbone_rows = read_csv(args.backbone_results)
    status_rows = [
        run_plot("ablation_comparison", lambda path: plot_ablation(ablation_rows, path), output_dir / "ablation_comparison.png"),
        run_plot("data_size_curve", lambda path: plot_data_size(data_size_rows, path), output_dir / "data_size_curve.png"),
        run_plot("backbone_comparison", lambda path: plot_backbone(backbone_rows, path), output_dir / "backbone_comparison.png"),
        run_plot("confusion_matrix_heatmap", lambda path: plot_confusion(metrics, path), output_dir / "confusion_matrix_heatmap.png"),
        run_plot(
            "training_accuracy_curve",
            lambda path: plot_training_curve(metrics, path, "train_accuracy", "val_accuracy", "Training Accuracy Curve", "Accuracy"),
            output_dir / "training_accuracy_curve.png",
        ),
        run_plot(
            "training_loss_curve",
            lambda path: plot_training_curve(metrics, path, "train_loss", "val_loss", "Training Loss Curve", "Loss"),
            output_dir / "training_loss_curve.png",
        ),
    ]
    write_status(output_dir, status_rows)
    failed = [row for row in status_rows if row["status"] == "failed"]
    print(f"plots: {output_dir}")
    if failed:
        print(f"plot failures: {len(failed)} (see plot_generation_status.csv)")


if __name__ == "__main__":
    main()
