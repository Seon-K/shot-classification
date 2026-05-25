from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path



DEFAULT_SIZES = ["100", "300", "500", "800", "full"]
FIELDNAMES = [
    "train_size",
    "train_video_count",
    "class_distribution",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_weighted_f1",
    "test_top2_accuracy",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video-level data size experiments for the shot classifier.")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="Train shot counts to target, plus optional 'full'.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm", choices=["none", "batch", "layer"], default="none")
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing metrics.json when present.")
    return parser.parse_args()


def load_embedding_file(path: str | Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu")


def group_indices_by_video(records: list[dict]) -> dict[str, list[int]]:
    by_video: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_video[str(record.get("video_id", ""))].append(idx)
    return dict(by_video)


def video_class_counts(records: list[dict], indices: list[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for idx in indices:
        counts[int(records[idx]["label"])] += 1
    return counts


def distribution_distance(candidate: Counter[int], desired: dict[int, float], class_ids: list[int]) -> float:
    return sum(abs(candidate.get(class_id, 0) - desired.get(class_id, 0.0)) for class_id in class_ids)


def select_video_subset(records: list[dict], target_size: int, seed: int) -> list[int]:
    by_video = group_indices_by_video(records)
    video_ids = sorted(by_video)
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    total_counts = Counter(int(record["label"]) for record in records)
    total = sum(total_counts.values())
    class_ids = sorted(total_counts)
    desired = {
        class_id: target_size * (total_counts[class_id] / total)
        for class_id in class_ids
    }
    per_video_counts = {
        video_id: video_class_counts(records, indices)
        for video_id, indices in by_video.items()
    }

    selected_videos: list[str] = []
    selected_counts: Counter[int] = Counter()
    selected_size = 0
    remaining = set(video_ids)

    while remaining and selected_size < target_size:
        best_video = min(
            remaining,
            key=lambda video_id: (
                distribution_distance(selected_counts + per_video_counts[video_id], desired, class_ids),
                abs((selected_size + len(by_video[video_id])) - target_size),
                video_id,
            ),
        )
        selected_videos.append(best_video)
        selected_counts.update(per_video_counts[best_video])
        selected_size += len(by_video[best_video])
        remaining.remove(best_video)

    selected_indices = [idx for video_id in selected_videos for idx in by_video[video_id]]
    return sorted(selected_indices)


def save_subset_embeddings(source_data: dict, indices: list[int], output_path: str | Path) -> None:
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = {
        "embeddings": source_data["embeddings"][indices],
        "records": [source_data["records"][idx] for idx in indices],
    }
    torch.save(subset, output_path)


def prepare_subset_embedding_dir(
    base_embedding_dir: Path,
    output_embedding_dir: Path,
    train_data: dict,
    indices: list[int],
) -> int:
    output_embedding_dir.mkdir(parents=True, exist_ok=True)
    save_subset_embeddings(train_data, indices, output_embedding_dir / "train_shot_embeddings.pt")
    for split in ["val", "test"]:
        shutil.copy2(
            base_embedding_dir / f"{split}_shot_embeddings.pt",
            output_embedding_dir / f"{split}_shot_embeddings.pt",
        )
    embeddings = train_data["embeddings"]
    return int(embeddings.shape[1])


def class_distribution(records: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("shot_type", record.get("label", "unknown"))) for record in records).items()))


def metrics_to_row(train_size: str, train_records: list[dict], metrics: dict, status: str, error_message: str = "") -> dict:
    test_metrics = metrics.get("test_metrics_best", {}) if metrics else {}
    train_videos = {str(record.get("video_id", "")) for record in train_records}
    return {
        "train_size": train_size,
        "train_video_count": len(train_videos),
        "class_distribution": json.dumps(class_distribution(train_records), ensure_ascii=False),
        "best_epoch": metrics.get("best_epoch", 0) if metrics else 0,
        "best_val_accuracy": metrics.get("best_val_accuracy", 0.0) if metrics else 0.0,
        "best_val_macro_f1": metrics.get("best_val_macro_f1", 0.0) if metrics else 0.0,
        "test_accuracy": test_metrics.get("accuracy", 0.0),
        "test_macro_f1": test_metrics.get("macro_f1", 0.0),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
        "test_weighted_f1": test_metrics.get("weighted_f1", 0.0),
        "test_top2_accuracy": test_metrics.get("top2_accuracy", 0.0),
        "status": status,
        "error_message": error_message,
    }


def run_training(args: argparse.Namespace, experiment_dir: Path, embedding_dir: Path, embedding_dim: int) -> dict:
    metrics_path = experiment_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    cmd = [
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
        str(embedding_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--weight-decay",
        str(args.weight_decay),
        "--norm",
        args.norm,
        "--loss-type",
        args.loss_type,
        "--patience",
        str(args.patience),
        "--model-name",
        args.model_name,
        "--pretrained",
        args.pretrained,
    ]
    if args.scheduler:
        cmd.append("--scheduler")
    if args.use_class_weight:
        cmd.append("--use-class-weight")
    if args.use_weighted_sampler:
        cmd.append("--use-weighted-sampler")

    subprocess.run(cmd, check=True)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


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
    base_embedding_dir = Path(args.embedding_dir)
    train_data = load_embedding_file(base_embedding_dir / "train_shot_embeddings.pt")
    all_records: list[dict] = train_data["records"]
    rows: list[dict] = []

    for size_value in args.sizes:
        experiment_name = f"data_size_{size_value}"
        experiment_dir = output_dir / experiment_name
        subset_embedding_dir = experiment_dir / "embeddings"
        print(f"[data-size] running {size_value}")
        try:
            if size_value == "full":
                indices = list(range(len(all_records)))
            else:
                indices = select_video_subset(all_records, int(size_value), seed=args.seed)
            selected_records = [all_records[idx] for idx in indices]
            embedding_dim = prepare_subset_embedding_dir(
                base_embedding_dir=base_embedding_dir,
                output_embedding_dir=subset_embedding_dir,
                train_data=train_data,
                indices=indices,
            )
            metrics = run_training(args, experiment_dir, subset_embedding_dir, embedding_dim)
            rows.append(metrics_to_row(size_value, selected_records, metrics, status="success"))
        except Exception as exc:  # noqa: BLE001 - runner should continue across experiments.
            error_message = str(exc).replace("\n", " ")[:500]
            print(f"[data-size] failed {size_value}: {error_message}")
            rows.append(metrics_to_row(size_value, [], {}, status="failed", error_message=error_message))

    results_path = output_dir / "data_size_results.csv"
    write_results(results_path, rows)
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
