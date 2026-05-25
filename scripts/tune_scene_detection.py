from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.video import (  # noqa: E402
    collect_labeled_cut_counts,
    detect_video_scenes,
    list_videos,
    summarize_scene_counts,
)


THRESHOLDS = [15, 20, 25, 27, 30, 35, 40]
FIELDNAMES = [
    "threshold",
    "video_count",
    "mismatch_count",
    "mismatch_ratio",
    "avg_abs_count_difference",
    "over_detected_count",
    "under_detected_count",
    "avg_scene_duration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep scene detection thresholds against labeled cut counts.")
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--dataset-dir", default="labeled_dataset")
    parser.add_argument("--output", default="artifacts/scenes/scene_threshold_sweep.csv")
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    parser.add_argument("--min-scene-len", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N videos for quick testing.")
    return parser.parse_args()


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    videos = list_videos(args.video_dir)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise RuntimeError(f"No videos found in {args.video_dir}")

    labeled_counts_all = collect_labeled_cut_counts(args.dataset_dir)
    rows: list[dict] = []
    for threshold in args.thresholds:
        all_records = []
        processed_video_ids = set()
        for video_path in videos:
            info, records = detect_video_scenes(
                video_path=video_path,
                threshold=threshold,
                min_scene_len=args.min_scene_len,
            )
            processed_video_ids.add(info.video_id)
            all_records.extend(records)
        labeled_counts = {
            video_id: count
            for video_id, count in labeled_counts_all.items()
            if video_id in processed_video_ids
        }
        summary = summarize_scene_counts(all_records, labeled_cut_counts=labeled_counts)
        row = {
            "threshold": threshold,
            "video_count": summary["video_count"],
            "mismatch_count": summary.get("count_mismatch_count", 0),
            "mismatch_ratio": summary.get("mismatch_ratio", 0.0),
            "avg_abs_count_difference": summary.get("avg_abs_count_difference", 0.0),
            "over_detected_count": summary.get("over_detected_count", 0),
            "under_detected_count": summary.get("under_detected_count", 0),
            "avg_scene_duration": summary.get("avg_scene_duration", 0.0),
        }
        rows.append(row)
        print(
            f"threshold={threshold}: mismatch_ratio={row['mismatch_ratio']:.4f}, "
            f"avg_abs_diff={row['avg_abs_count_difference']:.2f}"
        )

    write_rows(args.output, rows)
    best = min(rows, key=lambda row: (float(row["mismatch_ratio"]), float(row["avg_abs_count_difference"])))
    print(f"saved: {args.output}")
    print(
        "recommended threshold: "
        f"{best['threshold']} "
        f"(mismatch_ratio={float(best['mismatch_ratio']):.4f}, "
        f"avg_abs_diff={float(best['avg_abs_count_difference']):.2f})"
    )


if __name__ == "__main__":
    main()
