from __future__ import annotations

import argparse
import json
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
    write_scene_metadata,
    write_video_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scene metadata from source videos.")
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--dataset-dir", default="labeled_dataset")
    parser.add_argument("--output-dir", default="artifacts/scenes")
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N videos for quick testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    videos = list_videos(args.video_dir)
    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        raise RuntimeError(f"No videos found in {args.video_dir}")

    all_infos = []
    all_records = []
    for index, video_path in enumerate(videos, start=1):
        info, records = detect_video_scenes(
            video_path=video_path,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
        )
        all_infos.append(info)
        all_records.extend(records)
        print(f"[{index}/{len(videos)}] {info.video_id}: {len(records)} scenes")

    scene_csv = output_dir / "scene_metadata.csv"
    video_csv = output_dir / "video_metadata.csv"
    write_scene_metadata(scene_csv, all_records)
    write_video_metadata(video_csv, all_infos)

    processed_video_ids = {info.video_id for info in all_infos}
    labeled_cut_counts = {
        video_id: count
        for video_id, count in collect_labeled_cut_counts(args.dataset_dir).items()
        if video_id in processed_video_ids
    }
    summary = summarize_scene_counts(all_records, labeled_cut_counts=labeled_cut_counts)
    summary_path = output_dir / "scene_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"scene metadata: {scene_csv}")
    print(f"video metadata: {video_csv}")
    print(f"summary: {summary_path}")
    print(f"videos: {summary['video_count']}")
    print(f"scenes: {summary['scene_count']}")
    print(f"count mismatches: {summary.get('count_mismatch_count', 0)}")


if __name__ == "__main__":
    main()
