from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from scenedetect import ContentDetector, SceneManager, open_video


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    video_path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass(frozen=True)
class SceneRecord:
    video_id: str
    video_path: str
    scene_id: int
    cut_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    fps: float
    width: int
    height: int


def video_id_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def list_videos(video_dir: str | Path) -> list[Path]:
    video_dir = Path(video_dir)
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_video_info(video_path: str | Path) -> VideoInfo:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_sec = frame_count / fps if fps else 0.0
    return VideoInfo(
        video_id=video_id_from_path(video_path),
        video_path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        width=width,
        height=height,
    )


def detect_video_scenes(
    video_path: str | Path,
    threshold: float = 27.0,
    min_scene_len: int = 8,
) -> tuple[VideoInfo, list[SceneRecord]]:
    video_path = Path(video_path)
    info = read_video_info(video_path)

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
        )
    )
    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        scene_list = [(video.base_timecode, video.duration)]

    records: list[SceneRecord] = []
    for idx, (start_time, end_time) in enumerate(scene_list, start=1):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        records.append(
            SceneRecord(
                video_id=info.video_id,
                video_path=str(video_path),
                scene_id=idx,
                cut_id=f"{idx:03d}",
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=round(start_sec, 4),
                end_sec=round(end_sec, 4),
                duration_sec=round(end_sec - start_sec, 4),
                fps=round(info.fps, 4),
                width=info.width,
                height=info.height,
            )
        )

    return info, records


def collect_labeled_cut_counts(dataset_dir: str | Path) -> dict[str, int]:
    dataset_dir = Path(dataset_dir)
    cut_ids_by_video: dict[str, set[str]] = defaultdict(set)

    for image_path in dataset_dir.glob("*/*"):
        if not image_path.is_file() or "_cut_" not in image_path.stem:
            continue
        video_id, cut_id = image_path.stem.split("_cut_", 1)
        cut_ids_by_video[video_id].add(cut_id)

    return {video_id: len(cut_ids) for video_id, cut_ids in cut_ids_by_video.items()}


def write_scene_metadata(
    output_csv: str | Path,
    records: list[SceneRecord],
) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SceneRecord.__dataclass_fields__.keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_video_metadata(
    output_csv: str | Path,
    infos: list[VideoInfo],
) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(VideoInfo.__dataclass_fields__.keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(info) for info in infos)


def summarize_scene_counts(
    scene_records: list[SceneRecord],
    labeled_cut_counts: dict[str, int] | None = None,
) -> dict:
    detected_counts = Counter(record.video_id for record in scene_records)
    durations = [record.duration_sec for record in scene_records]
    scene_count = sum(detected_counts.values())
    video_count = len(detected_counts)
    summary = {
        "video_count": video_count,
        "scene_count": scene_count,
        "avg_scene_duration": sum(durations) / len(durations) if durations else 0.0,
        "min_scene_duration": min(durations) if durations else 0.0,
        "max_scene_duration": max(durations) if durations else 0.0,
        "avg_scene_count_per_video": scene_count / video_count if video_count else 0.0,
        "detected_counts": dict(sorted(detected_counts.items())),
    }

    if labeled_cut_counts is not None:
        mismatches = []
        for video_id, labeled_count in sorted(labeled_cut_counts.items()):
            detected_count = detected_counts.get(video_id, 0)
            diff = detected_count - labeled_count
            if diff != 0:
                mismatches.append(
                    {
                        "video_id": video_id,
                        "labeled_cuts": labeled_count,
                        "detected_scenes": detected_count,
                        "diff": diff,
                    }
                )
        abs_diffs = [abs(item["diff"]) for item in mismatches]
        over_detected = [item for item in mismatches if item["diff"] > 0]
        under_detected = [item for item in mismatches if item["diff"] < 0]
        summary["labeled_video_count"] = len(labeled_cut_counts)
        summary["count_mismatch_count"] = len(mismatches)
        summary["mismatch_ratio"] = len(mismatches) / len(labeled_cut_counts) if labeled_cut_counts else 0.0
        summary["over_detected_count"] = len(over_detected)
        summary["under_detected_count"] = len(under_detected)
        summary["avg_abs_count_difference"] = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        summary["top_10_mismatch_videos"] = sorted(
            mismatches,
            key=lambda item: abs(item["diff"]),
            reverse=True,
        )[:10]
        summary["count_mismatches"] = mismatches

    return summary
