from __future__ import annotations

# 인스타그램 링크를 받아 영상 다운로드부터 오버레이 생성까지 실행하는 메인 파이프라인입니다.
import json
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
import yt_dlp

import shotguide_batch_video_inference as base


# 프로젝트 기준 경로와 새 링크 처리용 입출력 폴더를 정의합니다.
ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos_new_links"
OUTPUT_ROOT = ROOT / "outputs_instagram_overlay"

# 새로 분석할 인스타그램 Reels 링크를 여기에 추가합니다.
INSTAGRAM_LINKS = [
    "https://www.instagram.com/reels/DYPNyiAp9EJ/",
    "https://www.instagram.com/reels/DYRs6scPeVC/",
]

# CLIP 임베딩 거리 기반 장면 전환 감지와 프레임 샘플링 설정입니다.
SAMPLE_INTERVAL_SEC = 0.25
DISTANCE_PERCENTILE = 88
MIN_SCENE_SEC = 0.75
MIN_DISTANCE = 0.055
NUM_FRAME_SAMPLES = 3
BATCH_SIZE = 32


def get_shortcode(url: str) -> str:
    # 인스타그램 URL에서 영상 고유 shortcode를 추출해 파일명에 사용합니다.
    parts = [part for part in urlparse(url).path.strip("/").split("/") if part]
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def download_instagram_video(url: str, index: int) -> Path:
    # yt-dlp로 인스타그램 영상을 mp4 파일로 다운로드합니다.
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    shortcode = get_shortcode(url)
    base_name = f"new_{index:03d}_{shortcode}"
    outtmpl = str(VIDEO_DIR / f"{base_name}.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4]/best",
        "quiet": False,
        "no_warnings": False,
        "overwrites": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = Path(ydl.prepare_filename(info))

    if not video_path.exists():
        mp4_path = VIDEO_DIR / f"{base_name}.mp4"
        if mp4_path.exists():
            return mp4_path
        raise FileNotFoundError(f"Downloaded file not found: {video_path}")
    return video_path


def sample_frame_indices_for_detection(video_path: Path, interval_sec=0.25):
    # 장면 전환 감지용으로 일정 시간 간격마다 검사할 프레임 번호를 만듭니다.
    info = base.get_video_info(video_path)
    step = max(1, int(round(info["fps"] * interval_sec)))
    indices = list(range(0, info["frame_count"], step))
    if indices and indices[-1] != info["frame_count"] - 1:
        indices.append(info["frame_count"] - 1)
    return indices, info


def read_frame_rgb(video_path: Path, frame_idx: int):
    # OpenCV로 특정 프레임을 읽고 CLIP 입력에 맞게 RGB로 변환합니다.
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def encode_detection_frames(video_path: Path, frame_indices, clip_model, clip_preprocess):
    # 전환 감지용 프레임들을 CLIP 이미지 임베딩으로 변환합니다.
    valid_indices = []
    tensors = []
    for frame_idx in tqdm(frame_indices, leave=False, desc=f"encode {video_path.stem}"):
        frame_rgb = read_frame_rgb(video_path, frame_idx)
        if frame_rgb is None:
            continue
        image = Image.fromarray(frame_rgb)
        tensors.append(clip_preprocess(image))
        valid_indices.append(frame_idx)

    features = []
    with torch.no_grad():
        for start in range(0, len(tensors), BATCH_SIZE):
            batch = torch.stack(tensors[start : start + BATCH_SIZE]).to(base.DEVICE)
            feat = clip_model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().float().numpy())
    return np.array(valid_indices, dtype=np.int64), np.vstack(features).astype("float32")


def find_distance_peaks(distances, frame_indices, fps):
    # 인접 프레임 임베딩 거리의 피크를 찾아 컷 전환 후보로 선택합니다.
    if len(distances) == 0:
        return [], float("nan")

    threshold = max(float(np.percentile(distances, DISTANCE_PERCENTILE)), float(MIN_DISTANCE))
    local_peaks = []
    for i, distance in enumerate(distances):
        left = distances[i - 1] if i > 0 else -1
        right = distances[i + 1] if i + 1 < len(distances) else -1
        if distance >= threshold and distance >= left and distance >= right:
            local_peaks.append((int(frame_indices[i + 1]), float(distance)))

    min_gap_frames = max(1, int(round(fps * MIN_SCENE_SEC)))
    selected = []
    for cut_frame, distance in sorted(local_peaks, key=lambda x: x[1], reverse=True):
        if all(abs(cut_frame - old_frame) >= min_gap_frames for old_frame, _ in selected):
            selected.append((cut_frame, distance))
    return sorted(selected, key=lambda x: x[0]), threshold


def build_scene_table(video_path: Path, cut_peaks, threshold, info):
    # 컷 시작 프레임 목록을 scene 단위 메타데이터 테이블로 변환합니다.
    cut_frames = [0] + [frame for frame, _ in cut_peaks]
    cut_frames = sorted(set(max(0, min(info["frame_count"] - 1, int(x))) for x in cut_frames))
    rows = []
    fps = info["fps"]
    for i, start_frame in enumerate(cut_frames):
        next_start = cut_frames[i + 1] if i + 1 < len(cut_frames) else info["frame_count"]
        end_frame = max(start_frame, next_start - 1)
        peak_distance = ""
        if i > 0:
            peak_distance = next((dist for frame, dist in cut_peaks if frame == start_frame), "")
        rows.append(
            {
                "video_path": str(video_path.resolve()),
                "video_name": video_path.name,
                "video_id": video_path.stem,
                "scene_index": i + 1,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
                "duration_sec": (end_frame - start_frame + 1) / fps,
                "threshold": threshold,
                "peak_distance": peak_distance,
                "fps": fps,
                "detector": "CLIP adjacent embedding distance",
            }
        )
    return pd.DataFrame(rows)


def detect_scenes_clip_distance(video_path: Path, clip_model, clip_preprocess):
    # CLIP 인접 임베딩 거리 방식으로 영상의 scene 구간을 감지합니다.
    frame_indices, info = sample_frame_indices_for_detection(video_path, SAMPLE_INTERVAL_SEC)
    valid_indices, embeddings = encode_detection_frames(video_path, frame_indices, clip_model, clip_preprocess)
    distances = 1.0 - np.sum(embeddings[1:] * embeddings[:-1], axis=1)
    cut_peaks, threshold = find_distance_peaks(distances, valid_indices, info["fps"])
    scene_df = build_scene_table(video_path, cut_peaks, threshold, info)
    distance_df = pd.DataFrame(
        {
            "frame_idx": valid_indices[1:],
            "time_sec": valid_indices[1:] / info["fps"],
            "clip_distance": distances,
        }
    )
    return scene_df, distance_df, threshold, info


def process_url(url: str, index: int, clip_model, clip_preprocess, head, idx_to_shot):
    # 링크 하나에 대해 다운로드, scene 감지, 프레임 추출, 예측, 오버레이 생성을 모두 수행합니다.
    video_path = download_instagram_video(url, index)
    video_output_dir = OUTPUT_ROOT / video_path.stem
    frames_dir = video_output_dir / "scene_frames"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    scene_df, distance_df, threshold, info = detect_scenes_clip_distance(video_path, clip_model, clip_preprocess)
    distance_df.to_csv(video_output_dir / "clip_distance_profile.csv", index=False, encoding="utf-8-sig")

    scene_df = base.extract_scene_frames(video_path, scene_df, frames_dir, num_samples=NUM_FRAME_SAMPLES)
    scene_df.to_csv(video_output_dir / "scene_metadata.csv", index=False, encoding="utf-8-sig")

    pred_df, embeddings = base.predict_scenes(scene_df, clip_model, clip_preprocess, head, idx_to_shot)
    pred_df.to_csv(video_output_dir / "scene_predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(video_output_dir / "scene_clip_embeddings.npz", embeddings=embeddings)

    overlay_path = video_output_dir / f"{video_path.stem}_overlay.mp4"
    base.render_overlay_video(video_path, pred_df, overlay_path)

    return {
        "url": url,
        "shortcode": get_shortcode(url),
        "video_path": str(video_path.resolve()),
        "output_dir": str(video_output_dir.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "scene_count": int(len(pred_df)),
        "duration_sec": float(info["duration_sec"]),
        "threshold": float(threshold),
        "text_scene_count": int(pred_df["pred_has_text"].sum()),
        "shot_label_counts": json.dumps(pred_df["pred_shot_type"].value_counts().to_dict(), ensure_ascii=False),
    }


def main():
    # 전체 링크 목록을 순회하며 결과 요약 CSV까지 저장합니다.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    clip_model, clip_preprocess, head, idx_to_shot = base.load_models()

    summary_rows = []
    for index, url in enumerate(INSTAGRAM_LINKS, start=1):
        print("=" * 80)
        print("processing:", url)
        summary_rows.append(process_url(url, index, clip_model, clip_preprocess, head, idx_to_shot))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "instagram_overlay_summary.csv", index=False, encoding="utf-8-sig")
    print("=" * 80)
    print(summary_df[["shortcode", "scene_count", "duration_sec", "text_scene_count", "shot_label_counts", "overlay_path"]])


if __name__ == "__main__":
    main()
