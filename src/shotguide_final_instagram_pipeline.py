from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import shotguide_batch_video_inference as base
import shotguide_instagram_overlay_pipeline as instagram_base


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "videos_new_links"
OUTPUT_ROOT = ROOT / "outputs" / "final_instagram_pipeline"

# 새로 분석할 인스타그램 Reels/Post 링크를 여기에 추가합니다.
# 실행하면 영상 다운로드, scene 분할, frame 추출, 예측, overlay 영상 생성까지 진행됩니다.
INSTAGRAM_LINKS = [
    "https://www.instagram.com/reels/DYPNyiAp9EJ/",
]

# 다운로드 파일명에 사용할 시작 번호입니다.
# 예: START_INDEX = 1이면 new_001_SHORTCODE.mp4 형식으로 저장됩니다.
START_INDEX = 1

# scene 하나에서 대표 프레임을 몇 장 추출해 평균 CLIP embedding을 만들지 설정합니다.
NUM_FRAME_SAMPLES = 3


def process_instagram_url(url: str, index: int, clip_model, clip_preprocess, head, idx_to_shot) -> dict:
    """인스타그램 링크 하나를 입력받아 최종 overlay 영상까지 생성합니다."""
    video_path = instagram_base.download_instagram_video(url, index)

    video_output_dir = OUTPUT_ROOT / video_path.stem
    frames_dir = video_output_dir / "scene_frames"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    scene_df, distance_df, threshold, video_info = instagram_base.detect_scenes_clip_distance(
        video_path,
        clip_model,
        clip_preprocess,
    )
    distance_df.to_csv(video_output_dir / "clip_distance_profile.csv", index=False, encoding="utf-8-sig")

    scene_df = base.extract_scene_frames(
        video_path,
        scene_df,
        frames_dir,
        num_samples=NUM_FRAME_SAMPLES,
    )
    scene_df.to_csv(video_output_dir / "scene_metadata.csv", index=False, encoding="utf-8-sig")

    pred_df, scene_embeddings = base.predict_scenes(
        scene_df,
        clip_model,
        clip_preprocess,
        head,
        idx_to_shot,
    )
    pred_df.to_csv(video_output_dir / "scene_predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(video_output_dir / "scene_clip_embeddings.npz", embeddings=scene_embeddings)

    overlay_path = video_output_dir / f"{video_path.stem}_overlay.mp4"
    base.render_overlay_video(video_path, pred_df, overlay_path)

    return {
        "url": url,
        "shortcode": instagram_base.get_shortcode(url),
        "downloaded_video_path": str(video_path.resolve()),
        "output_dir": str(video_output_dir.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "scene_count": int(len(pred_df)),
        "duration_sec": float(video_info["duration_sec"]),
        "threshold": float(threshold),
        "text_scene_count": int(pred_df["pred_has_text"].sum()),
        "shot_label_counts": json.dumps(
            pred_df["pred_shot_type"].value_counts().to_dict(),
            ensure_ascii=False,
        ),
    }


def run_pipeline(instagram_links: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """링크 목록 전체를 처리하고 성공/실패 요약을 저장합니다."""
    links = instagram_links if instagram_links is not None else INSTAGRAM_LINKS
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    clip_model, clip_preprocess, head, idx_to_shot = base.load_models()

    summary_rows = []
    failed_rows = []

    for offset, url in enumerate(links):
        index = START_INDEX + offset
        print("=" * 80)
        print(f"[{index:03d}] processing: {url}")

        try:
            result = process_instagram_url(
                url,
                index,
                clip_model,
                clip_preprocess,
                head,
                idx_to_shot,
            )
            summary_rows.append(result)
            print("overlay saved:", result["overlay_path"])
        except Exception as exc:
            failed_rows.append(
                {
                    "url": url,
                    "index": index,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print("failed:", type(exc).__name__, str(exc))

    summary_df = pd.DataFrame(summary_rows)
    failed_df = pd.DataFrame(failed_rows)

    if not summary_df.empty:
        summary_df.to_csv(OUTPUT_ROOT / "final_pipeline_summary.csv", index=False, encoding="utf-8-sig")
    if not failed_df.empty:
        failed_df.to_csv(OUTPUT_ROOT / "final_pipeline_failed_links.csv", index=False, encoding="utf-8-sig")

    return summary_df, failed_df


def main() -> None:
    summary_df, failed_df = run_pipeline()

    print("=" * 80)
    print("done")
    if not summary_df.empty:
        print(summary_df[["shortcode", "scene_count", "duration_sec", "text_scene_count", "shot_label_counts", "overlay_path"]])
    if not failed_df.empty:
        print("failed links:")
        print(failed_df)


if __name__ == "__main__":
    main()
