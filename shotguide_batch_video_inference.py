from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
import open_clip


ROOT = Path(__file__).resolve().parent
SPLIT_CSV = ROOT / "outputs_baseline" / "dataset_index_with_splits.csv"
VIDEO_DIR = ROOT / "videos"
CHECKPOINT_PATH = ROOT / "outputs_clip_embeddings" / "clip_vit_b32_multitask_head.pt"
OUTPUT_ROOT = ROOT / "outputs_video_batch_test"

BATCH_OFFSET = 5
BATCH_LIMIT = 3
PERCENTILE = 98
MIN_SCENE_SEC = 0.2
NUM_FRAME_SAMPLES = 3
TEXT_THRESHOLD = 0.5

OVERLAY_POSITION = "bottom"
OVERLAY_FONT_SCALE = 0.020
OVERLAY_MAX_WIDTH_RATIO = 0.92
OVERLAY_MAX_GUIDE_LINES = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipEmbeddingMultiTaskHead(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256, num_shot_classes=5, dropout=0.20):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shot_head = nn.Linear(hidden_dim, num_shot_classes)
        self.text_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        z = self.shared(x)
        return self.shot_head(z), self.text_head(z)


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps,
    }


def calculate_frame_diffs(video_path: Path, resize_size=(320, 180), sample_interval=1):
    cap = cv2.VideoCapture(str(video_path))
    prev_gray = None
    diffs = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(cv2.resize(frame, resize_size), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
        prev_gray = gray
        frame_idx += 1
    cap.release()
    return np.array(diffs, dtype=np.float32)


def get_adaptive_threshold(video_path: Path, percentile=98, fallback_threshold=50, min_threshold=15, max_threshold=120):
    diffs = calculate_frame_diffs(video_path)
    if len(diffs) == 0:
        return float(fallback_threshold)
    threshold = np.percentile(diffs, percentile)
    return float(np.clip(threshold, min_threshold, max_threshold))


def detect_cut_frames(video_path: Path, percentile=98, min_scene_sec=0.2, resize_size=(320, 180)):
    info = get_video_info(video_path)
    fps = info["fps"]
    min_scene_len = max(1, int(fps * min_scene_sec))
    threshold = get_adaptive_threshold(video_path, percentile=percentile)

    cap = cv2.VideoCapture(str(video_path))
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read first frame: {video_path}")

    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, resize_size), cv2.COLOR_BGR2GRAY)
    frame_idx = 0
    last_cut_frame = 0
    cut_frames = [0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(cv2.resize(frame, resize_size), cv2.COLOR_BGR2GRAY)
        diff_score = float(np.mean(cv2.absdiff(prev_gray, gray)))
        enough_gap = (frame_idx - last_cut_frame) >= min_scene_len
        if diff_score > threshold and enough_gap:
            cut_frames.append(frame_idx)
            last_cut_frame = frame_idx
        prev_gray = gray

    cap.release()
    return cut_frames, threshold, info


def build_scene_table(video_path: Path, cut_frames, threshold, info):
    fps = info["fps"]
    frame_count = info["frame_count"]
    rows = []
    for i, start_frame in enumerate(cut_frames):
        next_start = cut_frames[i + 1] if i + 1 < len(cut_frames) else frame_count
        end_frame = max(start_frame, next_start - 1)
        rows.append(
            {
                "video_path": str(video_path.resolve()),
                "video_name": video_path.name,
                "video_id": video_path.stem.split("_")[0],
                "scene_index": i + 1,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
                "duration_sec": (end_frame - start_frame + 1) / fps,
                "threshold": threshold,
                "fps": fps,
            }
        )
    return pd.DataFrame(rows)


def sample_frame_indices(start_frame, end_frame, num_samples=3):
    if end_frame <= start_frame:
        return [int(start_frame)]
    positions = np.linspace(0.20, 0.80, num_samples)
    indices = [int(round(start_frame + (end_frame - start_frame) * p)) for p in positions]
    return sorted(set(max(start_frame, min(end_frame, idx)) for idx in indices))


def read_frame_at(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def extract_scene_frames(video_path: Path, scene_df: pd.DataFrame, output_dir: Path, num_samples=3):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    for _, scene in scene_df.iterrows():
        frame_indices = sample_frame_indices(int(scene.start_frame), int(scene.end_frame), num_samples=num_samples)
        frame_paths = []
        for j, frame_idx in enumerate(frame_indices, start=1):
            frame = read_frame_at(cap, frame_idx)
            if frame is None:
                continue
            frame_path = output_dir / f"scene_{int(scene.scene_index):03d}_frame_{j:02d}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path.resolve()))
        row = scene.to_dict()
        row["sampled_frame_indices"] = json.dumps(frame_indices)
        row["sampled_frame_paths"] = json.dumps(frame_paths, ensure_ascii=False)
        rows.append(row)
    cap.release()
    return pd.DataFrame(rows)


def guide_text(shot_type, has_text):
    base = {
        "close-up": "피사체의 표정이나 디테일을 강조한 클로즈업 장면입니다.",
        "medium": "피사체와 주변 맥락을 함께 보여주는 미디엄 샷입니다.",
        "wide": "넓은 공간과 피사체 배치를 보여주는 와이드 샷입니다.",
        "object": "특정 제품이나 오브젝트가 중심이 되는 장면입니다.",
        "space": "공간의 분위기와 환경 정보가 중심이 되는 장면입니다.",
    }.get(shot_type, "장면 구도를 확인해야 하는 컷입니다.")
    if has_text:
        return base + " 화면 내 텍스트 정보도 함께 강조됩니다."
    return base + " 텍스트보다 시각적 구도가 중심입니다."


def load_korean_font(size=28):
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def overlay_text_pil(frame_bgr, lines, font, position="bottom"):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin = max(12, int(min(image.width, image.height) * 0.025))
    pad_x = max(12, int(image.width * 0.025))
    pad_y = max(10, int(image.height * 0.018))
    line_h = font.size + max(5, int(font.size * 0.28))
    box_w = int(image.width * OVERLAY_MAX_WIDTH_RATIO)
    box_h = pad_y * 2 + line_h * len(lines)
    x1 = (image.width - box_w) // 2
    y1 = image.height - margin - box_h if position == "bottom" else margin
    x2 = x1 + box_w
    y2 = y1 + box_h
    draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=(0, 0, 0, 185))

    y = y1 + pad_y
    for line in lines:
        draw.text((x1 + pad_x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h

    out = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def render_overlay_video(video_path: Path, pred_df: pd.DataFrame, output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    font = load_korean_font(size=max(16, int(height * OVERLAY_FONT_SCALE)))

    scenes = pred_df.sort_values("start_frame").reset_index(drop=True)
    scene_idx = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        while scene_idx + 1 < len(scenes) and frame_idx >= int(scenes.loc[scene_idx + 1, "start_frame"]):
            scene_idx += 1

        scene = scenes.loc[scene_idx]
        label = (
            f"Scene {int(scene.scene_index):03d} | {scene.pred_shot_type} | "
            f"text={int(scene.pred_has_text)} | shot {scene.shot_confidence:.2f} | text {scene.text_probability:.2f}"
        )
        guide = str(scene.guide_text)

        temp_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        temp_draw = ImageDraw.Draw(temp_img)
        wrapped = [label] + wrap_text(
            temp_draw,
            guide,
            font,
            max_width=int(width * OVERLAY_MAX_WIDTH_RATIO) - 48,
        )[:OVERLAY_MAX_GUIDE_LINES]
        frame = overlay_text_pil(frame, wrapped, font, position=OVERLAY_POSITION)
        writer.write(frame)

    cap.release()
    writer.release()
    return output_path


def choose_test_videos(offset=BATCH_OFFSET, limit=BATCH_LIMIT):
    split_df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    test_ids = set(split_df.loc[split_df["split"] == "test", "video_id"].astype(str).unique())
    videos = []
    for path in VIDEO_DIR.glob("*.mp4"):
        video_id = path.stem.split("_")[0]
        if video_id in test_ids:
            videos.append(path)
    return sorted(videos, key=lambda p: p.stat().st_size)[offset : offset + limit]


def encode_image_paths(image_paths, clip_model, clip_preprocess):
    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensors.append(clip_preprocess(image))
    if not tensors:
        return None
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        scene_feature = features.mean(dim=0, keepdim=True)
        scene_feature = scene_feature / scene_feature.norm(dim=-1, keepdim=True)
    return scene_feature.float()


def predict_scenes(scene_df, clip_model, clip_preprocess, head, idx_to_shot):
    prediction_rows = []
    scene_embeddings = []
    for _, scene in scene_df.iterrows():
        image_paths = json.loads(scene["sampled_frame_paths"])
        scene_feature = encode_image_paths(image_paths, clip_model, clip_preprocess)
        if scene_feature is None:
            continue

        with torch.no_grad():
            shot_logits, text_logits = head(scene_feature)
            shot_prob = torch.softmax(shot_logits, dim=1).cpu().numpy()[0]
            text_prob = torch.softmax(text_logits, dim=1).cpu().numpy()[0]

        pred_shot_idx = int(np.argmax(shot_prob))
        pred_shot = idx_to_shot[pred_shot_idx]
        has_text = bool(float(text_prob[1]) >= TEXT_THRESHOLD)

        row = scene.to_dict()
        row.update(
            {
                "pred_shot_type": pred_shot,
                "pred_has_text": int(has_text),
                "shot_confidence": float(shot_prob[pred_shot_idx]),
                "text_probability": float(text_prob[1]),
                "guide_text": guide_text(pred_shot, has_text),
            }
        )
        for i, label in idx_to_shot.items():
            row[f"prob_shot_{label}"] = float(shot_prob[i])
        prediction_rows.append(row)
        scene_embeddings.append(scene_feature.cpu().numpy()[0])

    return pd.DataFrame(prediction_rows), np.vstack(scene_embeddings).astype("float32")


def load_models():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    shot_to_idx = checkpoint["shot_to_idx"]
    idx_to_shot = {int(k): v for k, v in checkpoint["idx_to_shot"].items()}

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        checkpoint.get("clip_model", "ViT-B-32"),
        pretrained=checkpoint.get("clip_pretrained", "openai"),
        device=DEVICE,
    )
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    head = ClipEmbeddingMultiTaskHead(embedding_dim=512, hidden_dim=256, num_shot_classes=len(shot_to_idx)).to(DEVICE)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()
    return clip_model, clip_preprocess, head, idx_to_shot


def process_video(video_path: Path, clip_model, clip_preprocess, head, idx_to_shot):
    video_output_dir = OUTPUT_ROOT / video_path.stem
    frames_dir = video_output_dir / "scene_frames"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cut_frames, threshold, video_info = detect_cut_frames(
        video_path,
        percentile=PERCENTILE,
        min_scene_sec=MIN_SCENE_SEC,
    )
    scene_df = build_scene_table(video_path, cut_frames, threshold, video_info)
    scene_df = extract_scene_frames(video_path, scene_df, frames_dir, num_samples=NUM_FRAME_SAMPLES)
    scene_df.to_csv(video_output_dir / "scene_metadata.csv", index=False, encoding="utf-8-sig")

    pred_df, embeddings = predict_scenes(scene_df, clip_model, clip_preprocess, head, idx_to_shot)
    pred_df.to_csv(video_output_dir / "scene_predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(video_output_dir / "scene_clip_embeddings.npz", embeddings=embeddings)

    overlay_path = video_output_dir / f"{video_path.stem}_overlay.mp4"
    render_overlay_video(video_path, pred_df, overlay_path)

    label_counts = pred_df["pred_shot_type"].value_counts().to_dict()
    text_scene_count = int(pred_df["pred_has_text"].sum())
    return {
        "video_id": video_path.stem.split("_")[0],
        "video_name": video_path.name,
        "video_path": str(video_path.resolve()),
        "output_dir": str(video_output_dir.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "scene_count": int(len(pred_df)),
        "duration_sec": float(video_info["duration_sec"]),
        "threshold": float(threshold),
        "avg_scene_duration_sec": float(pred_df["duration_sec"].mean()),
        "min_scene_duration_sec": float(pred_df["duration_sec"].min()),
        "max_scene_duration_sec": float(pred_df["duration_sec"].max()),
        "text_scene_count": text_scene_count,
        "notext_scene_count": int(len(pred_df) - text_scene_count),
        "shot_label_counts": json.dumps(label_counts, ensure_ascii=False),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    videos = choose_test_videos()
    if not videos:
        raise RuntimeError("No test videos found.")

    print("device:", DEVICE)
    print("videos:")
    for path in videos:
        print(" -", path.name)

    clip_model, clip_preprocess, head, idx_to_shot = load_models()
    summary_rows = []
    all_scene_rows = []

    for video_path in tqdm(videos, desc="Batch videos"):
        summary = process_video(video_path, clip_model, clip_preprocess, head, idx_to_shot)
        summary_rows.append(summary)
        pred_path = Path(summary["output_dir"]) / "scene_predictions.csv"
        pred_df = pd.read_csv(pred_path)
        all_scene_rows.append(pred_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "batch_summary.csv", index=False, encoding="utf-8-sig")

    all_scene_df = pd.concat(all_scene_rows, ignore_index=True)
    all_scene_df.to_csv(OUTPUT_ROOT / "batch_scene_predictions.csv", index=False, encoding="utf-8-sig")

    print("saved:", OUTPUT_ROOT / "batch_summary.csv")
    print(summary_df[["video_id", "scene_count", "duration_sec", "threshold", "text_scene_count", "shot_label_counts"]])


if __name__ == "__main__":
    main()
