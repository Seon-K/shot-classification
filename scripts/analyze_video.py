from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.classifier import ShotClassifier, TextPresenceClassifier  # noqa: E402
from shortform.embedding import get_device, load_open_clip_model  # noqa: E402
from shortform.guide import build_temporal_summary, guide_for_label, summarize_guides, wrap_text  # noqa: E402
from shortform.video import detect_video_scenes  # noqa: E402


DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one video and create shot guide overlay.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--checkpoint", default="artifacts/classifier/shot_classifier.pt")
    parser.add_argument("--text-checkpoint", default="artifacts/text_classifier/text_classifier.pt")
    parser.add_argument("--output-dir", default="artifacts/demo")
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=int, default=8)
    parser.add_argument("--samples-per-scene", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--embedding-dim", type=int, default=None, help="Input embedding dimension. Defaults to training_config.json when available.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Classifier head hidden dimension. Defaults to training_config.json when available.")
    parser.add_argument("--dropout", type=float, default=None, help="Classifier head dropout. Defaults to training_config.json when available.")
    parser.add_argument("--norm", choices=["none", "batch", "layer"], default=None, help="Classifier head normalization. Defaults to training_config.json when available.")
    parser.add_argument("--model-name", default=None, help="CLIP model name (e.g., ViT-B-32). Defaults to training_config.json or ViT-B-32.")
    parser.add_argument("--pretrained", default=None, help="CLIP pretrained weights (e.g., openai). Defaults to training_config.json or openai.")
    parser.add_argument("--overlay-mode", choices=["compact", "detailed"], default="compact", help="Overlay density. compact shows core prediction; detailed adds temporal summary.")
    return parser.parse_args()




def load_training_config(checkpoint_path: str | Path) -> dict:
    config_path = Path(checkpoint_path).resolve().parent / "training_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_model_config(args: argparse.Namespace, checkpoint_path: str | Path) -> dict:
    config = load_training_config(checkpoint_path)
    return {
        "embedding_dim": args.embedding_dim if args.embedding_dim is not None else int(config.get("embedding_dim", 512)),
        "hidden_dim": args.hidden_dim if args.hidden_dim is not None else int(config.get("hidden_dim", 128)),
        "dropout": args.dropout if args.dropout is not None else float(config.get("dropout", 0.2)),
        "norm": args.norm if args.norm is not None else str(config.get("norm", "none")),
    }


def resolve_clip_config(args: argparse.Namespace, checkpoint_path: str | Path) -> dict:
    config = load_training_config(checkpoint_path)
    return {
        "model_name": args.model_name if args.model_name is not None else str(config.get("model_name", "ViT-B-32")),
        "pretrained": args.pretrained if args.pretrained is not None else str(config.get("pretrained", "openai")),
    }


def load_class_labels(manifest_path: str | Path) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    class_to_idx = manifest["class_to_idx"]
    return [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def sample_frame_indices(start_frame: int, end_frame: int, samples: int) -> list[int]:
    end_frame = max(start_frame + 1, end_frame)
    if samples <= 1:
        return [(start_frame + end_frame - 1) // 2]
    values = np.linspace(start_frame, end_frame - 1, num=samples)
    return sorted({int(round(value)) for value in values})


def read_frame_as_pil(capture: cv2.VideoCapture, frame_index: int) -> Image.Image | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@torch.no_grad()
def predict_scenes(
    video_path: str | Path,
    scene_records,
    clip_model,
    preprocess,
    classifier,
    text_classifier,
    labels: list[str],
    device: torch.device,
    samples_per_scene: int,
) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    rows: list[dict] = []
    for record in scene_records:
        tensors = []
        for frame_index in sample_frame_indices(
            record.start_frame,
            record.end_frame,
            samples=samples_per_scene,
        ):
            image = read_frame_as_pil(capture, frame_index)
            if image is None:
                continue
            tensors.append(preprocess(image))

        if not tensors:
            continue

        images = torch.stack(tensors).to(device)
        frame_embeddings = clip_model.encode_image(images)
        frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        shot_embedding = frame_embeddings.mean(dim=0, keepdim=True)
        shot_embedding = shot_embedding / shot_embedding.norm(dim=-1, keepdim=True)

        logits = classifier(shot_embedding.float())
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = probabilities.max(dim=0)
        label = labels[int(pred_idx)]
        confidence_value = float(confidence.item())

        text_logits = text_classifier(shot_embedding.float())
        text_probabilities = torch.softmax(text_logits, dim=1)[0]
        text_confidence, text_pred_idx = text_probabilities.max(dim=0)
        text_confidence_value = float(text_confidence.item())
        has_text = bool(int(text_pred_idx.item()))
        text_label = "text" if has_text else "notext"
        guide_text = guide_for_label(
            label,
            confidence_value,
            duration_sec=record.duration_sec,
            has_text=has_text,
            text_confidence=text_confidence_value,
        )

        rows.append(
            {
                "video_id": record.video_id,
                "scene_id": record.scene_id,
                "cut_id": record.cut_id,
                "start_sec": f"{record.start_sec:.4f}",
                "end_sec": f"{record.end_sec:.4f}",
                "duration_sec": f"{record.duration_sec:.4f}",
                "shot_type": label,
                "confidence": f"{confidence_value:.6f}",
                "text_label": text_label,
                "has_text": has_text,
                "text_confidence": f"{text_confidence_value:.6f}",
                "guide_text": guide_text,
            }
        )

    capture.release()
    return rows


def write_shot_log(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "scene_id",
        "cut_id",
        "start_sec",
        "end_sec",
        "duration_sec",
        "shot_type",
        "confidence",
        "text_label",
        "has_text",
        "text_confidence",
        "guide_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font with fallback to default if file not found."""
    try:
        return ImageFont.truetype(path, size=size)
    except (OSError, IOError):
        # Fallback to DejaVuSans if specified font not found
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except (OSError, IOError):
            # Final fallback to default PIL font
            return ImageFont.load_default()


def draw_overlay(
    frame,
    row: dict,
    font_path: str,
    total_cuts: int = 0,
    overlay_mode: str = "compact",
    temporal_summary: dict | None = None,
):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image.size
    title_font = load_font(font_path, max(24, width // 24))
    body_font = load_font(font_path, max(18, width // 34))

    text_value = "YES" if row["text_label"] == "text" else "NO"
    confidence_value = float(row["confidence"])
    cut_number = int(row.get("scene_id", row.get("cut_id", 0)))
    title = f"CUT {cut_number}/{total_cuts or '?'}"
    shot_line = f"SHOT: {row['shot_type']}  {confidence_value:.2f}"
    warning_line = "LOW CONFIDENCE" if confidence_value < 0.45 else ""
    text_line = f"TEXT: {text_value}  {float(row['text_confidence']):.2f}"
    guide_title = "GUIDE:"
    body_lines = wrap_text(row["guide_text"], width=24)[:3]
    lines = [title, shot_line]
    if warning_line:
        lines.append(warning_line)
    lines.append(text_line)
    if overlay_mode == "detailed":
        summary = temporal_summary or {}
        lines.extend(
            [
                f"AVG CUT: {float(summary.get('avg_duration', 0.0)):.2f}s",
                f"FAST RATIO: {float(summary.get('fast_cut_ratio', 0.0)):.2f}",
                f"DENSITY: {summary.get('text_density_level', 'unknown')}",
                f"DOMINANT: {summary.get('dominant_shot_type', 'unknown')}",
            ]
        )
    lines += [guide_title] + body_lines

    line_heights = []
    max_line_width = 0
    for idx, line in enumerate(lines):
        font = title_font if idx == 0 else body_font
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    padding = 18
    box_width = min(width - 32, max_line_width + padding * 2)
    box_height = sum(line_heights) + padding * 2 + 10 * (len(lines) - 1)
    x = 16
    y = max(16, height - box_height - 28)

    draw.rounded_rectangle(
        (x, y, x + box_width, y + box_height),
        radius=10,
        fill=(0, 0, 0, 170),
    )

    text_y = y + padding
    for idx, line in enumerate(lines):
        font = title_font if idx == 0 else body_font
        if idx == 0:
            fill = (255, 255, 255, 255)
        elif line == "LOW CONFIDENCE":
            fill = (255, 210, 120, 255)
        elif line.startswith(("TEXT:", "AVG CUT:", "FAST RATIO:", "DENSITY:", "DOMINANT:")):
            fill = (135, 220, 255, 255)
        else:
            fill = (235, 235, 235, 255)
        draw.text((x + padding, text_y), line, font=font, fill=fill)
        text_y += line_heights[idx] + 10

    progress_total = max(total_cuts, 1)
    progress_ratio = min(max(cut_number / progress_total, 0.0), 1.0)
    bar_x = x
    bar_y = min(height - 18, y + box_height + 8)
    bar_width = box_width
    bar_height = 6
    draw.rounded_rectangle(
        (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
        radius=3,
        fill=(255, 255, 255, 80),
    )
    draw.rounded_rectangle(
        (bar_x, bar_y, bar_x + int(bar_width * progress_ratio), bar_y + bar_height),
        radius=3,
        fill=(135, 220, 255, 230),
    )

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def create_overlay_video(
    video_path: str | Path,
    rows: list[dict],
    output_path: str | Path,
    font_path: str,
    overlay_mode: str = "compact",
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    sorted_rows = sorted(rows, key=lambda row: float(row["start_sec"]))
    temporal_summary = build_temporal_summary(sorted_rows)
    scene_index = 0
    active_row = sorted_rows[0] if sorted_rows else None

    for frame_index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        current_sec = frame_index / fps if fps else 0.0

        while (
            scene_index + 1 < len(sorted_rows)
            and current_sec >= float(sorted_rows[scene_index]["end_sec"])
        ):
            scene_index += 1
            active_row = sorted_rows[scene_index]

        if active_row is not None:
            frame = draw_overlay(
                frame,
                active_row,
                font_path=font_path,
                total_cuts=len(sorted_rows),
                overlay_mode=overlay_mode,
                temporal_summary=temporal_summary,
            )
        writer.write(frame)

    capture.release()
    writer.release()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir) / video_path.stem.split("_", 1)[0]
    labels = load_class_labels(args.manifest)
    device = get_device(args.device)

    clip_config = resolve_clip_config(args, args.checkpoint)
    clip_model, preprocess = load_open_clip_model(
        model_name=clip_config["model_name"],
        pretrained=clip_config["pretrained"],
        device=device,
    )
    shot_model_config = resolve_model_config(args, args.checkpoint)
    text_model_config = resolve_model_config(args, args.text_checkpoint)
    classifier = ShotClassifier(
        num_classes=len(labels),
        **shot_model_config,
    ).to(device)
    classifier.load_state_dict(torch.load(args.checkpoint, map_location=device))
    classifier.eval()
    text_classifier = TextPresenceClassifier(
        **text_model_config,
    ).to(device)
    text_classifier.load_state_dict(torch.load(args.text_checkpoint, map_location=device))
    text_classifier.eval()

    _, scenes = detect_video_scenes(
        video_path=video_path,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
    )
    rows = predict_scenes(
        video_path=video_path,
        scene_records=scenes,
        clip_model=clip_model,
        preprocess=preprocess,
        classifier=classifier,
        text_classifier=text_classifier,
        labels=labels,
        device=device,
        samples_per_scene=args.samples_per_scene,
    )

    shot_log_path = output_dir / "shot_log.csv"
    summary_path = output_dir / "guide_summary.txt"
    overlay_path = output_dir / "output_overlay.mp4"
    write_shot_log(shot_log_path, rows)
    summary_path.write_text(summarize_guides(rows), encoding="utf-8")
    create_overlay_video(
        video_path=video_path,
        rows=rows,
        output_path=overlay_path,
        font_path=args.font,
        overlay_mode=args.overlay_mode,
    )

    print(f"scenes: {len(rows)}")
    print(f"shot_log: {shot_log_path}")
    print(f"summary: {summary_path}")
    print(f"overlay: {overlay_path}")


if __name__ == "__main__":
    main()
