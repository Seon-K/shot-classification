from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from shortform.embedding import get_device, load_open_clip_model


DEFAULT_PROMPTS = {
    "close-up": "a close-up shot of the main subject",
    "medium": "a medium shot showing the upper body or action",
    "wide": "a wide shot showing the full body and background",
    "object": "a shot focused on an object, product, text, or screen",
    "space": "a shot focused on the surrounding space or location",
}


def load_class_mapping(manifest_path: str | Path) -> dict[str, int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return manifest["class_to_idx"]


@torch.no_grad()
def encode_text_prompts(
    prompts: dict[str, str],
    model,
    device: torch.device,
) -> torch.Tensor:
    import open_clip

    labels = list(prompts)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer([prompts[label] for label in labels]).to(device)
    embeddings = model.encode_text(tokens)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu()


def evaluate_zeroshot(
    shot_embedding_path: str | Path,
    text_embeddings: torch.Tensor,
    labels: list[str],
    confidence_threshold: float = 0.0,
) -> dict:
    shot_data = torch.load(shot_embedding_path, map_location="cpu")
    shot_embeddings: torch.Tensor = shot_data["embeddings"]
    records: list[dict] = shot_data["records"]

    similarities = shot_embeddings @ text_embeddings.T
    confidence, pred_indices = similarities.max(dim=1)

    rows: list[dict] = []
    correct = 0
    total = len(records)
    num_classes = len(labels)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for idx, record in enumerate(records):
        true_idx = int(record["label"])
        pred_idx = int(pred_indices[idx])
        pred_label = labels[pred_idx]
        if float(confidence[idx]) < confidence_threshold:
            pred_label = "uncertain"

        if pred_idx == true_idx:
            correct += 1
        confusion[true_idx, pred_idx] += 1

        rows.append(
            {
                "video_id": record["video_id"],
                "cut_id": record["cut_id"],
                "true_label": labels[true_idx],
                "pred_label": pred_label,
                "confidence": f"{float(confidence[idx]):.6f}",
            }
        )

    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "total": total,
        "rows": rows,
        "confusion": confusion.tolist(),
    }


def save_zeroshot_outputs(
    output_dir: str | Path,
    split: str,
    labels: list[str],
    result: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "cut_id", "true_label", "pred_label", "confidence"],
        )
        writer.writeheader()
        writer.writerows(result["rows"])

    summary = {
        "split": split,
        "labels": labels,
        "accuracy": result["accuracy"],
        "total": result["total"],
        "confusion_matrix": result["confusion"],
    }
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_text_embeddings(
    class_to_idx: dict[str, int],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device_name: str | None = None,
) -> tuple[list[str], torch.Tensor]:
    labels = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    prompts = {label: DEFAULT_PROMPTS[label] for label in labels}
    device = get_device(device_name)
    model, _ = load_open_clip_model(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    text_embeddings = encode_text_prompts(prompts, model=model, device=device)
    return labels, text_embeddings
