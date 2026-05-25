from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ManifestImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform) -> None:
        self.csv_path = Path(csv_path)
        self.transform = transform
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        image = self.transform(image)
        return image, record


def get_device(device_name: str | None = None) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_open_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: torch.device | None = None,
):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch가 필요합니다. `pip install open-clip-torch`를 실행하세요."
        ) from exc

    device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess


@torch.no_grad()
def extract_frame_embeddings(
    csv_path: str | Path,
    output_path: str | Path,
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict:
    dataset = ManifestImageDataset(csv_path, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    embeddings: list[torch.Tensor] = []
    records: list[dict] = []

    for images, batch_records in loader:
        images = images.to(device)
        batch_embeddings = model.encode_image(images)
        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
        embeddings.append(batch_embeddings.cpu())

        batch_size_actual = images.shape[0]
        for idx in range(batch_size_actual):
            records.append({key: batch_records[key][idx] for key in batch_records})

    result = {
        "embeddings": torch.cat(embeddings, dim=0),
        "records": records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result


def build_shot_embeddings(
    frame_embedding_path: str | Path,
    output_path: str | Path,
) -> dict:
    frame_data = torch.load(frame_embedding_path, map_location="cpu")
    frame_embeddings: torch.Tensor = frame_data["embeddings"]
    records: list[dict] = frame_data["records"]

    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped_indices[(record["video_id"], record["cut_id"])].append(idx)

    shot_embeddings: list[torch.Tensor] = []
    shot_records: list[dict] = []

    for (video_id, cut_id), indices in sorted(grouped_indices.items()):
        pooled = frame_embeddings[indices].mean(dim=0)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        first_record = records[indices[0]]
        shot_embeddings.append(pooled)
        shot_records.append(
            {
                "video_id": video_id,
                "cut_id": cut_id,
                "shot_type": first_record["shot_type"],
                "label": int(first_record["label"]),
                "has_text": first_record["has_text"],
                "frame_count": len(indices),
            }
        )

    result = {
        "embeddings": torch.stack(shot_embeddings, dim=0),
        "records": shot_records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result
