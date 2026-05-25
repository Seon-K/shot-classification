from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_CLASSES = ["close-up", "medium", "wide", "object", "space"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    source_folder: str
    video_id: str
    cut_id: str
    shot_type: str
    label: int
    has_text: bool


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def parse_label_folder(folder_name: str) -> tuple[str | None, bool | None]:
    if folder_name == "unclear":
        return None, None

    if folder_name.endswith("_text"):
        return folder_name.removesuffix("_text"), True

    if folder_name.endswith("_notext"):
        return folder_name.removesuffix("_notext"), False

    return folder_name, None


def parse_cut_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_cut_" not in stem:
        return stem, ""

    video_id, cut_suffix = stem.split("_cut_", 1)
    return video_id, cut_suffix


def scan_labeled_dataset(
    root: str | Path,
    classes: Iterable[str] = DEFAULT_CLASSES,
    include_unclear: bool = False,
) -> tuple[list[ImageRecord], dict[str, int], dict[str, int]]:
    root = Path(root)
    class_names = list(classes)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    folder_counts: Counter[str] = Counter()
    records: list[ImageRecord] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        shot_type, has_text = parse_label_folder(folder.name)
        image_paths = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        folder_counts[folder.name] = len(image_paths)

        if shot_type is None:
            if not include_unclear:
                continue
            shot_type = "unclear"
            has_text = False

        if shot_type not in class_to_idx:
            continue

        for image_path in image_paths:
            video_id, cut_id = parse_cut_name(image_path)
            records.append(
                ImageRecord(
                    path=str(image_path),
                    source_folder=folder.name,
                    video_id=video_id,
                    cut_id=cut_id,
                    shot_type=shot_type,
                    label=class_to_idx[shot_type],
                    has_text=bool(has_text),
                )
            )

    return records, class_to_idx, dict(folder_counts)


def split_records_by_video(
    records: list[ImageRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    by_video: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_video[record.video_id].append(record)

    video_ids = sorted(by_video)
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    total = len(video_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_videos = set(video_ids[:train_end])
    val_videos = set(video_ids[train_end:val_end])

    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []

    for video_id in video_ids:
        if video_id in train_videos:
            train_records.extend(by_video[video_id])
        elif video_id in val_videos:
            val_records.extend(by_video[video_id])
        else:
            test_records.extend(by_video[video_id])

    return train_records, val_records, test_records


class ShotImageDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        transform: Callable | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.records = records
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(record.label, dtype=torch.long)
        if not self.return_metadata:
            return image, label

        metadata = {
            "path": record.path,
            "video_id": record.video_id,
            "cut_id": record.cut_id,
            "shot_type": record.shot_type,
            "has_text": record.has_text,
        }
        return image, label, metadata


def build_dataloaders(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ShotImageDataset(train_records, transform=build_train_transform())
    val_dataset = ShotImageDataset(val_records, transform=build_eval_transform())
    test_dataset = ShotImageDataset(test_records, transform=build_eval_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def save_split_manifest(
    output_dir: str | Path,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    class_to_idx: dict[str, int],
    folder_counts: dict[str, int],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "class_to_idx": class_to_idx,
        "folder_counts": folder_counts,
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split_name, split_records in {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }.items():
        fieldnames = list(ImageRecord.__dataclass_fields__.keys())
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(record) for record in split_records)


def count_by_label(records: Iterable[ImageRecord]) -> dict[str, int]:
    return dict(Counter(record.shot_type for record in records))
