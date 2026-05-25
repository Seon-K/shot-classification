from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.data import (  # noqa: E402
    build_dataloaders,
    count_by_label,
    save_split_manifest,
    scan_labeled_dataset,
    split_records_by_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shot-type image dataloaders.")
    parser.add_argument("--data-root", default="labeled_dataset")
    parser.add_argument("--output-dir", default="artifacts/dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["close-up", "medium", "wide", "object", "space"],
        help="Shot type classes to train. Folder suffixes _text/_notext are merged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records, class_to_idx, folder_counts = scan_labeled_dataset(
        root=args.data_root,
        classes=args.classes,
        include_unclear=False,
    )
    if not records:
        raise RuntimeError(f"No training images found under {args.data_root}")

    train_records, val_records, test_records = split_records_by_video(
        records,
        seed=args.seed,
    )

    save_split_manifest(
        output_dir=args.output_dir,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        class_to_idx=class_to_idx,
        folder_counts=folder_counts,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    images, labels = next(iter(train_loader))

    print("데이터 준비 완료")
    print(f"전체 데이터: {len(records)}장")
    print(f"train: {len(train_records)}장 {count_by_label(train_records)}")
    print(f"val:   {len(val_records)}장 {count_by_label(val_records)}")
    print(f"test:  {len(test_records)}장 {count_by_label(test_records)}")
    print(f"라벨 매핑: {class_to_idx}")
    print(f"train batch image shape: {tuple(images.shape)}")
    print(f"train batch label shape: {tuple(labels.shape)}")
    print(f"저장 위치: {args.output_dir}")
    print("참고: DataLoader 객체 대신 split CSV와 manifest JSON을 저장합니다.")

    _ = val_loader, test_loader


if __name__ == "__main__":
    main()
