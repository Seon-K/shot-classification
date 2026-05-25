from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.embedding import (  # noqa: E402
    build_shot_embeddings,
    extract_frame_embeddings,
    get_device,
    load_open_clip_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP frame and shot embeddings.")
    parser.add_argument("--manifest-dir", default="artifacts/dataset")
    parser.add_argument("--output-dir", default="artifacts/embeddings")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    device = get_device(args.device)
    model, preprocess = load_open_clip_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )

    print(f"device: {device}")
    print(f"model: {args.model_name} ({args.pretrained})")

    for split in args.splits:
        csv_path = manifest_dir / f"{split}.csv"
        frame_output = output_dir / f"{split}_frame_embeddings.pt"
        shot_output = output_dir / f"{split}_shot_embeddings.pt"

        frame_result = extract_frame_embeddings(
            csv_path=csv_path,
            output_path=frame_output,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        shot_result = build_shot_embeddings(
            frame_embedding_path=frame_output,
            output_path=shot_output,
        )

        print(
            f"{split}: frame {tuple(frame_result['embeddings'].shape)} -> "
            f"shot {tuple(shot_result['embeddings'].shape)}"
        )


if __name__ == "__main__":
    main()
