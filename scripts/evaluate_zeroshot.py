from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.zeroshot import (  # noqa: E402
    build_text_embeddings,
    evaluate_zeroshot,
    load_class_mapping,
    save_zeroshot_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP zero-shot shot classification.")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--output-dir", default="artifacts/zeroshot")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_to_idx = load_class_mapping(args.manifest)
    labels, text_embeddings = build_text_embeddings(
        class_to_idx=class_to_idx,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device_name=args.device,
    )

    print(f"labels: {labels}")

    for split in args.splits:
        shot_embedding_path = Path(args.embedding_dir) / f"{split}_shot_embeddings.pt"
        result = evaluate_zeroshot(
            shot_embedding_path=shot_embedding_path,
            text_embeddings=text_embeddings,
            labels=labels,
            confidence_threshold=args.confidence_threshold,
        )
        save_zeroshot_outputs(
            output_dir=args.output_dir,
            split=split,
            labels=labels,
            result=result,
        )
        print(f"{split}: accuracy={result['accuracy']:.4f}, total={result['total']}")


if __name__ == "__main__":
    main()
