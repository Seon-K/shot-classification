from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.classifier import (  # noqa: E402
    build_embedding_loader,
    evaluate_classifier,
    save_training_outputs,
    train_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shot classifier on CLIP shot embeddings.")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    class_to_idx = manifest["class_to_idx"]
    num_classes = len(class_to_idx)

    embedding_dir = Path(args.embedding_dir)
    train_loader = build_embedding_loader(
        embedding_dir / "train_shot_embeddings.pt",
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = build_embedding_loader(
        embedding_dir / "val_shot_embeddings.pt",
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = build_embedding_loader(
        embedding_dir / "test_shot_embeddings.pt",
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device(args.device)
    model, history = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    test_metrics = evaluate_classifier(
        model,
        test_loader,
        device=device,
        num_classes=num_classes,
    )
    save_training_outputs(
        output_dir=args.output_dir,
        model=model,
        history=history,
        test_metrics=test_metrics,
        class_to_idx=class_to_idx,
    )

    last = history[-1]
    print("학습 완료")
    print(f"last train accuracy: {last['train_accuracy']:.4f}")
    print(f"last val accuracy:   {last['val_accuracy']:.4f}")
    print(f"best val accuracy:   {max(item['val_accuracy'] for item in history):.4f}")
    print(f"test accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"test macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
