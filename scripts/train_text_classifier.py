from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortform.classifier import (  # noqa: E402
    TextPresenceClassifier,
    build_embedding_loader,
    evaluate_classifier,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text/notext classifier on CLIP shot embeddings.")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--output-dir", default="artifacts/text_classifier")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    train_loader = build_embedding_loader(
        embedding_dir / "train_shot_embeddings.pt",
        batch_size=args.batch_size,
        shuffle=True,
        target="has_text",
    )
    val_loader = build_embedding_loader(
        embedding_dir / "val_shot_embeddings.pt",
        batch_size=args.batch_size,
        shuffle=False,
        target="has_text",
    )
    test_loader = build_embedding_loader(
        embedding_dir / "test_shot_embeddings.pt",
        batch_size=args.batch_size,
        shuffle=False,
        target="has_text",
    )

    device = torch.device(args.device)
    model = TextPresenceClassifier()
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=2,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    test_metrics = evaluate_classifier(model, test_loader, device=device, num_classes=2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "text_classifier.pt")

    import json

    summary = {
        "class_to_idx": {"notext": 0, "text": 1},
        "history": history,
        "best_val_accuracy": max(item["val_accuracy"] for item in history),
        "best_val_macro_f1": max(item["val_macro_f1"] for item in history),
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    last = history[-1]
    print("텍스트 여부 분류 학습 완료")
    print(f"last train accuracy: {last['train_accuracy']:.4f}")
    print(f"last val accuracy:   {last['val_accuracy']:.4f}")
    print(f"best val accuracy:   {summary['best_val_accuracy']:.4f}")
    print(f"test accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"test macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
