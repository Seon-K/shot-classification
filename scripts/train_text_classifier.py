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
    compute_class_weights,
    evaluate_classifier,
    load_embedding_dataset,
    save_training_outputs,
    train_model,
)
from shortform.utils.reproducibility import configure_reproducibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text/notext classifier on CLIP shot embeddings.")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--output-dir", default="artifacts/text_classifier")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use-class-weight", action="store_true", help="Apply inverse-frequency class weights to text/notext loss.")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Sample text/notext classes with inverse-frequency weights.")
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce", help="Training loss type.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience by validation macro F1. 0 disables it.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability in classifier head.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--scheduler", action="store_true", help="Use ReduceLROnPlateau on validation macro F1.")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Input embedding dimension from CLIP backbone.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of classifier head.")
    parser.add_argument("--norm", choices=["none", "batch", "layer"], default="none", help="Optional normalization after first linear layer.")
    parser.add_argument("--model-name", default="ViT-B-32", help="CLIP model name used to create embeddings; stored for reproducibility.")
    parser.add_argument("--pretrained", default="openai", help="CLIP pretrained tag used to create embeddings; stored for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility(seed=args.seed, project_root=PROJECT_ROOT)
    embedding_dir = Path(args.embedding_dir)
    train_embedding_path = embedding_dir / "train_shot_embeddings.pt"
    val_embedding_path = embedding_dir / "val_shot_embeddings.pt"
    test_embedding_path = embedding_dir / "test_shot_embeddings.pt"
    class_to_idx = {"notext": 0, "text": 1}
    class_names = ["notext", "text"]

    _, train_labels = load_embedding_dataset(train_embedding_path, target="has_text")
    class_weights = compute_class_weights(train_labels, 2) if args.use_class_weight else None

    train_loader = build_embedding_loader(
        train_embedding_path,
        batch_size=args.batch_size,
        shuffle=True,
        target="has_text",
        use_weighted_sampler=args.use_weighted_sampler,
        num_classes=2,
    )
    val_loader = build_embedding_loader(
        val_embedding_path,
        batch_size=args.batch_size,
        shuffle=False,
        target="has_text",
    )
    test_loader = build_embedding_loader(
        test_embedding_path,
        batch_size=args.batch_size,
        shuffle=False,
        target="has_text",
    )

    device = torch.device(args.device)
    model = TextPresenceClassifier(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, norm=args.norm)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=2,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        class_weights=class_weights,
        patience=args.patience,
        scheduler=args.scheduler,
        device=device,
    )
    test_metrics_last = evaluate_classifier(
        model,
        test_loader,
        device=device,
        num_classes=2,
        class_names=class_names,
    )
    best_model = TextPresenceClassifier(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, norm=args.norm).to(device)
    best_model.load_state_dict(model._best_state_dict)
    test_metrics_best = evaluate_classifier(
        best_model,
        test_loader,
        device=device,
        num_classes=2,
        class_names=class_names,
    )

    training_config = {
        "use_class_weight": args.use_class_weight,
        "use_weighted_sampler": args.use_weighted_sampler,
        "loss_type": args.loss_type,
        "patience": args.patience,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "norm": args.norm,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "seed": args.seed,
    }
    save_training_outputs(
        output_dir=args.output_dir,
        model=model,
        history=history,
        test_metrics_last=test_metrics_last,
        test_metrics_best=test_metrics_best,
        class_to_idx=class_to_idx,
        checkpoint_prefix="text_classifier",
        training_config=training_config,
    )

    last = history[-1]
    best = max(history, key=lambda item: item["val_macro_f1"])
    print("텍스트 여부 분류 학습 완료")
    print(f"last train accuracy: {last['train_accuracy']:.4f}")
    print(f"last val accuracy:   {last['val_accuracy']:.4f}")
    print(f"best epoch:          {model._best_epoch}")
    print(f"best val accuracy:   {best['val_accuracy']:.4f}")
    print(f"best val macro F1:   {best['val_macro_f1']:.4f}")
    print(f"test accuracy best:  {test_metrics_best['accuracy']:.4f}")
    print(f"test macro F1 best:  {test_metrics_best['macro_f1']:.4f}")
    print(f"저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
