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
    ShotClassifier,
    build_embedding_loader,
    collect_prediction_rows,
    compute_class_weights,
    create_experiment_summary,
    evaluate_classifier,
    load_embedding_dataset,
    save_error_samples,
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
    parser.add_argument("--use-class-weight", action="store_true", help="Apply inverse-frequency class weights to the loss.")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Sample minority classes more often during training.")
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce", help="Training loss type.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience by validation macro F1. 0 disables it.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability in classifier head.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--scheduler", action="store_true", help="Use ReduceLROnPlateau on validation macro F1.")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Input embedding dimension from CLIP backbone.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of classifier head.")
    parser.add_argument("--norm", choices=["none", "batch", "layer"], default="none", help="Optional normalization after first linear layer.")
    parser.add_argument("--error-sample-count", type=int, default=20, help="Number of high/low confidence errors to export.")
    parser.add_argument("--model-name", default="ViT-B-32", help="CLIP model name used to create embeddings; stored for reproducibility.")
    parser.add_argument("--pretrained", default="openai", help="CLIP pretrained tag used to create embeddings; stored for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    class_to_idx = manifest["class_to_idx"]
    class_names = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    num_classes = len(class_to_idx)

    embedding_dir = Path(args.embedding_dir)
    train_embedding_path = embedding_dir / "train_shot_embeddings.pt"
    val_embedding_path = embedding_dir / "val_shot_embeddings.pt"
    test_embedding_path = embedding_dir / "test_shot_embeddings.pt"

    _, train_labels = load_embedding_dataset(train_embedding_path)
    class_weights = compute_class_weights(train_labels, num_classes) if args.use_class_weight else None

    train_loader = build_embedding_loader(
        train_embedding_path,
        batch_size=args.batch_size,
        shuffle=True,
        use_weighted_sampler=args.use_weighted_sampler,
        num_classes=num_classes,
    )
    val_loader = build_embedding_loader(
        val_embedding_path,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = build_embedding_loader(
        test_embedding_path,
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
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        norm=args.norm,
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
        num_classes=num_classes,
        class_names=class_names,
    )

    best_model = ShotClassifier(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        norm=args.norm,
    ).to(device)
    best_model.load_state_dict(model._best_state_dict)
    test_metrics_best = evaluate_classifier(
        best_model,
        test_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    prediction_rows = collect_prediction_rows(
        best_model,
        test_embedding_path,
        device=device,
        class_names=class_names,
    )
    save_error_samples(
        Path(args.output_dir) / "error_samples",
        prediction_rows,
        high_count=args.error_sample_count,
        low_count=args.error_sample_count,
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
    }
    save_training_outputs(
        output_dir=args.output_dir,
        model=model,
        history=history,
        test_metrics_last=test_metrics_last,
        test_metrics_best=test_metrics_best,
        class_to_idx=class_to_idx,
        checkpoint_prefix="shot_classifier",
        training_config=training_config,
    )
    metrics = json.loads((Path(args.output_dir) / "metrics.json").read_text(encoding="utf-8"))
    create_experiment_summary(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        metrics=metrics,
        error_sample_dir=Path(args.output_dir) / "error_samples",
    )

    last = history[-1]
    best = max(history, key=lambda item: item["val_macro_f1"])
    print("학습 완료")
    print(f"last train accuracy: {last['train_accuracy']:.4f}")
    print(f"last val accuracy:   {last['val_accuracy']:.4f}")
    print(f"best epoch:          {model._best_epoch}")
    print(f"best val accuracy:   {best['val_accuracy']:.4f}")
    print(f"best val macro F1:   {best['val_macro_f1']:.4f}")
    print(f"test accuracy best:  {test_metrics_best['accuracy']:.4f}")
    print(f"test macro F1 best:  {test_metrics_best['macro_f1']:.4f}")
    print(f"test balanced acc:   {test_metrics_best['balanced_accuracy']:.4f}")
    print(f"test weighted F1:    {test_metrics_best['weighted_f1']:.4f}")
    print(f"test top-2 accuracy: {test_metrics_best['top2_accuracy']:.4f}")
    print(f"저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
