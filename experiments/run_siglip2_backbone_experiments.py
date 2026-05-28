from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import run_dinov3_backbone_experiments as common


ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "outputs" / "clip_embeddings" / "clip_embedding_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "siglip2_backbones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
common.OUTPUT_DIR = OUTPUT_DIR

DEVICE = common.DEVICE


@dataclass(frozen=True)
class SigLIP2BackboneConfig:
    name: str
    model_name: str
    pretrained: str = "webli"
    batch_size: int = 16


def create_siglip2_model(config: SigLIP2BackboneConfig):
    available = {(model_name, pretrained) for model_name, pretrained in open_clip.list_pretrained()}
    key = (config.model_name, config.pretrained)
    if key not in available:
        print(f"skipping unavailable open_clip model: {key}")
        return None, None
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=config.pretrained,
        device=DEVICE,
    )
    model.eval()
    return model, preprocess


def extract_embeddings(df: pd.DataFrame, config: SigLIP2BackboneConfig) -> tuple[np.ndarray | None, float]:
    emb_path = OUTPUT_DIR / f"{config.name}_embeddings.npz"
    if emb_path.exists():
        print("loading cached embeddings:", emb_path)
        started = time.perf_counter()
        return np.load(emb_path)["embeddings"].astype("float32"), time.perf_counter() - started

    model, preprocess = create_siglip2_model(config)
    if model is None:
        return None, 0.0

    loader = DataLoader(
        common.ImagePathDataset(df, preprocess),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    chunks = []
    order = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, indices in tqdm(loader, desc=f"{config.name} embeddings"):
            features = model.encode_image(images.to(DEVICE))
            features = features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-8)
            chunks.append(features.cpu().float().numpy())
            order.extend(indices.numpy().tolist())

    order_arr = np.array(order)
    assert np.all(order_arr == np.arange(len(df)))
    embeddings = np.concatenate(chunks, axis=0).astype("float32")
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        filepath=df["filepath"].to_numpy(),
        model_name=config.model_name,
        pretrained=config.pretrained,
    )
    return embeddings, elapsed


def main() -> None:
    common.seed_everything()
    print("device:", DEVICE)
    print("system:", platform.platform())
    df = pd.read_csv(META_CSV)
    backbones = [
        SigLIP2BackboneConfig("siglip2_vit_b16_256_webli", "ViT-B-16-SigLIP2-256", "webli", batch_size=16),
    ]
    head_configs = [
        common.HeadConfig("focal_multitask_512_256", loss_type="focal"),
    ]
    rows = []
    for backbone in backbones:
        print("extracting", backbone)
        embeddings, extraction_time = extract_embeddings(df, backbone)
        if embeddings is None:
            continue
        data, masks = common.build_data(df, embeddings)
        for head_config in head_configs:
            experiment_name = f"{backbone.name}_{head_config.name}"
            print("training", experiment_name)
            result = common.train_head(head_config, data)
            summary = common.save_metric_tables(experiment_name, df, masks, result, extraction_time, embeddings.shape[1])
            summary.update({"backbone": backbone.model_name, "pretrained": backbone.pretrained, **head_config.__dict__})
            rows.append(summary)
            print(json.dumps(summary, indent=2))

    summary_df = pd.DataFrame(rows).sort_values(["joint_accuracy", "shot_macro_f1"], ascending=False) if rows else pd.DataFrame()
    summary_df.to_csv(OUTPUT_DIR / "siglip2_backbone_experiment_summary.csv", index=False, encoding="utf-8-sig")
    if rows:
        best = summary_df.iloc[0].to_dict()
        with open(OUTPUT_DIR / "best_siglip2_backbone_experiment.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print("best:", best)


if __name__ == "__main__":
    main()
