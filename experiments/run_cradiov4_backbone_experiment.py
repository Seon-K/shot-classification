from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import run_dinov3_backbone_experiments as common


ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "outputs" / "clip_embeddings" / "clip_embedding_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_experiments" / "cradiov4_backbones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
common.OUTPUT_DIR = OUTPUT_DIR

DEVICE = common.DEVICE


@dataclass(frozen=True)
class CRADIOv4Config:
    name: str = "cradiov4_so400m_256"
    version: str = "c-radio_v4-so400m"
    image_size: int = 256
    batch_size: int = 4


class RadioImageTransform:
    def __init__(self, image_size: int):
        self.image_size = int(image_size)

    def __call__(self, image):
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        return torch.from_numpy(arr).permute(2, 0, 1)


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-8)


def load_cradiov4_model(config: CRADIOv4Config):
    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version=config.version,
        progress=True,
        skip_validation=True,
        trust_repo=True,
    )
    model.eval().to(DEVICE)
    return model


def extract_embeddings(df: pd.DataFrame, config: CRADIOv4Config) -> tuple[np.ndarray | None, float]:
    emb_path = OUTPUT_DIR / f"{config.name}_embeddings.npz"
    if emb_path.exists():
        print("loading cached embeddings:", emb_path)
        started = time.perf_counter()
        return np.load(emb_path)["embeddings"].astype("float32"), time.perf_counter() - started

    model = load_cradiov4_model(config)
    transform = RadioImageTransform(config.image_size)
    loader = DataLoader(
        common.ImagePathDataset(df, transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    chunks = []
    order = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, indices in tqdm(loader, desc=f"{config.name} embeddings"):
            images = images.to(DEVICE)
            nearest = model.get_nearest_supported_resolution(*images.shape[-2:])
            if (nearest.height, nearest.width) != tuple(images.shape[-2:]):
                images = torch.nn.functional.interpolate(
                    images,
                    size=(nearest.height, nearest.width),
                    mode="bilinear",
                    align_corners=False,
                )
            output = model(images)
            summary = output[0] if isinstance(output, tuple) else output.summary
            chunks.append(normalize_features(summary).cpu().float().numpy())
            order.extend(indices.numpy().tolist())

    order_arr = np.array(order)
    assert np.all(order_arr == np.arange(len(df)))
    embeddings = np.concatenate(chunks, axis=0).astype("float32")
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        filepath=df["filepath"].to_numpy(),
        version=config.version,
        image_size=config.image_size,
    )
    return embeddings, elapsed


def main() -> None:
    common.seed_everything()
    print("device:", DEVICE)
    print("system:", platform.platform())
    df = pd.read_csv(META_CSV)

    backbone = CRADIOv4Config()
    embeddings, extraction_time = extract_embeddings(df, backbone)
    if embeddings is None:
        print("C-RADIOv4 embeddings were not created.")
        return

    data, masks = common.build_data(df, embeddings)
    head_config = common.HeadConfig("focal_multitask_512_256", loss_type="focal")
    experiment_name = f"{backbone.name}_{head_config.name}"
    print("training", experiment_name)
    result = common.train_head(head_config, data)
    summary = common.save_metric_tables(experiment_name, df, masks, result, extraction_time, embeddings.shape[1])
    summary.update({
        "backbone": "C-RADIOv4-SO400M",
        "version": backbone.version,
        "image_size": backbone.image_size,
        **head_config.__dict__,
    })

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "cradiov4_backbone_experiment_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "best_cradiov4_backbone_experiment.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
