"""
Save a preview grid of the first validation batch (deterministic under the config seed).

Usage:
  python scripts/preview_val_batch.py --config configs/beta_vae_se.yaml

Outputs:
  - figures/val_preview_seed{seed}.png under cfg.paths.figures_dir
  - figures/val_preview_seed{seed}_paths.txt listing image paths and class names
"""
import argparse
import os
from pathlib import Path

import torch
from torchvision.utils import save_image

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
import sys
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.io import ensure_dirs
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms


def main():
    parser = argparse.ArgumentParser(description="Preview first validation batch.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = parser.parse_args()
    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    cfg = get_config()
    ensure_dirs()

    tf = get_test_transforms()
    _, test_loader = build_dataloaders(transform_train=tf, transform_test=tf)

    batch = next(iter(test_loader))
    imgs = batch["image"]
    class_names = batch.get("class_name", None)
    paths = batch.get("path", None)

    out_dir = Path(cfg.paths.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.data.seed
    grid_path = out_dir / f"val_preview_seed{seed}.png"
    save_image(imgs, grid_path, nrow=min(len(imgs), 8), normalize=True)

    meta_path = out_dir / f"val_preview_seed{seed}_paths.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for i in range(len(imgs)):
            cls = class_names[i] if class_names else ""
            p = paths[i] if paths else ""
            f.write(f"{i}: class={cls} path={p}\n")

    print(f"Saved validation preview to {grid_path}")
    print(f"Saved paths/classes to {meta_path}")


if __name__ == "__main__":
    main()
