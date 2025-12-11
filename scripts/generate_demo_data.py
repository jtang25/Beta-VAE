"""
Generate a small synthetic dataset under cfg.paths.processed_dir for quick smoke tests.

Creates grayscale images with simple patterns per class:
  - glioma: bright circle
  - meningioma: horizontal bands
  - pituitary: cross-hatch pattern
  - notumor: noise only

By default writes to data/processed/{train,test}/<class>/ using the active config.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config  # noqa: E402


def make_canvas(rng, size, base_level=0.3, noise=0.05):
    arr = rng.normal(base_level, noise, size=(size, size))
    return np.clip(arr, 0, 1)


def pattern_for_class(cls, rng, size):
    arr = make_canvas(rng, size, 0.25, 0.08)
    yy, xx = np.mgrid[:size, :size]
    if cls == "glioma":
        circle = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= (size // 4) ** 2
        arr[circle] += 0.35
    elif cls == "meningioma":
        band = (yy % (size // 8)) < (size // 16)
        arr[band] += 0.25
    elif cls == "pituitary":
        diag = ((xx + yy) % (size // 6)) < (size // 16)
        anti = ((xx - yy) % (size // 6)) < (size // 16)
        arr[diag | anti] += 0.25
    else:
        arr += rng.normal(0.0, 0.02, size=arr.shape)
    return np.clip(arr, 0, 1)


def write_split(proc_root, split, classes, per_class, size, seed):
    rng = np.random.default_rng(seed)
    for cls in classes:
        out_dir = Path(proc_root) / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(per_class):
            arr = pattern_for_class(cls, rng, size)
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            img.save(out_dir / f"{cls}_{idx}.png")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--train-per-class", type=int, default=24, help="Images per class for train split.")
    parser.add_argument("--test-per-class", type=int, default=12, help="Images per class for test split.")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    cfg = get_config()

    proc_root = Path(cfg.paths.processed_dir)
    size = cfg.data.image_size
    classes = ["glioma", "meningioma", "pituitary", "notumor"]

    write_split(proc_root, cfg.data.train_subdir, classes, args.train_per_class, size, seed=0)
    write_split(proc_root, cfg.data.test_subdir, classes, args.test_per_class, size, seed=1)

    print(f"Wrote synthetic data to {proc_root} (train/test splits).")


if __name__ == "__main__":
    main()
