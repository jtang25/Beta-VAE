"""
Create a GIF from recon panels, keeping only the bottom half (recons).

Usage:
  python scripts/make_recon_gif.py --config configs/beta_vae_se.yaml
  python scripts/make_recon_gif.py --config configs/beta_vae_se.yaml --pattern "recon_epoch*.png" --output recons_only.gif --duration 200
"""
import argparse
import glob
import os
from pathlib import Path
from PIL import Image

import sys
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config


def natural_sort_key(path):
    stem = Path(path).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    nums = [int(x) for x in digits] if digits else []
    return nums, path


def main():
    parser = argparse.ArgumentParser(description="Create GIF from recon panels (bottom half only).")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--pattern", type=str, default="recon_epoch*.png", help="Glob pattern within figures_dir.")
    parser.add_argument("--output", type=str, default="recons_only.gif", help="Output GIF filename (saved to figures_dir).")
    parser.add_argument("--duration", type=int, default=200, help="Frame duration (ms).")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    cfg = get_config()
    figures_dir = Path(cfg.paths.figures_dir)
    files = sorted(glob.glob(str(figures_dir / args.pattern)), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} found in {figures_dir}")

    frames = []
    for f in files:
        img = Image.open(f)
        w, h = img.size
        crop = img.crop((0, h // 2, w, h))
        frames.append(crop)

    out_path = figures_dir / args.output
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )
    print(f"Saved GIF to {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
