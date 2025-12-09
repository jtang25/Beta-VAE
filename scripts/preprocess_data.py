"""
Preprocess the brain tumour dataset into processed train/test splits.

Expected raw layout (per config): any subfolders are treated as classes, e.g.:
  data/braintumour/
    glioma/*.png|jpg|...
    meningioma/*.png|jpg|...
    pituitary/*.png|jpg|...
    notumor/*.png|jpg|...

Output (per config):
  data/processed/train/<class>/*.png
  data/processed/test/<class>/*.png

Steps:
  1) split_dataset.split_from_raw() copies raw files into processed/train & processed/test
  2) resize_and_normalize.preprocess_dataset() resizes to cfg.data.image_size and optionally normalizes
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config  # noqa: E402
from data_processing.split_dataset import split_from_raw, verify_processed  # noqa: E402
from data_processing.resize_and_normalize import preprocess_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Preprocess brain tumour dataset into processed/train|test splits.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (defaults to configs/beta_vae_se.yaml).")
    parser.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "global_z"], help="Normalization mode for resizing.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing processed dir before copying/splitting.")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    cfg = get_config()

    print(f"Using raw data from: {cfg.paths.raw_dir}")
    print(f"Writing processed data to: {cfg.paths.processed_dir}")
    print(f"Normalization mode: {args.normalization}")
    print("Classes are auto-detected from subfolders under raw_dir.")

    split_from_raw(overwrite=args.overwrite)
    preprocess_dataset(
        compute_stats=(args.normalization == "global_z"),
        normalization_mode=args.normalization
    )
    verify_processed()
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
