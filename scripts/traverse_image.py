"""
Run latent traversals on a specific image (not just the first batch).

Usage:
  python scripts/traverse_image.py --image path/to/image.png --config configs/beta_vae_se.yaml --checkpoint best

Outputs:
  - traversal_dim*.png under cfg.paths.figures_dir
  - traversal_tumor_<class>.png per class-direction if available
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.io import ensure_dirs
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms
from models.beta_vae import BetaVAE
from evaluation.traversal import run_traversals


def load_model(checkpoint_tag, device):
    cfg = get_config()
    if checkpoint_tag in ["best", "latest"]:
        path = Path(cfg.paths.models_dir) / f"{cfg.paths.run_id}_{checkpoint_tag}.pt"
    else:
        path = Path(checkpoint_tag)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint at {path}: {e}") from e
    model = BetaVAE()
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run latent traversals on a specific image.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file to traverse.")
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Checkpoint tag (best|latest) or explicit path.")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated latent indices to traverse (default uses config fallback).")
    parser.add_argument("--span", type=float, default=None, help="Traversal span (overrides config).")
    parser.add_argument("--steps", type=int, default=None, help="Traversal steps (overrides config).")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    cfg = get_config()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    tf = get_test_transforms()
    img = Image.open(args.image).convert("L" if cfg.data.grayscale else "RGB")
    img_t = tf(img).unsqueeze(0)

    _, test_loader = build_dataloaders(transform_train=tf, transform_test=tf)

    indices = None
    if args.indices:
        try:
            indices = [int(i.strip()) for i in args.indices.split(",") if i.strip() != ""]
        except ValueError:
            print("Could not parse --indices; ignoring.")

    span = args.span if args.span is not None else cfg.inference.edit_span
    steps = args.steps if args.steps is not None else cfg.evaluation.traversal_steps

    run_traversals(
        model,
        test_loader,
        device,
        indices=indices,
        steps=steps,
        span=span,
        images_override=img_t,
    )
    print(f"Saved traversals to {cfg.paths.figures_dir}")


if __name__ == "__main__":
    main()
