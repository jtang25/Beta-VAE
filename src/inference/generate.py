import os
import sys
from pathlib import Path
import torch
import numpy as np
from torchvision.utils import save_image

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from data_processing.augmentations import get_test_transforms
from utils.brain_tumor_utils.datautils import build_dataloaders
from models.beta_vae import BetaVAE

def load_model(weights="best"):
    cfg = get_config()
    path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_{weights}.pt"
    if not os.path.exists(path):
        path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_latest.pt"
    payload = torch.load(path, map_location="cpu")
    model = BetaVAE()
    model.load_state_dict(payload["model_state"])
    return model

def sample_random(model, n, out_dir, seed=None, filename="samples.png"):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    with torch.no_grad():
        imgs = model.sample_prior(n)
        save_image(imgs, os.path.join(out_dir, filename), nrow=int(np.sqrt(n)), normalize=True)

def edit_tumor_factor(model, batch, device, dim, steps, span, out_dir):
    with torch.no_grad():
        x = batch.to(device)
        mu, logvar = model.encode(x)
        base = mu[:1]
        vals = torch.linspace(-span, span, steps, device=device)
        out = []
        for v in vals:
            z = base.clone()
            z[:, dim] = v
            rec = model.decode(z)
            out.append(rec)
        grid = torch.cat(out, dim=0)
        save_image(grid, os.path.join(out_dir, f"edit_dim{dim}.png"), nrow=steps, normalize=True)

def interpolate(model, img_a, img_b, device, steps, out_dir):
    with torch.no_grad():
        x = torch.cat([img_a, img_b], dim=0).to(device)
        mu, logvar = model.encode(x)
        z0 = mu[0:1]
        z1 = mu[1:2]
        alphas = torch.linspace(0,1,steps, device=device)
        seq = []
        for a in alphas:
            z = (1-a)*z0 + a*z1
            rec = model.decode(z)
            seq.append(rec)
        grid = torch.cat(seq, dim=0)
        save_image(grid, os.path.join(out_dir, "interpolation.png"), nrow=steps, normalize=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate samples/traversals from a trained Beta-VAE.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--weights", type=str, default="best", help="Checkpoint tag (best or latest)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of prior samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Seed for sampling latent codes")
    args = parser.parse_args()
    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tf = get_test_transforms()
    _, test_loader = build_dataloaders(transform_train=test_tf, transform_test=test_tf)
    model = load_model(args.weights).to(device)
    out_dir = cfg.paths.figures_dir
    n = args.num_samples or cfg.inference.sample_grid_size
    sample_random(model, n, out_dir, seed=args.seed, filename="samples.png")

    tumor_dim = cfg.inference.tumor_latent_index
    if tumor_dim is not None:
        for batch in test_loader:
            edit_tumor_factor(model, batch["image"], device, tumor_dim, steps=cfg.evaluation.traversal_steps, span=3.0, out_dir=out_dir)
            break
    it = iter(test_loader)
    try:
        b1 = next(it)["image"]
        b2 = next(it)["image"]
        interpolate(model, b1[:1], b2[:1], device, steps=cfg.evaluation.traversal_steps, out_dir=out_dir)
    except StopIteration:
        pass

if __name__ == "__main__":
    main()
