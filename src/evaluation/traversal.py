import torch
import os
from pathlib import Path
import pandas as pd
from torchvision.utils import save_image
from utils.brain_tumor_utils.config_parser import get_config

def latent_traversal(model, images, device, out_dir, indices=None, steps=None, span=3.0):
    cfg = get_config()
    steps = steps if steps is not None else cfg.evaluation.traversal_steps
    if indices is None:
        indices = cfg.inference.traversal_latent_indices
        if not indices or len(indices) == 0:
            indices = list(range(min(model.latent_dim, 4)))
    model.eval()
    with torch.no_grad():
        x = images.to(device)
        mu, logvar = model.encode(x)
        base = mu[:1]
        vals = torch.linspace(-span, span, steps, device=device)
        for dim in indices:
            stack = []
            for v in vals:
                z = base.clone()
                z[:, dim] = v
                recon = model.decode(z)
                stack.append(recon)
            grid = torch.cat(stack, dim=0)
            save_image(grid, os.path.join(out_dir, f"traversal_dim{dim}.png"), nrow=steps, normalize=True)

def run_traversals(model, test_loader, device, indices=None, steps=None, span=3.0):
    cfg = get_config()
    out_dir = cfg.paths.figures_dir
    # Try to load a classifier direction from latent_usage.csv
    dir_vec = None
    usage_path = Path(cfg.paths.tables_dir) / "latent_usage.csv"
    if usage_path.exists():
        try:
            df = pd.read_csv(usage_path)
            if "logreg_weight" in df.columns:
                w = torch.tensor(df["logreg_weight"].to_numpy(), dtype=torch.float32, device=device)
                if w.norm() > 0:
                    dir_vec = w / w.norm()
        except Exception:
            dir_vec = None

    for batch in test_loader:
        imgs = batch["image"]
        latent_traversal(model, imgs, device, out_dir, indices=indices, steps=steps, span=span)
        # Also traverse along learned classifier direction if available
        if dir_vec is not None:
            model.eval()
            with torch.no_grad():
                x = imgs.to(device)
                mu, _ = model.encode(x)
                base = mu[:1]
                vals = torch.linspace(-span, span, steps if steps is not None else cfg.evaluation.traversal_steps, device=device)
                stack = []
                for v in vals:
                    z = base + v * dir_vec
                    recon = model.decode(z)
                    stack.append(recon)
                grid = torch.cat(stack, dim=0)
                save_image(grid, os.path.join(out_dir, "traversal_tumor_direction.png"), nrow=vals.numel(), normalize=True)
        break
