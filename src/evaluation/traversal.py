import torch
import os
from torchvision.utils import save_image
from utils.brain_tumor_utils.config_parser import get_config

def latent_traversal(model, images, device, out_dir):
    cfg = get_config()
    steps = cfg.evaluation.traversal_steps
    indices = cfg.inference.traversal_latent_indices
    if not indices or len(indices)==0:
        indices = list(range(min(model.latent_dim, 4)))
    model.eval()
    with torch.no_grad():
        x = images.to(device)
        mu, logvar = model.encode(x)
        base = mu[:1]
        span = 3.0
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

def run_traversals(model, test_loader, device):
    cfg = get_config()
    out_dir = cfg.paths.figures_dir
    for batch in test_loader:
        imgs = batch["image"]
        latent_traversal(model, imgs, device, out_dir)
        break
