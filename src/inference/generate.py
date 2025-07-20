import os
import torch
import numpy as np
from torchvision.utils import save_image
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

def sample_random(model, n, out_dir):
    with torch.no_grad():
        imgs = model.sample_prior(n)
        save_image(imgs, os.path.join(out_dir, "samples.png"), nrow=int(np.sqrt(n)), normalize=True)

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
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tf = get_test_transforms()
    _, test_loader = build_dataloaders(transform_train=test_tf, transform_test=test_tf)
    model = load_model("best").to(device)
    out_dir = cfg.paths.figures_dir
    sample_random(model, cfg.inference.sample_grid_size, out_dir)
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
