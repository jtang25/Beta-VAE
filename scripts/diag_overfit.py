from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_train_transforms, get_test_transforms
from models.beta_vae import BetaVAE
import torch, os, json
from utils.brain_tumor_utils.io import load_sharded_checkpoint

cfg = get_config()
train_tf = get_train_transforms()
test_tf = get_test_transforms()
train_loader, test_loader = build_dataloaders(train_tf, test_tf, train_limit=cfg.debug.train_samples, test_limit=cfg.debug.test_samples)

ckpt_path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_latest.pt"
ckpt = load_sharded_checkpoint(ckpt_path, map_location="cpu")
model = BetaVAE()
model.load_state_dict(ckpt["model_state"])
model.eval()

def full_mse(loader):
    mses = []
    with torch.no_grad():
        for b in loader:
            x = b["image"]
            r, mu, logvar, z = model.forward(x)
            mses.append(torch.mean((r-x)**2).item())
    return sum(mses)/len(mses)

train_mse = full_mse(train_loader)
val_mse = full_mse(test_loader)

b = next(iter(train_loader))
x = b["image"]
with torch.no_grad():
    r, mu, logvar, z = model.forward(x)
stats = {
    "train_mse_mean": train_mse,
    "val_mse_mean": val_mse,
    "mu_mean": float(mu.mean()),
    "mu_std": float(mu.std()),
    "z_mean": float(z.mean()),
    "z_std": float(z.std()),
    "logvar_mean": float(logvar.mean()),
    "logvar_std": float(logvar.std()),
    "x_min": float(x.min()),
    "x_max": float(x.max()),
    "r_min": float(r.min()),
    "r_max": float(r.max())
}
print(json.dumps(stats, indent=2))
