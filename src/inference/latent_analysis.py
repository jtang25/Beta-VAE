import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms
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

def extract_latents(model, data_loader, device):
    model.eval()
    lat = []
    labs = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device)
            mu, logvar = model.encode(x)
            lat.append(mu.cpu())
            labs.extend(batch["label"])
    L = torch.cat(lat, dim=0).numpy()
    y = np.array(labs)
    return L, y

def per_dimension_auc(L, y):
    out = []
    for i in range(L.shape[1]):
        scores = L[:, i]
        try:
            auc = roc_auc_score(y, scores if scores.var()>0 else np.zeros_like(scores))
        except:
            auc = float("nan")
        out.append((i, float(auc)))
    return out

def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tf = get_test_transforms()
    _, test_loader = build_dataloaders(transform_train=test_tf, transform_test=test_tf)
    model = load_model("best").to(device)
    L, y = extract_latents(model, test_loader, device)
    aucs = per_dimension_auc(L, y)
    import pandas as pd
    import json
    from utils.brain_tumor_utils.io import save_table, save_json
    df = pd.DataFrame(aucs, columns=["latent_dim","single_dim_auc"])
    save_table(df, "per_dimension_auc")
    best = max(aucs, key=lambda t: (t[1] if not np.isnan(t[1]) else -1))
    res = {"best_latent_dim": best[0], "best_auc": best[1]}
    save_json(res, "tumor_latent_candidate")
    print(res)

if __name__ == "__main__":
    main()
