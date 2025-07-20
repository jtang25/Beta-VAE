import os
import csv
import numpy as np
import torch
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms, get_train_transforms
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

def encode_loader(model, loader, device):
    model.eval()
    lat = []
    logvars = []
    labels = []
    paths = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            mu, logvar = model.encode(x)
            lat.append(mu.cpu())
            logvars.append(logvar.cpu())
            labels.extend(batch["label"])
            paths.extend(batch["path"])
    Z = torch.cat(lat, dim=0).numpy()
    LV = torch.cat(logvars, dim=0).numpy()
    return Z, LV, labels, paths

def write_embeddings(Z, LV, labels, paths, prefix):
    cfg = get_config()
    out_dir = cfg.paths.tables_dir
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_mu.npy"), Z)
    np.save(os.path.join(out_dir, f"{prefix}_logvar.npy"), LV)
    csv_path = os.path.join(out_dir, f"{prefix}_embeddings.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        header = ["path","label"] + [f"z{i}" for i in range(Z.shape[1])]
        w.writerow(header)
        for i in range(Z.shape[0]):
            w.writerow([paths[i], labels[i]] + list(Z[i]))
    return csv_path

def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = get_train_transforms()
    test_tf = get_test_transforms()
    train_loader, test_loader = build_dataloaders(transform_train=train_tf, transform_test=test_tf)
    model = load_model("best").to(device)
    Zt, LVt, Lt, Pt = encode_loader(model, train_loader, device)
    write_embeddings(Zt, LVt, Lt, Pt, "train_latents")
    Zv, LVv, Lv, Pv = encode_loader(model, test_loader, device)
    write_embeddings(Zv, LVv, Lv, Pv, "test_latents")

if __name__ == "__main__":
    main()
