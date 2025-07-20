import torch
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score, f1_score
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.logger import log_metrics
from utils.brain_tumor_utils.io import save_table
from torchvision import transforms
import torch.nn.functional as F

def mse(a,b):
    return torch.mean((a-b)**2).item()

def psnr(a,b):
    m = torch.mean((a-b)**2).item()
    if m == 0:
        return 99.0
    return 20 * math.log10(1.0) - 10 * math.log10(m)


def ssim(x, y, window_size=11, sigma=1.5):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(0)
    B, C, H, W = x.shape
    device = x.device
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2)/(2*sigma**2))
    g = g / g.sum()
    k2d = g[:, None] @ g[None, :]
    k2d = k2d / k2d.sum()
    window = k2d.view(1,1,window_size,window_size).repeat(C,1,1,1)
    pad = window_size // 2
    L = x.max() - x.min()
    if L <= 0:
        L = 1.0
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)
    denom = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_map = num / (denom + 1e-12)
    return ssim_map.mean().item()


def gather_reconstruction_metrics(model, data_loader, device):
    cfg = get_config()
    model.eval()
    rec_ms = []
    rec_ps = []
    rec_ss = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device)
            recon, mu, logvar, z = model.forward(x)
            for i in range(x.size(0)):
                xi = x[i:i+1]
                ri = recon[i:i+1]
                rec_ms.append(mse(ri, xi))
                rec_ps.append(psnr(ri, xi))
                rec_ss.append(ssim(ri, xi))
            labels.extend(batch["label"])
    d = {
        "mse_mean": float(np.mean(rec_ms)),
        "mse_std": float(np.std(rec_ms)),
        "psnr_mean": float(np.mean(rec_ps)),
        "psnr_std": float(np.std(rec_ps)),
        "ssim_mean": float(np.mean(rec_ss)),
        "ssim_std": float(np.std(rec_ss))
    }
    return d

def extract_latents(model, data_loader, device, limit=None):
    model.eval()
    latents = []
    labels = []
    paths = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device)
            mu, logvar = model.encode(x)
            latents.append(mu.cpu())
            labels.extend(batch["label"])
            paths.extend(batch["path"])
            if limit and len(labels) >= limit:
                break
    L = torch.cat(latents, dim=0)
    if limit:
        L = L[:limit]
        labels = labels[:limit]
        paths = paths[:limit]
    return L.numpy(), np.array(labels), paths

def logistic_probe(latents, labels, train_fraction=0.3, seed=42, binary=True):
    n = latents.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = int(train_fraction * n)
    train_idx = idx[:split]
    test_idx = idx[split:]
    if binary:
        y_train = labels[train_idx]
        y_test = labels[test_idx]
    else:
        y_train = labels[train_idx]
        y_test = labels[test_idx]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(latents[train_idx], y_train)
    probs = clf.predict_proba(latents[test_idx])
    if binary:
        auc = roc_auc_score(y_test, probs[:,1])
        f1 = f1_score(y_test, (probs[:,1] >= 0.5).astype(int))
        return {"probe_auc": float(auc), "probe_f1": float(f1)}
    else:
        macro_f1 = f1_score(y_test, np.argmax(probs, axis=1), average="macro")
        return {"probe_macro_f1": float(macro_f1)}

def latent_separability_scores(latents, labels, binary=True):
    out = {}
    if binary:
        try:
            sil = silhouette_score(latents, labels)
            out["silhouette"] = float(sil)
        except:
            out["silhouette"] = float("nan")
    else:
        try:
            sil = silhouette_score(latents, labels)
            out["silhouette"] = float(sil)
        except:
            out["silhouette"] = float("nan")
    return out

def evaluate_full(model, train_loader, test_loader, device):
    cfg = get_config()
    recon_metrics = gather_reconstruction_metrics(model, test_loader, device)
    lat_lim = cfg.evaluation.num_umap_samples
    latents, labels, paths = extract_latents(model, test_loader, device, limit=lat_lim)
    binary = cfg.data.class_mode == "binary"
    probe = logistic_probe(latents, labels, train_fraction=cfg.evaluation.probe_train_split, binary=binary)
    sep = latent_separability_scores(latents, labels, binary=binary)
    combined = {}
    combined.update(recon_metrics)
    combined.update(probe)
    combined.update(sep)
    save_table(
        __dict_to_frame(combined),
        f"metrics_summary"
    )
    log_metrics(combined, step=None, phase="eval")
    return combined

def __dict_to_frame(d):
    import pandas as pd
    return pd.DataFrame([{"metric":k,"value":v} for k,v in d.items()])
