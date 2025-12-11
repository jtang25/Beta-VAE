import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms
from models.beta_vae import BetaVAE
from utils.brain_tumor_utils.io import load_sharded_checkpoint

def load_model(weights="best"):
    cfg = get_config()
    path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_{weights}.pt"
    if not os.path.exists(path):
        path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_latest.pt"
    payload = load_sharded_checkpoint(path, map_location="cpu", num_shards=2)
    model = BetaVAE()
    model.load_state_dict(payload["model_state"])
    return model

def extract_latents(model, data_loader, device):
    model.eval()
    lat = []
    labs = []
    kl_chunks = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device)
            mu, logvar = model.encode(x)
            lat.append(mu.cpu())
            labs.extend(batch["label"])
            kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
            kl_chunks.append(kl.cpu())
    L = torch.cat(lat, dim=0).numpy()
    K = torch.cat(kl_chunks, dim=0).numpy()
    y = np.array(labs)
    return L, K, y

def per_dimension_auc(L, y):
    out = []
    classes = np.unique(y)
    multiclass = len(classes) > 2
    for i in range(L.shape[1]):
        scores = L[:, i]
        try:
            if multiclass:
                aucs = []
                for cls in classes:
                    y_bin = (y == cls).astype(int)
                    if y_bin.sum() == 0 or y_bin.sum() == len(y):
                        continue
                    aucs.append(roc_auc_score(y_bin, scores if scores.var() > 0 else np.zeros_like(scores)))
                auc = np.max(aucs) if aucs else float("nan")
            else:
                auc = roc_auc_score(y, scores if scores.var() > 0 else np.zeros_like(scores))
        except Exception:
            auc = float("nan")
        out.append((i, float(auc)))
    return out

def per_dimension_abs_auc(L, y):
    out = []
    classes = np.unique(y)
    multiclass = len(classes) > 2
    for i in range(L.shape[1]):
        scores = np.abs(L[:, i])
        try:
            if multiclass:
                aucs = []
                for cls in classes:
                    y_bin = (y == cls).astype(int)
                    if y_bin.sum() == 0 or y_bin.sum() == len(y):
                        continue
                    aucs.append(roc_auc_score(y_bin, scores if scores.var() > 0 else np.zeros_like(scores)))
                auc = np.max(aucs) if aucs else float("nan")
            else:
                auc = roc_auc_score(y, scores if scores.var() > 0 else np.zeros_like(scores))
        except Exception:
            auc = float("nan")
        out.append((i, float(auc)))
    return out

def logistic_weights(L, y):
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(L, y)
    coef = clf.coef_
    if coef.ndim == 1:
        coef = coef[None, :]
    max_abs = np.max(np.abs(coef), axis=0)
    order = np.argsort(max_abs)[::-1]
    return order, coef, clf.classes_

def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tf = get_test_transforms()
    _, test_loader = build_dataloaders(transform_train=test_tf, transform_test=test_tf)
    model = load_model("best").to(device)
    L, K, y = extract_latents(model, test_loader, device)
    aucs = per_dimension_auc(L, y)
    aucs_abs = per_dimension_abs_auc(L, y)
    kl_mean = K.mean(axis=0)
    mu_var = L.var(axis=0)
    order, coef, classes = logistic_weights(L, y)
    import pandas as pd
    import json
    from utils.brain_tumor_utils.io import save_table, save_json
    df = pd.DataFrame(aucs, columns=["latent_dim","single_dim_auc"])
    save_table(df, "per_dimension_auc")

    usage_payload = {
        "latent_dim": np.arange(L.shape[1]),
        "kl_mean": kl_mean,
        "mu_var": mu_var,
        "single_dim_auc": [a[1] for a in aucs],
        "single_dim_auc_abs": [a[1] for a in aucs_abs],
        "logreg_weight_maxabs": np.max(np.abs(coef), axis=0)
    }
    idx_to_class = {v: k for k, v in getattr(test_loader.dataset, "class_to_idx", {}).items()}
    for cls_idx, cls_name in enumerate(classes):
        cname = idx_to_class.get(cls_name, f"class{cls_name}")
        usage_payload[f"logreg_weight_{cname}"] = coef[cls_idx]
    usage_df = pd.DataFrame(usage_payload)
    save_table(usage_df.sort_values("kl_mean", ascending=False), "latent_usage")

    best = max(aucs, key=lambda t: (t[1] if not np.isnan(t[1]) else -1))
    best_abs = max(aucs_abs, key=lambda t: (t[1] if not np.isnan(t[1]) else -1))

    top_logreg = []
    for d in order[:10]:
        weights_per_class = {
            str(idx_to_class.get(cls, cls)): float(coef_row[d])
            for cls, coef_row in zip(classes, coef)
        }
        top_logreg.append({
            "latent_dim": int(d),
            "abs_weight_max": float(np.max(np.abs(coef[:, d]))),
            "weights": weights_per_class,
            "kl_mean": float(kl_mean[d]),
            "mu_var": float(mu_var[d]),
            "single_dim_auc": float([a[1] for a in aucs][d])
        })

    traversal_order_auc = [int(i) for i, _ in sorted(aucs, key=lambda t: (t[1] if not np.isnan(t[1]) else -1), reverse=True)]
    traversal_order_kl = [int(i) for i in np.argsort(-kl_mean)]

    corr = np.corrcoef(L, rowvar=False)
    triu_idx = np.triu_indices_from(corr, k=1)
    corr_pairs = []
    for i, j, c in zip(triu_idx[0], triu_idx[1], corr[triu_idx]):
        corr_pairs.append((i, j, float(c)))
    corr_pairs_sorted = sorted(corr_pairs, key=lambda t: abs(t[2]), reverse=True)[:20]
    corr_df = pd.DataFrame(corr_pairs, columns=["i","j","corr"])
    save_table(corr_df, "latent_corr_pairs")

    res = {
        "best_auc_dim": int(best[0]),
        "best_auc": float(best[1]),
        "best_abs_auc_dim": int(best_abs[0]),
        "best_abs_auc": float(best_abs[1]),
        "top_logreg_dims": top_logreg
    }
    res["traversal_order_auc"] = traversal_order_auc
    res["traversal_order_kl"] = traversal_order_kl
    res["class_balance"] = {
        "counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    }
    res["top_corr_pairs"] = [{"i": int(i), "j": int(j), "corr": c} for i, j, c in corr_pairs_sorted]

    save_json(res, "latent_ranking_summary")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
