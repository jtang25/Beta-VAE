import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score, f1_score, confusion_matrix
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.logger import log_metrics
from utils.brain_tumor_utils.io import save_table, save_figure
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
    class_to_idx = getattr(data_loader.dataset, "class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else {}
    per_class = {}
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device)
            recon, mu, logvar, z = model.forward(x)
            for i in range(x.size(0)):
                xi = x[i:i+1]
                ri = recon[i:i+1]
                m = mse(ri, xi)
                p = psnr(ri, xi)
                s = ssim(ri, xi)
                rec_ms.append(m)
                rec_ps.append(p)
                rec_ss.append(s)
                lbl = batch["label"][i]
                lbl_name = batch.get("class_name", [None])[i] if isinstance(batch.get("class_name", None), list) else None
                cname = lbl_name or idx_to_class.get(lbl, str(lbl))
                if cname not in per_class:
                    per_class[cname] = {"mse": [], "psnr": [], "ssim": []}
                per_class[cname]["mse"].append(m)
                per_class[cname]["psnr"].append(p)
                per_class[cname]["ssim"].append(s)
            labels.extend(batch["label"])
    d = {
        "mse_mean": float(np.mean(rec_ms)),
        "mse_std": float(np.std(rec_ms)),
        "psnr_mean": float(np.mean(rec_ps)),
        "psnr_std": float(np.std(rec_ps)),
        "ssim_mean": float(np.mean(rec_ss)),
        "ssim_std": float(np.std(rec_ss))
    }
    for cname, vals in per_class.items():
        d[f"per_class/{cname}/mse_mean"] = float(np.mean(vals["mse"])) if vals["mse"] else float("nan")
        d[f"per_class/{cname}/psnr_mean"] = float(np.mean(vals["psnr"])) if vals["psnr"] else float("nan")
        d[f"per_class/{cname}/ssim_mean"] = float(np.mean(vals["ssim"])) if vals["ssim"] else float("nan")
        d[f"per_class/{cname}/count"] = int(len(vals["mse"]))
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

def logistic_probe(latents, labels, train_fraction=0.3, seed=42, binary=True, return_model=False):
    n = latents.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = int(train_fraction * n)
    train_idx = idx[:split]
    test_idx = idx[split:]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(latents[train_idx], y_train)
    probs = clf.predict_proba(latents[test_idx])
    preds = np.argmax(probs, axis=1) if probs.shape[1] > 1 else (probs[:, 0] >= 0.5).astype(int)
    classes = np.unique(labels)
    res = {}
    if binary:
        auc = roc_auc_score(y_test, probs[:,1])
        f1 = f1_score(y_test, (probs[:,1] >= 0.5).astype(int))
        cm = confusion_matrix(y_test, (probs[:,1] >= 0.5).astype(int), labels=classes)
        res = {
            "probe_auc": float(auc),
            "probe_f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classes": classes.tolist(),
        }
    else:
        macro_f1 = f1_score(y_test, preds, average="macro")
        try:
            macro_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except Exception:
            macro_auc = float("nan")
        cm = confusion_matrix(y_test, preds, labels=classes)
        per_class_f1 = f1_score(y_test, preds, average=None, labels=classes)
        res = {
            "probe_macro_f1": float(macro_f1),
            "probe_macro_auc": float(macro_auc),
            "confusion_matrix": cm.tolist(),
            "classes": classes.tolist(),
            "per_class_f1": per_class_f1.tolist(),
        }
    if return_model:
        return res, clf, classes
    return res


def compute_probe_directions(probe_model, classes, class_map=None, device="cpu"):
    if probe_model is None or not hasattr(probe_model, "coef_"):
        return {}
    coef = probe_model.coef_
    if coef.ndim == 1:
        coef = coef[None, :]
    idx_to_class = {v: k for k, v in class_map.items()} if class_map else {}
    dirs = {}
    for cls_idx, row in enumerate(coef):
        cname = idx_to_class.get(classes[cls_idx], str(classes[cls_idx]))
        vec = torch.tensor(row, dtype=torch.float32, device=device)
        if torch.norm(vec) > 0:
            dirs[cname] = vec / torch.norm(vec)
    return dirs


def traversal_probe_validation(probe_model, classes, latents, labels, class_dirs, steps=7, span=3.0, class_map=None):
    if probe_model is None or not class_dirs:
        return {}, None
    import pandas as pd

    vals = np.linspace(-span, span, steps)
    idx_to_class = {v: k for k, v in class_map.items()} if class_map else {}
    summary_rows = []
    for cls_idx, cls_id in enumerate(classes):
        cname = idx_to_class.get(cls_id, str(cls_id))
        dir_vec = class_dirs.get(cname)
        if dir_vec is None:
            continue
        dir_np = dir_vec.cpu().numpy()
        mask = labels == cls_id
        base = latents[mask].mean(axis=0) if mask.any() else latents.mean(axis=0)
        samples = np.array([base + v * dir_np for v in vals])
        try:
            probs = probe_model.predict_proba(samples)
        except Exception:
            continue
        class_pos = np.where(classes == cls_id)[0]
        if class_pos.size == 0:
            continue
        cls_probs = probs[:, class_pos[0]]
        delta = float(cls_probs[-1] - cls_probs[0])
        try:
            corr = float(np.corrcoef(vals, cls_probs)[0, 1])
        except Exception:
            corr = float("nan")
        summary_rows.append({
            "class": cname,
            "start_prob": float(cls_probs[0]),
            "end_prob": float(cls_probs[-1]),
            "delta": delta,
            "corr": corr,
        })
    if not summary_rows:
        return {}, None
    df = pd.DataFrame(summary_rows)
    save_table(df, "traversal_probe_validation")
    metrics = {}
    for row in summary_rows:
        metrics[f"traversal_probe/{row['class']}/delta"] = row["delta"]
        metrics[f"traversal_probe/{row['class']}/corr"] = row["corr"]
    return metrics, df


def save_logreg_weight_heatmap(probe_model, classes, class_map=None, name="latent_logreg_weights"):
    if probe_model is None or not hasattr(probe_model, "coef_"):
        return None
    coef = probe_model.coef_
    if coef.ndim == 1:
        coef = coef[None, :]
    idx_to_class = {v: k for k, v in class_map.items()} if class_map else {}
    class_labels = [idx_to_class.get(int(c), str(int(c))) for c in classes]
    vmax = np.max(np.abs(coef))
    vmax = float(vmax) if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(coef, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    xticks = np.arange(coef.shape[1])
    step = max(1, coef.shape[1] // 16)
    ax.set_xticks(xticks[::step])
    ax.set_xticklabels([str(i) for i in xticks[::step]], rotation=90)
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("class")
    ax.set_title("Logistic regression weights")
    fig.colorbar(im, ax=ax, label="weight")
    path = save_figure(fig, name)
    plt.close(fig)
    return path


def save_recon_traversal_comparison(model, data_loader, device, class_dirs=None, span=3.0, steps=7):
    cfg = get_config()
    class_dirs = class_dirs or {}
    first_batch = None
    for batch in data_loader:
        first_batch = batch
        break
    if first_batch is None:
        return None
    x = first_batch["image"][:1].to(device)
    label = int(first_batch.get("label", [0])[0]) if "label" in first_batch else None
    class_map = getattr(data_loader.dataset, "class_to_idx", {})
    idx_to_class = {v: k for k, v in class_map.items()} if class_map else {}
    cname = idx_to_class.get(label, str(label)) if label is not None else None
    direction = None
    if cname and cname in class_dirs:
        direction = class_dirs[cname]
    elif class_dirs:
        direction = next(iter(class_dirs.values()))
    if direction is None:
        direction = torch.zeros((model.latent_dim,), device=device)
        direction[0] = 1.0
    direction = direction.view(1, -1).to(device)
    span = span if span is not None else getattr(cfg.inference, "edit_span", 3.0)
    with torch.no_grad():
        recon, mu, logvar, _ = model.forward(x)
        base = mu
        z_neg = base - span * direction
        z_pos = base + span * direction
        end_neg = model.decode(z_neg)
        end_pos = model.decode(z_pos)
    imgs = [x, recon, end_neg, end_pos]
    titles = ["original", "reconstruction", f"traverse -{span}", f"traverse +{span}"]
    fig, axes = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3))
    for ax, img, title in zip(axes, imgs, titles):
        arr = img[0].detach().cpu().permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = arr[..., 0]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(title)
    path = save_figure(fig, "recon_vs_traversal")
    plt.close(fig)
    return path

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
    class_map = getattr(test_loader.dataset, "class_to_idx", {})
    recon_metrics = gather_reconstruction_metrics(model, test_loader, device)
    lat_lim = cfg.evaluation.num_umap_samples
    latents, labels, paths = extract_latents(model, test_loader, device, limit=lat_lim)
    binary = cfg.data.class_mode == "binary"
    probe, probe_model, probe_classes = logistic_probe(
        latents,
        labels,
        train_fraction=cfg.evaluation.probe_train_split,
        binary=binary,
        return_model=True,
    )
    class_dirs = compute_probe_directions(probe_model, probe_classes, class_map, device=device)
    traversal_metrics, _ = traversal_probe_validation(
        probe_model,
        probe_classes,
        latents,
        labels,
        class_dirs,
        steps=cfg.evaluation.traversal_steps,
        span=getattr(cfg.inference, "edit_span", 3.0),
        class_map=class_map,
    )
    sep = latent_separability_scores(latents, labels, binary=binary)
    combined = {}
    combined.update(recon_metrics)
    combined.update(probe)
    combined.update(traversal_metrics)
    combined.update(sep)
    save_table(
        __dict_to_frame(combined),
        f"metrics_summary"
    )
    if "confusion_matrix" in probe and "classes" in probe:
        import pandas as pd
        cm = pd.DataFrame(probe["confusion_matrix"], columns=[f"pred_{c}" for c in probe["classes"]], index=[f"true_{c}" for c in probe["classes"]])
        save_table(cm.reset_index(), "confusion_matrix")
    save_logreg_weight_heatmap(probe_model, probe_classes, class_map)
    save_recon_traversal_comparison(
        model,
        test_loader,
        device,
        class_dirs=class_dirs,
        span=getattr(cfg.inference, "edit_span", 3.0),
        steps=cfg.evaluation.traversal_steps,
    )
    log_metrics(combined, step=None, phase="eval")
    return combined

def __dict_to_frame(d):
    import pandas as pd
    return pd.DataFrame([{"metric":k,"value":v} for k,v in d.items()])
