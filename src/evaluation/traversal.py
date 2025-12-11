import torch
import os
import warnings
import numpy as np
from pathlib import Path
import pandas as pd
from torchvision.utils import save_image
from utils.brain_tumor_utils.config_parser import get_config
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

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

def run_traversals(model, test_loader, device, indices=None, steps=None, span=3.0, images_override=None):
    cfg = get_config()
    out_dir = cfg.paths.figures_dir
    class_dirs = {}
    usage_path = Path(cfg.paths.tables_dir) / "latent_usage.csv"
    if usage_path.exists():
        try:
            df = pd.read_csv(usage_path)
            for col in df.columns:
                if col.startswith("logreg_weight_"):
                    name = col.replace("logreg_weight_", "")
                    w = torch.tensor(df[col].to_numpy(), dtype=torch.float32, device=device)
                    if w.norm() > 0:
                        class_dirs[name] = w / w.norm()
        except Exception:
            class_dirs = {}

    if not class_dirs:
        latents = []
        labels = []
        class_map = getattr(test_loader.dataset, "class_to_idx", None)
        idx_to_class = {v: k for k, v in class_map.items()} if class_map else {}
        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device)
                mu, _ = model.encode(x)
                latents.append(mu.cpu())
                labels.extend(batch["label"])
            if latents and len(labels) >= 2:
                L = torch.cat(latents, dim=0).numpy()
                y = np.array(labels)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        clf = LogisticRegression(max_iter=2000, multi_class="auto")
                        clf.fit(L, y)
                        coef = clf.coef_
                        if coef.ndim == 1:
                            coef = coef[None, :]
                        for cls_idx, row in enumerate(coef):
                            name = idx_to_class.get(clf.classes_[cls_idx], f"class{clf.classes_[cls_idx]}")
                            w = torch.tensor(row, dtype=torch.float32, device=device)
                            if w.norm() > 0:
                                class_dirs[name] = w / w.norm()
                except Exception:
                    class_dirs = {}

    imgs = None
    if images_override is not None:
        imgs = images_override
    else:
        for batch in test_loader:
            imgs = batch["image"]
            break

    if imgs is None:
        return

    latent_traversal(model, imgs, device, out_dir, indices=indices, steps=steps, span=span)
    tumor_dirs = {k: v for k, v in class_dirs.items() if "notumor" not in k.lower()}
    if tumor_dirs:
        model.eval()
        with torch.no_grad():
            x = imgs.to(device)
            mu, _ = model.encode(x)
            base = mu[:1]
            vals = torch.linspace(-span, span, steps if steps is not None else cfg.evaluation.traversal_steps, device=device)
            for cls_name, dir_vec in tumor_dirs.items():
                stack = []
                for v in vals:
                    z = base + v * dir_vec
                    recon = model.decode(z)
                    stack.append(recon)
                grid = torch.cat(stack, dim=0)
                save_image(
                    grid,
                    os.path.join(out_dir, f"traversal_tumor_{cls_name}.png"),
                    nrow=vals.numel(),
                    normalize=True
                )
