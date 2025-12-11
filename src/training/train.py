import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
from torchvision.utils import save_image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.logger import init_logger, log_config, log_metrics
from utils.brain_tumor_utils.io import ensure_dirs
from utils.brain_tumor_utils.datautils import build_dataloaders
from models.beta_vae import BetaVAE
from data_processing.augmentations import get_train_transforms, get_test_transforms
from training.callbacks import EarlyStopping, CheckpointManager, GradScalerWrapper, build_scheduler, get_optimizer
from training.schedulers import BetaScheduler, CapacityScheduler

def compute_probe_metrics(latents, labels):
    """
    Compute global linear-probe AUC plus per-dimension AUC/correlation/R2 on latent means.
    """
    out = {
        "latent_probe_auc": float("nan"),
        "best_dim_auc": float("nan"),
        "best_dim_corr": float("nan"),
        "best_dim_r2": float("nan"),
    }
    if latents is None or len(latents) < 2:
        return out
    lat = np.asarray(latents)
    y = np.asarray(labels)
    classes = np.unique(y)
    if len(classes) < 2:
        return out
    try:
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        clf.fit(lat, y)
        prob = clf.predict_proba(lat)
        if len(classes) == 2:
            out["latent_probe_auc"] = float(roc_auc_score(y, prob[:, 1]))
        else:
            out["latent_probe_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except Exception:
        pass

    best_auc = []
    best_corr = []
    best_r2 = []
    for k in range(lat.shape[1]):
        z = lat[:, k]
        if np.allclose(z, z[0]):
            continue
        if len(classes) == 2:
            try:
                best_auc.append(roc_auc_score(y, z))
            except Exception:
                pass
        else:
            per_class_auc = []
            for cls in classes:
                y_bin = (y == cls).astype(int)
                if y_bin.sum() == 0 or y_bin.sum() == len(y):
                    continue
                try:
                    per_class_auc.append(roc_auc_score(y_bin, z))
                except Exception:
                    continue
            if per_class_auc:
                best_auc.append(np.max(per_class_auc))

        for cls in classes:
            y_bin = (y == cls).astype(int)
            if np.std(y_bin) == 0:
                continue
            if np.std(z) > 0:
                c = np.corrcoef(z, y_bin)[0, 1]
                if not np.isnan(c):
                    best_corr.append(abs(c))
            try:
                best_r2.append(r2_score(y_bin, z))
            except Exception:
                pass

    if best_auc:
        out["best_dim_auc"] = float(np.max(best_auc))
    if best_corr:
        out["best_dim_corr"] = float(np.max(best_corr))
    if best_r2:
        out["best_dim_r2"] = float(np.max(best_r2))
    return out

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def batch_to_device(batch, device):
    x = batch["image"]
    if isinstance(x, (list, tuple)):
        x = torch.stack(x, dim=0)
    x = x.to(device, non_blocking=True)
    return x

def sample_reconstructions(model,
                           data_loader,
                           device,
                           out_dir,
                           epoch,
                           fixed_paths=None,
                           transform=None,
                           max_images=8,
                           batch=None):
    """
    Save a panel of original vs reconstruction PLUS diagnostics.

    If fixed_paths is provided (list of filesystem paths), those images are loaded
    with `transform` and used. Otherwise the *first* batch from `data_loader` is used.

    Saves:
      - recon_epoch{epoch}.png (top originals, bottom recon)
      - recon_epoch{epoch}_diff.png (|recon - original|)
      - recon_epoch{epoch}_stats.pt (dict with stats)
    Prints per-image MSE and mean pairwise reconstruction L2 distance.
    """
    import os
    from PIL import Image
    from torchvision.utils import save_image
    import torch

    model.eval()

    if batch is not None:
        x = batch["image"]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, 0)
        filenames = batch.get("path", None)
    elif fixed_paths and len(fixed_paths) > 0:
        imgs = []
        for p in fixed_paths[:max_images]:
            img = Image.open(p).convert("L" if model.cfg.data.grayscale else "RGB")
            if transform:
                img = transform(img)
            else:
                from torchvision import transforms
                img = transforms.ToTensor()(img)
            imgs.append(img)
        x = torch.stack(imgs, 0)
        filenames = fixed_paths[:max_images]
    else:
        batch = next(iter(data_loader))
        x = batch["image"]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, 0)
        filenames = batch.get("path", None)

    if x.size(0) > max_images:
        x = x[:max_images]
        if isinstance(filenames, list):
            filenames = filenames[:max_images]

    x = x.to(device)
    with torch.no_grad():
        recon, mu, logvar, z = model.forward(x, deterministic=True)

    recon_clamped = recon.clamp(0, 1)

    per_img_mse = ((recon_clamped - x) ** 2).flatten(1).mean(dim=1)

    rflat = recon_clamped.flatten(1)
    pairwise = torch.cdist(rflat, rflat)
    if rflat.size(0) > 1:
        mean_pairwise = (pairwise.sum() - pairwise.diag().sum()) / (pairwise.numel() - rflat.size(0))
    else:
        mean_pairwise = torch.tensor(0.0, device=device)

    diff = (recon_clamped - x).abs()

    panel = torch.cat([x, recon_clamped], dim=0)
    save_image(panel,
               os.path.join(out_dir, f"recon_epoch{epoch}.png"),
               nrow=x.size(0),
               normalize=True)

    save_image(diff,
               os.path.join(out_dir, f"recon_epoch{epoch}_diff.png"),
               nrow=x.size(0),
               normalize=True)

    stats = {
        "epoch": epoch,
        "filenames": filenames,
        "per_image_mse": per_img_mse.cpu().tolist(),
        "mean_per_image_mse": per_img_mse.mean().item(),
        "mean_pairwise_recon_L2": mean_pairwise.item(),
        "x_min": x.min().item(),
        "x_max": x.max().item(),
        "recon_min": recon_clamped.min().item(),
        "recon_max": recon_clamped.max().item(),
        "recon_mean": recon_clamped.mean().item(),
        "recon_std": recon_clamped.std().item()
    }
    torch.save(stats, os.path.join(out_dir, f"recon_epoch{epoch}_stats.pt"))

    print(f"[RECON DEBUG] epoch {epoch} per-image MSE: {per_img_mse.cpu().numpy()}")
    print(f"[RECON DEBUG] epoch {epoch} mean pairwise recon L2: {mean_pairwise.item():.6f}")

    model.train()


def weight_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def train(resume="none"):
    cfg = get_config()
    ensure_dirs()
    init_logger()
    log_config()
    set_seeds(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug_cfg = getattr(cfg, "debug", None)
    debug_enabled = bool(debug_cfg and debug_cfg.enabled)
    epochs = debug_cfg.epochs if debug_enabled else cfg.training.epochs

    train_tf = get_train_transforms()
    test_tf = get_test_transforms()

    train_limit = debug_cfg.train_samples if (debug_enabled and hasattr(debug_cfg, "train_samples")) else None
    test_limit = debug_cfg.test_samples if (debug_enabled and hasattr(debug_cfg, "test_samples")) else None

    train_loader, test_loader = build_dataloaders(
        transform_train=train_tf,
        transform_test=test_tf,
        train_limit=train_limit,
        test_limit=test_limit
    )

    model = BetaVAE()
    model.apply(weight_init)
    model.to(device)

    optimizer = get_optimizer(model)
    scheduler = build_scheduler(optimizer)
    beta_scheduler = BetaScheduler(cfg, total_epochs=epochs)
    capacity_scheduler = CapacityScheduler(cfg, total_epochs=epochs)
    loss_cfg = getattr(cfg, "loss", None)
    capacity_weight_cfg = getattr(loss_cfg, "capacity_weight", None) if loss_cfg else None
    early = EarlyStopping(patience=20, min_delta=0.0, mode="min")
    ckpt = CheckpointManager(model, optimizer)
    amp = GradScalerWrapper(enabled=cfg.training.mixed_precision)
    amp.set_params(model.parameters())

    start_epoch = 1
    total_steps = 0
    if resume in ["best", "latest"]:
        path = os.path.join(cfg.paths.models_dir, f"{cfg.paths.run_id}_{resume}.pt")
        if os.path.exists(path):
            payload = torch.load(path, map_location=device)
            model.load_state_dict(payload.get("model_state", payload))
            if "optim_state" in payload:
                optimizer.load_state_dict(payload["optim_state"])
            start_epoch = payload.get("epoch", 0) + 1
            total_steps = payload.get("total_steps", 0)
            if scheduler:
                for _ in range(start_epoch - 1):
                    scheduler.step()
            print(f"[RESUME] Loaded checkpoint '{resume}' from {path}, restarting at epoch {start_epoch}")
        else:
            print(f"[RESUME] Requested '{resume}' but checkpoint not found at {path}; starting fresh.")

    figures_dir = cfg.paths.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    fixed_paths = None
    if debug_cfg and hasattr(debug_cfg, "fixed_recon_paths"):
        fixed_paths = list(debug_cfg.fixed_recon_paths)
        missing = [p for p in fixed_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Some fixed_recon_paths do not exist:\n" + "\n".join(missing)
            )

    for epoch in range(start_epoch, epochs + 1):
        val_preview_batch = None
        beta = beta_scheduler.value(epoch - 1)
        capacity = capacity_scheduler.value(epoch) if capacity_scheduler.enabled else None
        free_bits = 0.0
        if capacity is None and hasattr(cfg, "loss") and hasattr(cfg.loss, "free_bits"):
            free_bits = cfg.loss.free_bits
        model.set_beta(beta)
        model.train()
        running = {
            "total": 0.0,
            "recon": 0.0,
            "recon_base": 0.0,
            "recon_lpips": 0.0,
            "recon_ffl": 0.0,
            "kl": 0.0
        }

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            x = batch["image"]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, 0)
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg.training.mixed_precision):
                losses = model.loss(
                    x,
                    beta=beta,
                    capacity=capacity,
                    free_bits=free_bits,
                    capacity_weight=capacity_weight_cfg if capacity is not None else None,
                )
                loss = losses["total"]

            amp.backward(loss, optimizer,
                         clip=cfg.training.grad_clip if cfg.training.grad_clip > 0 else 0.0)

            if scheduler and cfg.optimization.scheduler not in ["cosine"]:
                scheduler.step()

            running["total"] += losses["total"].item()
            running["recon"] += losses["recon"].item()
            running["recon_base"] += losses.get("recon_base", losses["recon"]).item()
            running["recon_lpips"] += losses.get("recon_lpips", torch.tensor(0.0)).item()
            running["recon_ffl"] += losses.get("recon_ffl", torch.tensor(0.0)).item()
            running["kl"] += losses["kl_mean"].item()
            train_kl_mean_epoch = running["kl"] / (i + 1)
            train_kl_effective_last = losses["kl_effective"].item()
            kl_per_dim_mean = losses["kl_per_dim"].mean().item()
            loss_mode = losses.get("mode", None)

            total_steps += 1

            if total_steps % cfg.logging.log_every_n_steps == 0:
                denom = i + 1
                mu_stats = losses["mu"].mean().item()
                z_std = losses["z"].std().item()
                metrics = {
                    "epoch": epoch,
                    "beta": float(beta),
                    "capacity": float(capacity) if capacity is not None else 0.0,
                    "train_total_loss": running["total"] / denom,
                    "train_recon_loss": running["recon"] / denom,
                    "train_recon_base": running["recon_base"] / denom,
                    "train_recon_lpips": running["recon_lpips"] / denom,
                    "train_recon_ffl": running["recon_ffl"] / denom,
                    "train_kl": running["kl"] / denom,
                    "train_kl_mean": train_kl_mean_epoch,
                    "train_kl_effective_last": train_kl_effective_last,
                    "train_kl_per_dim_mean": kl_per_dim_mean,
                    "loss_mode": loss_mode,
                    "mu_mean_batch": mu_stats,
                    "z_std_batch": z_std,
                    "lr": optimizer.param_groups[0]["lr"]
                }
                log_metrics(metrics, step=total_steps, phase="train")

            if debug_enabled and i + 1 >= debug_cfg.max_train_batches:
                break

        if scheduler and cfg.optimization.scheduler == "cosine":
            scheduler.step()

        model.eval()
        val_tot = val_rec = val_kl = 0.0
        val_recon_base = 0.0
        val_recon_lpips = 0.0
        val_recon_ffl = 0.0
        val_latents = []
        val_labels = []
        val_batches = 0
        with torch.no_grad():
            for j, batch in enumerate(test_loader):
                x = batch["image"]
                if isinstance(x, (list, tuple)):
                    x = torch.stack(x, 0)
                x = x.to(device, non_blocking=True)
                if val_preview_batch is None:
                    val_preview_batch = batch
                l = model.loss(
                    x,
                    beta=beta,
                    capacity=capacity,
                    free_bits=free_bits,
                    capacity_weight=capacity_weight_cfg if capacity is not None else None,
                )
                val_tot += l["total"].item()
                val_rec += l["recon"].item()
                val_recon_base += l.get("recon_base", l["recon"]).item()
                val_recon_lpips += l.get("recon_lpips", torch.tensor(0.0)).item()
                val_recon_ffl += l.get("recon_ffl", torch.tensor(0.0)).item()
                val_kl += l["kl_mean"].item()
                val_kl_per_dim_mean = l["kl_per_dim"].mean().item()
                val_loss_mode = l.get("mode", None)
                val_latents.append(l["mu"].cpu())
                val_labels.extend(batch["label"])
                val_batches += 1
                if debug_enabled and j + 1 >= debug_cfg.max_val_batches:
                    break

        val_total = val_tot / max(1, val_batches)
        val_recon = val_rec / max(1, val_batches)
        val_recon_base_avg = val_recon_base / max(1, val_batches)
        val_recon_lpips_avg = val_recon_lpips / max(1, val_batches)
        val_recon_ffl_avg = val_recon_ffl / max(1, val_batches)
        val_kl_avg = val_kl / max(1, val_batches)
        final_train_kl_mean = running["kl"] / (i + 1)
        final_train_kl_effective = train_kl_effective_last
        probe_metrics = {"latent_probe_auc": float("nan"),
                         "best_dim_auc": float("nan"),
                         "best_dim_corr": float("nan"),
                         "best_dim_r2": float("nan")}
        if val_latents and len(val_labels) >= 2:
            lat_arr = torch.cat(val_latents, dim=0).numpy()
            probe_metrics = compute_probe_metrics(lat_arr, val_labels)
        capacity_logged = float(capacity) if capacity is not None else 0.0
        metrics = {
            "epoch": epoch,
            "beta": float(beta),
            "capacity": capacity_logged,
            "val_total_loss": val_total,
            "val_recon_loss": val_recon,
            "val_recon_base": val_recon_base_avg,
            "val_recon_lpips": val_recon_lpips_avg,
            "val_recon_ffl": val_recon_ffl_avg,
            "val_kl": val_kl_avg,
            "val_kl_per_dim_mean": val_kl_per_dim_mean,
            "loss_mode": val_loss_mode,
            "train_kl_mean": final_train_kl_mean,
            "train_kl_effective_last": final_train_kl_effective,
            "latent_probe_auc": probe_metrics["latent_probe_auc"],
            "best_dim_auc": probe_metrics["best_dim_auc"],
            "best_dim_corr": probe_metrics["best_dim_corr"],
            "best_dim_r2": probe_metrics["best_dim_r2"],
        }
        log_metrics(metrics, step=total_steps, phase="val")

        ckpt.save_latest(epoch, total_steps, {"val_total": val_total})
        ckpt.save_best(epoch, total_steps, {"val_total": val_total}, monitor_value=val_total)

        sample_reconstructions(
            model,
            test_loader,
            device,
            figures_dir,
            epoch,
            fixed_paths=fixed_paths,
            transform=test_tf,
            batch=val_preview_batch,
        )

        early.update(val_total)
        if early.should_stop:
            break

    return

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Beta-VAE model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (optional)")
    parser.add_argument("--resume", type=str, choices=["best", "latest", "none"], default="none",
                        help="Resume from a checkpoint in models_dir (best|latest|none)")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    train(resume=args.resume)
