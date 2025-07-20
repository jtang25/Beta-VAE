import os
import random
import numpy as np
import torch
from torchvision.utils import save_image
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.logger import init_logger, log_config, log_metrics
from utils.brain_tumor_utils.io import ensure_dirs
from utils.brain_tumor_utils.datautils import build_dataloaders
from models.beta_vae import BetaVAE
from data_processing.augmentations import get_train_transforms, get_test_transforms
from training.callbacks import EarlyStopping, CheckpointManager, GradScalerWrapper, build_scheduler, get_optimizer
from training.schedulers import BetaScheduler, CapacityScheduler

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
                           max_images=8):
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

    # --------- Build input batch ---------
    if fixed_paths and len(fixed_paths) > 0:
        imgs = []
        for p in fixed_paths[:max_images]:
            img = Image.open(p).convert("L" if model.cfg.data.grayscale else "RGB")
            if transform:
                img = transform(img)
            else:
                # Fallback: ToTensor
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

    # Clamp just in case (should already be in [0,1] after sigmoid)
    recon_clamped = recon.clamp(0, 1)

    # Per-image MSE
    per_img_mse = ((recon_clamped - x) ** 2).flatten(1).mean(dim=1)

    # Diversity (pairwise L2 between reconstructions)
    rflat = recon_clamped.flatten(1)
    pairwise = torch.cdist(rflat, rflat)
    # exclude diagonal
    if rflat.size(0) > 1:
        mean_pairwise = (pairwise.sum() - pairwise.diag().sum()) / (pairwise.numel() - rflat.size(0))
    else:
        mean_pairwise = torch.tensor(0.0, device=device)

    # Diff image
    diff = (recon_clamped - x).abs()

    # Assemble panel: originals first row then recon second row
    panel = torch.cat([x, recon_clamped], dim=0)
    save_image(panel,
               os.path.join(out_dir, f"recon_epoch{epoch}.png"),
               nrow=x.size(0),
               normalize=True)

    save_image(diff,
               os.path.join(out_dir, f"recon_epoch{epoch}_diff.png"),
               nrow=x.size(0),
               normalize=True)

    # Stats
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

def train():
    cfg = get_config()
    ensure_dirs()
    init_logger()
    log_config()
    set_seeds(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debug / run length config
    debug_cfg = getattr(cfg, "debug", None)
    debug_enabled = bool(debug_cfg and debug_cfg.enabled)
    epochs = debug_cfg.epochs if debug_enabled else cfg.training.epochs

    # Transforms
    train_tf = get_train_transforms()
    test_tf = get_test_transforms()

    # Optional sample limits
    train_limit = debug_cfg.train_samples if (debug_enabled and hasattr(debug_cfg, "train_samples")) else None
    test_limit = debug_cfg.test_samples if (debug_enabled and hasattr(debug_cfg, "test_samples")) else None

    # Data loaders
    train_loader, test_loader = build_dataloaders(
        transform_train=train_tf,
        transform_test=test_tf,
        train_limit=train_limit,
        test_limit=test_limit
    )

    # Model
    model = BetaVAE()
    model.apply(weight_init)
    model.to(device)

    # Optim / sched
    optimizer = get_optimizer(model)
    scheduler = build_scheduler(optimizer)
    beta_scheduler = BetaScheduler(cfg, total_epochs=epochs)
    capacity_scheduler = CapacityScheduler(cfg, total_epochs=epochs)
    early = EarlyStopping(patience=20, min_delta=0.0, mode="min")
    ckpt = CheckpointManager(model, optimizer)
    amp = GradScalerWrapper(enabled=cfg.training.mixed_precision)
    amp.set_params(model.parameters())

    figures_dir = cfg.paths.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    # Pull (optional) fixed recon paths from debug section
    fixed_paths = None
    if debug_cfg and hasattr(debug_cfg, "fixed_recon_paths"):
        fixed_paths = list(debug_cfg.fixed_recon_paths)  # ensure it's a plain list
        # (Optional) early validation: warn if any missing
        missing = [p for p in fixed_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Some fixed_recon_paths do not exist:\n" + "\n".join(missing)
            )

    total_steps = 0

    for epoch in range(1, epochs + 1):
        # Beta schedule update (even if beta=0 for deterministic overfit)
        beta = beta_scheduler.value(epoch - 1)
        capacity = capacity_scheduler.value(epoch - 1)
        free_bits = 0.0
        if hasattr(cfg, "loss") and hasattr(cfg.loss, "free_bits"):
            free_bits = cfg.loss.free_bits
        model.set_beta(beta)
        model.train()
        running = {"total": 0.0, "recon": 0.0, "kl": 0.0}

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            x = batch["image"]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, 0)
            x = x.to(device, non_blocking=True)

            # Forward / loss
            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                use_capacity = capacity is not None
                if use_capacity:
                    losses = model.loss(x, capacity=capacity, capacity_weight=cfg.loss.capacity_weight)
                else:
                    losses = model.loss(x, beta=beta, free_bits=free_bits)
                loss = losses["total"]

            # Backward + step
            amp.backward(loss, optimizer,
                         clip=cfg.training.grad_clip if cfg.training.grad_clip > 0 else 0.0)

            # Per-step scheduler (for non-cosine types)
            if scheduler and cfg.optimization.scheduler not in ["cosine"]:
                scheduler.step()

            # Accumulate
            running["total"] += losses["total"].item()
            running["recon"] += losses["recon"].item()
            running["kl"] += losses["kl_mean"].item()
            train_kl_mean_epoch = running["kl"] / (i + 1)
            train_kl_effective_last = losses["kl_effective"].item()

            total_steps += 1

            # Logging every N steps
            if total_steps % cfg.logging.log_every_n_steps == 0:
                denom = i + 1
                mu_stats = losses["mu"].mean().item()
                z_std = losses["z"].std().item()
                log_metrics({
                    "epoch": epoch,
                    "beta": float(beta),
                    "capacity": float(capacity) if capacity is not None else None,
                    "train_total_loss": running["total"] / denom,
                    "train_recon_loss": running["recon"] / denom,
                    "train_kl": running["kl"] / denom,
                    "train_kl_mean": train_kl_mean_epoch,
                    "train_kl_effective_last": train_kl_effective_last,
                    "mu_mean_batch": mu_stats,
                    "z_std_batch": z_std,
                    "lr": optimizer.param_groups[0]["lr"]
                }, step=total_steps, phase="train")

            # Debug early break
            if debug_enabled and i + 1 >= debug_cfg.max_train_batches:
                break

        # Epoch-level scheduler (e.g. cosine)
        if scheduler and cfg.optimization.scheduler == "cosine":
            scheduler.step()

        # -------- Validation --------
        model.eval()
        val_tot = val_rec = val_kl = 0.0
        val_batches = 0
        with torch.no_grad():
            for j, batch in enumerate(test_loader):
                x = batch["image"]
                if isinstance(x, (list, tuple)):
                    x = torch.stack(x, 0)
                x = x.to(device, non_blocking=True)
                if capacity is not None:
                    l = model.loss(x, capacity=capacity, capacity_weight=cfg.loss.capacity_weight)
                else:
                    l = model.loss(x, beta=beta, free_bits=free_bits)
                val_tot += l["total"].item()
                val_rec += l["recon"].item()
                val_kl += l["kl_mean"].item()
                val_batches += 1
                if debug_enabled and j + 1 >= debug_cfg.max_val_batches:
                    break

        val_total = val_tot / max(1, val_batches)
        val_recon = val_rec / max(1, val_batches)
        val_kl_avg = val_kl / max(1, val_batches)
        final_train_kl_mean = running["kl"] / (i + 1)
        final_train_kl_effective = train_kl_effective_last
        log_metrics({
            "epoch": epoch,
            "beta": float(beta),
            "capacity": float(capacity) if capacity is not None else None,
            "val_total_loss": val_total,
            "val_recon_loss": val_recon,
            "val_kl": val_kl_avg,
            "train_kl_mean": final_train_kl_mean,
            "train_kl_effective_last": final_train_kl_effective,
        }, step=total_steps, phase="val")

        # Checkpoints
        ckpt.save_latest(epoch, {"val_total": val_total})
        ckpt.save_best(epoch, {"val_total": val_total}, monitor_value=val_total)

        # Recon panel (fixed set or first batch)
        sample_reconstructions(
            model,
            test_loader,
            device,
            figures_dir,
            epoch,
            fixed_paths=fixed_paths,
            transform=test_tf,
        )

        # Early stopping
        early.update(val_total)
        if early.should_stop:
            break

    return

if __name__ == "__main__":
    train()
