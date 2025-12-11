"""
Plot train (total/recon/KL) and validation (total) losses from a log file.

Usage:
  python scripts/plot_phase_losses.py --config configs/beta_vae_se.yaml
  # or override paths explicitly
  python scripts/plot_phase_losses.py --log outputs/logs/beta_vae_se.log --out outputs/figures/beta_vae_se_losses.png
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config


def parse_metrics(log_path: Path) -> pd.DataFrame:
    """Extract JSON payloads from lines containing METRICS."""
    rows = []
    pattern = re.compile(r"METRICS (\{.*\})")
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            try:
                rows.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def plot_train_val_losses(df: pd.DataFrame, out_path: Path) -> None:
    if "phase" not in df.columns:
        raise ValueError("No phase column found in metrics log.")

    df_train = df[df["phase"] == "train"].copy()
    if len(df_train) > 7:
        df_train = df_train.iloc[7:]
    df_val = df[df["phase"] == "val"].copy()
    if df_train.empty and df_val.empty:
        raise ValueError("No train or val metrics found in the log.")

    if "step" in df_train and df_train["step"].notna().any():
        x_train = df_train["step"]
    else:
        x_train = df_train.index

    if "step" in df_val and df_val["step"].notna().any():
        x_val = df_val["step"]
        x_val_label = "step"
    elif "epoch" in df_val and df_val["epoch"].notna().any():
        x_val = df_val["epoch"] * 180
        x_val_label = "epoch (scaled to step)"
    else:
        x_val = df_val.index
        x_val_label = "index"

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    ax_recon = axes[0]
    train_recon_line = None
    val_recon_line = None

    if not df_train.empty and "train_recon_loss" in df_train and not df_train["train_recon_loss"].isna().all():
        train_recon_line = ax_recon.plot(
            x_train,
            df_train["train_recon_loss"],
            label="train_recon_loss",
            color="tab:blue",
        )[0]

    if not df_val.empty and "val_recon_loss" in df_val and not df_val["val_recon_loss"].isna().all():
        val_recon_line = ax_recon.plot(
            x_val,
            df_val["val_recon_loss"],
            label="val_recon_loss",
            color="tab:orange",
            linestyle="--",
        )[0]

    if train_recon_line is None and val_recon_line is None:
        ax_recon.text(0.5, 0.5, "No recon losses found", ha="center", va="center")

    ax_recon.set_title("Reconstruction losses")
    ax_recon.set_xlabel("step")
    ax_recon.set_ylabel("recon loss")
    ax_recon.grid(True, linestyle="--", alpha=0.4)

    recon_lines = [l for l in (train_recon_line, val_recon_line) if l is not None]
    recon_labels = [l.get_label() for l in recon_lines]
    if recon_lines:
        ax_recon.legend(recon_lines, recon_labels, loc="upper right")

    ax_kl = axes[1]
    train_kl_line = None
    val_kl_line = None

    if not df_train.empty and "train_kl" in df_train and not df_train["train_kl"].isna().all():
        train_kl_line = ax_kl.plot(
            x_train,
            df_train["train_kl"],
            label="train_kl",
            color="tab:red",
        )[0]

    if not df_val.empty and "val_kl" in df_val and not df_val["val_kl"].isna().all():
        val_kl_line = ax_kl.plot(
            x_val,
            df_val["val_kl"],
            label="val_kl",
            color="tab:green",
        )[0]

    if train_kl_line is None and val_kl_line is None:
        ax_kl.text(0.5, 0.5, "No KL losses found", ha="center", va="center")

    ax_kl.set_title("KL losses")
    ax_kl.set_xlabel("step")
    ax_kl.set_ylabel("KL")
    ax_kl.grid(True, linestyle="--", alpha=0.4)

    kl_lines = [l for l in (train_kl_line, val_kl_line) if l is not None]
    kl_labels = [l.get_label() for l in kl_lines]
    if kl_lines:
        ax_kl.legend(kl_lines, kl_labels, loc="upper right")

    ax_metrics = axes[2]
    ax_metrics_r2 = ax_metrics.twinx()

    auc_line = None
    best_auc_line = None
    best_corr_line = None
    best_r2_line = None

    if "epoch" in df_val and df_val["epoch"].notna().any():
        x_metrics = df_val["epoch"]
        x_metrics_label = "epoch"
    else:
        x_metrics = x_val
        x_metrics_label = x_val_label

    if not df_val.empty and "latent_probe_auc" in df_val and not df_val["latent_probe_auc"].isna().all():
        auc_line = ax_metrics.plot(
            x_metrics,
            df_val["latent_probe_auc"],
            label="latent_probe_auc",
            color="tab:blue",
        )[0]

    if not df_val.empty and "best_dim_auc" in df_val and not df_val["best_dim_auc"].isna().all():
        best_auc_line = ax_metrics.plot(
            x_metrics,
            df_val["best_dim_auc"],
            label="best_dim_auc",
            color="tab:orange",
            linestyle="--",
        )[0]

    if not df_val.empty and "best_dim_corr" in df_val and not df_val["best_dim_corr"].isna().all():
        best_corr_line = ax_metrics.plot(
            x_metrics,
            df_val["best_dim_corr"],
            label="best_dim_corr",
            color="tab:green",
            linestyle=":",
        )[0]

    if not df_val.empty and "best_dim_r2" in df_val and not df_val["best_dim_r2"].isna().all():
        best_r2_line = ax_metrics_r2.plot(
            x_metrics,
            df_val["best_dim_r2"],
            label="best_dim_r2",
            color="tab:red",
        )[0]

    if all(v is None for v in (auc_line, best_auc_line, best_corr_line, best_r2_line)):
        ax_metrics.text(0.5, 0.5, "No probe metrics found", ha="center", va="center")

    ax_metrics.set_title("Latent probe metrics")
    ax_metrics.set_xlabel(x_metrics_label)
    ax_metrics.set_ylabel("AUC / Corr")
    ax_metrics.grid(True, linestyle="--", alpha=0.4)

    ax_metrics_r2.set_ylabel("best_dim_r2", color="tab:red")
    ax_metrics_r2.tick_params(axis="y", labelcolor="tab:red")

    metric_lines = [l for l in (auc_line, best_auc_line, best_corr_line) if l is not None]
    metric_labels = [l.get_label() for l in metric_lines]
    if best_r2_line is not None:
        metric_lines.append(best_r2_line)
        metric_labels.append(best_r2_line.get_label())
    if metric_lines:
        ax_metrics.legend(metric_lines, metric_labels, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot train/val losses from log.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (sets run_id).")
    parser.add_argument("--log", type=str, default=None, help="Override log path.")
    parser.add_argument("--out", type=str, default=None, help="Override output figure path.")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    cfg = get_config()
    log_path = Path(args.log) if args.log else Path(cfg.paths.outputs_dir) / "logs" / f"{cfg.paths.run_id}.log"
    out_path = Path(args.out) if args.out else Path(cfg.paths.figures_dir) / f"{cfg.paths.run_id}_losses.png"

    df = parse_metrics(log_path)
    plot_train_val_losses(df, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
