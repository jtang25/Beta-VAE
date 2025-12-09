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

# Ensure src/ is on sys.path
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
    df_val = df[df["phase"] == "val"].copy()
    if df_train.empty and df_val.empty:
        raise ValueError("No train or val metrics found in the log.")

    x_train = df_train["step"] if "step" in df_train and df_train["step"].notna().any() else df_train.index
    x_val = df_val["step"] if "step" in df_val and df_val["step"].notna().any() else df_val.index

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax_train = axes[0]
    ax_train_kl = ax_train.twinx()

    recon_line = None
    kl_line = None

    if "train_recon_loss" in df_train and not df_train["train_recon_loss"].isna().all():
        recon_line = ax_train.plot(
            x_train,
            df_train["train_recon_loss"],
            label="train_recon_loss",
            color="tab:blue",
        )[0]

    if "train_kl" in df_train and not df_train["train_kl"].isna().all():
        kl_line = ax_train_kl.plot(
            x_train,
            df_train["train_kl"],
            label="train_kl",
            color="tab:red",
        )[0]

    if recon_line is None and kl_line is None:
        ax_train.text(0.5, 0.5, "No train recon/KL found", ha="center", va="center")

    ax_train.set_title("Train losses")
    ax_train.set_xlabel("step")
    ax_train.set_ylabel("recon loss", color="tab:blue")
    ax_train.tick_params(axis="y", labelcolor="tab:blue")
    ax_train.grid(True, linestyle="--", alpha=0.4)

    ax_train_kl.set_ylabel("KL", color="tab:red")
    ax_train_kl.tick_params(axis="y", labelcolor="tab:red")

    lines = [l for l in (recon_line, kl_line) if l is not None]
    labels = [l.get_label() for l in lines]
    if lines:
        ax_train.legend(lines, labels, loc="upper right")

    ax_val = axes[1]
    ax_val_kl = ax_val.twinx()

    val_total_line = None
    val_recon_line = None
    val_kl_line = None

    if not df_val.empty and "val_total_loss" in df_val and not df_val["val_total_loss"].isna().all():
        val_total_line = ax_val.plot(
            x_val,
            df_val["val_total_loss"],
            label="val_total_loss",
            color="tab:orange",
            marker="o",
        )[0]

    if not df_val.empty and "val_recon_loss" in df_val and not df_val["val_recon_loss"].isna().all():
        val_recon_line = ax_val.plot(
            x_val,
            df_val["val_recon_loss"],
            label="val_recon_loss",
            color="tab:blue",
            linestyle="--",
            marker="x",
        )[0]

    if not df_val.empty and "val_kl" in df_val and not df_val["val_kl"].isna().all():
        val_kl_line = ax_val_kl.plot(
            x_val,
            df_val["val_kl"],
            label="val_kl",
            color="tab:red",
            marker="s",
        )[0]

    if all(v is None for v in (val_total_line, val_recon_line, val_kl_line)):
        ax_val.text(0.5, 0.5, "No val metrics found", ha="center", va="center")

    ax_val.set_title("Validation losses")
    ax_val.set_xlabel("step")
    ax_val.set_ylabel("total/recon loss", color="tab:blue")
    ax_val.tick_params(axis="y", labelcolor="tab:blue")
    ax_val.grid(True, linestyle="--", alpha=0.4)

    ax_val_kl.set_ylabel("KL", color="tab:red")
    ax_val_kl.tick_params(axis="y", labelcolor="tab:red")

    val_lines = [l for l in (val_total_line, val_recon_line, val_kl_line) if l is not None]
    val_labels = [l.get_label() for l in val_lines]
    if val_lines:
        ax_val.legend(val_lines, val_labels, loc="upper right")

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
