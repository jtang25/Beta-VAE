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

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=False)

    train_total = None
    val_total = None

    if not df_train.empty and "train_total_loss" in df_train and not df_train["train_total_loss"].isna().all():
        train_total = ax.plot(
            x_train,
            df_train["train_total_loss"],
            label="train_total_loss",
            color="tab:blue",
        )[0]

    if not df_val.empty and "val_total_loss" in df_val and not df_val["val_total_loss"].isna().all():
        val_total = ax.plot(
            x_val,
            df_val["val_total_loss"],
            label="val_total_loss",
            color="tab:orange",
            linestyle="--",
        )[0]

    if train_total is None and val_total is None:
        ax.text(0.5, 0.5, "No total losses found", ha="center", va="center")

    ax.set_title("Total loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", alpha=0.4)

    lines = [l for l in (train_total, val_total) if l is not None]
    labels = [l.get_label() for l in lines]
    if lines:
        ax.legend(lines, labels, loc="upper right")

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
