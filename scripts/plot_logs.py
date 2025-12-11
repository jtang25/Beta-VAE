"""
Parse training logs and plot key metrics for the train phase.

Usage:
  python scripts/plot_logs.py --config configs/beta_vae_se.yaml

Outputs:
  - saves a PNG under outputs/figures/{run_id}_train_metrics.png
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config


def parse_metrics(log_path):
    """
    Extract JSON payloads from lines containing 'METRICS '.
    Returns a pandas DataFrame.
    """
    rows = []
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    pattern = re.compile(r"METRICS (\\{.*\\})")
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            try:
                payload = json.loads(m.group(1))
                rows.append(payload)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def plot_train_metrics(df, out_path):
    cols = [
        "train_total_loss",
        "train_recon_loss",
        "train_kl",
        "train_kl_mean",
        "train_kl_effective_last",
    ]
    df_train = df[df["phase"] == "train"].copy()
    if df_train.empty:
        raise ValueError("No train-phase metrics found in the log.")
    x = df_train["step"] if "step" in df_train and df_train["step"].notna().any() else df_train.index

    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 12), sharex=True)
    for ax, c in zip(axes, cols):
        if c not in df_train:
            ax.text(0.5, 0.5, f"{c} not found", ha="center", va="center")
            ax.set_ylabel(c)
            continue
        ax.plot(x, df_train[c], label=c, color="tab:blue")
        ax.set_ylabel(c)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("step")
    fig.suptitle("Train metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot train metrics from log.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = parser.parse_args()
    if args.config:
        import os
        os.environ["CONFIG_PATH"] = args.config

    cfg = get_config()
    log_path = Path(cfg.paths.outputs_dir) / "logs" / f"{cfg.paths.run_id}.log"
    out_path = Path(cfg.paths.figures_dir) / f"{cfg.paths.run_id}_train_metrics.png"

    df = parse_metrics(log_path)
    plot_train_metrics(df, out_path)
    print(f"Saved train metrics plot to {out_path}")


if __name__ == "__main__":
    main()
