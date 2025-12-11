import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms
from models.beta_vae import BetaVAE
from utils.brain_tumor_utils.io import load_sharded_checkpoint


def load_model(weights="best"):
    cfg = get_config()
    path = Path(cfg.paths.models_dir) / f"{cfg.paths.run_id}_{weights}.pt"
    if not path.exists():
        path = Path(cfg.paths.models_dir) / f"{cfg.paths.run_id}_latest.pt"
    payload = load_sharded_checkpoint(str(path), map_location="cpu")
    model = BetaVAE()
    model.load_state_dict(payload["model_state"])
    return model


def extract_latents(model, loader, device, limit=None):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            mu, _ = model.encode(x)
            latents.append(mu.cpu())
            labels.extend(batch["label"])
            if limit and len(labels) >= limit:
                break
    L = torch.cat(latents, dim=0)
    if limit:
        L = L[:limit]
        labels = labels[:limit]
    return L.numpy(), np.array(labels)


def make_umap_gif(latents, labels, out_path, n_neighbors=15, min_dist=0.1, frames=60, elev=30, class_names=None):
    import umap
    import imageio.v2 as imageio

    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    emb = reducer.fit_transform(latents)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    uniq = np.unique(labels)
    def color_for(lbl):
        return cmap(int(lbl) % 10)
    colors = [color_for(i) for i in labels]
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colors, s=8, alpha=0.8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    if len(uniq) <= 10:
        handles = [plt.Line2D([0], [0], marker="o", color="w", label=str(u),
                              markerfacecolor=color_for(u), markersize=6)
                   for u in uniq]
        if class_names:
            for h in handles:
                try:
                    lbl_val = float(h.get_label()) if "." in h.get_label() else int(h.get_label())
                except Exception:
                    lbl_val = h.get_label()
                h.set_label(class_names.get(lbl_val, class_names.get(str(lbl_val), h.get_label())))
        ax.legend(handles=handles, title="class", loc="upper right")

    images = []
    for azim in np.linspace(0, 360, frames, endpoint=False):
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape((h, w, 3))
        images.append(frame)
    imageio.mimsave(out_path, images, duration=0.08, loop=0)
    plt.close(fig)
    return out_path


def split_image_into_columns(img: Image.Image, num_cols: int = 7) -> List[Image.Image]:
    w, h = img.size
    cols = []
    for i in range(num_cols):
        left = round(i * w / num_cols)
        right = round((i + 1) * w / num_cols)
        cols.append(img.crop((left, 0, right, h)))
    return cols


def class_from_name(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0]


def make_traversal_grid(saved_dir, out_path, titles=None, grid_title="Traversal Grid"):
    saved_dir = Path(saved_dir)
    files = sorted(saved_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNGs found in {saved_dir}")
    titles = titles or ["-3", "-2", "-1", "0", "+1", "+2", "+3"]
    rows = len(files)
    cols = 7
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    fig.suptitle(grid_title, fontsize=18, weight="bold")
    if rows == 1:
        axes = [axes]
    for r, fp in enumerate(files):
        img = Image.open(fp).convert("RGB")
        slices = split_image_into_columns(img, num_cols=cols)
        for c, sl in enumerate(slices):
            ax = axes[r][c]
            ax.imshow(sl)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=12, weight="bold")
            if c == 0:
                ax.set_ylabel(class_from_name(fp), rotation=0, labelpad=35, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate 3D UMAP GIF and traversal grid.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--weights", type=str, default="best", help="Checkpoint tag to load.")
    parser.add_argument("--umap-gif", type=str, default="outputs/figures/umap3d.gif", help="Output path for rotating UMAP GIF.")
    parser.add_argument("--umap-limit", type=int, default=None, help="Optional max samples for UMAP (defaults to config.evaluation.num_umap_samples).")
    parser.add_argument("--umap-frames", type=int, default=60, help="Number of frames in rotation GIF.")
    parser.add_argument("--umap-elev", type=float, default=30.0, help="Elevation angle for view.")
    parser.add_argument("--saved-dir", type=str, default="outputs/saved", help="Dir with saved traversal-like PNGs.")
    parser.add_argument("--grid-out", type=str, default="outputs/figures/traversal_grid.png", help="Output path for grid PNG.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP GIF generation.")
    parser.add_argument("--skip-grid", action="store_true", help="Skip traversal grid generation.")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.skip_umap:
        tf = get_test_transforms()
        _, test_loader = build_dataloaders(transform_train=tf, transform_test=tf)
        model = load_model(args.weights).to(device)
        umap_limit = args.umap_limit if args.umap_limit is not None else getattr(get_config().evaluation, "num_umap_samples", None)
        latents, labels = extract_latents(model, test_loader, device, limit=umap_limit)
        class_map = getattr(getattr(test_loader, "dataset", None), "class_to_idx", {}) or {}
        idx_to_class = {v: k for k, v in class_map.items()} if class_map else {}
        Path(args.umap_gif).parent.mkdir(parents=True, exist_ok=True)
        make_umap_gif(latents, labels, args.umap_gif, frames=args.umap_frames, elev=args.umap_elev, class_names=idx_to_class)
        print(f"[OK] Saved UMAP GIF to {args.umap_gif}")

    if not args.skip_grid:
        Path(args.grid_out).parent.mkdir(parents=True, exist_ok=True)
        make_traversal_grid(args.saved_dir, args.grid_out)
        print(f"[OK] Saved traversal grid to {args.grid_out}")


if __name__ == "__main__":
    main()
