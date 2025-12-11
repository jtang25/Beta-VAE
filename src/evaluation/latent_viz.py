import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.io import save_figure
from .recon_metrics import extract_latents

def reduce_latents(latents, method="umap", n_neighbors=15, min_dist=0.1, seed=42, n_components=2):
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed, n_components=n_components)
            emb = reducer.fit_transform(latents)
            return emb
        except Exception:
            method = "pca"
    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=seed)
        return pca.fit_transform(latents)
    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, random_state=seed, init="random", learning_rate="auto")
        return tsne.fit_transform(latents)

def plot_latent_scatter(emb, labels, title, binary=True, class_names=None):
    plt.figure(figsize=(5,5))
    if binary:
        colors = ["#1f77b4","#d62728"]
        for c in [0,1]:
            mask = labels == c
            plt.scatter(emb[mask,0], emb[mask,1], s=10, alpha=0.7, c=colors[c], label=str(c))
    else:
        uniq = sorted(np.unique(labels))
        cmap = plt.get_cmap("tab10")
        for i,c in enumerate(uniq):
            mask = labels == c
            cname = class_names.get(c, str(c)) if class_names else str(c)
            plt.scatter(emb[mask,0], emb[mask,1], s=10, alpha=0.7, c=[cmap(i)], label=cname)
    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_latent_scatter3d_matplotlib(emb, labels, title, class_names=None):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    uniq = sorted(np.unique(labels))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(uniq):
        mask = labels == c
        cname = class_names.get(c, str(c)) if class_names else str(c)
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2], s=10, alpha=0.7, color=cmap(i), label=cname)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def generate_latent_visualizations(model, test_loader, device):
    cfg = get_config()
    lim = cfg.evaluation.num_umap_samples
    latents, labels, paths = extract_latents(model, test_loader, device, limit=lim)
    binary = cfg.data.class_mode == "binary"
    class_map = getattr(test_loader.dataset, "class_to_idx", {})
    idx_to_class = {v: k for k, v in class_map.items()} if class_map else None
    emb_umap = reduce_latents(latents, method="umap", n_components=2)
    fig = plot_latent_scatter(emb_umap, labels, "Latent Scatter (UMAP/PCA)", binary=binary, class_names=idx_to_class)
    save_figure(fig, "latent_scatter")
    plt.close(fig)
    try:
        emb_tsne = reduce_latents(latents, method="tsne", n_components=2)
        fig_tsne = plot_latent_scatter(emb_tsne, labels, "Latent Scatter (t-SNE)", binary=binary, class_names=idx_to_class)
        save_figure(fig_tsne, "latent_scatter_tsne")
        plt.close(fig_tsne)
    except Exception:
        pass
    per_dim_violin(latents, labels, binary)

    try:
        emb_umap3 = reduce_latents(latents, method="umap", n_components=3)
        if emb_umap3.shape[1] == 3:
            plot_latent_scatter3d_matplotlib(emb_umap3, labels, "Latent Scatter (UMAP 3D)", class_names=idx_to_class)
    except Exception:
        pass

    try:
        emb_tsne3 = reduce_latents(latents, method="tsne", n_components=3)
        if emb_tsne3.shape[1] == 3:
            plot_latent_scatter3d_matplotlib(emb_tsne3, labels, "Latent Scatter (t-SNE 3D)", class_names=idx_to_class)
    except Exception:
        pass

def per_dim_violin(latents, labels, binary=True):
    import matplotlib.pyplot as plt
    cfg = get_config()
    k = latents.shape[1]
    cols = min(4, k)
    rows = int(np.ceil(k/cols))
    plt.figure(figsize=(3*cols, 2.4*rows))
    for i in range(k):
        ax = plt.subplot(rows, cols, i+1)
        if binary:
            g0 = latents[labels==0, i]
            g1 = latents[labels==1, i]
            ax.violinplot([g0, g1], showextrema=False)
            ax.set_xticks([1,2])
            ax.set_xticklabels(["0","1"])
        else:
            groups = [latents[labels==c, i] for c in sorted(np.unique(labels))]
            ax.violinplot(groups, showextrema=False)
            ax.set_xticks(range(1,len(groups)+1))
            ax.set_xticklabels([str(c) for c in sorted(np.unique(labels))], rotation=90)
        ax.set_title(f"z{i}")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    save_figure(plt.gcf(), "latent_per_dim_violin")
    plt.close()
