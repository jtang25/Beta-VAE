import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.io import save_figure
from .recon_metrics import extract_latents

def reduce_latents(latents, method="umap", n_neighbors=15, min_dist=0.1, seed=42):
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
            emb = reducer.fit_transform(latents)
            return emb
        except:
            method = "pca"
    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(latents)

def plot_latent_scatter(emb, labels, title, binary=True):
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
            plt.scatter(emb[mask,0], emb[mask,1], s=10, alpha=0.7, c=[cmap(i)], label=str(c))
    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def generate_latent_visualizations(model, test_loader, device):
    cfg = get_config()
    lim = cfg.evaluation.num_umap_samples
    latents, labels, paths = extract_latents(model, test_loader, device, limit=lim)
    binary = cfg.data.class_mode == "binary"
    emb = reduce_latents(latents, method="umap")
    fig = plot_latent_scatter(emb, labels, "Latent Scatter", binary=binary)
    save_figure(fig, "latent_scatter")
    plt.close(fig)
    per_dim_violin(latents, labels, binary)

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
    plt.tight_layout()
    save_figure(plt.gcf(), "latent_per_dim_violin")
    plt.close()
