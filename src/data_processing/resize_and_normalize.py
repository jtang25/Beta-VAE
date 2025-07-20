import os
import json
from PIL import Image
import torch
from torchvision import transforms
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.io import ensure_dirs

def _gather_image_paths(root):
    out = []
    for cls in sorted(os.listdir(root)):
        cpath = os.path.join(root, cls)
        if not os.path.isdir(cpath):
            continue
        for f in os.listdir(cpath):
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                out.append(os.path.join(cpath,f))
    return out

def compute_global_stats(split_dirs, sample_limit=None):
    cfg = get_config()
    imgs = []
    for d in split_dirs:
        imgs.extend(_gather_image_paths(d))
    if sample_limit is not None:
        imgs = imgs[:sample_limit]
    size = cfg.data.image_size
    to_tensor = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    s = 0
    ss = 0
    n = 0
    for p in imgs:
        im = Image.open(p).convert("L" if cfg.data.grayscale else "RGB")
        t = to_tensor(im)
        pixels = t.view(-1)
        s += pixels.sum()
        ss += (pixels**2).sum()
        n += pixels.numel()
    mean = (s / n).item()
    var = (ss / n - mean**2)
    std = var.sqrt().item()
    return {"mean": mean, "std": std}

def normalize_and_resize(split_root, stats=None, mode="minmax", overwrite=False):
    cfg = get_config()
    size = cfg.data.image_size
    classes = [c for c in os.listdir(split_root) if os.path.isdir(os.path.join(split_root,c))]
    for cls in classes:
        cls_dir = os.path.join(split_root, cls)
        for f in os.listdir(cls_dir):
            if not f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                continue
            path = os.path.join(cls_dir, f)
            im = Image.open(path).convert("L" if cfg.data.grayscale else "RGB")
            im = im.resize((size,size))
            if mode == "global_z" and stats is not None:
                t = transforms.ToTensor()(im)
                t = (t - stats["mean"]) / (stats["std"] + 1e-8)
                arr = (t - t.min()) / (t.max() - t.min() + 1e-8)
                arr = arr.mul(255).clamp(0,255).byte()
                if cfg.data.grayscale:
                    out_im = Image.fromarray(arr.squeeze(0).numpy(), mode="L")
                else:
                    out_im = Image.merge("RGB",[Image.fromarray(arr[i].numpy(), mode="L") for i in range(3)])
                out_im.save(path)
            else:
                im.save(path)

def write_stats(stats):
    cfg = get_config()
    stats_dir = os.path.join("data","intermediate")
    os.makedirs(stats_dir, exist_ok=True)
    path = os.path.join(stats_dir, "norm_stats.json")
    with open(path,"w") as f:
        json.dump(stats,f,indent=2)
    return path

def preprocess_dataset(compute_stats=True, normalization_mode="minmax"):
    cfg = get_config()
    ensure_dirs()
    train_root = os.path.join(cfg.paths.processed_dir, cfg.data.train_subdir)
    test_root = os.path.join(cfg.paths.processed_dir, cfg.data.test_subdir)
    stats = None
    if compute_stats and normalization_mode == "global_z":
        stats = compute_global_stats([train_root])
        write_stats(stats)
    normalize_and_resize(train_root, stats=stats, mode=normalization_mode)
    normalize_and_resize(test_root, stats=stats, mode=normalization_mode)
    return stats
