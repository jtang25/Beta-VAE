import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .config_parser import get_config

_tumor_classes = ["glioma","meningioma","pituitary"]

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, sample_limit=None):
        self.cfg = get_config()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(root_dir, self.cfg.data.train_subdir if split=="train" else self.cfg.data.test_subdir)
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir,d))])
        self.original_classes = classes
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png",".jpg",".jpeg",".tif",".bmp",".tiff")):
                    self.samples.append((os.path.join(cls_dir,fname), cls))
        rng = random.Random(self.cfg.data.seed if split=="train" else self.cfg.data.seed+1)
        rng.shuffle(self.samples)
        if sample_limit is not None:
            self.samples = self.samples[:sample_limit]

        self.class_mode = self.cfg.data.class_mode
        if self.class_mode == "multiclass":
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
        else:
            self.class_to_idx = {"healthy":0,"tumor":1}
        self._compute_labels()

    def _compute_labels(self):
        labels = []
        for path, cls in self.samples:
            if self.class_mode == "multiclass":
                labels.append(self.class_to_idx[cls])
            else:
                lab = 0 if cls == "notumor" else 1
                labels.append(lab)
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("L" if self.cfg.data.grayscale else "RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return {"image": img, "label": label, "class_name": cls, "path": path}

def build_dataloaders(transform_train=None, transform_test=None, train_limit=None, test_limit=None):
    cfg = get_config()
    train_ds = BrainTumorDataset(cfg.paths.processed_dir, "train", transform=transform_train, sample_limit=train_limit)
    test_ds = BrainTumorDataset(cfg.paths.processed_dir, "test", transform=transform_test, sample_limit=test_limit)
    if getattr(cfg.model, "deterministic_overfit", False) and getattr(cfg.debug, "enabled", False):
        test_ds = train_ds
    g = torch.Generator()
    g.manual_seed(cfg.data.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        generator=g
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
    )
    return train_loader, test_loader
