import os
import shutil
import random
from math import floor
from utils.brain_tumor_utils.config_parser import get_config


def _find_presplit_dirs(raw_root):
    subs = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    train_dir = None
    test_dir = None
    for s in subs:
        if s.lower() in ["training", "train"]:
            train_dir = os.path.join(raw_root, s)
        if s.lower() in ["testing", "test", "validation", "val"]:
            test_dir = os.path.join(raw_root, s)
    if train_dir and test_dir:
        return train_dir, test_dir
    return None, None


def _copy_class_split(src_split_dir, dst_split_dir):
    classes = [c for c in os.listdir(src_split_dir) if os.path.isdir(os.path.join(src_split_dir, c))]
    for cls in classes:
        cls_src = os.path.join(src_split_dir, cls)
        cls_dst = os.path.join(dst_split_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)
        for f in os.listdir(cls_src):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                shutil.copy2(os.path.join(cls_src, f), os.path.join(cls_dst, f))


def split_from_raw(overwrite=False):
    cfg = get_config()
    raw = cfg.paths.raw_dir
    proc = cfg.paths.processed_dir
    train_dir = os.path.join(proc, cfg.data.train_subdir)
    test_dir = os.path.join(proc, cfg.data.test_subdir)

    if (os.path.exists(train_dir) or os.path.exists(test_dir)) and not overwrite:
        return
    if overwrite and os.path.isdir(proc):
        shutil.rmtree(proc)

    pre_train, pre_test = _find_presplit_dirs(raw)
    if pre_train and pre_test:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        _copy_class_split(pre_train, train_dir)
        _copy_class_split(pre_test, test_dir)
        return

    classes = [c for c in os.listdir(raw) if os.path.isdir(os.path.join(raw, c))]
    try:
        train_ratio = cfg.data.train_ratio
    except AttributeError:
        train_ratio = 0.8
    for split_dir in [train_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
    rng = random.Random(cfg.data.seed)
    for cls in classes:
        cls_raw = os.path.join(raw, cls)
        files = [f for f in os.listdir(cls_raw) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
        rng.shuffle(files)
        n_train = floor(len(files) * train_ratio)
        train_files = files[:n_train]
        test_files = files[n_train:]
        tgt_train_cls = os.path.join(train_dir, cls)
        tgt_test_cls = os.path.join(test_dir, cls)
        os.makedirs(tgt_train_cls, exist_ok=True)
        os.makedirs(tgt_test_cls, exist_ok=True)
        for f in train_files:
            shutil.copy2(os.path.join(cls_raw, f), os.path.join(tgt_train_cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(cls_raw, f), os.path.join(tgt_test_cls, f))


def verify_processed():
    cfg = get_config()
    train_dir = os.path.join(cfg.paths.processed_dir, cfg.data.train_subdir)
    test_dir = os.path.join(cfg.paths.processed_dir, cfg.data.test_subdir)
    for d in [train_dir, test_dir]:
        if not os.path.isdir(d):
            raise RuntimeError(f"Missing split directory {d}")
        classes = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
        if len(classes) == 0:
            raise RuntimeError(f"No class folders in {d}")
    return True
