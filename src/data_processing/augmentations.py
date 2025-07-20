import torch
from torchvision import transforms
from utils.brain_tumor_utils.config_parser import get_config

def get_train_transforms():
    cfg = get_config()
    size = cfg.data.image_size
    t = []
    t.append(transforms.Resize((size, size)))
    if cfg.augmentation.use_augmentations:
        if cfg.augmentation.horizontal_flip:
            t.append(transforms.RandomHorizontalFlip())
        if cfg.augmentation.rotation_degrees and cfg.augmentation.rotation_degrees > 0:
            t.append(transforms.RandomRotation(degrees=cfg.augmentation.rotation_degrees))
        if cfg.augmentation.brightness and cfg.augmentation.brightness > 0:
            t.append(transforms.ColorJitter(brightness=cfg.augmentation.brightness))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def get_test_transforms():
    cfg = get_config()
    size = cfg.data.image_size
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
