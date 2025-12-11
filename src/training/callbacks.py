import torch
import os
import math
from utils.brain_tumor_utils.config_parser import get_config

class GradScalerWrapper:
    def __init__(self, enabled):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.enabled = enabled and torch.cuda.is_available()
        try:
            self.scaler = torch.amp.GradScaler(device_type=device_type, enabled=self.enabled)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=self.enabled)
        self.params = None
    def set_params(self, params):
        self.params = list(params)
    def backward(self, loss, optimizer, clip=0.0):
        if self.enabled:
            self.scaler.scale(loss).backward()
            if clip > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.params, clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.params, clip)
            optimizer.step()

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False
    def update(self, value):
        if self.best is None:
            self.best = value
            return
        improve = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improve:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True

class CheckpointManager:
    def __init__(self, model, optimizer):
        cfg = get_config()
        self.model = model
        self.optimizer = optimizer
        self.dir = cfg.paths.models_dir
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = cfg.paths.run_id
        self.best_value = None
    def save_latest(self, epoch, total_steps, extra):
        path = os.path.join(self.dir, f"{self.run_id}_latest.pt")
        torch.save({
            "epoch": epoch,
            "total_steps": total_steps,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            **extra
        }, path)
    def save_best(self, epoch, total_steps, extra, monitor_value):
        if self.best_value is None or monitor_value < self.best_value:
            self.best_value = monitor_value
            path = os.path.join(self.dir, f"{self.run_id}_best.pt")
            torch.save({
                "epoch": epoch,
                "total_steps": total_steps,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                **extra
            }, path)


def get_optimizer(model):
    cfg = get_config()
    opt_cfg = cfg.optimization
    params = model.parameters()
    if opt_cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    if opt_cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    if opt_cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay, momentum=0.9)
    raise ValueError("unsupported optimizer")

def build_scheduler(optimizer):
    cfg = get_config()
    sch = cfg.optimization.scheduler.lower()
    if sch == "none":
        return None
    if sch == "cosine":
        epochs = cfg.training.epochs if not cfg.debug.enabled else cfg.debug.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sch == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    raise ValueError("unsupported scheduler")
