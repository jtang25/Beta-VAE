import os
import json
import torch
from datetime import datetime
from .config_parser import get_config

def ensure_dirs():
    cfg = get_config()
    for k in ["outputs_dir","models_dir","figures_dir","tables_dir"]:
        path = getattr(cfg.paths, k)
        os.makedirs(path, exist_ok=True)
    log_dir = os.path.join(cfg.paths.outputs_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

def run_artifact_dir():
    cfg = get_config()
    d = os.path.join(cfg.paths.outputs_dir, cfg.paths.run_id)
    os.makedirs(d, exist_ok=True)
    return d

def model_checkpoint_path(epoch=None, tag=None):
    cfg = get_config()
    base = cfg.paths.models_dir
    os.makedirs(base, exist_ok=True)
    if tag:
        return os.path.join(base, f"{cfg.paths.run_id}_{tag}.pt")
    if epoch is not None:
        return os.path.join(base, f"{cfg.paths.run_id}_epoch{epoch}.pt")
    return os.path.join(base, f"{cfg.paths.run_id}_latest.pt")

def save_checkpoint(model, optimizer, epoch, metrics=None, tag=None, extra=None):
    path = model_checkpoint_path(epoch if tag is None else None, tag=tag)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
        "extra": extra or {}
    }
    torch.save(payload, path)
    return path

def save_json(data, name):
    cfg = get_config()
    out = os.path.join(cfg.paths.outputs_dir, f"{name}.json")
    with open(out,"w") as f:
        json.dump(data,f,indent=2)
    return out

def save_table(df, name):
    cfg = get_config()
    path = os.path.join(cfg.paths.tables_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path

def save_figure(fig, name):
    cfg = get_config()
    path = os.path.join(cfg.paths.figures_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    return path
