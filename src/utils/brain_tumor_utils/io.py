import os
import json
import torch
from datetime import datetime
from .config_parser import get_config


def _shard_paths(base_path, num_shards):
    root, ext = os.path.splitext(base_path)
    suffix = ext if ext else ".pt"
    return [f"{root}_shard{i}{suffix}" for i in range(num_shards)]


def save_sharded_checkpoint(base_path, payload, num_shards=2):
    """
    Save a checkpoint split across multiple files to reduce per-file size.
    Other fields are duplicated across shards; model_state is partitioned.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    model_state = payload.get("model_state")
    if model_state is None:
        raise ValueError("payload missing model_state for sharded checkpoint save")
    keys = sorted(model_state.keys())
    if num_shards < 1:
        num_shards = 1
    shards = [[] for _ in range(num_shards)]
    for idx, k in enumerate(keys):
        shards[idx % num_shards].append(k)

    for shard_idx, shard_keys in enumerate(shards):
        shard_state = {k: model_state[k] for k in shard_keys}
        shard_payload = dict(payload)
        shard_payload["model_state"] = shard_state
        shard_payload["shard_id"] = shard_idx
        shard_payload["num_shards"] = num_shards
        torch.save(shard_payload, _shard_paths(base_path, num_shards)[shard_idx])
    if os.path.exists(base_path):
        os.remove(base_path)
    return _shard_paths(base_path, num_shards)


def load_sharded_checkpoint(base_path, map_location=None, num_shards=2):
    """
    Load checkpoint saved via save_sharded_checkpoint. Falls back to single-file checkpoints.
    """
    shard_paths = _shard_paths(base_path, num_shards)
    if all(os.path.exists(p) for p in shard_paths):
        merged = {}
        meta = {}
        for p in shard_paths:
            part = torch.load(p, map_location=map_location)
            merged.update(part.get("model_state", {}))
            if not meta:
                meta = {k: v for k, v in part.items() if k != "model_state"}
        meta["model_state"] = merged
        return meta
    if os.path.exists(base_path):
        return torch.load(base_path, map_location=map_location)
    raise FileNotFoundError(f"No checkpoint found at {base_path} or shards")

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
    save_sharded_checkpoint(path, payload, num_shards=2)
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
