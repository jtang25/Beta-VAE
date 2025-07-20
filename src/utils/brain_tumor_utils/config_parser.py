import os
import yaml
from copy import deepcopy
from threading import Lock

_required_top_keys = [
    "paths","data","model","training","optimization",
    "beta_schedule","augmentation","evaluation","inference","logging","experiment","debug"
]

class _Frozen:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = _Frozen(v)
            setattr(self, k, v)
    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, _Frozen):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out
    def __getitem__(self, item):
        return getattr(self, item)
    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError("Frozen config is immutable")
        super().__setattr__(key, value)

_config_cache = None
_config_lock = Lock()

def _validate(cfg):
    missing = [k for k in _required_top_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required top-level keys: {missing}")
    if cfg["data"]["class_mode"] not in ["binary","multiclass"]:
        raise ValueError("data.class_mode must be binary or multiclass")
    if cfg["beta_schedule"]["type"] not in ["constant","linear","cyclical"]:
        raise ValueError("beta_schedule.type invalid")
    return True

def load_config(path=None):
    path = path or os.environ.get("CONFIG_PATH","configs/beta_vae_se.yaml")
    with open(path,"r") as f:
        raw = yaml.safe_load(f)
    _validate(raw)
    return raw

def get_config(path=None):
    global _config_cache
    if _config_cache is None:
        with _config_lock:
            if _config_cache is None:
                raw = load_config(path)
                frozen = _Frozen(deepcopy(raw))
                _config_cache = frozen
    return _config_cache
