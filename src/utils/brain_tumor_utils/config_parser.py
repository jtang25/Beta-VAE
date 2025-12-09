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

def _resolve_config_path(path=None):
    """
    Pick a usable config path using the following priority:
      1) explicit function argument
      2) CONFIG_PATH environment variable
      3) default config (configs/beta_vae_se.yaml)
      4) known fallback (configs/overfit_capacity.yaml)
    Raises FileNotFoundError if none are present.
    """
    candidates = []
    if path:
        candidates.append(path)
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append("configs/beta_vae_se.yaml")
    candidates.append("configs/overfit_capacity.yaml")

    tried = []
    for cand in candidates:
        if not cand:
            continue
        cand = os.path.expanduser(str(cand))
        tried.append(cand)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Config file not found. Set CONFIG_PATH or pass a path. Tried: {tried}"
    )

def load_config(path=None):
    cfg_path = _resolve_config_path(path)
    with open(cfg_path,"r") as f:
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
