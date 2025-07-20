# training/schedulers.py
import math
import inspect

def _to_mapping(obj):
    """
    Convert an attribute-style / OmegaConf object to a plain dict recursively (1 level is enough here).
    If it's already a dict, return as-is.
    """
    if isinstance(obj, dict):
        return obj
    # Some OmegaConf nodes have .keys(); if so we can just iterate
    if hasattr(obj, "keys") and callable(getattr(obj, "keys")):
        return {k: getattr(obj, k) for k in obj.keys()}
    # Fallback: dir() and filter callables / private
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        v = getattr(obj, k)
        if inspect.ismethod(v) or inspect.isfunction(v):
            continue
        out[k] = v
    return out

class BetaScheduler:
    """
    Supports config section 'beta_schedule' with keys:
      type: constant|linear|cosine|cyclical
      start_beta / start
      end_beta / end
      warmup_epochs / warmup
      cycle_length / cycle
    Fallback to model.beta if schedule not provided.
    """
    def __init__(self, root_cfg, total_epochs):
        root_map = _to_mapping(root_cfg)

        # Extract sub-config (if missing, create minimal)
        if "beta_schedule" in root_map:
            raw_bs = root_map["beta_schedule"]
        else:
            # fallback to model.beta
            model_beta = 1.0
            if "model" in root_map and isinstance(root_map["model"], (dict, object)):
                model_map = _to_mapping(root_map["model"])
                model_beta = model_map.get("beta", model_beta)
            raw_bs = {"type": "constant", "end_beta": model_beta}

        bs = _to_mapping(raw_bs)

        # Normalize keys
        self.type = bs.get("type", "constant")
        self.start = bs.get("start_beta", bs.get("start",
                      bs.get("end_beta", bs.get("end", 1.0))))
        self.end = bs.get("end_beta", bs.get("end", self.start))
        self.warm = bs.get("warmup_epochs", bs.get("warmup", 0))
        self.cycle = bs.get("cycle_length", bs.get("cycle", 0))
        self.total_epochs = total_epochs

    def value(self, epoch: int):
        if self.type == "constant":
            return self.end
        if self.type == "linear":
            if self.warm <= 0:
                return self.end
            ratio = min(1.0, epoch / float(self.warm))
            return self.start + (self.end - self.start) * ratio
        if self.type == "cosine":
            # smooth transition start->end over total_epochs
            if self.total_epochs <= 1:
                return self.end
            return ( self.start +
                     0.5 * (self.end - self.start) *
                     (1 - math.cos(math.pi * epoch / (self.total_epochs - 1))) )
        if self.type in ("cyclical", "cyc"):
            if self.cycle <= 0:
                return self.end
            pos = (epoch % self.cycle) / float(self.cycle)
            return self.start + (self.end - self.start) * pos
        return self.end


class CapacityScheduler:
    """
    Expects config path loss.capacity_schedule with keys:
      enabled (bool)
      C_start
      C_end
      warmup_epochs
      total_epochs
    Returns None if disabled.
    """
    def __init__(self, root_cfg, total_epochs):
        root_map = _to_mapping(root_cfg)
        loss_map = _to_mapping(root_map.get("loss", {}))
        cap_raw = loss_map.get("capacity_schedule", {})
        cs = _to_mapping(cap_raw)

        self.enabled = bool(cs.get("enabled", False))
        self.C0 = cs.get("C_start", 0.0)
        self.C1 = cs.get("C_end", self.C0)
        self.warm = cs.get("warmup_epochs", 0)
        # total_epochs in schedule can override training epochs
        self.total = cs.get("total_epochs", total_epochs)
        self.total_epochs = total_epochs  # keep original training loop length

    def value(self, epoch: int):
        if not self.enabled:
            return None
        if epoch < self.warm:
            return self.C0
        span = max(1, (self.total - self.warm))
        prog = min(1.0, (epoch - self.warm) / span)
        return self.C0 + prog * (self.C1 - self.C0)
