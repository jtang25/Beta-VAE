import logging
import os
import sys
import json
from datetime import datetime
from .config_parser import get_config

_logger = None

def init_logger(name="beta_vae_se"):
    global _logger
    if _logger is not None:
        return _logger
    cfg = get_config()
    level = getattr(logging, cfg.logging.log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if cfg.logging.log_to_file:
            run_id = cfg.paths.run_id
            log_dir = os.path.join(cfg.paths.outputs_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"{run_id}.log"))
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    _logger = logger
    return logger

def log_config():
    logger = init_logger()
    cfg = get_config().to_dict()
    logger.info("CONFIG " + json.dumps(cfg))

def log_metrics(metrics, step=None, phase="train"):
    logger = init_logger()
    payload = {"phase": phase, "step": step}
    payload.update(metrics)
    logger.info("METRICS " + json.dumps(payload))
