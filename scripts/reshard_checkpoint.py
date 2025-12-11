import argparse
import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.io import load_sharded_checkpoint, save_sharded_checkpoint


def _resolve_base_path(checkpoint: str, cfg) -> Path:
    """Turn a tag or path into a concrete checkpoint base path (without shard suffixes)."""
    if checkpoint in ("latest", "best"):
        base = Path(cfg.paths.models_dir) / f"{cfg.paths.run_id}_{checkpoint}.pt"
    else:
        base = Path(checkpoint)
    if base.suffix == "":
        base = base.with_suffix(".pt")
    return base


def _find_existing_shards(base_path: Path):
    """Return shard files matching the base path, e.g., beta_vae_se_latest_shard*.pt."""
    root = base_path.with_suffix("") if base_path.suffix else base_path
    suffix = base_path.suffix or ".pt"
    pattern = f"{root.name}_shard*{suffix}"
    return sorted(base_path.parent.glob(pattern))


def _infer_current_shard_count(base_path: Path) -> int:
    shards = _find_existing_shards(base_path)
    if shards:
        return len(shards)
    if base_path.exists():
        return 1
    raise FileNotFoundError(f"No checkpoint shards or file found for base path: {base_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine existing shards and reshard a checkpoint to a higher shard count."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config; defaults to project config resolution.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint tag (best|latest) or explicit base path (with or without .pt).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Desired number of shards for the output checkpoint (must exceed current count).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output base path for the new shards. Defaults to the input checkpoint base.",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    cfg = get_config()

    input_base = _resolve_base_path(args.checkpoint, cfg)
    output_base = Path(args.output) if args.output else input_base
    if output_base.suffix == "":
        output_base = output_base.with_suffix(".pt")

    current_shards = _infer_current_shard_count(input_base)
    if args.num_shards <= current_shards:
        raise ValueError(
            f"Requested shard count ({args.num_shards}) must be greater than existing shard count ({current_shards})."
        )

    shard_files = _find_existing_shards(input_base)
    if shard_files:
        print(f"Found {len(shard_files)} shard(s):")
        for p in shard_files:
            print(f"  - {p}")
    else:
        print(f"No shards found; using single checkpoint file at {input_base}")

    payload = load_sharded_checkpoint(
        str(input_base),
        map_location="cpu",
        num_shards=current_shards,
    )
    new_paths = save_sharded_checkpoint(str(output_base), payload, num_shards=args.num_shards)

    print(f"\nResharded checkpoint saved to {len(new_paths)} shard(s):")
    for p in new_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
