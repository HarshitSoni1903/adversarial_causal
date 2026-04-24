"""Shared utilities: config loading, seed management, checkpoint paths, bucketing."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    """Set seeds for random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------
# Bucketing
# ------------------------------------------------------------------

def bucket_investment(inv: int | float, rule: str) -> int:
    """Return bucket index 0..4 for a raw investment integer.

    Two rules, both tested to resolve paper ambiguity:
      paper_range:      0-4→0, 5-8→1, 9-12→2, 13-16→3, 17-20→4
      nearest_neighbor: argmin |inv - center| where centers = [0,5,10,15,20]
    """
    inv = int(round(inv))
    if rule == "paper_range":
        if inv <= 4:
            return 0
        elif inv <= 8:
            return 1
        elif inv <= 12:
            return 2
        elif inv <= 16:
            return 3
        else:
            return 4
    elif rule == "nearest_neighbor":
        centers = [0, 5, 10, 15, 20]
        return int(np.argmin([abs(inv - c) for c in centers]))
    else:
        raise ValueError(f"Unknown bucketing_rule: {rule!r}")


def action_to_idx(action: float, actions: list[int], rule: str = "nearest_neighbor") -> int:
    """Map a raw investment value to its nearest bucket index.

    Wraps bucket_investment for backward compatibility; uses nearest_neighbor
    by default so existing callers are unaffected.
    """
    return bucket_investment(int(round(action)), rule)


def idx_to_action(idx: int, action_values: list[int]) -> int:
    """Convert a bucket index back to the corresponding action value."""
    return action_values[idx]


# ------------------------------------------------------------------
# Checkpoint / output paths
# ------------------------------------------------------------------

def _world_mode_str(config: dict) -> str:
    """Return 'w0' or 'w1' based on world.mode."""
    mode = config.get("world", {}).get("mode", 0)
    return "w1" if mode == 1 else "w0"


def create_run_id(config: dict) -> str:
    """Create a unique run ID: w0_YYYYMMDD_HHMMSS."""
    mode = _world_mode_str(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{timestamp}"


def get_output_dir(config: dict, run_id: str | None = None) -> str:
    """Return the run-specific output directory: outputs/<run_id>/."""
    base = config["export"]["output_dir"]
    path = Path(base) / (run_id or _world_mode_str(config))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_save_dir(config: dict) -> str:
    """Return adversary checkpoint save directory from config."""
    return config["adversary"].get("save_dir", "checkpoints/")
