"""Shared utilities: config loading, seed management, checkpoint paths, and action encoding."""

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


def _world_mode_str(config: dict) -> str:
    """Return 'w0' or 'w1' based on communication setting."""
    return "w1" if config["world"]["communication"] else "w0"


def create_run_id(config: dict) -> str:
    """Create a unique run ID: w0_20260404_153022 or w1_20260404_153022.

    Each run gets a timestamped folder so results are never overwritten.
    """
    mode = _world_mode_str(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{timestamp}"


def get_checkpoint_dir(config: dict, run_id: str | None = None) -> str:
    """Return the run-specific checkpoint directory.

    Structure: checkpoints/<run_id>/
    If run_id is None, falls back to checkpoints/<w0|w1>/ for backward compat.
    """
    base = config["checkpoints"]["base_dir"]
    if run_id:
        path = Path(base) / run_id
    else:
        path = Path(base) / _world_mode_str(config)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_output_dir(config: dict, run_id: str | None = None) -> str:
    """Return the run-specific output directory.

    Structure: outputs/<run_id>/
    If run_id is None, falls back to outputs/<w0|w1>/ for backward compat.
    """
    base = config["export"]["output_dir"]
    if run_id:
        path = Path(base) / run_id
    else:
        path = Path(base) / _world_mode_str(config)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def set_all_seeds(seed: int) -> None:
    """Set seeds for random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_to_idx(action: float, actions: list[int]) -> int:
    """Map a raw investment value to its nearest bucket index."""
    return int(np.argmin([abs(action - a) for a in actions]))


def idx_to_action(idx: int, actions: list[int]) -> int:
    """Convert a bucket index back to the corresponding action value."""
    return actions[idx]
