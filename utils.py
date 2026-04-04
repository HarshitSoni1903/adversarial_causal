"""Shared utilities: config loading, seed management, checkpoint paths, and action encoding."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def get_checkpoint_dir(config: dict) -> str:
    """Return the world-mode-specific checkpoint directory, creating it if needed.

    Maps config["world"]["communication"] to w0/ or w1/ under the base dir.
    """
    base = config["checkpoints"]["base_dir"]
    mode = "w1" if config["world"]["communication"] else "w0"
    path = Path(base) / mode
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
