"""Shared utilities: config loading, seed management, and action encoding helpers."""

import random
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


def action_to_idx(action: float, actions: list[int]) -> int:
    """Map a raw investment value to its nearest bucket index."""
    return int(np.argmin([abs(action - a) for a in actions]))


def idx_to_action(idx: int, actions: list[int]) -> int:
    """Convert a bucket index back to the corresponding action value."""
    return actions[idx]
