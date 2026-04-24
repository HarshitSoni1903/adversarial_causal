"""Abstract base class for all adversary agents.

The game loop constructs the full state vector and passes it as a single
numpy array. This removes state-building concerns from individual agents and
ensures round_norm is present during both training and simulation.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):

    def __init__(self, name: str, policy: str, config: dict) -> None:
        self.name = name
        self.policy = policy
        self.config = config
        self.cumulative_reward: float = 0.0

    @abstractmethod
    def act(self, state: np.ndarray) -> float:
        """Return repayment proportion in [0.0, 1.0] given a pre-built state vector."""

    @abstractmethod
    def observe(self, reward: float, done: bool) -> None:
        """Receive reward after acting. done=True on the final round."""

    def reset(self) -> None:
        """Reset episode state (cumulative reward, Q-learner buffers, etc.)."""
        self.cumulative_reward = 0.0
