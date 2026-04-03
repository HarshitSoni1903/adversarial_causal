"""Abstract base class for all adversary agents.

Defines the interface that game.py expects: act() receives observation
components separately, observe() receives the reward signal, and reset()
prepares the agent for a new episode.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):

    def __init__(self, name: str, policy: str, config: dict) -> None:
        self.name = name
        self.policy = policy
        self.config = config
        self.cumulative_reward = 0.0

    @abstractmethod
    def act(
        self,
        own_obs: np.ndarray,
        others_obs: np.ndarray,
        rnn_hidden: np.ndarray,
        round_scaled: float,
    ) -> float:
        """Return repayment proportion in [0.0, 1.0]."""

    @abstractmethod
    def observe(self, reward: float, done: bool) -> None:
        """Receive reward after acting. done=True on final step."""

    def reset(self) -> None:
        """Reset for a new episode."""
        self.cumulative_reward = 0.0
