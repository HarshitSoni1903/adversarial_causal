"""Random adversary agent: uniform random repayment, no learning."""

import random

import numpy as np

from agents.base import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, name: str, policy: str, config: dict) -> None:
        super().__init__(name, "random", config)
        self.repayment_actions: list[float] = config["adversary"].get(
            "repayment_actions", [0.0, 0.25, 0.50, 0.75, 1.0],
        )

    def act(self, state: np.ndarray) -> float:
        return random.choice(self.repayment_actions)

    def observe(self, reward: float, done: bool) -> None:
        pass
