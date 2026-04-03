"""Random adversary agent: uniform random repayment, no learning."""

import random

import numpy as np

from agents.base import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, name: str, agent_cfg: dict, full_config: dict) -> None:
        super().__init__(name, "random", full_config)
        self.repayment_actions: list[float] = agent_cfg.get(
            "repayment_actions", full_config["dqn"].get(
                "repayment_actions", [0.0, 0.25, 0.50, 0.75, 1.0],
            ),
        )

    def act(
        self,
        own_obs: np.ndarray,
        others_obs: np.ndarray,
        rnn_hidden: np.ndarray,
        round_scaled: float,
    ) -> float:
        """Uniform random choice from repayment_actions."""
        return random.choice(self.repayment_actions)

    def observe(self, reward: float, done: bool) -> None:
        pass
