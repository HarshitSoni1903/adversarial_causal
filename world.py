"""World: constructs observation vectors for adversary agents (multi-agent extension).

At N=1 the others observation is always empty, so W0 and W1 produce identical
adversary states. This is a verified invariant of the N=1 Dezfouli replication.

At N>=2:
  W0 (mode=0): others observation zeroed — adversaries play independently.
  W1 (mode=1): others observation contains real values — adversaries can see
               each other's last-d-round history.

The world is part of the multi-agent EXTENSION layer. Core Dezfouli replication
(N=1) does not use any world observation.
"""

import numpy as np


class World:

    def __init__(
        self,
        mode: int,
        observation_depth: int,
        agent_names: list[str],
    ) -> None:
        self.mode = mode                        # 0=W0, 1=W1
        self.communication = (mode == 1)        # backward-compat alias
        self.observation_depth = observation_depth
        self.agent_names = agent_names
        self.history: dict[str, list[tuple[float, float]]] = {
            name: [] for name in agent_names
        }

    def reset(self) -> None:
        for name in self.agent_names:
            self.history[name] = []

    def record_step(
        self, agent_name: str, investment: float, repayment: float,
    ) -> None:
        self.history[agent_name].append((investment, repayment))

    def get_others_observation(self, agent_name: str) -> np.ndarray:
        """Last d (investment, repayment) pairs per other agent, zeroed in W0.

        Shape: (observation_depth * 2 * (N-1),)
        At N=1: empty array (0-dim).
        """
        d = self.observation_depth
        result: list[float] = []
        for name in self.agent_names:
            if name == agent_name:
                continue
            hist = self.history[name]
            window = hist[-d:] if self.communication else []
            pad_len = d - len(window)
            padded = [(0.0, 0.0)] * pad_len + list(window)
            for inv, rep in padded:
                result.extend([inv, rep])
        return np.array(result, dtype=np.float32)

    def others_obs_dim(self, agent_name: str | None = None) -> int:
        """Dimension of the others observation for any agent (same for all)."""
        n_others = len(self.agent_names) - 1
        return self.observation_depth * 2 * n_others

    def total_obs_dim(self) -> int:
        """Total others-observation dimension (same for all agents)."""
        return self.others_obs_dim()
