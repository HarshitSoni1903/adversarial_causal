"""World: constructs observation vectors for adversary agents.

The world tracks game history and builds two observation components:

- own observation: the agent's last d steps of (investment, repayment),
  flattened to shape (d * 2,). This captures the agent's local interaction
  history, analogous to the h_t compression input from Dezfouli et al.

- others observation: the most recent (investment, repayment) pair from
  each other agent, in fixed agent_names order (skipping self). Shape is
  ((num_agents - 1) * 2,). Built dynamically by iterating the agent list,
  so adding agents to the YAML automatically adjusts the vector size.

The communication switch (W0/W1) controls whether the others vector
contains real values (W1) or is zeroed out (W0). The shape stays fixed
either way, so the DQN architecture doesn't change between world modes.

Components are returned separately so they can be logged independently
in the CSV export. The caller (agent or game loop) concatenates them
with rnn_hidden and round_number into the final DQN state vector.
"""

import numpy as np


class World:

    def __init__(
        self,
        communication: bool,
        observation_depth: int,
        agent_names: list[str],
    ) -> None:
        self.communication = communication
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

    def get_own_observation(self, agent_name: str) -> np.ndarray:
        """Agent's own last d (investment, repayment) pairs, flattened."""
        d = self.observation_depth
        hist = self.history[agent_name]
        window = hist[-d:]
        pad_len = d - len(window)
        padded = [(0.0, 0.0)] * pad_len + window
        return np.array(padded, dtype=np.float32).flatten()

    def get_others_observation(self, agent_name: str) -> np.ndarray:
        """Most recent (investment, repayment) per other agent, zeroed under W0."""
        pairs: list[tuple[float, float]] = []
        for name in self.agent_names:
            if name == agent_name:
                continue
            if self.communication and self.history[name]:
                pairs.append(self.history[name][-1])
            else:
                pairs.append((0.0, 0.0))

        if not pairs:
            return np.array([], dtype=np.float32)
        return np.array(pairs, dtype=np.float32).flatten()

    def get_full_observation(self, agent_name: str) -> dict[str, np.ndarray]:
        """Return own and others observations as separate vectors."""
        return {
            "own": self.get_own_observation(agent_name),
            "others": self.get_others_observation(agent_name),
        }

    def own_obs_dim(self) -> int:
        return self.observation_depth * 2

    def others_obs_dim(self) -> int:
        return (len(self.agent_names) - 1) * 2

    def total_obs_dim(self) -> int:
        """Total observation dim (own + others). Does NOT include rnn_hidden or round."""
        return self.own_obs_dim() + self.others_obs_dim()
