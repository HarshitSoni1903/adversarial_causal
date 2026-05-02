"""World: tracks per-dyad history and provides cross-pair observations.

Two independent edge flags control information flow:
  ii_edge — investors observe each other (cross-investor GRU update).
  aa_edge — trustees observe each other (cross-pair window in DQN state).

The World stores per-dyad (invest, repay) history and the previous-round
(action_onehot, repay_prop) for the snapshot-safe cross-investor update.
Cross-step inputs come from World, not from the other investor's cache.
"""

import numpy as np


class World:

    def __init__(
        self,
        ii_edge: int,
        aa_edge: int,
        observation_depth: int,
        dyad_pairs: list[tuple[str, str]],
        n_actions: int,
    ) -> None:
        self.ii_edge = ii_edge
        self.aa_edge = aa_edge
        self.observation_depth = observation_depth
        self.n_dyads = len(dyad_pairs)
        self.n_actions = n_actions
        self.history: list[list[tuple[float, float]]] = [[] for _ in range(self.n_dyads)]
        self._last_ah: list[np.ndarray] = [
            np.zeros(n_actions, dtype=np.float32) for _ in range(self.n_dyads)
        ]
        self._last_rp: list[float] = [0.0] * self.n_dyads

    def reset(self) -> None:
        for k in range(self.n_dyads):
            self.history[k] = []
            self._last_ah[k] = np.zeros(self.n_actions, dtype=np.float32)
            self._last_rp[k] = 0.0

    def record_dyad_step(
        self,
        k: int,
        invest: float,
        repay: float,
        action_oh: np.ndarray,
        repay_prop: float,
    ) -> None:
        self.history[k].append((invest, repay))
        self._last_ah[k] = action_oh.copy()
        self._last_rp[k] = repay_prop

    def get_other_pair_window(self, k: int) -> np.ndarray:
        """Last observation_depth (invest, repay) pairs from the other dyad, zero-padded.

        Returns shape (2 * observation_depth,).
        """
        other_k = 1 - k
        d = self.observation_depth
        hist = self.history[other_k]
        window = hist[-d:]
        pad_len = d - len(window)
        padded = [(0.0, 0.0)] * pad_len + list(window)
        result: list[float] = []
        for inv, rep in padded:
            result.extend([inv, rep])
        return np.array(result, dtype=np.float32)

    def get_other_pair_last_action_repay(self, k: int) -> tuple[np.ndarray, float]:
        """Previous-round (action_onehot, repay_prop) from the other dyad.

        Returns zeros at t=0 (before any round is recorded).
        """
        other_k = 1 - k
        return self._last_ah[other_k].copy(), self._last_rp[other_k]
