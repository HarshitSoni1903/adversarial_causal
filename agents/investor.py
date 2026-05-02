"""Frozen BehavioralRNN investor: single-trustee GRU state management.

One RNNInvestor instance per dyad. State is reset each episode.

Per-round call order (enforced by game.py):
  1. decay()         — apply trust decay h ← γ·h
  2. cross_step()    — optional ii_edge GRU update from other dyad
  3. act()           — self GRU step, sample action, cache state info
  4. observe_outcome() — store actual investment and repay_prop for next round
"""

from __future__ import annotations

import numpy as np
import torch

from models.behavioral_rnn import load_behavioral_rnn
from utils import bucket_investment


class RNNInvestor:

    def __init__(self, config: dict, trustee_name: str, device: str = "cpu") -> None:
        rnn_cfg = config["behavioral_rnn"]
        self.model = load_behavioral_rnn(rnn_cfg["save_path"], device=device)
        self.device = device
        self.action_values: list[int] = rnn_cfg["action_values"]
        self.n_actions: int = rnn_cfg["n_actions"]
        self.inference_sample: bool = rnn_cfg.get("inference_sample", True)
        self.trust_decay: float = float(rnn_cfg.get("trust_decay", 1.0))
        if not (0.0 < self.trust_decay <= 1.0):
            raise ValueError(f"trust_decay must be in (0, 1], got {self.trust_decay}")
        self.bucketing_rule: str = config["data"]["bucketing_rule"]
        self.trustee_name = trustee_name
        self.reset()

    def reset(self) -> None:
        hs = self.model.hidden_size
        self._h = torch.zeros(1, 1, hs, device=self.device)
        self._prev_ah = torch.zeros(self.n_actions, dtype=torch.float32)
        self._prev_rp: float = 0.0
        self._h_predecision = np.zeros(hs, dtype=np.float32)
        self._policy_vec = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float32)
        self._action_onehot = np.zeros(self.n_actions, dtype=np.float32)

    def decay(self) -> None:
        if self.trust_decay < 1.0:
            self._h = self._h * self.trust_decay

    def cross_step(self, other_action_oh: np.ndarray, other_repay_prop: float) -> None:
        ah = torch.tensor(other_action_oh, dtype=torch.float32, device=self.device)
        h_new, _ = self.model.step_forward(self._h, ah, float(other_repay_prop))
        self._h = h_new

    def act(self) -> float:
        """Run one GRU step, sample action, cache state info. Returns desired investment."""
        h_new, policy_vec = self.model.step_forward(
            self._h,
            self._prev_ah.to(self.device),
            self._prev_rp,
        )
        self._h = h_new

        h_arr = h_new.squeeze().cpu().numpy()
        p_arr = policy_vec.cpu().numpy()

        if self.inference_sample:
            action_idx = int(torch.multinomial(policy_vec, 1).item())
        else:
            action_idx = int(policy_vec.argmax().item())

        ah = np.zeros(self.n_actions, dtype=np.float32)
        ah[action_idx] = 1.0

        self._h_predecision = h_arr
        self._policy_vec = p_arr
        self._action_onehot = ah

        return float(self.action_values[action_idx])

    def observe_outcome(self, actual_investment: float, repay_prop: float) -> None:
        """Store actual round outcomes as inputs for the next GRU step."""
        idx = bucket_investment(int(round(actual_investment)), self.bucketing_rule)
        ah = torch.zeros(self.n_actions, dtype=torch.float32)
        ah[idx] = 1.0
        self._prev_ah = ah
        self._prev_rp = float(np.clip(repay_prop, 0.0, 1.0))

    def get_rnn_info(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (h_predecision, policy_vec, action_onehot) cached after act()."""
        return (
            self._h_predecision.copy(),
            self._policy_vec.copy(),
            self._action_onehot.copy(),
        )
