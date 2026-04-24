"""Frozen BehavioralRNN investor: per-agent GRU state management.

The investor is a surrogate for the human participant in Dezfouli et al. (2020).
It is frozen (no learning) and drives investment decisions via incremental GRU steps.

Per-agent state maintained across rounds:
  _h[name]       : (1, 1, hidden_size) GRU hidden state
  _prev_ah[name] : (n_actions,) one-hot of last action (zeros at t=0)
  _prev_rp[name] : float repayment proportion from last round (0.0 at t=0)

act(agent_name) caches (h_predecision, policy_vec, action_onehot) between
act() and observe_outcome(). game.py calls get_rnn_info() to build adversary states.
"""

from __future__ import annotations

import numpy as np
import torch

from models.behavioral_rnn import load_behavioral_rnn
from utils import bucket_investment


class RNNInvestor:

    def __init__(self, config: dict, agent_names: list[str], device: str = "cpu") -> None:
        rnn_cfg = config["behavioral_rnn"]
        self.model = load_behavioral_rnn(rnn_cfg["save_path"], device=device)
        self.device = device
        self.action_values: list[int] = rnn_cfg["action_values"]
        self.n_actions: int = rnn_cfg["n_actions"]
        self.inference_sample: bool = rnn_cfg.get("inference_sample", True)
        self.bucketing_rule: str = config["data"]["bucketing_rule"]
        self.agent_names = agent_names

        # Per-agent GRU state (reset each episode)
        self._h: dict[str, torch.Tensor] = {}
        self._prev_ah: dict[str, torch.Tensor] = {}
        self._prev_rp: dict[str, float] = {}
        # Cached after act(), read by game.py via get_rnn_info()
        self._h_predecision: dict[str, np.ndarray] = {}
        self._policy_vec: dict[str, np.ndarray] = {}
        self._action_onehot: dict[str, np.ndarray] = {}

        self.reset()

    def reset(self) -> None:
        hs = self.model.hidden_size
        zero_ah = torch.zeros(self.n_actions, dtype=torch.float32)
        for name in self.agent_names:
            self._h[name] = torch.zeros(1, 1, hs, device=self.device)
            self._prev_ah[name] = zero_ah.clone()
            self._prev_rp[name] = 0.0
            self._h_predecision[name] = np.zeros(hs, dtype=np.float32)
            self._policy_vec[name] = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float32)
            self._action_onehot[name] = np.zeros(self.n_actions, dtype=np.float32)

    def act(self, agent_name: str) -> float:
        """Run one GRU step, sample action, cache state info. Returns desired investment."""
        h_new, policy_vec = self.model.step_forward(
            self._h[agent_name],
            self._prev_ah[agent_name].to(self.device),
            self._prev_rp[agent_name],
        )
        self._h[agent_name] = h_new

        h_arr = h_new.squeeze().cpu().numpy()   # (hidden_size,)
        p_arr = policy_vec.cpu().numpy()         # (n_actions,)

        if self.inference_sample:
            action_idx = int(torch.multinomial(policy_vec, 1).item())
        else:
            action_idx = int(policy_vec.argmax().item())

        ah = np.zeros(self.n_actions, dtype=np.float32)
        ah[action_idx] = 1.0

        self._h_predecision[agent_name] = h_arr
        self._policy_vec[agent_name] = p_arr
        self._action_onehot[agent_name] = ah

        return float(self.action_values[action_idx])

    def observe_outcome(
        self, agent_name: str, actual_investment: float, repay_prop: float,
    ) -> None:
        """Store actual round outcomes as inputs for the next GRU step."""
        idx = bucket_investment(int(round(actual_investment)), self.bucketing_rule)
        ah = torch.zeros(self.n_actions, dtype=torch.float32)
        ah[idx] = 1.0
        self._prev_ah[agent_name] = ah
        self._prev_rp[agent_name] = float(np.clip(repay_prop, 0.0, 1.0))

    def get_rnn_info(self, agent_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (h_predecision, policy_vec, action_onehot) cached after act()."""
        return (
            self._h_predecision[agent_name].copy(),
            self._policy_vec[agent_name].copy(),
            self._action_onehot[agent_name].copy(),
        )

    def get_hidden_state(self, agent_name: str) -> np.ndarray:
        """Return current GRU hidden state as (hidden_size,) numpy array."""
        return self._h[agent_name].squeeze().cpu().detach().numpy().copy()

    def set_hidden_state(self, agent_name: str, h_arr: np.ndarray) -> None:
        """Set GRU hidden state from (hidden_size,) numpy array."""
        hs = self.model.hidden_size
        self._h[agent_name] = torch.tensor(
            h_arr, dtype=torch.float32, device=self.device,
        ).reshape(1, 1, hs)
