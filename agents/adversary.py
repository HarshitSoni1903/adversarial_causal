"""DQN-based adversary agent (MAX and FAIR policies).

State is pre-built by game.py as a flat numpy array:
  [rnn_hidden(5), policy_vec(5), investor_action_onehot(5), round_norm(1), others_obs(?)]
  = 16-dim at N=1 / W0; wider with W1 + N>1.

Reward shaping:
  MAX : rl_reward = investment_multiplied - repayment  (per-step profit)
  FAIR: rl_reward = 0 every step; terminal = -|agent_total - investor_from_i_total|

Delayed transition: state_t is stored with reward_{t} when state_{t+1} arrives.
"""

from __future__ import annotations

import numpy as np

from agents.base import BaseAgent
from models.q_learner import QLearner


class DQNAdversary(BaseAgent):

    def __init__(
        self,
        name: str,
        policy: str,
        config: dict,
        state_dim: int,
    ) -> None:
        super().__init__(name, policy, config)
        adv_cfg = config["adversary"]
        self.repayment_actions: list[float] = adv_cfg["repayment_actions"]
        self.q_learner = QLearner(state_dim, len(self.repayment_actions), adv_cfg)
        self.eval_mode: bool = False

        self._prev_state: np.ndarray | None = None
        self._prev_action: int | None = None
        self._prev_reward: float = 0.0
        self._update_counter: int = 0
        # FAIR tracking: bilateral episode totals
        self._episode_agent_total: float = 0.0
        self._episode_investor_from_i: float = 0.0

    def act(self, state: np.ndarray) -> float:
        """Select repayment proportion given pre-built state vector."""
        if not self.eval_mode and self._prev_state is not None:
            self.q_learner.store_transition(
                self._prev_state, self._prev_action, self._prev_reward,
                state, False,
            )
            if self._update_counter % 10 == 0:
                self.q_learner.update()
            self._update_counter += 1

        action_idx = self.q_learner.select_action(state, greedy=self.eval_mode)
        self._prev_state = state.copy()
        self._prev_action = action_idx
        self._prev_reward = 0.0
        return self.repayment_actions[action_idx]

    def observe(self, reward: float, done: bool) -> None:
        """Receive per-step agent profit and compute RL signal.

        reward = investment_multiplied - repayment (adversary's gross profit this round).
        """
        if self.policy == "max":
            rl_reward = reward
        elif self.policy == "fair":
            self._episode_agent_total += reward
            rl_reward = (
                -abs(self._episode_agent_total - self._episode_investor_from_i)
                if done else 0.0
            )
        else:
            rl_reward = reward

        if done:
            if not self.eval_mode and self._prev_state is not None:
                terminal_state = np.zeros_like(self._prev_state)
                self.q_learner.store_transition(
                    self._prev_state, self._prev_action, rl_reward, terminal_state, True,
                )
                if self._update_counter % 10 == 0:
                    self.q_learner.update()
                self._update_counter += 1
            self._prev_state = None
            self._prev_action = None
        else:
            self._prev_reward = rl_reward

    def accumulate_investor_reward(self, investor_from_i: float) -> None:
        """Called by game.py with investor's per-round earnings from this agent.

        investor_from_i = repayment - investment (bilateral net).
        Accumulated for FAIR terminal reward computation.
        """
        self._episode_investor_from_i += investor_from_i

    def set_eval_mode(self, mode: bool = True) -> None:
        self.eval_mode = mode

    def reset(self) -> None:
        super().reset()
        self._prev_state = None
        self._prev_action = None
        self._prev_reward = 0.0
        self._update_counter = 0
        self._episode_agent_total = 0.0
        self._episode_investor_from_i = 0.0

    def save(self, path: str | None = None) -> None:
        self.q_learner.save(path or self._save_path())

    def load(self, path: str | None = None) -> None:
        self.q_learner.load(path or self._save_path())

    def _save_path(self) -> str:
        for d in self.config["game"]["dyads"]:
            t = d["trustee"]
            if t["name"] == self.name:
                path = t["save_path"]
                if "{condition}" in path:
                    path = path.replace("{condition}", self.config.get("_condition", ""))
                return path
        return f"checkpoints/{self.name}.pt"
