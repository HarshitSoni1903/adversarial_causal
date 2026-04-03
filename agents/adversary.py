"""DQN-based adversary agent for both MAX and FAIR policies.

A single class handles both policy types. The only difference is the
reward signal used for Q-learning:

- MAX: rl_reward = agent's profit each step (investment_multiplied - repayment).
- FAIR: rl_reward = 0 every step except the final one, where
  rl_reward = -|agent_total - investor_total|.

Per-agent parameters in agent_cfg override global dqn defaults. To add a
new customizable parameter, just add it to the agent's config entry in YAML —
no code changes needed.

The game loop must call set_investor_reward() after each step so the FAIR
agent can track the investor's running total.
"""

import numpy as np

from agents.base import BaseAgent
from models.q_learner import QLearner


class DQNAdversary(BaseAgent):

    def __init__(
        self,
        name: str,
        policy: str,
        agent_cfg: dict,
        full_config: dict,
        state_dim: int,
        num_actions: int,
    ) -> None:
        super().__init__(name, policy, full_config)
        dqn_cfg = full_config["dqn"]

        self.repayment_actions: list[float] = agent_cfg.get(
            "repayment_actions", dqn_cfg.get(
                "repayment_actions", [0.0, 0.25, 0.50, 0.75, 1.0],
            ),
        )
        self.q_learner = QLearner(state_dim, num_actions, dqn_cfg)

        self._prev_state: np.ndarray | None = None
        self._prev_action: int = 0
        self._pending_reward: float = 0.0
        self._episode_agent_total = 0.0
        self._episode_investor_total = 0.0
        self._last_investment_received = 0.0

    def act(
        self,
        own_obs: np.ndarray,
        others_obs: np.ndarray,
        rnn_hidden: np.ndarray,
        round_scaled: float,
    ) -> float:
        """Assemble state, select action, return repayment proportion."""
        state = np.concatenate([
            own_obs, others_obs, rnn_hidden, [round_scaled],
        ]).astype(np.float32)

        action_idx = self.q_learner.select_action(state)

        if self._prev_state is not None:
            self.q_learner.store_transition(
                self._prev_state, self._prev_action, self._pending_reward,
                state, False,
            )
            self.q_learner.update()

        self._prev_state = state
        self._prev_action = action_idx
        self._pending_reward = 0.0

        return self.repayment_actions[action_idx]

    def observe(self, reward: float, done: bool) -> None:
        """Compute policy-specific reward and store terminal transition if done."""
        if self.policy == "max":
            rl_reward = reward
        elif self.policy == "fair":
            self._episode_agent_total += reward
            if done:
                rl_reward = -abs(self._episode_agent_total - self._episode_investor_total)
            else:
                rl_reward = 0.0
        else:
            rl_reward = reward

        if done:
            if self._prev_state is not None:
                terminal_state = np.zeros_like(self._prev_state)
                self.q_learner.store_transition(
                    self._prev_state, self._prev_action, rl_reward,
                    terminal_state, True,
                )
                self.q_learner.update()
                self._prev_state = None
        else:
            self._pending_reward = rl_reward

    def set_investor_reward(self, investor_reward: float) -> None:
        """Called by game.py after each step so FAIR can track investor earnings."""
        self._episode_investor_total += investor_reward

    def set_investment_received(self, investment_multiplied: float) -> None:
        """Called by game.py before act() with the tripled investment amount."""
        self._last_investment_received = investment_multiplied

    def reset(self) -> None:
        super().reset()
        self._prev_state = None
        self._prev_action = 0
        self._pending_reward = 0.0
        self._episode_agent_total = 0.0
        self._episode_investor_total = 0.0
        self._last_investment_received = 0.0

    def save(self, path: str | None = None) -> None:
        path = path or f"{self.config['dqn']['save_dir']}{self.name}.pt"
        self.q_learner.save(path)

    def load(self, path: str | None = None) -> None:
        path = path or f"{self.config['dqn']['save_dir']}{self.name}.pt"
        self.q_learner.load(path)

    def set_greedy(self, greedy: bool = True) -> None:
        """Set epsilon to 0 for pure exploitation (simulation mode)."""
        if greedy:
            self.q_learner.epsilon = 0.0
