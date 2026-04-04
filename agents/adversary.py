"""DQN-based adversary agent for both MAX and FAIR policies.

A single class handles both policy types. The only difference is the
reward signal used for Q-learning:

- MAX: rl_reward = agent's profit each step (investment_multiplied - repayment).
- FAIR: rl_reward = 0 every step except the final one, where
  rl_reward = -|agent_total - investor_total|.

Learns online during the game via delayed transition storage:

  Step t:  act()     → receives state_t, picks action_t, caches them
  Step t:  observe() → receives reward_t, caches it
  Step t+1: act()    → receives state_{t+1}, NOW stores (state_t, action_t,
                        reward_t, state_{t+1}, False) and calls q_learner.update()

On the final step (done=True in observe()), the terminal transition is
stored immediately with done=True.

Per-agent parameters in agent_cfg override global dqn defaults.

The game loop must call set_investor_reward() after each step so the FAIR
agent can track the investor's running total.
"""

import numpy as np

from agents.base import BaseAgent
from models.q_learner import QLearner
from utils import get_checkpoint_dir


class DQNAdversary(BaseAgent):

    def __init__(
        self,
        name: str,
        policy: str,
        agent_cfg: dict,
        full_config: dict,
        state_dim: int,
        num_actions: int,
        run_id: str | None = None,
    ) -> None:
        super().__init__(name, policy, full_config)
        dqn_cfg = full_config["dqn"]
        self._run_id = run_id

        self.repayment_actions: list[float] = agent_cfg.get(
            "repayment_actions", dqn_cfg.get(
                "repayment_actions", [0.0, 0.25, 0.50, 0.75, 1.0],
            ),
        )
        self.q_learner = QLearner(state_dim, num_actions, dqn_cfg)
        self.eval_mode: bool = False

        self._prev_state: np.ndarray | None = None
        self._prev_action: int | None = None
        self._prev_reward: float = 0.0
        self._current_state: np.ndarray | None = None

        self._episode_agent_total = 0.0
        self._episode_investor_total = 0.0
        self._last_investment_received = 0.0
        self._update_counter: int = 0

    def act(
        self,
        own_obs: np.ndarray,
        others_obs: np.ndarray,
        rnn_hidden: np.ndarray,
        round_scaled: float,
    ) -> float:
        """Assemble state, select action, return repayment proportion.

        If a previous transition is pending (prev_state from step t-1),
        this call provides state_{t} as the next_state to complete and
        store that transition.
        """
        state = np.concatenate([
            own_obs, others_obs, rnn_hidden, [round_scaled],
        ]).astype(np.float32)

        if not self.eval_mode and self._prev_state is not None:
            self.q_learner.store_transition(
                self._prev_state, self._prev_action, self._prev_reward,
                state, False,
            )
            if self._update_counter % 10 == 0:
                self.q_learner.update()
            self._update_counter += 1

        action_idx = self.q_learner.select_action(state, greedy=self.eval_mode)

        self._current_state = state
        self._prev_state = state
        self._prev_action = action_idx
        self._prev_reward = 0.0

        return self.repayment_actions[action_idx]

    def observe(self, reward: float, done: bool) -> None:
        """Receive reward and compute policy-specific RL signal.

        Caches the reward for the delayed transition (stored in the next act()).
        On done=True, stores the terminal transition immediately.

        MAX reward (Dezfouli original):
        Pure profit per step: rl_reward = investment_multiplied - repayment.
        Trust-building emerges naturally from the discount factor — repaying
        more now leads to larger investments later, which means larger future
        profits. The Q-learner discovers this through temporal credit assignment
        without needing an artificial trust bonus.

        FAIR reward (bilateral):
        0 on all steps except terminal, where it's the negative absolute gap
        between THIS agent's total earnings and the investor's earnings FROM
        THIS AGENT ONLY (repayment - investment, summed). Each agent's fairness
        is evaluated on its own bilateral relationship with the investor.
        """
        if self.policy == "max":
            rl_reward = reward  # pure profit: investment_multiplied - repayment
        elif self.policy == "fair":
            self._episode_agent_total += reward
            if done:
                # Bilateral gap: agent's earnings vs investor's earnings from THIS agent only
                rl_reward = -abs(self._episode_agent_total - self._episode_investor_total)
            else:
                rl_reward = 0.0
        else:
            rl_reward = reward

        if done:
            if not self.eval_mode and self._prev_state is not None:
                terminal_state = np.zeros_like(self._prev_state)
                self.q_learner.store_transition(
                    self._prev_state, self._prev_action, rl_reward,
                    terminal_state, True,
                )
                if self._update_counter % 10 == 0:
                    self.q_learner.update()
                self._update_counter += 1
            self._prev_state = None
            self._prev_action = None
        else:
            self._prev_reward = rl_reward

    def set_investor_reward(self, investor_reward: float) -> None:
        """Called by game.py after each step with the investor's reward FROM THIS AGENT.

        investor_reward = repayment - investment (bilateral, not total).
        Used by FAIR to compute the bilateral earnings gap at episode end.
        """
        self._episode_investor_total += investor_reward

    def set_investment_received(self, investment_multiplied: float) -> None:
        """Called by game.py before act() with the tripled investment amount."""
        self._last_investment_received = investment_multiplied

    def set_eval_mode(self, mode: bool = True) -> None:
        """Toggle evaluation mode: greedy actions, no learning."""
        self.eval_mode = mode

    def reset(self) -> None:
        """Reset episode state. Q-learner weights persist across episodes."""
        super().reset()
        self._prev_state = None
        self._prev_action = None
        self._prev_reward = 0.0
        self._current_state = None
        self._episode_agent_total = 0.0
        self._episode_investor_total = 0.0
        self._last_investment_received = 0.0
        self._update_counter = 0

    def save(self, path: str | None = None) -> None:
        path = path or f"{get_checkpoint_dir(self.config, self._run_id)}/{self.name}.pt"
        self.q_learner.save(path)

    def load(self, path: str | None = None) -> None:
        path = path or f"{get_checkpoint_dir(self.config, self._run_id)}/{self.name}.pt"
        self.q_learner.load(path)
