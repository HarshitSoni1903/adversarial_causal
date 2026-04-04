"""Frozen RNN investor wrapper with optional Q-learner for strategic allocation.

When learns=False (default): the frozen BehavioralRNN makes all investment
decisions independently per agent. No learning occurs. This is the original
Dezfouli surrogate investor.

When learns=True: the frozen RNN still runs every step (producing hidden
states and behavioral features), but a Q-learner makes the actual investment
decisions using cross-agent information. The RNN provides features; the
Q-learner provides strategy. Neither modifies the other.

Q-learner state vector for deciding investment in agent_i:
  [rnn_hidden_all (64*k), prev_returns (k), cumulative_returns (k),
   wealth_scaled (1), round_scaled (1)]
  state_dim = 64*k + 2*k + 2

Uses incremental GRU evaluation: each call to act() feeds only the current
step through the GRU (with the cached hidden state), O(1) per step.

Reward signal: the total investor reward for the round (sum across all agents),
delivered via receive_round_reward() after all agents have acted. Within a
round, intermediate transitions get reward 0 (sparse-reward MDP).

Actions are returned in TRAINING scale (0-20). The game loop maps to the
simulation's endowment and clamps to available wealth.
"""

from __future__ import annotations

import numpy as np
import torch

from models.behavioral_rnn import load_behavioral_rnn
from models.q_learner import QLearner
from utils import get_checkpoint_dir


class RNNInvestor:

    def __init__(
        self,
        config: dict,
        agent_names: list[str],
        device: str = "cpu",
    ) -> None:
        rnn_cfg = config["behavioral_rnn"]
        data_cfg = config["data"]

        self.model = load_behavioral_rnn(rnn_cfg["save_path"], device=device)
        self.device = device
        self.rnn_actions: list[int] = rnn_cfg["actions"]
        self.original_endowment: int = data_cfg["original_endowment"]
        self.original_rounds: int = data_cfg["original_rounds"]
        self.multiplier: int = data_cfg["multiplier"]
        self.agent_names = agent_names
        self.config = config

        self._tracking: dict[str, dict] = {}
        self._hidden_cache: dict[str, torch.Tensor] = {}

        # --- Optional Q-learner ---
        self.learns: bool = config.get("investor", {}).get("learns", False)
        self.q_learner: QLearner | None = None
        self.eval_mode: bool = False

        if self.learns:
            inv_cfg = config["investor"]
            self.ql_actions: list[int] = inv_cfg["actions"]
            k = len(agent_names)
            rnn_hidden_size = rnn_cfg["hidden_size"]
            state_dim = rnn_hidden_size * k + 2 * k + 2
            action_dim = len(self.ql_actions)
            self.q_learner = QLearner(state_dim, action_dim, inv_cfg, device)

            self._initial_endowment: float = float(config["game"]["endowment"])
            self._current_wealth: float = self._initial_endowment

            self._prev_returns: dict[str, float] = {}
            self._cumulative_returns: dict[str, float] = {}
            self._prev_state: np.ndarray | None = None
            self._prev_action: int | None = None
            self._prev_reward: float = 0.0
            self._update_counter: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-agent tracking and hidden state caches."""
        for name in self.agent_names:
            self._tracking[name] = {
                "prev_repay_prop": None,
                "prev_action": None,
                "prev_reward": None,
                "step_count": 0,
            }
            self._hidden_cache[name] = torch.zeros(
                1, 1, self.model.hidden_size, device=self.device,
            )

        if self.learns:
            self._current_wealth = self._initial_endowment
            self._prev_returns = {n: 0.0 for n in self.agent_names}
            self._cumulative_returns = {n: 0.0 for n in self.agent_names}
            self._prev_state = None
            self._prev_action = None
            self._prev_reward = 0.0
            self._update_counter = 0

    # ------------------------------------------------------------------
    # RNN encoding / stepping
    # ------------------------------------------------------------------

    def _encode_step(self, agent_name: str) -> list[float]:
        """Encode current step for an agent, rescaled to training distribution."""
        trk = self._tracking[agent_name]
        round_scale = max(self.original_rounds - 1, 1)
        endow = self.original_endowment

        round_scaled = min(trk["step_count"] / round_scale, 1.0)

        if trk["prev_repay_prop"] is None:
            return [round_scaled, 0.0, 0.0, 0.0]

        prev_repay_prop = float(np.clip(trk["prev_repay_prop"], 0.0, 1.0))
        prev_invest_scaled = min(trk["prev_action"] / endow, 1.0)
        prev_reward_scaled = float(np.clip(trk["prev_reward"] / endow, -1.0, 1.0))

        return [round_scaled, prev_repay_prop, prev_invest_scaled, prev_reward_scaled]

    @torch.no_grad()
    def _step_rnn(self, agent_name: str) -> None:
        """Run one incremental GRU step to update hidden state cache."""
        enc = self._encode_step(agent_name)
        x = torch.tensor([[enc]], dtype=torch.float32, device=self.device)
        h_prev = self._hidden_cache[agent_name]
        _, h_new = self.model.rnn(x, h_prev)
        self._hidden_cache[agent_name] = h_new

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        agent_name: str,
        *,
        round_num: int = 0,
        max_rounds: int = 1,
    ) -> float:
        """Decide investment for a specific agent (training scale 0-20).

        When learns=False: frozen RNN samples an action.
        When learns=True: frozen RNN updates hidden state, Q-learner picks action.
        """
        with torch.no_grad():
            self._step_rnn(agent_name)

        if not self.learns:
            with torch.no_grad():
                h = self.model.dropout(self._hidden_cache[agent_name][-1])
                logits = self.model.head(h)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                action_idx = int(torch.multinomial(probs, 1).item())
            return float(self.rnn_actions[action_idx])

        # --- Q-learner mode ---
        state = self._build_ql_state(round_num, max_rounds)

        if not self.eval_mode and self._prev_state is not None:
            self.q_learner.store_transition(
                self._prev_state, self._prev_action,
                self._prev_reward, state, False,
            )
            if self._update_counter % 10 == 0:
                self.q_learner.update()
            self._update_counter += 1
            self._prev_reward = 0.0

        action_idx = self.q_learner.select_action(state, greedy=self.eval_mode)
        investment = float(self.ql_actions[action_idx])

        self._prev_state = state
        self._prev_action = action_idx

        return investment

    def _build_ql_state(
        self, round_num: int, max_rounds: int,
    ) -> np.ndarray:
        """Build cross-agent state vector for the Q-learner."""
        all_hiddens = [
            self._hidden_cache[n].squeeze().cpu().numpy()
            for n in self.agent_names
        ]
        prev_rets = np.array(
            [self._prev_returns[n] for n in self.agent_names], dtype=np.float32,
        )
        cum_rets = np.array(
            [self._cumulative_returns[n] for n in self.agent_names], dtype=np.float32,
        )
        wealth_scaled = np.float32(self._current_wealth / self._initial_endowment)
        round_scaled = np.float32(round_num / max(max_rounds - 1, 1))

        return np.concatenate([
            *all_hiddens, prev_rets, cum_rets,
            [wealth_scaled, round_scaled],
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Observation / feedback from game
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_hidden_state(self, agent_name: str) -> np.ndarray:
        """Return cached GRU hidden state as 1D numpy array (hidden_size,)."""
        h = self._hidden_cache[agent_name]
        return h.squeeze().cpu().numpy()

    def observe_outcome(
        self,
        agent_name: str,
        investment: float,
        repayment: float,
        reward: float,
    ) -> None:
        """Update tracking after a round resolves. Values are in simulation scale."""
        trk = self._tracking[agent_name]
        inv_mult = investment * self.multiplier
        trk["prev_repay_prop"] = repayment / inv_mult if inv_mult > 0 else 0.0
        trk["prev_action"] = investment
        trk["prev_reward"] = reward
        trk["step_count"] += 1

        if self.learns:
            self._prev_returns[agent_name] = reward
            self._cumulative_returns[agent_name] += reward
            self._current_wealth += reward

    def receive_round_reward(self, total_round_reward: float) -> None:
        """Called by game.py after all agents act in a round.

        Sets the reward for the Q-learner's delayed transition storage.
        The transition crossing the round boundary gets this reward;
        within-round transitions get 0 (sparse-reward MDP).
        """
        if self.learns:
            self._prev_reward = total_round_reward

    def observe_done(self) -> None:
        """Called at episode end to store the terminal Q-learner transition."""
        if self.learns and not self.eval_mode and self._prev_state is not None:
            terminal_state = np.zeros_like(self._prev_state)
            self.q_learner.store_transition(
                self._prev_state, self._prev_action, self._prev_reward,
                terminal_state, True,
            )
            if self._update_counter % 10 == 0:
                self.q_learner.update()
            self._update_counter += 1
            self._prev_state = None
            self._prev_action = None

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def set_eval_mode(self, mode: bool = True) -> None:
        """Toggle evaluation mode: greedy actions, no learning."""
        if self.learns:
            self.eval_mode = mode

    def decay_epsilon(self) -> None:
        """Decay Q-learner epsilon once per episode."""
        if self.learns:
            self.q_learner.decay_epsilon()

    def save(self, path: str | None = None) -> None:
        """Save Q-learner weights."""
        if self.learns:
            save_name = self.config["investor"]["save_name"]
            path = path or f"{get_checkpoint_dir(self.config)}/{save_name}"
            self.q_learner.save(path)

    def load(self, path: str | None = None) -> None:
        """Load Q-learner weights."""
        if self.learns:
            save_name = self.config["investor"]["save_name"]
            path = path or f"{get_checkpoint_dir(self.config)}/{save_name}"
            self.q_learner.load(path)
