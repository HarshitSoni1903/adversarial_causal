"""Frozen RNN investor wrapper for simulation.

Wraps the trained BehavioralRNN to act as the investor surrogate. This is
NOT an RL agent — it's a supervised model predicting what a human investor
would do, based on the sequence of past rounds.

The investor maintains a SEPARATE hidden state cache per adversary agent,
because it's playing k independent MRTT games simultaneously (one per
trustee). This matches how the RNN was trained: one investor vs one trustee.

Uses incremental GRU evaluation: each call to act() feeds only the current
step through the GRU (with the cached hidden state), producing identical
results to replaying the full sequence but in O(1) per step instead of O(t).

IMPORTANT: act() must be called before get_hidden_state() each round,
because get_hidden_state() returns the cached value updated by act().

Actions are returned in TRAINING scale (0-20). The game loop is responsible
for mapping to the simulation's endowment and clamping to available wealth.
"""

import numpy as np
import torch

from models.behavioral_rnn import load_behavioral_rnn


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
        self.actions: list[int] = rnn_cfg["actions"]
        self.original_endowment: int = data_cfg["original_endowment"]
        self.original_rounds: int = data_cfg["original_rounds"]
        self.multiplier: int = data_cfg["multiplier"]
        self.agent_names = agent_names

        self._tracking: dict[str, dict] = {}
        self._hidden_cache: dict[str, torch.Tensor] = {}
        self.reset()

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
    def act(self, agent_name: str) -> float:
        """Predict investment via incremental GRU step. Returns training scale (0-20).

        Updates the hidden state cache for this agent. Must be called before
        get_hidden_state() each round.
        """
        enc = self._encode_step(agent_name)
        x = torch.tensor([[enc]], dtype=torch.float32, device=self.device)
        h_prev = self._hidden_cache[agent_name]

        output, h_new = self.model.rnn(x, h_prev)
        self._hidden_cache[agent_name] = h_new

        h = self.model.dropout(h_new[-1])
        logits = self.model.head(h)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        action_idx = int(torch.multinomial(probs, 1).item())
        return float(self.actions[action_idx])

    @torch.no_grad()
    def get_hidden_state(self, agent_name: str) -> np.ndarray:
        """Return cached GRU hidden state as 1D numpy array of shape (hidden_size,).

        Returns the hidden state last updated by act(). Do NOT call this
        before act() in the same round — the cache won't reflect the
        current step yet.
        """
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
