"""BehavioralRNN: GRU learner model from Dezfouli et al. (2020).

Architecture (faithful to paper):
  - Input: 6-dim = prev_action_onehot(5) + prev_repay_prop(1)
  - Prepend zero dummy so round-0 action is in the training loss
  - GRU hidden_size=5, dropout=0.2, linear head → 5 action logits

Two usage modes:
  Training  : forward(x)  — x is (batch, n_rounds, 6), returns per-step logits
              (batch, n_rounds, n_actions). Loss is summed over all time steps.
  Simulation: step_forward(h_prev, action_onehot, repay_prop) — single incremental
              GRU step, returns (h_new, policy_vec). Called once per agent per round.
"""

import torch
import torch.nn as nn
from torch import Tensor


class BehavioralRNN(nn.Module):

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 5,
        n_actions: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        """Full-sequence forward for training.

        Args:
            x: (batch, seq_len, input_size=6)

        Returns:
            logits: (batch, seq_len, n_actions)
        """
        h, _ = self.rnn(x)                     # (batch, seq_len, hidden_size)
        h = self.dropout(h)
        return self.head(h)                     # (batch, seq_len, n_actions)

    @torch.no_grad()
    def step_forward(
        self,
        h_prev: Tensor,
        action_onehot: Tensor,
        repay_prop: float,
    ) -> tuple[Tensor, Tensor]:
        """Single incremental GRU step for simulation.

        Called once per agent per round with the PREVIOUS round's features
        (or zeros at t=0). Returns the pre-decision hidden state and the
        action probability distribution derived from it.

        Args:
            h_prev: (1, 1, hidden_size) — previous hidden state
            action_onehot: (n_actions,) tensor — one-hot of previous action
            repay_prop: float — previous repayment proportion

        Returns:
            h_new: (1, 1, hidden_size)
            policy_vec: (n_actions,) softmax distribution
        """
        inp = torch.cat([
            action_onehot.float(),
            torch.tensor([repay_prop], dtype=torch.float32, device=h_prev.device),
        ]).unsqueeze(0).unsqueeze(0)            # (1, 1, 6)
        _, h_new = self.rnn(inp, h_prev)
        # Dropout is disabled during eval/no_grad; apply linear head directly
        policy_vec = torch.softmax(
            self.head(h_new.squeeze(0)),        # (1, n_actions)
            dim=-1,
        ).squeeze(0)                            # (n_actions,)
        return h_new, policy_vec

    def predict_probs(self, x: Tensor) -> Tensor:
        """Return per-step action probabilities (batch, seq_len, n_actions)."""
        return torch.softmax(self.forward(x), dim=-1)

    def freeze(self) -> None:
        """Freeze all parameters and switch to eval mode."""
        self.requires_grad_(False)
        self.eval()


def load_behavioral_rnn(checkpoint_path: str, device: str = "cpu") -> BehavioralRNN:
    """Load a trained BehavioralRNN from checkpoint, freeze it, and return."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BehavioralRNN(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        n_actions=ckpt["n_actions"],
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.freeze()
    return model.to(device)
