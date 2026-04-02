"""BehavioralRNN: GRU learner model from Dezfouli et al. (2020).

Supervised model trained to predict a human investor's next discretized
action given the sequence of past rounds. After training, this model is
frozen and used as the investor surrogate in adversary training and
simulation. Model definition only — no training logic here.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class BehavioralRNN(nn.Module):

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 16,
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

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Return logits of shape (batch, n_actions)."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h = self.dropout(h_n[-1])
        return self.head(h)

    def get_hidden_state(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Return raw GRU hidden state h_n of shape (batch, hidden_size).

        No dropout or linear head — this vector is passed to the adversary
        as part of its RL state.
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        return h_n[-1]

    def predict_probs(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Return action probabilities of shape (batch, n_actions)."""
        return torch.softmax(self.forward(x, lengths), dim=-1)

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
        n_actions=len(ckpt["actions"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.freeze()
    return model.to(device)
