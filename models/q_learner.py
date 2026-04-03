"""Generic DQN agent with experience replay and target network.

Self-contained RL module — no game logic, no imports from agents/ or world/.
Implements the 5-method interface (select_action, store_transition, update,
save, load) so it can be swapped for A2C/PPO later by writing a new file
with the same methods.
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DQNNetwork(nn.Module):
    """Fully connected Q-network."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_layers: list[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QLearner:
    """DQN with experience replay buffer and periodic target network updates."""

    def __init__(
        self, state_dim: int, action_dim: int, config: dict,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        hidden_layers: list[int] = config.get("hidden_layers", [128, 128])
        self.gamma: float = config.get("gamma", 0.99)
        self.lr: float = config.get("lr", 1e-4)
        self.batch_size: int = config.get("batch_size", 64)
        self.target_update_freq: int = config.get("target_update_freq", 1000)
        self.epsilon: float = config.get("epsilon_start", 0.2)
        self.epsilon_decay: float = config.get("epsilon_decay", 0.9999)
        self.epsilon_min: float = config.get("epsilon_min", 0.01)
        buf_size: int = config.get("replay_buffer_size", 400000)

        self.policy_net = _DQNNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = _DQNNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self._buffer_list: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._buffer_max: int = buf_size
        self._buffer_idx: int = 0
        self.step_count = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = (state, action, reward, next_state, done)
        if len(self._buffer_list) < self._buffer_max:
            self._buffer_list.append(transition)
        else:
            self._buffer_list[self._buffer_idx] = transition
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_max

    def update(self) -> float | None:
        """Sample a batch, compute TD loss, and step. Returns loss or None if buffer too small.

        Does NOT decay epsilon — call decay_epsilon() once per episode instead.
        """
        buf_len = len(self._buffer_list)
        if buf_len < self.batch_size:
            return None

        indices = np.random.randint(0, buf_len, size=self.batch_size)
        batch = [self._buffer_list[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(ns).max(1).values
            targets = r + self.gamma * q_next * (1.0 - d)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        """Decay epsilon once per episode. Call at the end of each training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, path)

    def load(self, path: str) -> None:
        """Load saved state. Sets epsilon to epsilon_min for evaluation."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = self.epsilon_min
        self.step_count = ckpt.get("step_count", 0)
