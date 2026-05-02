"""Agent factory: creates adversary agents from config.

State dimension formula (two-dyad mode):
  base  = rnn_hidden(5) + policy_vec(5) + action_onehot(5) + round_norm(1) = 16
  flags = ii_edge(1) + aa_edge(1) = 2
  cross = 2 * observation_depth if aa_edge else 0
  total = 18 (aa_edge=0)  or  26 (aa_edge=1, obs_depth=4)
"""

from agents.adversary import DQNAdversary
from agents.base import BaseAgent
from agents.random_agent import RandomAgent


def compute_state_dim(config: dict, aa_edge: int | None = None) -> int:
    """Compute adversary state dimension from config."""
    rnn_hidden = config["behavioral_rnn"]["hidden_size"]
    n_actions = config["behavioral_rnn"]["n_actions"]
    base = rnn_hidden + n_actions + n_actions + 1
    flags = 2
    if aa_edge is None:
        aa_edge = config["edges"]["aa_edge"]
    obs_depth = config["game"]["observation_depth"]
    cross = 2 * obs_depth if aa_edge else 0
    return base + flags + cross


def create_agent(agent_cfg: dict, full_config: dict, state_dim: int) -> BaseAgent:
    agent_type = agent_cfg["type"]
    name = agent_cfg["name"]
    if agent_type == "random":
        return RandomAgent(name, "random", full_config)
    elif agent_type in ("max", "fair"):
        return DQNAdversary(name, agent_type, full_config, state_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")
