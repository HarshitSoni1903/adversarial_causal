"""Agent factory: creates adversary agents from config.

State dimension formula:
  base = rnn_hidden(5) + policy_vec(5) + investor_action_onehot(5) + round_norm(1) = 16
  W1/N>1 extension = world.others_obs_dim()
  total = base + others_obs_dim
"""

from agents.adversary import DQNAdversary
from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from world import World


def compute_state_dim(config: dict, world: World) -> int:
    """Compute adversary state dimension: 16 base + others_obs extension."""
    rnn_hidden = config["behavioral_rnn"]["hidden_size"]
    n_actions = config["behavioral_rnn"]["n_actions"]
    # rnn_hidden + policy_vec + action_onehot + round_norm
    base = rnn_hidden + n_actions + n_actions + 1
    return base + world.others_obs_dim()


def create_agent(agent_cfg: dict, full_config: dict, state_dim: int) -> BaseAgent:
    agent_type = agent_cfg["type"]
    name = agent_cfg["name"]
    if agent_type == "random":
        return RandomAgent(name, "random", full_config)
    elif agent_type in ("max", "fair"):
        return DQNAdversary(name, agent_type, full_config, state_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")


def create_all_agents(config: dict, state_dim: int) -> list[BaseAgent]:
    return [
        create_agent(a_cfg, config, state_dim)
        for a_cfg in config["game"]["agents"]
    ]
