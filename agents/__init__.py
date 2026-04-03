"""Agent factory: creates agent instances from config entries.

Per-agent parameters in the agents list override global dqn defaults.
Each agent constructor receives its own agent_cfg dict to read overrides from.
"""

from agents.adversary import DQNAdversary
from agents.base import BaseAgent
from agents.random_agent import RandomAgent


def create_agent(
    agent_cfg: dict, full_config: dict, state_dim: int,
) -> BaseAgent:
    """Create a single agent from a config entry.

    agent_cfg is the per-agent dict (e.g. {"name": "fair_1", "policy": "fair",
    "repayment_actions": [0.25, 0.50, 0.75, 1.0]}). Fields not present here
    fall back to full_config["dqn"] defaults inside each agent's constructor.
    """
    policy = agent_cfg["policy"]
    name = agent_cfg["name"]

    if policy == "random":
        return RandomAgent(name, agent_cfg, full_config)
    elif policy in ("max", "fair"):
        repayment_actions = agent_cfg.get(
            "repayment_actions",
            full_config["dqn"].get("repayment_actions", [0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        num_actions = len(repayment_actions)
        return DQNAdversary(name, policy, agent_cfg, full_config, state_dim, num_actions)
    else:
        raise ValueError(f"Unknown policy: {policy}")


def create_all_agents(
    full_config: dict, state_dim: int,
) -> list[BaseAgent]:
    """Create all agents from config['agents'] list."""
    return [create_agent(cfg, full_config, state_dim) for cfg in full_config["agents"]]
