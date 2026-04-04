"""Agent factory: creates agents from config entries.

Usage:
    agents = create_all_agents(config, state_dim, run_id="w0_20260404_153022")
"""

from agents.adversary import DQNAdversary
from agents.base import BaseAgent
from agents.random_agent import RandomAgent


def create_agent(
    agent_cfg: dict,
    full_config: dict,
    state_dim: int,
    run_id: str | None = None,
) -> BaseAgent:
    """Create a single agent from its config entry."""
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
        return DQNAdversary(
            name, policy, agent_cfg, full_config,
            state_dim, num_actions, run_id=run_id,
        )
    else:
        raise ValueError(f"Unknown policy: {policy}")


def create_all_agents(
    full_config: dict,
    state_dim: int,
    run_id: str | None = None,
) -> list[BaseAgent]:
    """Create all agents from config['agents'] list."""
    return [
        create_agent(cfg, full_config, state_dim, run_id=run_id)
        for cfg in full_config["agents"]
    ]
