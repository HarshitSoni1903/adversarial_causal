"""Game loop: orchestrates rounds between the frozen RNN investor and adversary agents.

Handles endowments, multiplier, wealth tracking, and termination. Produces a
full game log DataFrame matching the CSV export contract (one row per timestep
per agent). Does NOT train models — training is a separate phase.

The investor and agent interfaces are duck-typed. The game passes observation
components (own, others, rnn_hidden, round_scaled) separately to each agent,
letting the agent decide how to concatenate them into its DQN state vector.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from world import World


class InvestorProtocol(Protocol):
    def act(self, agent_name: str) -> float: ...
    def get_hidden_state(self, agent_name: str) -> np.ndarray: ...
    def observe_outcome(
        self, agent_name: str, investment: float,
        repayment: float, reward: float,
    ) -> None: ...
    def reset(self) -> None: ...


class AgentProtocol(Protocol):
    name: str
    policy: str
    cumulative_reward: float
    def act(
        self, own_obs: np.ndarray, others_obs: np.ndarray,
        rnn_hidden: np.ndarray, round_scaled: float,
    ) -> float: ...
    def observe(self, reward: float, done: bool) -> None: ...
    def reset(self) -> None: ...


class Game:

    def __init__(
        self,
        config: dict,
        world: World,
        investor: Any,
        agents: list[Any],
    ) -> None:
        game_cfg = config["game"]
        self.max_rounds: int = game_cfg["max_rounds"]
        self.endowment: float = float(game_cfg["endowment"])
        self.multiplier: int = game_cfg["multiplier"]
        self.min_investment: float = float(game_cfg["min_investment"])

        world_cfg = config["world"]
        self.communication: bool = world_cfg["communication"]
        self.observation_depth: int = world_cfg["observation_depth"]

        self.world = world
        self.investor = investor
        self.agents = agents
        self.export_dir = Path(config["export"]["output_dir"])
        self.json_log_interval: int = config["export"].get("json_log_interval", 100)

    def run(self) -> pd.DataFrame:
        """Run the full game and return a log DataFrame."""
        self.world.reset()
        self.investor.reset()
        for agent in self.agents:
            agent.reset()

        investor_wealth = self.endowment
        investor_cumulative = 0.0
        log_rows: list[dict] = []
        world_mode = 1 if self.communication else 0
        round_denom = max(self.max_rounds - 1, 1)

        recent_repay: dict[str, list[float]] = defaultdict(list)

        pbar = tqdm(range(self.max_rounds), desc="Round", unit="rnd")
        for t in pbar:
            if investor_wealth < self.min_investment:
                break

            for agent in self.agents:
                investment = self.investor.act(agent.name)
                investment = max(self.min_investment, min(investment, investor_wealth))

                if investor_wealth < self.min_investment:
                    break

                investment_multiplied = investment * self.multiplier

                if hasattr(agent, "set_investment_received"):
                    agent.set_investment_received(investment_multiplied)

                obs = self.world.get_full_observation(agent.name)
                rnn_hidden = self.investor.get_hidden_state(agent.name)
                round_scaled = t / round_denom

                repay_pct = agent.act(
                    own_obs=obs["own"],
                    others_obs=obs["others"],
                    rnn_hidden=rnn_hidden,
                    round_scaled=round_scaled,
                )
                repayment = repay_pct * investment_multiplied

                investor_reward = repayment - investment
                agent_reward = investment_multiplied - repayment

                investor_wealth += investor_reward
                investor_cumulative += investor_reward
                agent.cumulative_reward += agent_reward

                self.world.record_step(agent.name, investment, repayment)
                self.investor.observe_outcome(
                    agent.name, investment, repayment, investor_reward,
                )
                if hasattr(agent, "set_investor_reward"):
                    agent.set_investor_reward(investor_reward)
                agent.observe(agent_reward, done=False)

                recent_repay[agent.name].append(repay_pct)

                if t % self.json_log_interval == 0:
                    hidden_json = json.dumps(rnn_hidden.tolist())
                    obs_json = json.dumps({
                        "own": obs["own"].tolist(),
                        "others": obs["others"].tolist(),
                    })
                else:
                    hidden_json = ""
                    obs_json = ""

                log_rows.append({
                    "timestep": t,
                    "agent_name": agent.name,
                    "agent_type": agent.policy,
                    "world_mode": world_mode,
                    "observation_depth": self.observation_depth,
                    "investor_wealth": round(investor_wealth, 2),
                    "investment": round(investment, 2),
                    "investment_multiplied": round(investment_multiplied, 2),
                    "repayment": round(repayment, 2),
                    "repayment_pct": round(repay_pct, 2),
                    "investor_reward": round(investor_reward, 2),
                    "agent_reward": round(agent_reward, 2),
                    "investor_cumulative": round(investor_cumulative, 2),
                    "agent_cumulative": round(agent.cumulative_reward, 2),
                    "rnn_hidden_state": hidden_json,
                    "observation_window": obs_json,
                })

            if t % 100 == 99 or t == 0:
                postfix = {"w": f"{investor_wealth:.0f}"}
                for agent in self.agents:
                    buf = recent_repay[agent.name][-100:]
                    avg_r = np.mean(buf) if buf else 0.0
                    postfix[agent.name] = (
                        f"{'+' if agent.cumulative_reward >= 0 else ''}"
                        f"{agent.cumulative_reward:.0f}"
                        f"(r={avg_r:.2f})"
                    )
                pbar.set_postfix(postfix)

            if investor_wealth < self.min_investment:
                break

        pbar.close()

        for agent in self.agents:
            agent.observe(0.0, done=True)

        log_df = pd.DataFrame(log_rows)
        self._save_summary_plot(log_df)
        return log_df

    def _save_summary_plot(self, log_df: pd.DataFrame) -> None:
        """Save investor wealth and agent cumulative reward plots."""
        if log_df.empty:
            return

        self.export_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        first_agent = self.agents[0].name
        wealth_data = log_df[log_df["agent_name"] == first_agent]
        ax1.plot(wealth_data["timestep"], wealth_data["investor_wealth"])
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Wealth")
        ax1.set_title("Investor Wealth Over Time")
        ax1.axhline(y=self.endowment, color="gray", linestyle="--", alpha=0.5)

        for agent in self.agents:
            agent_data = log_df[log_df["agent_name"] == agent.name]
            ax2.plot(
                agent_data["timestep"], agent_data["agent_cumulative"],
                label=f"{agent.name} ({agent.policy})",
            )
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_title("Agent Cumulative Rewards")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.export_dir / "game_summary.png", dpi=150)
        plt.close(fig)
