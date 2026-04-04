"""Game loop: orchestrates rounds between the frozen RNN investor and adversary agents.

Supports both single-run simulation (run()) and multi-episode training
(run_training()). Both use _run_episode() internally to avoid duplicating
the game loop.

The investor and agent interfaces are duck-typed. The game passes observation
components (own, others, rnn_hidden, round_scaled) separately to each agent,
letting the agent decide how to concatenate them into its DQN state vector.
"""

import json
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
    learns: bool
    def act(self, agent_name: str, *, round_num: int = 0, max_rounds: int = 1) -> float: ...
    def get_hidden_state(self, agent_name: str) -> np.ndarray: ...
    def observe_outcome(
        self, agent_name: str, investment: float,
        repayment: float, reward: float,
    ) -> None: ...
    def receive_round_reward(self, total_round_reward: float) -> None: ...
    def observe_done(self) -> None: ...
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
        training_mode: bool = True,
    ) -> None:
        game_cfg = config["game"]
        self.max_rounds: int = game_cfg["max_rounds"]
        self.endowment: float = float(game_cfg["endowment"])
        self.multiplier: int = game_cfg["multiplier"]
        self.min_investment: float = float(game_cfg["min_investment"])

        world_cfg = config["world"]
        self.communication: bool = world_cfg["communication"]
        self.observation_depth: int = world_cfg["observation_depth"]

        self.training_rounds: int = config["dqn"].get("training_rounds", self.max_rounds)

        self.world = world
        self.investor = investor
        self.agents = agents
        self.training_mode = training_mode
        # Output dir organized by world mode: outputs/w0/ or outputs/w1/
        mode = "w1" if self.communication else "w0"
        self.export_dir = Path(config["export"]["output_dir"]) / mode
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.json_log_interval: int = config["export"].get("json_log_interval", 100)

    # ------------------------------------------------------------------
    # Core episode runner
    # ------------------------------------------------------------------

    def _run_episode(
        self, log_details: bool = False, rounds_override: int | None = None,
    ) -> dict:
        """Run one full game episode.

        Args:
            log_details: if True, build full DataFrame log (slower).
            rounds_override: if set, use this instead of self.max_rounds.

        Returns:
            dict with keys: total_rounds, final_wealth, investor_cumulative,
            agent_rewards (dict[name, float]), agent_repay_means (dict[name, float]),
            and optionally "log_df" (pd.DataFrame) when log_details=True.
        """
        n_rounds = rounds_override if rounds_override is not None else self.max_rounds

        self.world.reset()
        self.investor.reset()
        for agent in self.agents:
            agent.reset()

        investor_wealth = self.endowment
        investor_cumulative = 0.0
        world_mode = 1 if self.communication else 0
        round_denom = max(n_rounds - 1, 1)

        log_rows: list[dict] = [] if log_details else []
        repay_sums: dict[str, float] = {a.name: 0.0 for a in self.agents}
        repay_counts: dict[str, int] = {a.name: 0 for a in self.agents}
        total_rounds = 0

        for t in range(n_rounds):
            if investor_wealth < self.min_investment:
                break
            total_rounds = t + 1
            round_investor_reward = 0.0

            for agent in self.agents:
                investment = self.investor.act(
                    agent.name, round_num=t, max_rounds=n_rounds,
                )
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
                repay_pct = float(np.clip(repay_pct, 0.0, 1.0))
                repayment = repay_pct * investment_multiplied

                investor_reward = repayment - investment
                agent_reward = investment_multiplied - repayment

                investor_wealth += investor_reward
                investor_cumulative += investor_reward
                round_investor_reward += investor_reward
                agent.cumulative_reward += agent_reward

                self.world.record_step(agent.name, investment, repayment)
                self.investor.observe_outcome(
                    agent.name, investment, repayment, investor_reward,
                )
                if hasattr(agent, "set_investor_reward"):
                    agent.set_investor_reward(investor_reward)
                agent.observe(agent_reward, done=False)

                repay_sums[agent.name] += repay_pct
                repay_counts[agent.name] += 1

                if log_details:
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

            self.investor.receive_round_reward(round_investor_reward)

            if investor_wealth < self.min_investment:
                break

        for agent in self.agents:
            agent.observe(0.0, done=True)
        self.investor.observe_done()

        result: dict[str, Any] = {
            "total_rounds": total_rounds,
            "final_wealth": investor_wealth,
            "investor_cumulative": investor_cumulative,
            "agent_rewards": {a.name: a.cumulative_reward for a in self.agents},
            "agent_repay_means": {
                a.name: repay_sums[a.name] / max(repay_counts[a.name], 1)
                for a in self.agents
            },
        }
        if log_details:
            result["log_df"] = pd.DataFrame(log_rows)
        return result

    # ------------------------------------------------------------------
    # Public: single simulation run with full logging
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Run a single game with full DataFrame logging and summary plot."""
        result = self._run_episode(log_details=True)
        log_df = result["log_df"]
        self._save_summary_plot(log_df)
        return log_df

    # ------------------------------------------------------------------
    # Public: multi-episode training
    # ------------------------------------------------------------------

    def run_training(
        self,
        num_episodes: int,
        eval_interval: int = 100,
        eval_episodes: int = 10,
    ) -> dict:
        """Run multiple episodes for online adversary training.

        Returns dict with training_returns (list of per-episode dicts with
        investor_return and per-agent returns) and eval_history.
        """
        training_returns: list[dict] = []
        eval_history: list[dict] = []

        pbar = tqdm(range(1, num_episodes + 1), desc="Training", unit="ep")
        for ep in pbar:
            stats = self._run_episode(log_details=False, rounds_override=self.training_rounds)
            ep_returns = {
                "investor_return": stats["investor_cumulative"],
                **stats["agent_rewards"],
            }
            training_returns.append(ep_returns)

            for agent in self.agents:
                if hasattr(agent, "q_learner"):
                    agent.q_learner.decay_epsilon()
            self.investor.decay_epsilon()

            eps_val = self._get_epsilon()
            if ep % 10 == 0 or ep == 1:
                inv_ret = stats["investor_cumulative"]
                postfix: dict[str, str] = {
                    "inv_ret": f"{'+' if inv_ret >= 0 else ''}{inv_ret:.0f}",
                }
                for a in self.agents:
                    r = stats["agent_rewards"][a.name]
                    postfix[a.name] = f"{'+' if r >= 0 else ''}{r:.0f}"
                postfix["eps"] = f"{eps_val:.3f}"
                pbar.set_postfix(postfix)

            if ep % eval_interval == 0:
                eval_stats = self._run_eval(eval_episodes)
                eval_history.append({"episode": ep, **eval_stats})
                print(
                    f"  [Eval @ ep {ep}] "
                    f"mean_wealth={eval_stats['mean_wealth']:.0f}  "
                    + "  ".join(
                        f"{a.name}={eval_stats['mean_agent_rewards'][a.name]:+.0f}"
                        f"(r={eval_stats['mean_agent_repay'][a.name]:.2f})"
                        for a in self.agents
                    )
                )

        pbar.close()

        for agent in self.agents:
            if hasattr(agent, "save"):
                agent.save()
                print(f"Saved {agent.name} weights")

        self.investor.save()
        if self.investor.learns:
            print("Saved investor Q-learner weights")

        self._save_training_returns_csv(training_returns)

        return {"training_returns": training_returns, "eval_history": eval_history}

    def _save_training_returns_csv(self, training_returns: list[dict]) -> None:
        """Save per-episode returns CSV for downstream analysis in R."""
        if not training_returns:
            return

        self.export_dir.mkdir(parents=True, exist_ok=True)
        agent_names = [a.name for a in self.agents]

        rows: list[dict] = []
        for ep_idx, ret in enumerate(training_returns, start=1):
            row: dict[str, float | int] = {"episode": ep_idx}
            row["investor_return"] = round(ret["investor_return"], 2)
            for name in agent_names:
                row[f"{name}_return"] = round(ret[name], 2)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.export_dir / "training_returns.csv"
        df.to_csv(out_path, index=False)
        print(f"Training returns saved to {out_path}  ({len(df)} rows)")

    def _run_eval(self, num_episodes: int) -> dict:
        """Run evaluation episodes with agents in eval mode."""
        self.investor.set_eval_mode(True)
        for agent in self.agents:
            if hasattr(agent, "set_eval_mode"):
                agent.set_eval_mode(True)

        wealth_list: list[float] = []
        agent_rewards_list: list[dict[str, float]] = []
        agent_repay_list: list[dict[str, float]] = []

        for _ in range(num_episodes):
            stats = self._run_episode(log_details=False, rounds_override=self.training_rounds)
            wealth_list.append(stats["final_wealth"])
            agent_rewards_list.append(stats["agent_rewards"])
            agent_repay_list.append(stats["agent_repay_means"])

        self.investor.set_eval_mode(False)
        for agent in self.agents:
            if hasattr(agent, "set_eval_mode"):
                agent.set_eval_mode(False)

        return {
            "mean_wealth": float(np.mean(wealth_list)),
            "mean_agent_rewards": {
                a.name: float(np.mean([r[a.name] for r in agent_rewards_list]))
                for a in self.agents
            },
            "mean_agent_repay": {
                a.name: float(np.mean([r[a.name] for r in agent_repay_list]))
                for a in self.agents
            },
        }

    def _get_epsilon(self) -> float:
        """Get current epsilon from the first DQN agent or investor Q-learner."""
        if self.investor.learns:
            return self.investor.q_learner.epsilon
        for agent in self.agents:
            if hasattr(agent, "q_learner"):
                return agent.q_learner.epsilon
        return 0.0

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

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
