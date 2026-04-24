"""Game loop: orchestrates rounds between the frozen RNN investor and adversary agents.

Two-phase per-round loop (Dezfouli D3 trust-ranked allocation):
  Phase 1: investor.act() for each agent → desired investments (RNN samples)
  Phase 2: _allocate() distributes shared endowment budget (trust-ranked, greedy)
  Phase 3: for each agent — build adversary state, get repayment, compute rewards

Investor state is NEVER carried over between rounds (per-round fresh endowment).
investor_cumulative tracks net profit/loss accumulated over the episode.
investor_wealth = endowment + investor_cumulative at any point.

Adversary state: [rnn_hidden(5), policy_vec(5), investor_action_onehot(5), round_norm(1),
                  others_obs(W1/N>1 only)] — built by _build_adversary_state().
"""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from world import World


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
        self.multiplier: int = game_cfg["multiplier"]
        self.observation_depth: int = game_cfg.get("observation_depth", 4)

        # Effective wallet scales with N so each agent can always get its full share.
        # At N=1 this equals the Dezfouli endowment of 20.
        if "endowment_per_agent" in game_cfg:
            self.endowment: float = float(game_cfg["endowment_per_agent"]) * len(agents)
        else:
            self.endowment = float(game_cfg["endowment"])

        self.cfg = config
        self.world = world
        self.investor = investor
        self.agents = agents
        self._agent_config_idx: dict[str, int] = {a.name: i for i, a in enumerate(agents)}
        self._investor_action_values: list[int] = sorted(
            config["behavioral_rnn"]["action_values"]
        )

        from utils import get_output_dir
        self.export_dir = Path(get_output_dir(config))
        self.json_log_interval: int = config["export"].get("json_log_interval", 1)

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------

    def _floor_to_bucket(self, amount: float) -> float:
        """Floor amount down to the nearest investor action value."""
        result = 0.0
        for v in self._investor_action_values:
            if v <= amount:
                result = float(v)
        return result

    def _allocate(self, desired: dict[str, float]) -> dict[str, float]:
        """Trust-ranked sequential allocation from shared endowment budget.

        Sort agents by desired investment descending; config-list order breaks ties.
        Fill greedily; clip and round DOWN to nearest bucket.
        """
        budget = self.endowment
        order = sorted(
            desired.keys(),
            key=lambda n: (-desired[n], self._agent_config_idx[n]),
        )
        actual: dict[str, float] = {}
        for name in order:
            alloc = self._floor_to_bucket(min(desired[name], budget))
            actual[name] = alloc
            budget -= alloc
        return actual

    # ------------------------------------------------------------------
    # Spillover
    # ------------------------------------------------------------------

    def _apply_spillover(self) -> None:
        """Blend investor RNN hidden states across agents (simultaneous update).

        Snapshot all hidden states before modifying any, so agent i's blend
        uses agent j's original state, not a partially-updated one.
        Called only when N>1 and spillover_alpha>0.
        """
        alpha = float(self.cfg["behavioral_rnn"].get("spillover_alpha", 0.0))
        n = len(self.agents)
        current_h = {
            i: self.investor.get_hidden_state(self.agents[i].name)
            for i in range(n)
        }
        for i in range(n):
            others_mean = np.mean(
                [current_h[j] for j in range(n) if j != i], axis=0
            )
            blended = (1.0 - alpha) * current_h[i] + alpha * others_mean
            self.investor.set_hidden_state(self.agents[i].name, blended)

    # ------------------------------------------------------------------
    # Adversary state builder
    # ------------------------------------------------------------------

    def _build_adversary_state(self, agent: Any, t: int, n_rounds: int) -> np.ndarray:
        """Build 16-dim (W0/N=1) or wider (W1/N>1) adversary state vector.

        [rnn_hidden(5), policy_vec(5), investor_action_onehot(5), round_norm(1), others_obs(?)]
        """
        h, policy, ah = self.investor.get_rnn_info(agent.name)
        round_norm = np.float32(t / max(n_rounds - 1, 1))
        others_obs = self.world.get_others_observation(agent.name)
        return np.concatenate([h, policy, ah, [round_norm], others_obs]).astype(np.float32)

    # ------------------------------------------------------------------
    # Core episode runner
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        log_details: bool = False,
        rounds_override: int | None = None,
    ) -> dict:
        """Run one full game episode.

        Returns dict with: total_rounds, final_wealth, investor_cumulative,
        agent_rewards, agent_repay_means, and optionally log_df.
        """
        n_rounds = rounds_override if rounds_override is not None else self.max_rounds

        self.world.reset()
        self.investor.reset()
        for agent in self.agents:
            agent.reset()

        investor_cumulative = 0.0
        log_rows: list[dict] = []
        repay_sums: dict[str, float] = {a.name: 0.0 for a in self.agents}
        repay_counts: dict[str, int] = {a.name: 0 for a in self.agents}

        for t in range(n_rounds):
            done = (t == n_rounds - 1)

            # Phase 1: collect desired investments from RNN
            desired: dict[str, float] = {}
            for agent in self.agents:
                desired[agent.name] = self.investor.act(agent.name)

            # Phase 2: trust-ranked allocation
            actual = self._allocate(desired)

            # Phase 3: process repayments for each agent
            for agent in self.agents:
                investment = actual[agent.name]
                investment_multiplied = investment * self.multiplier

                state = self._build_adversary_state(agent, t, n_rounds)
                repay_pct = float(np.clip(agent.act(state), 0.0, 1.0))
                repayment = repay_pct * investment_multiplied

                investor_reward = repayment - investment
                agent_reward = investment_multiplied - repayment

                investor_cumulative += investor_reward
                agent.cumulative_reward += agent_reward

                repay_prop = repayment / investment_multiplied if investment_multiplied > 0 else 0.0
                self.investor.observe_outcome(agent.name, investment, repay_prop)
                self.world.record_step(agent.name, investment, repayment)

                agent.observe(agent_reward, done=done)
                if hasattr(agent, "accumulate_investor_reward"):
                    agent.accumulate_investor_reward(investor_reward)

                repay_sums[agent.name] += repay_pct
                repay_counts[agent.name] += 1

                if log_details:
                    log_rows.append({
                        "timestep": t,
                        "agent_name": agent.name,
                        "agent_type": agent.policy,
                        "world_mode": self.world.mode,
                        "observation_depth": self.observation_depth,
                        "investor_wealth": round(self.endowment + investor_cumulative, 2),
                        "investment_desired": round(desired[agent.name], 2),
                        "investment": round(investment, 2),
                        "investment_multiplied": round(investment_multiplied, 2),
                        "repayment": round(repayment, 2),
                        "repayment_pct": round(repay_pct, 4),
                        "investor_reward": round(investor_reward, 2),
                        "agent_reward": round(agent_reward, 2),
                        "investor_cumulative": round(investor_cumulative, 2),
                        "agent_cumulative": round(agent.cumulative_reward, 2),
                    })

            # Spillover: blend investor hidden states across agents after each round.
            # N=1 guard: len check makes this a no-op for the Dezfouli replication.
            if (
                len(self.agents) > 1
                and float(self.cfg["behavioral_rnn"].get("spillover_alpha", 0.0)) > 0.0
            ):
                self._apply_spillover()

        result: dict = {
            "total_rounds": n_rounds,
            "final_wealth": self.endowment + investor_cumulative,
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
    # Public: single simulation run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Run a single episode with full logging and summary plot."""
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
        eval_interval: int = 10000,
        eval_episodes: int = 2000,
    ) -> dict:
        """Train adversary DQNs via independent Q-learning (IQL).

        Runs num_episodes episodes; eval at eval_interval; saves agent checkpoints.
        """
        training_returns: list[dict] = []
        eval_history: list[dict] = []

        pbar = tqdm(range(1, num_episodes + 1), desc="Training", unit="ep")
        for ep in pbar:
            stats = self._run_episode(log_details=False)
            ep_returns: dict = {
                "investor_return": stats["investor_cumulative"],
                **stats["agent_rewards"],
            }
            for name, rp in stats["agent_repay_means"].items():
                ep_returns[f"{name}_repay"] = rp
            training_returns.append(ep_returns)

            for agent in self.agents:
                if hasattr(agent, "q_learner"):
                    agent.q_learner.decay_epsilon()

            if ep % 10 == 0 or ep == 1:
                eps_val = self._get_epsilon()
                postfix: dict = {"eps": f"{eps_val:.3f}"}
                for a in self.agents:
                    postfix[a.name] = f"{stats['agent_rewards'][a.name]:+.0f}"
                pbar.set_postfix(postfix)

            if ep % eval_interval == 0:
                eval_stats = self._run_eval(eval_episodes)
                eval_history.append({"episode": ep, **eval_stats})
                print(
                    f"  [Eval @ ep {ep}] "
                    + "  ".join(
                        f"{a.name}(reward={eval_stats['mean_agent_rewards'][a.name]:+.0f}"
                        f", repay={eval_stats['mean_agent_repay'][a.name]:.2f})"
                        for a in self.agents
                    )
                )

        pbar.close()

        for agent in self.agents:
            if hasattr(agent, "save"):
                agent.save()
                print(f"Saved {agent.name}")

        self._save_training_returns_csv(training_returns)
        return {"training_returns": training_returns, "eval_history": eval_history}

    def _run_eval(self, num_episodes: int) -> dict:
        """Run evaluation episodes (greedy, no learning)."""
        for agent in self.agents:
            if hasattr(agent, "set_eval_mode"):
                agent.set_eval_mode(True)

        wealth_list: list[float] = []
        rewards_list: list[dict] = []
        repay_list: list[dict] = []

        for _ in range(num_episodes):
            stats = self._run_episode(log_details=False)
            wealth_list.append(stats["final_wealth"])
            rewards_list.append(stats["agent_rewards"])
            repay_list.append(stats["agent_repay_means"])

        for agent in self.agents:
            if hasattr(agent, "set_eval_mode"):
                agent.set_eval_mode(False)

        return {
            "mean_wealth": float(np.mean(wealth_list)),
            "mean_agent_rewards": {
                a.name: float(np.mean([r[a.name] for r in rewards_list]))
                for a in self.agents
            },
            "mean_agent_repay": {
                a.name: float(np.mean([r[a.name] for r in repay_list]))
                for a in self.agents
            },
        }

    def _get_epsilon(self) -> float:
        for agent in self.agents:
            if hasattr(agent, "q_learner"):
                return agent.q_learner.epsilon
        return 0.0

    def _save_training_returns_csv(self, training_returns: list[dict]) -> None:
        if not training_returns:
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        agent_names = [a.name for a in self.agents]
        rows: list[dict] = []
        for ep_idx, ret in enumerate(training_returns, start=1):
            row: dict = {"episode": ep_idx, "investor_return": round(ret["investor_return"], 2)}
            for name in agent_names:
                row[f"{name}_return"] = round(ret[name], 2)
                repay_key = f"{name}_repay"
                if repay_key in ret:
                    row[repay_key] = round(ret[repay_key], 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = self.export_dir / "training_returns.csv"
        df.to_csv(out_path, index=False)
        print(f"Training returns: {out_path}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _save_summary_plot(self, log_df: pd.DataFrame) -> None:
        if log_df.empty:
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        first = self.agents[0].name
        wealth = log_df[log_df["agent_name"] == first]
        ax1.plot(wealth["timestep"], wealth["investor_wealth"])
        ax1.axhline(y=self.endowment, color="gray", ls="--", alpha=0.5)
        ax1.set(xlabel="Round", ylabel="Wealth", title="Investor Wealth Over Time")

        for agent in self.agents:
            rows = log_df[log_df["agent_name"] == agent.name]
            ax2.plot(rows["timestep"], rows["agent_cumulative"],
                     label=f"{agent.name} ({agent.policy})")
        ax2.set(xlabel="Round", ylabel="Cumulative Reward", title="Agent Cumulative Rewards")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.export_dir / "game_summary.png", dpi=150)
        plt.close(fig)
