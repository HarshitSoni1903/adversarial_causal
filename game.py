"""Game loop: orchestrates rounds between frozen RNN investors and adversary agents.

Per-round flow (two-dyad mode):
  Step A: decay all investors  h_k ← γ·h_k
  Step B: cross-investor update if ii_edge  (snapshot from World, apply to each)
  Step C: self step + decision  desired_k = investor_k.act()
  Step D: per-dyad allocation  invest_k = floor_to_bucket(min(desired_k, endowment))
  Step E: trustee phase  build state, get repayment, compute rewards, bookkeeping
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
        investors: list[Any],
        agents: list[Any],
    ) -> None:
        game_cfg = config["game"]
        self.max_rounds: int = game_cfg["max_rounds"]
        self.multiplier: int = game_cfg["multiplier"]
        self.observation_depth: int = game_cfg.get("observation_depth", 4)
        self.endowment_per_investor: float = float(game_cfg["endowment_per_investor"])

        edges = config["edges"]
        self.ii_edge: int = edges["ii_edge"]
        self.aa_edge: int = edges["aa_edge"]

        self.cfg = config
        self.world = world
        self.investors = investors
        self.agents = agents
        self._investor_action_values: list[int] = sorted(
            config["behavioral_rnn"]["action_values"]
        )

        from utils import get_output_dir
        self.export_dir = Path(get_output_dir(config))
        self.json_log_interval: int = config["export"].get("json_log_interval", 1)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _floor_to_bucket(self, amount: float) -> float:
        result = 0.0
        for v in self._investor_action_values:
            if v <= amount:
                result = float(v)
        return result

    def _allocate(self, desired: dict[str, float]) -> dict[str, float]:
        """Per-dyad floor allocation: each agent gets min(desired, endowment), bucketed."""
        return {
            name: self._floor_to_bucket(min(desired[name], self.endowment_per_investor))
            for name in desired
        }

    # ------------------------------------------------------------------
    # Adversary state builder
    # ------------------------------------------------------------------

    def _build_adversary_state(self, k: int, t: int, n_rounds: int) -> np.ndarray:
        """Build adversary state for dyad k at round t.

        [rnn_hidden(5), policy_vec(5), action_onehot(5), round_norm(1),
         ii_edge(1), aa_edge(1), cross_window(8 if aa_edge else 0)]
        """
        h, policy, ah = self.investors[k].get_rnn_info()
        round_norm = np.float32(t / max(n_rounds - 1, 1))
        flags = np.array([self.ii_edge, self.aa_edge], dtype=np.float32)
        parts = [h, policy, ah, [round_norm], flags]
        if self.aa_edge:
            parts.append(self.world.get_other_pair_window(k))
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # Core episode runner
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        log_details: bool = False,
        rounds_override: int | None = None,
    ) -> dict:
        n_rounds = rounds_override if rounds_override is not None else self.max_rounds

        self.world.reset()
        for inv in self.investors:
            inv.reset()
        for agent in self.agents:
            agent.reset()

        dyad_cumulatives = [0.0] * len(self.investors)
        total_investor_cumulative = 0.0
        log_rows: list[dict] = []
        repay_sums: dict[str, float] = {a.name: 0.0 for a in self.agents}
        repay_counts: dict[str, int] = {a.name: 0 for a in self.agents}

        for t in range(n_rounds):
            done = (t == n_rounds - 1)

            # Step A: decay all investors
            for inv in self.investors:
                inv.decay()

            # Step B: cross-investor update (snapshot from World first)
            if self.ii_edge:
                prev = [
                    self.world.get_other_pair_last_action_repay(k)
                    for k in range(len(self.investors))
                ]
                for k, inv in enumerate(self.investors):
                    other_k = 1 - k
                    inv.cross_step(*prev[other_k])

            # Step C: self step + decision
            desired: dict[str, float] = {}
            for k, (inv, agent) in enumerate(zip(self.investors, self.agents)):
                desired[agent.name] = inv.act()

            # Step D: per-dyad allocation
            actual = self._allocate(desired)

            # Step E: per-dyad trustee phase
            for k, (inv, agent) in enumerate(zip(self.investors, self.agents)):
                investment = actual[agent.name]
                investment_multiplied = investment * self.multiplier

                state = self._build_adversary_state(k, t, n_rounds)
                repay_pct = float(np.clip(agent.act(state), 0.0, 1.0))
                repayment = repay_pct * investment_multiplied

                investor_reward = repayment - investment
                agent_reward = investment_multiplied - repayment

                total_investor_cumulative += investor_reward
                dyad_cumulatives[k] += investor_reward
                agent.cumulative_reward += agent_reward

                repay_prop = repayment / investment_multiplied if investment_multiplied > 0 else 0.0

                # Step F: bookkeeping — observe_outcome sets _prev_ah from actual investment
                inv.observe_outcome(investment, repay_prop)
                self.world.record_dyad_step(
                    k, investment, repayment,
                    inv._prev_ah.numpy(), repay_prop,
                )

                agent.observe(agent_reward, done=done)
                if hasattr(agent, "accumulate_investor_reward"):
                    agent.accumulate_investor_reward(investor_reward)

                repay_sums[agent.name] += repay_pct
                repay_counts[agent.name] += 1

                if log_details:
                    log_rows.append({
                        "timestep": t,
                        "dyad_idx": k,
                        "agent_name": agent.name,
                        "agent_type": agent.policy,
                        "ii_edge": self.ii_edge,
                        "aa_edge": self.aa_edge,
                        "observation_depth": self.observation_depth,
                        "investor_wealth": round(
                            self.endowment_per_investor + dyad_cumulatives[k], 2
                        ),
                        "investor_cumulative": round(total_investor_cumulative, 2),
                        "dyad_investor_cumulative": round(dyad_cumulatives[k], 2),
                        "investment_desired": round(desired[agent.name], 2),
                        "investment": round(investment, 2),
                        "investment_multiplied": round(investment_multiplied, 2),
                        "repayment": round(repayment, 2),
                        "repayment_pct": round(repay_pct, 4),
                        "investor_reward": round(investor_reward, 2),
                        "agent_reward": round(agent_reward, 2),
                        "agent_cumulative": round(agent.cumulative_reward, 2),
                    })

        result: dict = {
            "total_rounds": n_rounds,
            "final_wealth": (
                len(self.investors) * self.endowment_per_investor + total_investor_cumulative
            ),
            "investor_cumulative": total_investor_cumulative,
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
        ax1.axhline(y=self.endowment_per_investor, color="gray", ls="--", alpha=0.5)
        ax1.set(xlabel="Round", ylabel="Wealth", title="Investor Wealth Over Time (dyad 0)")

        for agent in self.agents:
            rows = log_df[log_df["agent_name"] == agent.name]
            ax2.plot(rows["timestep"], rows["agent_cumulative"],
                     label=f"{agent.name} ({agent.policy})")
        ax2.set(xlabel="Round", ylabel="Cumulative Reward", title="Agent Cumulative Rewards")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.export_dir / "game_summary.png", dpi=150)
        plt.close(fig)
