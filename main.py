"""Entry point: wires together all components and runs the simulation pipeline.

Currently implements a 'simulate' mode that runs the frozen RNN investor
against adversary agents in the trust game and exports the full game log
to CSV for downstream analysis in R.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from agents import create_all_agents
from agents.investor import RNNInvestor
from game import Game
from utils import load_config, set_all_seeds
from world import World


def main() -> None:
    cfg = load_config()
    set_all_seeds(cfg["seed"])

    agent_names = [a["name"] for a in cfg["agents"]]

    world = World(
        communication=cfg["world"]["communication"],
        observation_depth=cfg["world"]["observation_depth"],
        agent_names=agent_names,
    )

    rnn_hidden_size: int = cfg["behavioral_rnn"]["hidden_size"]
    state_dim = world.own_obs_dim() + world.others_obs_dim() + rnn_hidden_size + 1
    print(f"State dim: {state_dim}  "
          f"(own={world.own_obs_dim()} + others={world.others_obs_dim()} "
          f"+ rnn_hidden={rnn_hidden_size} + round=1)")

    agents = create_all_agents(cfg, state_dim)
    print(f"Agents: {[(a.name, a.policy) for a in agents]}")

    investor = RNNInvestor(cfg, agent_names)
    print("Investor loaded (frozen BehavioralRNN)")

    game = Game(cfg, world, investor, agents)
    print(f"Game configured: endowment={cfg['game']['endowment']}, "
          f"max_rounds={cfg['game']['max_rounds']}, "
          f"multiplier={cfg['game']['multiplier']}")

    print("\nRunning simulation...")
    log_df = game.run()

    # --- Export CSV atomically ---
    export_cfg = cfg["export"]
    out_dir = Path(export_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / export_cfg["filename"]

    with tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".csv", delete=False,
    ) as tmp:
        log_df.to_csv(tmp, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.rename(out_path)
    print(f"\nGame log saved to {out_path}  ({len(log_df)} rows)")

    # --- Summary ---
    if log_df.empty:
        print("No rounds played.")
        return

    total_rounds = log_df["timestep"].max() + 1
    final_wealth = log_df["investor_wealth"].iloc[-1]

    print(f"\n{'='*50}")
    print("SIMULATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total rounds:       {total_rounds}")
    print(f"  Final investor wealth: {final_wealth:.2f}")
    print(f"  Investor cumulative:   {log_df['investor_cumulative'].iloc[-1]:.2f}")

    print(f"\n  {'Agent':<12} {'Policy':<8} {'Cumulative':>12} {'Mean Repay%':>12}")
    for agent in agents:
        agent_rows = log_df[log_df["agent_name"] == agent.name]
        mean_repay = agent_rows["repayment_pct"].mean()
        print(f"  {agent.name:<12} {agent.policy:<8} "
              f"{agent.cumulative_reward:>12.2f} {mean_repay:>11.1%}")

    # --- Quick validation ---
    expected_cols = [
        "timestep", "agent_name", "agent_type", "world_mode",
        "observation_depth", "investor_wealth", "investment",
        "investment_multiplied", "repayment", "repayment_pct",
        "investor_reward", "agent_reward", "investor_cumulative",
        "agent_cumulative", "rnn_hidden_state", "observation_window",
    ]
    missing = set(expected_cols) - set(log_df.columns)
    if missing:
        print(f"\n  WARNING: Missing columns: {missing}")
    else:
        print(f"\n  CSV columns: OK ({len(expected_cols)} expected)")

    sample_hidden = json.loads(log_df["rnn_hidden_state"].iloc[0])
    sample_obs = json.loads(log_df["observation_window"].iloc[0])
    print(f"  rnn_hidden_state: valid JSON, length={len(sample_hidden)}")
    print(f"  observation_window: valid JSON, keys={list(sample_obs.keys())}")


if __name__ == "__main__":
    main()
