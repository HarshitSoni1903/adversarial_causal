"""Entry point: wires together all components and runs the pipeline.

Usage:
    python main.py train                     # Train, auto-generates run ID
    python main.py simulate --run w0_20260404_153022   # Simulate a specific run
    python main.py simulate --run latest     # Simulate the most recent run

Each run creates timestamped folders:
    checkpoints/<run_id>/    — model weights
    outputs/<run_id>/        — plots, CSVs, summaries
    outputs/<run_id>/config.yaml  — snapshot of config used

Run IDs follow the format: w0_YYYYMMDD_HHMMSS or w1_YYYYMMDD_HHMMSS
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from agents import create_all_agents
from agents.investor import RNNInvestor
from game import Game
from utils import (
    create_run_id,
    get_checkpoint_dir,
    get_output_dir,
    load_config,
    set_all_seeds,
)
from world import World


# ------------------------------------------------------------------
# Run management
# ------------------------------------------------------------------

def _find_latest_run(cfg: dict) -> str:
    """Find the most recent run_id matching the current world mode."""
    mode_prefix = "w1" if cfg["world"]["communication"] else "w0"
    ckpt_base = Path(cfg["checkpoints"]["base_dir"])
    if not ckpt_base.exists():
        raise FileNotFoundError("No checkpoints directory found")

    runs = sorted(
        [d.name for d in ckpt_base.iterdir()
         if d.is_dir() and d.name.startswith(mode_prefix + "_")],
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(
            f"No runs found for {mode_prefix} in {ckpt_base}"
        )
    return runs[0]


def _save_config_snapshot(cfg: dict, out_dir: Path) -> None:
    """Save a copy of the config used for this run."""
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------
# Build components
# ------------------------------------------------------------------

def _build_components(
    cfg: dict, run_id: str,
) -> tuple[World, list, RNNInvestor, int]:
    """Build world, agents, investor, and compute state_dim."""
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

    agents = create_all_agents(cfg, state_dim, run_id=run_id)
    print(f"Agents: {[(a.name, a.policy) for a in agents]}")

    investor = RNNInvestor(cfg, agent_names, run_id=run_id)
    if investor.learns:
        k = len(agent_names)
        ql_state_dim = k + rnn_hidden_size * k + 2 * k + 2
        print(f"Investor loaded (frozen RNN + Q-learner, state_dim={ql_state_dim})")
    else:
        print("Investor loaded (frozen BehavioralRNN)")

    return world, agents, investor, state_dim


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def _save_training_curves(
    training_stats: dict, agents: list, save_dir: Path,
) -> None:
    """Save training curves: smoothed returns + per-episode returns + repayment rates."""
    returns = training_stats["training_returns"]
    if not returns:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    window = max(1, len(returns) // 50)

    # --- Plot 1: Training returns (smoothed) + eval snapshots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    inv_vals = [r["investor_return"] for r in returns]
    inv_smooth = np.convolve(inv_vals, np.ones(window) / window, mode="valid")
    ax1.plot(inv_smooth, label="investor", color="black", linewidth=1.5)

    for agent in agents:
        vals = [r[agent.name] for r in returns]
        smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed, label=f"{agent.name} ({agent.policy})", alpha=0.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward (smoothed)")
    ax1.set_title("Training Returns")
    ax1.legend()

    eval_hist = training_stats["eval_history"]
    if eval_hist:
        eps = [e["episode"] for e in eval_hist]
        ax2.plot(eps, [e["mean_wealth"] for e in eval_hist], "k-",
                 label="investor wealth", linewidth=1.5)
        for agent in agents:
            ax2.plot(
                eps,
                [e["mean_agent_rewards"][agent.name] for e in eval_hist],
                label=agent.name, alpha=0.8,
            )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Mean Reward / Wealth")
        ax2.set_title("Evaluation Snapshots")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Per-episode raw returns ---
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(inv_vals, alpha=0.3, color="black", linewidth=0.5)
    ax.plot(inv_smooth, color="black", linewidth=1.5)
    ax.set_title("Investor Return per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, agent in enumerate(agents[:3]):
        ax = axes[(i + 1) // 2, (i + 1) % 2]
        vals = [r[agent.name] for r in returns]
        smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
        c = colors[i % len(colors)]
        ax.plot(vals, alpha=0.3, color=c, linewidth=0.5)
        ax.plot(smoothed, color=c, linewidth=1.5)
        ax.set_title(f"{agent.name} ({agent.policy}) Return per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")

    plt.tight_layout()
    plt.savefig(save_dir / "per_episode_returns.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: Repayment rates from eval ---
    if eval_hist:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        eps = [e["episode"] for e in eval_hist]
        for agent in agents:
            ax3.plot(
                eps,
                [e["mean_agent_repay"][agent.name] for e in eval_hist],
                label=f"{agent.name} ({agent.policy})", marker="o", markersize=3,
            )
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Mean Repayment %")
        ax3.set_title("Agent Repayment Rates Over Training")
        ax3.legend()
        ax3.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(save_dir / "repayment_rates.png", dpi=150)
        plt.close(fig3)

    print(f"Training curves saved to {save_dir}/")


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------

def _export_csv(log_df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    """Export game log atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    with tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".csv", delete=False,
    ) as tmp:
        log_df.to_csv(tmp, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.rename(out_path)
    print(f"Game log saved to {out_path}  ({len(log_df)} rows)")


# ------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------

def _build_summary(
    log_df: pd.DataFrame, agents: list, cfg: dict, run_id: str,
) -> str:
    """Build simulation summary as a string."""
    if log_df.empty:
        return "No rounds played."

    lines = []
    total_rounds = log_df["timestep"].max() + 1
    final_wealth = log_df["investor_wealth"].iloc[-1]
    mode = "W1 (communication)" if cfg["world"]["communication"] else "W0 (independent)"

    lines.append(f"{'='*60}")
    lines.append("SIMULATION SUMMARY")
    lines.append(f"{'='*60}")
    lines.append(f"  Run ID:                {run_id}")
    lines.append(f"  World mode:            {mode}")
    lines.append(f"  Total rounds:          {total_rounds}")
    lines.append(f"  Final investor wealth: {final_wealth:.2f}")
    lines.append(f"  Investor cumulative:   {log_df['investor_cumulative'].iloc[-1]:.2f}")
    lines.append("")
    lines.append(f"  {'Agent':<12} {'Policy':<8} {'Cumulative':>12} {'Mean Repay%':>12}")
    for agent in agents:
        agent_rows = log_df[log_df["agent_name"] == agent.name]
        mean_repay = agent_rows["repayment_pct"].mean()
        lines.append(f"  {agent.name:<12} {agent.policy:<8} "
                      f"{agent.cumulative_reward:>12.2f} {mean_repay:>11.1%}")

    expected_cols = [
        "timestep", "agent_name", "agent_type", "world_mode",
        "observation_depth", "investor_wealth", "investment",
        "investment_multiplied", "repayment", "repayment_pct",
        "investor_reward", "agent_reward", "investor_cumulative",
        "agent_cumulative", "rnn_hidden_state", "observation_window",
    ]
    missing = set(expected_cols) - set(log_df.columns)
    if missing:
        lines.append(f"\n  WARNING: Missing columns: {missing}")
    else:
        lines.append(f"\n  CSV columns: OK ({len(expected_cols)} expected)")

    json_rows = log_df[log_df["rnn_hidden_state"] != ""]
    if not json_rows.empty:
        first_json_row = json_rows.iloc[0]
        sample_hidden = json.loads(first_json_row["rnn_hidden_state"])
        sample_obs = json.loads(first_json_row["observation_window"])
        lines.append(f"  rnn_hidden_state: valid JSON, length={len(sample_hidden)}")
        lines.append(f"  observation_window: valid JSON, keys={list(sample_obs.keys())}")

    return "\n".join(lines)


def _build_training_summary(
    stats: dict, agents: list, cfg: dict, game: Game, run_id: str,
) -> str:
    """Build training summary as a string."""
    lines = []
    returns = stats["training_returns"]
    mode = "W1 (communication)" if cfg["world"]["communication"] else "W0 (independent)"

    lines.append(f"\n{'='*60}")
    lines.append("TRAINING COMPLETE")
    lines.append(f"{'='*60}")
    lines.append(f"  Run ID:         {run_id}")
    lines.append(f"  World mode:     {mode}")
    lines.append(f"  Episodes:       {len(returns)}")
    lines.append(f"  Final epsilon:  {game._get_epsilon():.4f}")

    last_10 = returns[-10:] if len(returns) >= 10 else returns
    lines.append(f"\n  Last 10 episodes (mean):")
    lines.append(f"    Investor return: "
                 f"{np.mean([r['investor_return'] for r in last_10]):+.0f}")
    for agent in agents:
        mean_r = np.mean([r[agent.name] for r in last_10])
        lines.append(f"    {agent.name} ({agent.policy}): {mean_r:+.0f}")

    if stats["eval_history"]:
        last_eval = stats["eval_history"][-1]
        lines.append(f"\n  Last eval (ep {last_eval['episode']}):")
        lines.append(f"    Mean wealth: {last_eval['mean_wealth']:.0f}")
        for agent in agents:
            r = last_eval["mean_agent_rewards"][agent.name]
            rp = last_eval["mean_agent_repay"][agent.name]
            lines.append(f"    {agent.name}: reward={r:+.0f}  repay={rp:.2f}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------

def cmd_train(cfg: dict) -> None:
    """Train adversaries online. Creates a new run folder."""
    run_id = create_run_id(cfg)
    print(f"Run ID: {run_id}")

    # Save config snapshot
    out_dir = Path(get_output_dir(cfg, run_id))
    _save_config_snapshot(cfg, out_dir)

    world, agents, investor, _ = _build_components(cfg, run_id)
    dqn_cfg = cfg["dqn"]

    game = Game(cfg, world, investor, agents, training_mode=True, run_id=run_id)
    print(f"Game configured: endowment={cfg['game']['endowment']}, "
          f"max_rounds={cfg['game']['max_rounds']}, "
          f"multiplier={cfg['game']['multiplier']}")

    num_episodes = dqn_cfg["training_episodes"]
    eval_interval = dqn_cfg["eval_interval"]
    eval_episodes = dqn_cfg["eval_episodes"]
    print(f"\nTraining: {num_episodes} episodes, "
          f"eval every {eval_interval} ({eval_episodes} eval episodes)")

    stats = game.run_training(
        num_episodes=num_episodes,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )

    _save_training_curves(stats, agents, out_dir)

    # Print and save summary
    summary = _build_training_summary(stats, agents, cfg, game, run_id)
    print(summary)
    (out_dir / "training_summary.txt").write_text(summary)
    print(f"Summary saved to {out_dir / 'training_summary.txt'}")
    print(f"\nRun saved: checkpoints/{run_id}/  and  outputs/{run_id}/")


def cmd_simulate(cfg: dict, run_id: str) -> None:
    """Load trained weights from a specific run, simulate, export CSV."""
    print(f"Run ID: {run_id}")

    world, agents, investor, _ = _build_components(cfg, run_id)

    for agent in agents:
        if hasattr(agent, "load"):
            try:
                agent.load()
                print(f"Loaded weights for {agent.name}")
            except FileNotFoundError:
                print(f"  WARNING: No checkpoint for {agent.name}, using random weights")
        if hasattr(agent, "set_eval_mode"):
            agent.set_eval_mode(True)

    if investor.learns:
        try:
            investor.load()
            print("Loaded investor Q-learner weights")
        except FileNotFoundError:
            print("  WARNING: No investor checkpoint, using random weights")
        investor.set_eval_mode(True)

    game = Game(cfg, world, investor, agents, training_mode=False, run_id=run_id)
    print(f"Game configured: endowment={cfg['game']['endowment']}, "
          f"max_rounds={cfg['game']['max_rounds']}, "
          f"multiplier={cfg['game']['multiplier']}")

    print("\nRunning simulation (eval mode)...")
    log_df = game.run()

    out_dir = Path(get_output_dir(cfg, run_id))
    _export_csv(log_df, out_dir, cfg["export"]["filename"])

    # Print and save summary
    summary = _build_summary(log_df, agents, cfg, run_id)
    print(summary)
    (out_dir / "simulation_summary.txt").write_text(summary)
    print(f"Summary saved to {out_dir / 'simulation_summary.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial MRTT Pipeline")
    parser.add_argument(
        "mode", nargs="?", default="train",
        choices=["train", "simulate"],
        help="'train' or 'simulate' (default: train)",
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run ID for simulate mode (e.g., w0_20260404_153022). "
             "Use 'latest' for most recent run. Required for simulate.",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_all_seeds(cfg["seed"])

    if args.mode == "train":
        cmd_train(cfg)

    elif args.mode == "simulate":
        if args.run is None or args.run == "latest":
            run_id = _find_latest_run(cfg)
            print(f"Using latest run: {run_id}")
        else:
            run_id = args.run
        cmd_simulate(cfg, run_id)


if __name__ == "__main__":
    main()
