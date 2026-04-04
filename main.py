"""Entry point: wires together all components and runs the pipeline.

Usage:
    python main.py train      # Train adversaries online, save weights
    python main.py simulate   # Load trained weights, run one game, export CSV

Outputs are organized by world mode:
    outputs/w0/   — communication=false results
    outputs/w1/   — communication=true results
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import create_all_agents
from agents.investor import RNNInvestor
from game import Game
from utils import get_checkpoint_dir, load_config, set_all_seeds
from world import World


def _get_output_dir(cfg: dict) -> Path:
    """Return world-mode-specific output directory (outputs/w0/ or outputs/w1/)."""
    base = Path(cfg["export"]["output_dir"])
    mode = "w1" if cfg["world"]["communication"] else "w0"
    out = base / mode
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_components(cfg: dict) -> tuple[World, list, RNNInvestor, int]:
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

    agents = create_all_agents(cfg, state_dim)
    print(f"Agents: {[(a.name, a.policy) for a in agents]}")

    investor = RNNInvestor(cfg, agent_names)
    if investor.learns:
        k = len(agent_names)
        ql_state_dim = k + rnn_hidden_size * k + 2 * k + 2
        print(f"Investor loaded (frozen RNN + Q-learner, state_dim={ql_state_dim})")
    else:
        print("Investor loaded (frozen BehavioralRNN)")

    return world, agents, investor, state_dim


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

    # --- Plot 2: Per-episode raw returns (with smoothed overlay) ---
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

    # --- Plot 3: Repayment rates from eval snapshots ---
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


def _export_csv(log_df: pd.DataFrame, cfg: dict) -> None:
    """Export game log atomically to world-mode-specific output folder."""
    out_dir = _get_output_dir(cfg)
    out_path = out_dir / cfg["export"]["filename"]

    with tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".csv", delete=False,
    ) as tmp:
        log_df.to_csv(tmp, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.rename(out_path)
    print(f"Game log saved to {out_path}  ({len(log_df)} rows)")


def _build_summary(log_df: pd.DataFrame, agents: list, cfg: dict) -> str:
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
    stats: dict, agents: list, cfg: dict, game: Game,
) -> str:
    """Build training summary as a string."""
    lines = []
    returns = stats["training_returns"]
    mode = "W1 (communication)" if cfg["world"]["communication"] else "W0 (independent)"

    lines.append(f"\n{'='*60}")
    lines.append("TRAINING COMPLETE")
    lines.append(f"{'='*60}")
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
    """Train adversaries online against frozen RNN investor."""
    world, agents, investor, _ = _build_components(cfg)
    dqn_cfg = cfg["dqn"]

    game = Game(cfg, world, investor, agents, training_mode=True)
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

    out_dir = _get_output_dir(cfg)
    _save_training_curves(stats, agents, out_dir)

    # Print and save training summary
    summary = _build_training_summary(stats, agents, cfg, game)
    print(summary)
    summary_path = out_dir / "training_summary.txt"
    summary_path.write_text(summary)
    print(f"Summary saved to {summary_path}")


def cmd_simulate(cfg: dict) -> None:
    """Load trained weights, run one game, export CSV."""
    world, agents, investor, _ = _build_components(cfg)

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
            print("  WARNING: No investor checkpoint, using random Q-learner weights")
        investor.set_eval_mode(True)

    game = Game(cfg, world, investor, agents, training_mode=False)
    print(f"Game configured: endowment={cfg['game']['endowment']}, "
          f"max_rounds={cfg['game']['max_rounds']}, "
          f"multiplier={cfg['game']['multiplier']}")

    print("\nRunning simulation (eval mode)...")
    log_df = game.run()

    _export_csv(log_df, cfg)

    # Print and save simulation summary
    summary = _build_summary(log_df, agents, cfg)
    print(summary)
    out_dir = _get_output_dir(cfg)
    summary_path = out_dir / "simulation_summary.txt"
    summary_path.write_text(summary)
    print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial MRTT Pipeline")
    parser.add_argument(
        "mode", nargs="?", default="train",
        choices=["train", "simulate"],
        help="'train' to train adversaries, 'simulate' to run with trained weights",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_all_seeds(cfg["seed"])

    if args.mode == "train":
        cmd_train(cfg)
    elif args.mode == "simulate":
        cmd_simulate(cfg)


if __name__ == "__main__":
    main()
