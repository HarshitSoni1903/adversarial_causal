"""Entry point: trains agents then simulates in one command.

Usage:
    python main.py              # Train + simulate with current config
    python main.py --sim-only w0_20260404_153022   # Re-simulate an existing run

Each run creates timestamped folders:
    checkpoints/<run_id>/         — model weights
    outputs/<run_id>/             — plots, CSVs, summaries, config snapshot
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from agents import create_all_agents
from agents.investor import RNNInvestor
from game import Game
from utils import load_config, set_all_seeds
from world import World


# ------------------------------------------------------------------
# Run ID & paths
# ------------------------------------------------------------------

def _create_run_id(cfg: dict) -> str:
    mode = "w1" if cfg["world"]["communication"] else "w0"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{ts}"


def _ckpt_dir(cfg: dict, run_id: str) -> Path:
    p = Path(cfg["checkpoints"]["base_dir"]) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _out_dir(cfg: dict, run_id: str) -> Path:
    p = Path(cfg["export"]["output_dir"]) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------
# Build components
# ------------------------------------------------------------------

def _build(cfg: dict, run_id: str):
    agent_names = [a["name"] for a in cfg["agents"]]

    world = World(
        communication=cfg["world"]["communication"],
        observation_depth=cfg["world"]["observation_depth"],
        agent_names=agent_names,
    )

    rnn_hidden = cfg["behavioral_rnn"]["hidden_size"]
    state_dim = world.own_obs_dim() + world.others_obs_dim() + rnn_hidden + 1
    print(f"State dim: {state_dim}  "
          f"(own={world.own_obs_dim()} + others={world.others_obs_dim()} "
          f"+ rnn_hidden={rnn_hidden} + round=1)")

    agents = create_all_agents(cfg, state_dim, run_id=run_id)
    print(f"Agents: {[(a.name, a.policy) for a in agents]}")

    investor = RNNInvestor(cfg, agent_names, run_id=run_id)
    if investor.learns:
        k = len(agent_names)
        ql_dim = k + rnn_hidden * k + 2 * k + 2
        print(f"Investor: frozen RNN + Q-learner (state_dim={ql_dim})")
    else:
        print("Investor: frozen BehavioralRNN only")

    return world, agents, investor


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def _save_curves(stats: dict, agents: list, save_dir: Path) -> None:
    returns = stats["training_returns"]
    if not returns:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    window = max(1, len(returns) // 50)

    # Plot 1: smoothed returns + eval
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    inv = [r["investor_return"] for r in returns]
    inv_s = np.convolve(inv, np.ones(window)/window, mode="valid")
    ax1.plot(inv_s, label="investor", color="black", lw=1.5)
    for a in agents:
        v = [r[a.name] for r in returns]
        ax1.plot(np.convolve(v, np.ones(window)/window, mode="valid"),
                 label=f"{a.name} ({a.policy})", alpha=0.8)
    ax1.set(xlabel="Episode", ylabel="Cumulative Reward (smoothed)", title="Training Returns")
    ax1.legend()

    eh = stats["eval_history"]
    if eh:
        eps = [e["episode"] for e in eh]
        ax2.plot(eps, [e["mean_wealth"] for e in eh], "k-", label="investor wealth", lw=1.5)
        for a in agents:
            ax2.plot(eps, [e["mean_agent_rewards"][a.name] for e in eh], label=a.name, alpha=0.8)
        ax2.set(xlabel="Episode", ylabel="Mean Reward / Wealth", title="Evaluation Snapshots")
        ax2.legend()
    plt.tight_layout(); plt.savefig(save_dir/"training_curves.png", dpi=150); plt.close(fig)

    # Plot 2: per-episode raw returns
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    ax.plot(inv, alpha=0.3, color="black", lw=0.5); ax.plot(inv_s, color="black", lw=1.5)
    ax.set(title="Investor Return per Episode", xlabel="Episode", ylabel="Return")
    colors = ["tab:blue","tab:orange","tab:green"]
    for i, a in enumerate(agents[:3]):
        ax = axes[(i+1)//2, (i+1)%2]
        v = [r[a.name] for r in returns]
        s = np.convolve(v, np.ones(window)/window, mode="valid")
        ax.plot(v, alpha=0.3, color=colors[i], lw=0.5); ax.plot(s, color=colors[i], lw=1.5)
        ax.set(title=f"{a.name} ({a.policy})", xlabel="Episode", ylabel="Return")
    plt.tight_layout(); plt.savefig(save_dir/"per_episode_returns.png", dpi=150); plt.close(fig2)

    # Plot 3: repayment rates
    if eh:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        eps = [e["episode"] for e in eh]
        for a in agents:
            ax3.plot(eps, [e["mean_agent_repay"][a.name] for e in eh],
                     label=f"{a.name} ({a.policy})", marker="o", ms=3)
        ax3.set(xlabel="Episode", ylabel="Mean Repayment %", title="Repayment Rates", ylim=(-0.05,1.05))
        ax3.legend(); plt.tight_layout()
        plt.savefig(save_dir/"repayment_rates.png", dpi=150); plt.close(fig3)


def _print_investor_qvals(investor):
    if not investor.learns:
        return
    print("\n  Investor Q-values per agent (greedy):")
    for i, name in enumerate(investor.agent_names):
        state = investor._build_ql_state(name, 25, 50)  # mid-episode state
        with torch.no_grad():
            q = investor.q_learner.policy_net(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        actions = investor.ql_actions
        best = q.argmax(1).item()
        vals = "  ".join(f"inv_{actions[j]}={'>' if j==best else ' '}{q[0,j]:.1f}" for j in range(len(actions)))
        print(f"    {name}: {vals}")


# ------------------------------------------------------------------
# Summary builders
# ------------------------------------------------------------------

def _training_summary(stats, agents, cfg, game, run_id) -> str:
    lines = []
    rets = stats["training_returns"]
    mode = "W1 (communication)" if cfg["world"]["communication"] else "W0 (independent)"
    lines.append(f"\n{'='*60}")
    lines.append("TRAINING COMPLETE")
    lines.append(f"{'='*60}")
    lines.append(f"  Run ID:       {run_id}")
    lines.append(f"  World mode:   {mode}")
    lines.append(f"  Episodes:     {len(rets)}")
    lines.append(f"  Epsilon:      {game._get_epsilon():.4f}")
    last = rets[-10:] if len(rets) >= 10 else rets
    lines.append(f"\n  Last 10 episodes (mean):")
    lines.append(f"    Investor: {np.mean([r['investor_return'] for r in last]):+.0f}")
    for a in agents:
        lines.append(f"    {a.name} ({a.policy}): {np.mean([r[a.name] for r in last]):+.0f}")
    if stats["eval_history"]:
        ev = stats["eval_history"][-1]
        lines.append(f"\n  Last eval (ep {ev['episode']}):")
        lines.append(f"    Wealth: {ev['mean_wealth']:.0f}")
        for a in agents:
            lines.append(f"    {a.name}: reward={ev['mean_agent_rewards'][a.name]:+.0f}"
                         f"  repay={ev['mean_agent_repay'][a.name]:.2f}")
    return "\n".join(lines)


def _simulation_summary(log_df, agents, cfg, run_id) -> str:
    if log_df.empty:
        return "No rounds played."
    lines = []
    mode = "W1 (communication)" if cfg["world"]["communication"] else "W0 (independent)"
    lines.append(f"\n{'='*60}")
    lines.append("SIMULATION SUMMARY")
    lines.append(f"{'='*60}")
    lines.append(f"  Run ID:          {run_id}")
    lines.append(f"  World mode:      {mode}")
    lines.append(f"  Total rounds:    {log_df['timestep'].max() + 1}")
    lines.append(f"  Final wealth:    {log_df['investor_wealth'].iloc[-1]:.2f}")
    lines.append(f"  Investor cumul:  {log_df['investor_cumulative'].iloc[-1]:.2f}")
    lines.append("")
    lines.append(f"  {'Agent':<12} {'Policy':<8} {'Cumulative':>12} {'Mean Repay%':>12}")
    for a in agents:
        rows = log_df[log_df["agent_name"] == a.name]
        cumul = rows["agent_cumulative"].iloc[-1] if not rows.empty else 0.0
        repay = rows["repayment_pct"].mean() if not rows.empty else 0.0
        lines.append(f"  {a.name:<12} {a.policy:<8} {cumul:>12.2f} {repay:>11.1%}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(cfg: dict) -> None:
    """Train + simulate in one shot. One run_id for everything."""
    run_id = _create_run_id(cfg)
    out = _out_dir(cfg, run_id)
    ckpt = _ckpt_dir(cfg, run_id)

    # Save config snapshot
    with open(out / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"{'='*60}")
    print(f"RUN: {run_id}")
    print(f"{'='*60}")

    # --- Build ---
    world, agents, investor = _build(cfg, run_id)
    dqn = cfg["dqn"]
    print(f"Game: endowment={cfg['game']['endowment']}, "
          f"multiplier={cfg['game']['multiplier']}, "
          f"train_rounds={dqn['training_rounds']}, "
          f"sim_rounds={cfg['game']['max_rounds']}")

    # --- Phase 1: TRAIN ---
    print(f"\n--- TRAINING ({dqn['training_episodes']} episodes) ---")
    game = Game(cfg, world, investor, agents, training_mode=True, run_id=run_id)
    stats = game.run_training(
        num_episodes=dqn["training_episodes"],
        eval_interval=dqn["eval_interval"],
        eval_episodes=dqn["eval_episodes"],
    )
    _save_curves(stats, agents, out)
    _print_investor_qvals(investor)
    train_summary = _training_summary(stats, agents, cfg, game, run_id)
    print(train_summary)
    (out / "training_summary.txt").write_text(train_summary)

    # Save training returns CSV
    rets_df = pd.DataFrame(stats["training_returns"])
    rets_df.insert(0, "episode", range(1, len(rets_df)+1))
    rets_df.to_csv(out / "training_returns.csv", index=False)

    # --- Phase 2: SIMULATE (same agents, same weights, eval mode) ---
    print(f"\n--- SIMULATION ({cfg['game']['max_rounds']} rounds, eval mode) ---")

    # Reset everything for simulation but keep trained weights
    world.reset()
    investor.reset()
    for a in agents:
        a.reset()

    # Set eval mode
    investor.set_eval_mode(True)
    for a in agents:
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    # Run full simulation
    game_sim = Game(cfg, world, investor, agents, training_mode=False, run_id=run_id)
    log_df = game_sim.run()

    # Export CSV
    csv_path = out / cfg["export"]["filename"]
    with tempfile.NamedTemporaryFile(mode="w", dir=out, suffix=".csv", delete=False) as tmp:
        log_df.to_csv(tmp, index=False)
        Path(tmp.name).rename(csv_path)
    print(f"Game log: {csv_path} ({len(log_df)} rows)")

    # Print investor decisions at simulation start
    _print_investor_qvals(investor)

    # Summary from CSV data (not stale agent objects)
    sim_summary = _simulation_summary(log_df, agents, cfg, run_id)
    print(sim_summary)
    (out / "simulation_summary.txt").write_text(sim_summary)

    print(f"\n{'='*60}")
    print(f"All outputs in: {out}/")
    print(f"Checkpoints in: {ckpt}/")
    print(f"{'='*60}")


def run_simulate_only(cfg: dict, run_id: str) -> None:
    """Re-simulate an existing run with its saved weights."""
    print(f"Re-simulating run: {run_id}")
    out = _out_dir(cfg, run_id)

    world, agents, investor = _build(cfg, run_id)

    # Load weights
    for a in agents:
        if hasattr(a, "load"):
            try:
                a.load()
                print(f"  Loaded {a.name}")
            except FileNotFoundError:
                print(f"  WARNING: No checkpoint for {a.name}")
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    if investor.learns:
        try:
            investor.load()
            print("  Loaded investor Q-learner")
        except FileNotFoundError:
            print("  WARNING: No investor checkpoint")
        investor.set_eval_mode(True)

    game = Game(cfg, world, investor, agents, training_mode=False, run_id=run_id)
    log_df = game.run()

    csv_path = out / cfg["export"]["filename"]
    with tempfile.NamedTemporaryFile(mode="w", dir=out, suffix=".csv", delete=False) as tmp:
        log_df.to_csv(tmp, index=False)
        Path(tmp.name).rename(csv_path)
    print(f"Game log: {csv_path} ({len(log_df)} rows)")

    _print_investor_qvals(investor)
    sim_summary = _simulation_summary(log_df, agents, cfg, run_id)
    print(sim_summary)
    (out / "simulation_summary.txt").write_text(sim_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial MRTT Pipeline")
    parser.add_argument(
        "--sim-only", type=str, default=None,
        help="Re-simulate an existing run by ID (e.g. w0_20260404_153022)",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_all_seeds(cfg["seed"])

    if args.sim_only:
        run_simulate_only(cfg, args.sim_only)
    else:
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
