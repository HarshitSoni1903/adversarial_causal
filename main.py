"""Entry point: parse → train RNN → train DQN adversaries → verify → smoke test.

Usage:
    python main.py                          # Full pipeline
    python main.py --skip-parse             # Skip MRTT CSV parsing
    python main.py --skip-rnn               # Skip BehavioralRNN training
    python main.py --skip-dqn              # Skip DQN adversary training
    python main.py --sim-only [run_id]      # Re-simulate saved weights (latest if omitted)
    python main.py --smoke-test             # N=2 allocation/W0/W1 smoke test only
    python main.py --experiment-matrix      # Full 5-condition experiment matrix
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from agents import compute_state_dim, create_all_agents
from agents.investor import RNNInvestor
from game import Game
from utils import load_config, set_all_seeds
from world import World


# ------------------------------------------------------------------
# Run ID & paths
# ------------------------------------------------------------------

def _create_run_id(cfg: dict) -> str:
    mode = "w1" if cfg["world"]["mode"] == 1 else "w0"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{ts}"


def _out_dir(cfg: dict, run_id: str) -> Path:
    p = Path(cfg["export"]["output_dir"]) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_latest_run_id(cfg: dict) -> str | None:
    """Return most-recent output subdirectory matching the world mode."""
    mode = "w1" if cfg["world"]["mode"] == 1 else "w0"
    out_base = Path(cfg["export"]["output_dir"])
    dirs = sorted(out_base.glob(f"{mode}_*"), reverse=True)
    return dirs[0].name if dirs else None


# ------------------------------------------------------------------
# Build components
# ------------------------------------------------------------------

def _build(cfg: dict):
    agent_names = [a["name"] for a in cfg["game"]["agents"]]

    world = World(
        mode=cfg["world"]["mode"],
        observation_depth=cfg["game"].get("observation_depth", 4),
        agent_names=agent_names,
    )

    state_dim = compute_state_dim(cfg, world)
    rnn_h = cfg["behavioral_rnn"]["hidden_size"]
    n_act = cfg["behavioral_rnn"]["n_actions"]
    print(f"Adversary state dim: {state_dim}  "
          f"(rnn_h={rnn_h} + policy={n_act} + ah={n_act} + round=1 "
          f"+ others_obs={world.others_obs_dim()})")

    agents = create_all_agents(cfg, state_dim)
    print(f"Agents: {[(a.name, a.policy) for a in agents]}")

    investor = RNNInvestor(cfg, agent_names)
    print("Investor: frozen BehavioralRNN")

    return world, agents, investor


# ------------------------------------------------------------------
# Pipeline steps
# ------------------------------------------------------------------

def step_parse(cfg: dict) -> None:
    print("\n--- STEP 1: PARSE MRTT DATA ---")
    result = subprocess.run(
        [sys.executable, "data/parse_mrtt.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Parse step failed")


def step_train_rnn(cfg: dict) -> None:
    print("\n--- STEP 2: TRAIN BEHAVIORAL RNN ---")
    result = subprocess.run(
        [sys.executable, "models/train_behavioral.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("RNN training failed")


def step_train_dqn(cfg: dict, run_id: str, out: Path) -> tuple:
    print("\n--- STEP 3: TRAIN DQN ADVERSARIES ---")
    world, agents, investor = _build(cfg)
    adv_cfg = cfg["adversary"]

    # Patch export dir to match this run's output dir
    cfg_patched = dict(cfg)
    cfg_patched["export"] = dict(cfg["export"])
    cfg_patched["export"]["output_dir"] = str(out.parent)

    game = Game(cfg, world, investor, agents)
    # Override export dir to this run's folder
    game.export_dir = out

    stats = game.run_training(
        num_episodes=adv_cfg["training_episodes"],
        eval_interval=adv_cfg["eval_interval"],
        eval_episodes=adv_cfg["eval_episodes"],
    )
    _save_training_curves(stats, agents, out)
    summary = _training_summary(stats, agents, cfg, game, run_id)
    print(summary)
    (out / "training_summary.txt").write_text(summary)
    return world, agents, investor


def step_verify(
    cfg: dict, world: World, agents: list, investor: RNNInvestor, out: Path, run_id: str,
    n_episodes: int = 1000,
) -> None:
    """Figure 5 verification: run n_episodes eval episodes, plot mean repayment by round."""
    print(f"\n--- STEP 4: FIGURE 5 VERIFICATION ({n_episodes} episodes) ---")

    for a in agents:
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    game = Game(cfg, world, investor, agents)
    game.export_dir = out

    all_rows: list[pd.DataFrame] = []
    for ep_i in range(n_episodes):
        world.reset()
        investor.reset()
        for a in agents:
            a.reset()
        stats = game._run_episode(log_details=True)
        df = stats["log_df"]
        df = df.copy()
        df["episode"] = ep_i
        all_rows.append(df)

    fig5_df = pd.concat(all_rows, ignore_index=True)
    fig5_csv = out / "fig5_verification.csv"
    fig5_df.to_csv(fig5_csv, index=False)
    print(f"Fig5 data: {fig5_csv}  ({len(fig5_df)} rows)")

    _save_fig5_plot(fig5_df, agents, out)
    _print_fig5_summary(fig5_df, agents)

    for a in agents:
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(False)


# ------------------------------------------------------------------
# Smoke test: N=2 allocation, W0/W1 invariance at N=1
# ------------------------------------------------------------------

def run_smoke_test(cfg: dict) -> None:
    """Structural and behavioral smoke tests.

    Test A — N=1 structural invariant:
      At N=1, others_obs_dim() == 0 for both W0 and W1.
      State dim must be exactly 16 regardless of mode.

    Test B — N=1 behavioral invariant (if checkpoint available):
      Load trained weights; run 10 greedy episodes each under W0 and W1
      with the same seed. At N=1, state vectors are identical → outcomes match.

    Test C — N=1 allocation:
      Per-round investment ≤ endowment for 10 episodes.

    Test D — N=2 allocation (if config has 2 agents):
      Per-round sum of investments ≤ endowment for 10 episodes under W0 + W1.
    """
    print("\n=== SMOKE TEST ===")
    agent_names = [a["name"] for a in cfg["game"]["agents"]]
    obs_depth = cfg["game"].get("observation_depth", 4)
    endowment_per_agent = cfg["game"].get("endowment_per_agent", cfg["game"].get("endowment", 20))

    # ------------------------------------------------------------------
    # Test A: structural dim invariant
    # ------------------------------------------------------------------
    print(f"\n[A] N={len(agent_names)} structural state-dim check")
    for mode_val in (0, 1):
        world = World(mode=mode_val, observation_depth=obs_depth, agent_names=agent_names)
        sd = compute_state_dim(cfg, world)
        ood = world.others_obs_dim()
        expected_sd = 16 + ood
        ok = (sd == expected_sd)
        print(f"  W{mode_val}: others_obs_dim={ood}  state_dim={sd}  {'OK' if ok else 'FAIL'}")
    if len(agent_names) == 1:
        # At N=1 others_obs_dim must be 0 for both modes
        for mode_val in (0, 1):
            w = World(mode=mode_val, observation_depth=obs_depth, agent_names=agent_names)
            assert w.others_obs_dim() == 0, f"Expected 0 at N=1, got {w.others_obs_dim()}"
        print("  N=1 W0==W1 state_dim invariant: PASS (others_obs=0 for both)")

    # ------------------------------------------------------------------
    # Test B: behavioral invariant — load checkpoint, same seed, compare
    # ------------------------------------------------------------------
    if len(agent_names) == 1:
        print(f"\n[B] N=1 W0==W1 behavioral invariant (greedy, same seed)")
        checkpoint_ok = True
        results = {}
        for mode_val in (0, 1):
            world = World(mode=mode_val, observation_depth=obs_depth, agent_names=agent_names)
            sd = compute_state_dim(cfg, world)
            agents = create_all_agents(cfg, sd)
            for a in agents:
                try:
                    a.load()
                except Exception:
                    checkpoint_ok = False
                    break
                if hasattr(a, "set_eval_mode"):
                    a.set_eval_mode(True)
            if not checkpoint_ok:
                break
            investor = RNNInvestor(cfg, agent_names)
            game = Game(cfg, world, investor, agents)
            set_all_seeds(cfg["seed"])
            rep_means = []
            for _ in range(10):
                world.reset(); investor.reset()
                for a in agents:
                    a.reset()
                stats = game._run_episode(log_details=False)
                rep_means.append(stats["agent_repay_means"][agent_names[0]])
            results[mode_val] = round(np.mean(rep_means), 6)

        if not checkpoint_ok:
            print("  SKIP — no checkpoint available (run training first)")
        else:
            diff = abs(results[0] - results[1])
            ok = diff < 1e-6
            print(f"  W0 mean_repay={results[0]:.4f}  W1 mean_repay={results[1]:.4f}")
            print(f"  W0==W1: {'PASS' if ok else 'FAIL'} (diff={diff:.2e})")

    # ------------------------------------------------------------------
    # Test C: N=1 allocation bound
    # ------------------------------------------------------------------
    print(f"\n[C] N=1 allocation bound (10 episodes)")
    world = World(mode=0, observation_depth=obs_depth, agent_names=agent_names)
    sd = compute_state_dim(cfg, world)
    agents = create_all_agents(cfg, sd)
    investor = RNNInvestor(cfg, agent_names)
    game = Game(cfg, world, investor, agents)
    set_all_seeds(cfg["seed"])
    violations = 0
    effective_endowment_c = endowment_per_agent * len(agent_names)
    for _ in range(10):
        world.reset(); investor.reset()
        for a in agents:
            a.reset()
        stats = game._run_episode(log_details=True)
        per_round = stats["log_df"].groupby("timestep")["investment"].sum()
        violations += int((per_round > effective_endowment_c + 1e-6).any())
    print(f"  allocation_violations={violations}/10  {'PASS' if violations == 0 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Test D: N=2 allocation (if config has 2 agents)
    # ------------------------------------------------------------------
    n_agents = len(agent_names)
    if n_agents >= 2:
        print(f"\n[D] N={n_agents} allocation bound (10 episodes, W0 + W1)")
        effective_endowment_d = endowment_per_agent * n_agents
        for mode_val in (0, 1):
            world = World(mode=mode_val, observation_depth=obs_depth, agent_names=agent_names)
            sd = compute_state_dim(cfg, world)
            agents = create_all_agents(cfg, sd)
            investor = RNNInvestor(cfg, agent_names)
            game = Game(cfg, world, investor, agents)
            set_all_seeds(cfg["seed"])
            violations = 0
            for _ in range(10):
                world.reset(); investor.reset()
                for a in agents:
                    a.reset()
                stats = game._run_episode(log_details=True)
                per_round = stats["log_df"].groupby("timestep")["investment"].sum()
                violations += int((per_round > effective_endowment_d + 1e-6).any())
            print(f"  W{mode_val}: violations={violations}/10  "
                  f"{'PASS' if violations == 0 else 'FAIL'}")

    print("\n=== SMOKE TEST DONE ===")


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def _save_training_curves(stats: dict, agents: list, save_dir: Path) -> None:
    returns = stats["training_returns"]
    if not returns:
        return
    window = max(1, len(returns) // 50)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    inv = [r["investor_return"] for r in returns]
    inv_s = np.convolve(inv, np.ones(window) / window, mode="valid")
    ax1.plot(inv_s, label="investor", color="black", lw=1.5)
    for a in agents:
        v = [r[a.name] for r in returns]
        ax1.plot(np.convolve(v, np.ones(window) / window, mode="valid"),
                 label=f"{a.name} ({a.policy})", alpha=0.8)
    ax1.set(xlabel="Episode", ylabel="Return (smoothed)", title="Training Returns")
    ax1.legend()

    eh = stats["eval_history"]
    if eh:
        eps = [e["episode"] for e in eh]
        ax2.plot(eps, [e["mean_wealth"] for e in eh], "k-", label="investor wealth")
        for a in agents:
            ax2.plot(eps, [e["mean_agent_rewards"][a.name] for e in eh], label=a.name)
        ax2.set(xlabel="Episode", ylabel="Mean Reward", title="Eval Snapshots")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    if eh:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for a in agents:
            ax3.plot(eps, [e["mean_agent_repay"][a.name] for e in eh],
                     label=f"{a.name} ({a.policy})", marker="o", ms=3)
        ax3.set(xlabel="Episode", ylabel="Mean Repayment %",
                title="Eval Repayment Rates", ylim=(-0.05, 1.05))
        ax3.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "repayment_rates.png", dpi=150)
        plt.close(fig3)


def _save_fig5_plot(fig5_df: pd.DataFrame, agents: list, save_dir: Path) -> None:
    """Figure 5 equivalent: mean repayment % by round, averaged across episodes."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for a in agents:
        rows = fig5_df[fig5_df["agent_name"] == a.name]
        by_round = rows.groupby("timestep")["repayment_pct"].mean()
        ax.plot(by_round.index, by_round.values,
                label=f"{a.name} ({a.policy})", marker="o", ms=4)
    ax.set(xlabel="Round", ylabel="Mean Repayment %",
           title="Fig 5: Repayment by Round (1000 ep)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    ax = axes[1]
    for a in agents:
        rows = fig5_df[fig5_df["agent_name"] == a.name]
        by_round = rows.groupby("timestep")["investment"].mean()
        ax.plot(by_round.index, by_round.values,
                label=f"{a.name} ({a.policy})", marker="o", ms=4)
    ax.set(xlabel="Round", ylabel="Mean Investment",
           title="Fig 5: Investment by Round (1000 ep)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "fig5_verification.png", dpi=150)
    plt.close(fig)


def _save_round_analysis(log_df: pd.DataFrame, agents: list, save_dir: Path,
                          label: str = "") -> None:
    if log_df.empty:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{label}_" if label else ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (axes[0, 0], "repayment_pct", "Repayment %", (-0.05, 1.05)),
        (axes[0, 1], "investment", "Investment", None),
        (axes[1, 0], "investor_reward", "Investor Reward", None),
        (axes[1, 1], "agent_reward", "Agent Reward", None),
    ]
    for ax, col, ylabel, ylim in panels:
        for a in agents:
            rows = log_df[log_df["agent_name"] == a.name]
            by_round = rows.groupby("timestep")[col].mean()
            ax.plot(by_round.index, by_round.values,
                    label=f"{a.name} ({a.policy})", marker="o", ms=3)
        ax.set(xlabel="Round", ylabel=ylabel, title=f"{ylabel} by Round")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()

    plt.suptitle(f"Per-Round Analysis{' (' + label + ')' if label else ''}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}round_analysis.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------

def _training_summary(stats: dict, agents: list, cfg: dict, game: Game, run_id: str) -> str:
    rets = stats["training_returns"]
    mode = "W1 (communication)" if cfg["world"]["mode"] == 1 else "W0 (independent)"
    lines = [
        f"\n{'='*60}", "TRAINING COMPLETE", f"{'='*60}",
        f"  Run ID:     {run_id}",
        f"  World:      {mode}",
        f"  Episodes:   {len(rets)}",
        f"  Epsilon:    {game._get_epsilon():.4f}",
    ]
    last = rets[-100:] if len(rets) >= 100 else rets
    lines.append(f"\n  Last {len(last)} episodes (mean):")
    lines.append(f"    Investor: {np.mean([r['investor_return'] for r in last]):+.0f}")
    for a in agents:
        lines.append(f"    {a.name} ({a.policy}): {np.mean([r[a.name] for r in last]):+.0f}")
    if stats["eval_history"]:
        ev = stats["eval_history"][-1]
        lines.append(f"\n  Last eval (ep {ev['episode']}):")
        lines.append(f"    Wealth: {ev['mean_wealth']:.0f}")
        for a in agents:
            lines.append(
                f"    {a.name}: reward={ev['mean_agent_rewards'][a.name]:+.0f}"
                f"  repay={ev['mean_agent_repay'][a.name]:.2f}"
            )
    return "\n".join(lines)


def _simulation_summary(log_df: pd.DataFrame, agents: list, cfg: dict,
                         run_id: str, label: str = "SIMULATION SUMMARY") -> str:
    if log_df.empty:
        return "No rounds played."
    mode = "W1 (communication)" if cfg["world"]["mode"] == 1 else "W0 (independent)"
    lines = [
        f"\n{'='*60}", label, f"{'='*60}",
        f"  Run ID:         {run_id}",
        f"  World mode:     {mode}",
        f"  Total rounds:   {log_df['timestep'].max() + 1}",
        f"  Final wealth:   {log_df['investor_wealth'].iloc[-1]:.2f}",
        f"  Inv cumulative: {log_df['investor_cumulative'].iloc[-1]:.2f}",
        "",
        f"  {'Agent':<12} {'Policy':<8} {'Cumulative':>12} {'Mean Repay%':>12}",
    ]
    for a in agents:
        rows = log_df[log_df["agent_name"] == a.name]
        cumul = rows["agent_cumulative"].iloc[-1] if not rows.empty else 0.0
        repay = rows["repayment_pct"].mean() if not rows.empty else 0.0
        lines.append(f"  {a.name:<12} {a.policy:<8} {cumul:>12.2f} {repay:>11.1%}")
    return "\n".join(lines)


def _print_fig5_summary(fig5_df: pd.DataFrame, agents: list) -> None:
    print("\n  Figure 5 Summary — Mean repayment % by round:")
    for a in agents:
        rows = fig5_df[fig5_df["agent_name"] == a.name]
        by_round = rows.groupby("timestep")["repayment_pct"].mean()
        vals = "  ".join(f"R{r}={v:.0%}" for r, v in by_round.items())
        print(f"    {a.name} ({a.policy}): {vals}")


# ------------------------------------------------------------------
# Simulate only (reload saved weights)
# ------------------------------------------------------------------

def run_simulate_only(cfg: dict, run_id: str) -> None:
    print(f"Re-simulating run: {run_id}")
    out = _out_dir(cfg, run_id)

    world, agents, investor = _build(cfg)

    for a in agents:
        if hasattr(a, "load"):
            try:
                a.load()
                print(f"  Loaded {a.name}")
            except (FileNotFoundError, Exception) as e:
                print(f"  WARNING: {a.name} load failed: {e}")
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    game = Game(cfg, world, investor, agents)
    game.export_dir = out
    log_df = game.run()

    csv_path = out / cfg["export"]["filename"]
    log_df.to_csv(csv_path, index=False)
    print(f"Game log: {csv_path} ({len(log_df)} rows)")

    sim_summary = _simulation_summary(log_df, agents, cfg, run_id)
    print(sim_summary)
    (out / "simulation_summary.txt").write_text(sim_summary)
    _save_round_analysis(log_df, agents, out, label="eval")

    step_verify(cfg, world, agents, investor, out, run_id)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(cfg: dict, skip_parse: bool, skip_rnn: bool, skip_dqn: bool) -> None:
    run_id = _create_run_id(cfg)
    out = _out_dir(cfg, run_id)

    with open(out / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"{'='*60}")
    print(f"RUN: {run_id}")
    print(f"{'='*60}")
    epa = cfg["game"].get("endowment_per_agent", cfg["game"].get("endowment", 20))
    n_agents = len(cfg["game"]["agents"])
    print(f"  endowment={epa}×{n_agents}={epa*n_agents}  "
          f"multiplier={cfg['game']['multiplier']}  "
          f"max_rounds={cfg['game']['max_rounds']}  "
          f"world_mode=W{cfg['world']['mode']}")

    if not skip_parse:
        step_parse(cfg)
    else:
        print("\n--- STEP 1: PARSE (skipped) ---")

    if not skip_rnn:
        step_train_rnn(cfg)
    else:
        print("\n--- STEP 2: TRAIN RNN (skipped) ---")

    if not skip_dqn:
        world, agents, investor = step_train_dqn(cfg, run_id, out)
    else:
        print("\n--- STEP 3: TRAIN DQN (skipped) ---")
        world, agents, investor = _build(cfg)
        for a in agents:
            try:
                a.load()
                print(f"  Loaded {a.name}")
            except Exception:
                print(f"  WARNING: no checkpoint for {a.name}")

    step_verify(cfg, world, agents, investor, out, run_id)

    print(f"\n{'='*60}")
    print(f"All outputs in: {out}/")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial MRTT Pipeline")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-rnn", action="store_true")
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument(
        "--sim-only", nargs="?", const="", default=None,
        help="Re-simulate saved weights. Pass run_id or omit to auto-detect latest.",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run N=1/N=2 allocation smoke tests and exit.")
    parser.add_argument("--experiment-matrix", action="store_true",
                        help="Run full 5-condition experiment matrix.")
    args = parser.parse_args()

    cfg = load_config()
    set_all_seeds(cfg["seed"])

    # Validate spillover_alpha at startup
    alpha = cfg["behavioral_rnn"].get("spillover_alpha", 0.0)
    if not (0.0 <= alpha <= 0.3):
        raise ValueError(
            f"spillover_alpha={alpha} out of range [0.0, 0.3]. "
            "Values >0.3 risk RNN extrapolation; adjust config."
        )

    if args.smoke_test:
        run_smoke_test(cfg)
    elif args.experiment_matrix:
        run_experiment_matrix(cfg)
    elif args.sim_only is not None:
        run_id = args.sim_only or _find_latest_run_id(cfg)
        if not run_id:
            print("No run found. Run training first.")
            return
        run_simulate_only(cfg, run_id)
    else:
        run_pipeline(
            cfg,
            skip_parse=args.skip_parse,
            skip_rnn=args.skip_rnn,
            skip_dqn=args.skip_dqn,
        )


# ===========================================================================
# EXPERIMENT MATRIX
# ===========================================================================

# (condition_id, N, world_mode, spillover_alpha, skip_training)
EXPERIMENT_CONDITIONS = [
    ("N1_W0_a0", 1, 0, 0.0, True),
    ("N2_W0_a0", 2, 0, 0.0, False),
    ("N2_W1_a0", 2, 1, 0.0, False),
    ("N2_W0_a3", 2, 0, 0.3, False),
    ("N2_W1_a3", 2, 1, 0.3, False),
]


def _build_condition_config(base_cfg: dict, condition: str, N: int,
                             world_mode: int, alpha: float) -> dict:
    """Deep-copy base config and override per-condition fields."""
    cfg = copy.deepcopy(base_cfg)
    cfg["world"]["mode"] = world_mode
    cfg["behavioral_rnn"]["spillover_alpha"] = alpha
    cfg["_condition"] = condition

    if N == 1:
        cfg["game"]["agents"] = [{
            "name": "max_1", "type": "max",
            "save_path": "checkpoints/adversary_max_1.pt",  # existing
        }]
    else:
        cfg["game"]["agents"] = [
            {"name": "max_1", "type": "max",
             "save_path": f"checkpoints/adversary_{condition}_max_1.pt"},
            {"name": "fair_1", "type": "fair",
             "save_path": f"checkpoints/adversary_{condition}_fair_1.pt"},
        ]
    return cfg


def _matrix_train(cfg: dict, cond_dir: Path) -> float:
    """Train DQNs for a condition. Returns elapsed seconds."""
    world, agents, investor = _build(cfg)
    adv_cfg = cfg["adversary"]
    game = Game(cfg, world, investor, agents)
    game.export_dir = cond_dir
    t0 = time.time()
    stats = game.run_training(
        num_episodes=adv_cfg["training_episodes"],
        eval_interval=adv_cfg["eval_interval"],
        eval_episodes=adv_cfg["eval_episodes"],
    )
    elapsed = time.time() - t0
    _save_training_curves(stats, agents, cond_dir)
    return elapsed


def _matrix_eval(cfg: dict, cond_dir: Path, n_episodes: int = 1000
                  ) -> tuple[pd.DataFrame, list[dict], list]:
    """Load trained agents and run eval episodes. Returns (log_df, ep_stats, agents)."""
    world, agents, investor = _build(cfg)

    for a in agents:
        if hasattr(a, "load"):
            try:
                a.load()
                print(f"    Loaded {a.name}")
            except Exception as exc:
                print(f"    WARNING: could not load {a.name}: {exc}")
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    game = Game(cfg, world, investor, agents)
    game.export_dir = cond_dir

    all_rows: list[pd.DataFrame] = []
    ep_stats: list[dict] = []

    for ep_i in range(n_episodes):
        world.reset()
        investor.reset()
        for a in agents:
            a.reset()
        result = game._run_episode(log_details=True)
        df = result["log_df"].copy()
        df["episode"] = ep_i
        all_rows.append(df)
        stat: dict = {
            "episode": ep_i,
            "investor_cumulative": result["investor_cumulative"],
        }
        for a in agents:
            stat[f"{a.name}_reward"] = result["agent_rewards"][a.name]
        ep_stats.append(stat)

    log_df = pd.concat(all_rows, ignore_index=True)
    log_df.to_csv(cond_dir / "game_log.csv", index=False)
    return log_df, ep_stats, agents


def _audit_invariants(log_df: pd.DataFrame, condition: str,
                       endowment_per_agent: float) -> list[str]:
    """Verify per-row and per-round invariants. Returns list of error strings."""
    errors: list[str] = []
    n_agent_names = log_df["agent_name"].nunique()
    wallet = endowment_per_agent * n_agent_names

    # Reward conservation: investor_reward + agent_reward == 2 * investment
    row_sum = log_df["investor_reward"] + log_df["agent_reward"]
    expected = 2.0 * log_df["investment"]
    bad = ~np.isclose(row_sum.values, expected.values, atol=1e-3)
    if bad.any():
        errors.append(
            f"{condition}: reward invariant violated in {bad.sum()}/{len(bad)} rows"
        )

    # Budget: sum of investments per (episode, round) <= wallet
    per_round = log_df.groupby(["episode", "timestep"])["investment"].sum()
    violations = int((per_round > wallet + 1e-6).sum())
    if violations > 0:
        errors.append(
            f"{condition}: budget violated in {violations} rounds (wallet={wallet})"
        )

    return errors


def _save_condition_plots(log_df: pd.DataFrame, agents: list,
                           cond_dir: Path, condition: str) -> None:
    """Save Figure-5-style per-round plots for a condition."""
    cond_dir.mkdir(parents=True, exist_ok=True)

    for col, fname, ylabel in [
        ("repayment_pct", "figure5c_mean_repayment.png", "Mean Repayment %"),
        ("investment", "figure5c_mean_investment.png", "Mean Investment"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for a in agents:
            rows = log_df[log_df["agent_name"] == a.name]
            by_round = rows.groupby("timestep")[col].mean()
            ax.plot(by_round.index, by_round.values,
                    label=f"{a.name} ({a.policy})", marker="o", ms=4)
        ax.set(xlabel="Round", ylabel=ylabel,
               title=f"{ylabel} by Round — {condition}")
        if col == "repayment_pct":
            ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.savefig(cond_dir / fname, dpi=150)
        plt.close(fig)


def _save_condition_summary(
    log_df: pd.DataFrame,
    ep_stats: list[dict],
    agents: list,
    cond_dir: Path,
    condition: str,
    N: int,
    world_mode: int,
    alpha: float,
    train_time: float,
) -> dict:
    """Build and persist summary.json for one condition. Returns summary dict."""
    agent_summaries: dict = {}
    for a in agents:
        rows = log_df[log_df["agent_name"] == a.name]
        by_round = rows.groupby("timestep")["repayment_pct"].mean().to_dict()
        rewards = [ep.get(f"{a.name}_reward", 0.0) for ep in ep_stats]

        fair_gaps: list[float] = []
        if a.policy == "fair":
            for ep_i in log_df["episode"].unique():
                ep_rows = rows[rows["episode"] == ep_i]
                inv_from_i = float(ep_rows["investor_reward"].sum())
                agent_total = float(ep_rows["agent_reward"].sum())
                fair_gaps.append(abs(inv_from_i - agent_total))

        agent_summaries[a.name] = {
            "policy": a.policy,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_repayment_by_round": {str(k): float(v) for k, v in by_round.items()},
            "fair_gap_mean": float(np.mean(fair_gaps)) if fair_gaps else None,
        }

    inv_cumulatives = [ep["investor_cumulative"] for ep in ep_stats]
    summary = {
        "condition": condition,
        "N": N,
        "world_mode": world_mode,
        "spillover_alpha": alpha,
        "n_eval_episodes": len(ep_stats),
        "investor_earnings_mean": float(np.mean(inv_cumulatives)),
        "investor_earnings_std": float(np.std(inv_cumulatives)),
        "train_time_seconds": train_time,
        "agents": agent_summaries,
    }
    with open(cond_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _bootstrap_diff_ci(
    data_a: np.ndarray, data_b: np.ndarray,
    n_boot: int = 5000, ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(b) - mean(a) from independent samples."""
    rng = np.random.default_rng(42)
    diffs = np.array([
        np.mean(rng.choice(data_b, len(data_b), replace=True))
        - np.mean(rng.choice(data_a, len(data_a), replace=True))
        for _ in range(n_boot)
    ])
    obs = float(np.mean(data_b) - np.mean(data_a))
    lo = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    return obs, lo, hi


def _produce_2x2_grid(all_log_dfs: dict, matrix_dir: Path) -> None:
    """Save 2×2 grid: W0/W1 (rows) × α=0/α=0.3 (cols), mean repayment by round."""
    layout = [
        ("N2_W0_a0", "W0, α=0.0"),
        ("N2_W0_a3", "W0, α=0.3"),
        ("N2_W1_a0", "W1, α=0.0"),
        ("N2_W1_a3", "W1, α=0.3"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for (cond, title), ax in zip(layout, axes.flat):
        if cond not in all_log_dfs:
            ax.set_title(f"{title}\n(missing)")
            continue
        log_df = all_log_dfs[cond]
        ax.set_title(title, fontsize=12)
        for aname in sorted(log_df["agent_name"].unique()):
            rows = log_df[log_df["agent_name"] == aname]
            policy = rows["agent_type"].iloc[0] if not rows.empty else aname
            by_round = rows.groupby("timestep")["repayment_pct"].mean()
            ax.plot(by_round.index, by_round.values,
                    label=f"{aname} ({policy})", marker="o", ms=4)
        ax.set(xlabel="Round", ylabel="Mean Repayment %")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

    plt.suptitle(
        "2×2 Comparison: World Mode × Spillover α\n"
        "Mean Repayment % by Round (1000 eval episodes)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(matrix_dir / "comparison_2x2_grid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {matrix_dir / 'comparison_2x2_grid.png'}")


def _produce_main_effects_table(all_summaries: dict, matrix_dir: Path) -> pd.DataFrame:
    """Build and save main_effects_table.csv."""
    rows: list[dict] = []
    for cond, summary in all_summaries.items():
        inv_mean = summary["investor_earnings_mean"]
        for aname, asum in summary["agents"].items():
            fair_gap = asum["fair_gap_mean"]
            rows.append({
                "Condition": cond,
                "Type": asum["policy"].upper(),
                "Agent": aname,
                "Investor_earnings": round(inv_mean, 2),
                "Agent_earnings": round(asum["mean_reward"], 2),
                "FAIR_gap": round(fair_gap, 2) if fair_gap is not None else "-",
            })
    df = pd.DataFrame(rows)
    csv_path = matrix_dir / "main_effects_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMain effects table ({csv_path}):")
    print(df.to_string(index=False))
    return df


def _produce_effect_decomposition(all_ep_stats: dict, matrix_dir: Path) -> pd.DataFrame:
    """Compute planned comparisons with bootstrap 95% CIs."""

    def _inv(cond: str) -> np.ndarray | None:
        if cond not in all_ep_stats:
            return None
        return np.array([ep["investor_cumulative"] for ep in all_ep_stats[cond]])

    d_w0a0 = _inv("N2_W0_a0")
    d_w1a0 = _inv("N2_W1_a0")
    d_w0a3 = _inv("N2_W0_a3")
    d_w1a3 = _inv("N2_W1_a3")

    rows: list[dict] = []

    def _add(label: str, comp: str, a: np.ndarray | None, b: np.ndarray | None) -> None:
        if a is None or b is None:
            return
        obs, lo, hi = _bootstrap_diff_ci(a, b)
        rows.append({
            "Effect": label,
            "Comparison": comp,
            "Point_est": round(obs, 3),
            "CI_lo_95": round(lo, 3),
            "CI_hi_95": round(hi, 3),
        })

    _add("W effect (alpha=0.0)", "N2_W1_a0 − N2_W0_a0", d_w0a0, d_w1a0)
    _add("W effect (alpha=0.3)", "N2_W1_a3 − N2_W0_a3", d_w0a3, d_w1a3)
    _add("Spillover effect (W0)", "N2_W0_a3 − N2_W0_a0", d_w0a0, d_w0a3)
    _add("Spillover effect (W1)", "N2_W1_a3 − N2_W1_a0", d_w1a0, d_w1a3)

    # Interaction: (W1a3 − W0a3) − (W1a0 − W0a0)
    if all(x is not None for x in [d_w0a0, d_w1a0, d_w0a3, d_w1a3]):
        rng = np.random.default_rng(43)
        interactions = [
            (np.mean(rng.choice(d_w1a3, len(d_w1a3), replace=True))
             - np.mean(rng.choice(d_w0a3, len(d_w0a3), replace=True)))
            - (np.mean(rng.choice(d_w1a0, len(d_w1a0), replace=True))
               - np.mean(rng.choice(d_w0a0, len(d_w0a0), replace=True)))
            for _ in range(5000)
        ]
        obs_int = float((np.mean(d_w1a3) - np.mean(d_w0a3))
                        - (np.mean(d_w1a0) - np.mean(d_w0a0)))
        rows.append({
            "Effect": "W × Spillover interaction",
            "Comparison": "(W1a3−W0a3) − (W1a0−W0a0)",
            "Point_est": round(obs_int, 3),
            "CI_lo_95": round(float(np.percentile(interactions, 2.5)), 3),
            "CI_hi_95": round(float(np.percentile(interactions, 97.5)), 3),
        })

    df = pd.DataFrame(rows)
    csv_path = matrix_dir / "effect_decomposition.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nEffect decomposition ({csv_path}):")
    print(df.to_string(index=False))
    return df


def _write_experiment_report(
    all_summaries: dict,
    main_eff_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    matrix_dir: Path,
) -> None:
    """Write experiment_report.md."""
    lines = [
        "# Experiment Matrix Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Condition Summaries",
        "",
    ]
    for cond, s in all_summaries.items():
        lines += [
            f"### {cond}",
            f"- N={s['N']}, World=W{s['world_mode']}, α={s['spillover_alpha']}",
            f"- Investor earnings: {s['investor_earnings_mean']:.2f} "
            f"± {s['investor_earnings_std']:.2f}  (n={s['n_eval_episodes']})",
        ]
        for aname, asum in s["agents"].items():
            fg = f", FAIR gap={asum['fair_gap_mean']:.2f}" if asum["fair_gap_mean"] is not None else ""
            lines.append(
                f"- {aname} ({asum['policy']}): reward={asum['mean_reward']:.2f}"
                f" ± {asum['std_reward']:.2f}{fg}"
            )
        if s["train_time_seconds"] > 0:
            lines.append(f"- Training time: {s['train_time_seconds']:.1f}s")
        lines.append("")

    lines += ["## Main Effects Table", ""]
    lines.append(main_eff_df.to_string(index=False))
    lines += ["", "## Effect Decomposition (95% bootstrap CI)", ""]
    if not effects_df.empty:
        lines.append(effects_df.to_string(index=False))
    lines.append("")

    lines += ["## Observations", ""]

    def _row(effect: str) -> dict | None:
        if effects_df.empty:
            return None
        match = effects_df[effects_df["Effect"] == effect]
        return match.iloc[0].to_dict() if not match.empty else None

    # Observation 1
    r = _row("W effect (alpha=0.0)")
    lines.append("### 1. Does W1 help adversaries at alpha=0?")
    if r:
        pe, lo, hi = r["Point_est"], r["CI_lo_95"], r["CI_hi_95"]
        direction = "increases" if pe > 0 else "decreases"
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not significant at 95%)"
        lines.append(
            f"W1 {direction} investor earnings by {pe:+.2f} [{lo:.2f}, {hi:.2f}]. {sig}."
        )
    lines.append("")

    # Observation 2
    r = _row("Spillover effect (W0)")
    lines.append("### 2. Does alpha=0.3 change anything at W0?")
    if r:
        pe, lo, hi = r["Point_est"], r["CI_lo_95"], r["CI_hi_95"]
        direction = "increases" if pe > 0 else "decreases"
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not significant at 95%)"
        lines.append(
            f"Spillover at W0 {direction} investor earnings by {pe:+.2f} [{lo:.2f}, {hi:.2f}]. {sig}."
        )
    lines.append("")

    # Observation 3
    r = _row("W × Spillover interaction")
    lines.append("### 3. Is there a W×alpha interaction?")
    if r:
        pe, lo, hi = r["Point_est"], r["CI_lo_95"], r["CI_hi_95"]
        direction = "amplifies" if pe > 0 else "dampens"
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not significant at 95%)"
        lines.append(
            f"W×α interaction = {pe:+.2f} [{lo:.2f}, {hi:.2f}]. "
            f"Spillover {direction} the W effect. {sig}."
        )
    lines.append("")

    lines += [
        "## Limitations",
        "",
        "- Single training run per condition; no error bars on DQN convergence.",
        "- 1000 eval episodes; CIs reflect sampling variance only.",
        "- Alpha restricted to [0.0, 0.3] to avoid RNN extrapolation risk.",
        "- N=1 replication uses frozen existing checkpoint (no retraining).",
        "- Spillover operates on investor RNN hidden states only; "
          "no RNN input augmentation.",
    ]

    report_path = matrix_dir / "experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nExperiment report: {report_path}")


def run_experiment_matrix(cfg: dict) -> None:
    """Run the full 5-condition experiment matrix (Parts C–E)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = Path(f"outputs/experiment_matrix_{timestamp}")
    matrix_dir.mkdir(parents=True, exist_ok=True)

    endowment_per_agent = float(
        cfg["game"].get("endowment_per_agent", cfg["game"].get("endowment", 20))
    )

    print(f"\n{'='*60}")
    print("EXPERIMENT MATRIX")
    print(f"Output: {matrix_dir}")
    print(f"Conditions: {[c[0] for c in EXPERIMENT_CONDITIONS]}")
    print(f"{'='*60}")

    all_log_dfs: dict = {}
    all_ep_stats: dict = {}
    all_summaries: dict = {}
    all_errors: list[str] = []
    training_times: dict = {}

    for condition, N, world_mode, alpha, skip_training in EXPERIMENT_CONDITIONS:
        print(f"\n{'─'*55}")
        print(f"Condition: {condition}  (N={N}, W{world_mode}, α={alpha})")
        print(f"{'─'*55}")

        cond_dir = matrix_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        cond_cfg = _build_condition_config(cfg, condition, N, world_mode, alpha)
        # Save condition config for reproducibility
        with open(cond_dir / "config.yaml", "w") as f:
            safe_cfg = {k: v for k, v in cond_cfg.items() if not k.startswith("_")}
            yaml.dump(safe_cfg, f, default_flow_style=False, sort_keys=False)

        # --- Training ---
        train_time = 0.0
        if skip_training:
            print("  Training: skipped (using existing checkpoint)")
        else:
            adv_ep = cond_cfg["adversary"]["training_episodes"]
            print(f"  Training: {adv_ep} episodes...")
            t_train = time.time()
            train_time = _matrix_train(cond_cfg, cond_dir)
            print(f"  Training time: {train_time:.1f}s")
        training_times[condition] = train_time

        # --- Evaluation ---
        print("  Evaluating: 1000 episodes...")
        t_eval = time.time()
        log_df, ep_stats, agents = _matrix_eval(cond_cfg, cond_dir, n_episodes=1000)
        eval_time = time.time() - t_eval
        print(f"  Eval time: {eval_time:.1f}s")

        all_log_dfs[condition] = log_df
        all_ep_stats[condition] = ep_stats

        # --- Invariant audit ---
        errors = _audit_invariants(log_df, condition, endowment_per_agent)
        all_errors.extend(errors)
        if errors:
            print(f"  INVARIANT VIOLATIONS:")
            for e in errors:
                print(f"    {e}")
            raise RuntimeError(
                f"Invariant audit failed for {condition}. Halting. Errors:\n"
                + "\n".join(errors)
            )
        else:
            print("  Invariant audit: PASS")

        # --- N1 Fig5D replication check ---
        if condition == "N1_W0_a0":
            max_rows = log_df[log_df["agent_name"] == "max_1"]
            by_round = max_rows.groupby("timestep")["repayment_pct"].mean()
            r0 = float(by_round.iloc[0]) if len(by_round) > 0 else float("nan")
            rlast = float(by_round.iloc[-1]) if len(by_round) > 0 else float("nan")
            if r0 > 0.5 and rlast < 0.3:
                print(f"  Fig 5D replication: PASS (R0={r0:.1%} → R9={rlast:.1%})")
            else:
                print(
                    f"  Fig 5D replication: WARNING — "
                    f"R0={r0:.1%} → R9={rlast:.1%} (expected ~74%→7%)"
                )

        # --- Plots & summary ---
        _save_condition_plots(log_df, agents, cond_dir, condition)
        summary = _save_condition_summary(
            log_df, ep_stats, agents, cond_dir,
            condition, N, world_mode, alpha, train_time,
        )
        all_summaries[condition] = summary

        inv_mean = summary["investor_earnings_mean"]
        print(f"  Investor mean earnings: {inv_mean:+.1f}")
        for aname, asum in summary["agents"].items():
            print(f"  {aname} ({asum['policy']}): reward={asum['mean_reward']:+.1f}")

    # --- Aggregate outputs ---
    print(f"\n{'─'*55}")
    print("Producing aggregate outputs...")

    _produce_2x2_grid(all_log_dfs, matrix_dir)
    main_eff_df = _produce_main_effects_table(all_summaries, matrix_dir)
    effects_df = _produce_effect_decomposition(all_ep_stats, matrix_dir)
    _write_experiment_report(all_summaries, main_eff_df, effects_df, matrix_dir)

    # --- Invariant summary ---
    if all_errors:
        print(f"\nINVARIANT AUDIT FAILURES ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
    else:
        print("\nInvariant audit: ALL CONDITIONS PASS")

    # --- Training time summary ---
    total_train = sum(training_times.values())
    print(f"\nTraining times (total: {total_train:.1f}s):")
    for cond, t in training_times.items():
        status = "skipped" if t == 0.0 else f"{t:.1f}s"
        print(f"  {cond}: {status}")

    print(f"\n{'='*60}")
    print(f"Experiment matrix complete.")
    print(f"All outputs in: {matrix_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
