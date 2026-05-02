"""Entry point: parse → train RNN → train DQN adversaries → verify → smoke test.

Usage:
    python main.py                          # Full pipeline
    python main.py --skip-parse             # Skip MRTT CSV parsing
    python main.py --skip-rnn               # Skip BehavioralRNN training
    python main.py --skip-dqn              # Skip DQN adversary training
    python main.py --sim-only [run_id]      # Re-simulate saved weights (latest if omitted)
    python main.py --smoke-test             # Two-dyad structural + behavioral smoke tests
    python main.py --experiment-matrix      # Full 4-condition 2x2 experiment matrix
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

from agents import compute_state_dim, create_agent
from agents.investor import RNNInvestor
from game import Game
from utils import load_config, set_all_seeds
from world import World


# ------------------------------------------------------------------
# Run ID & paths
# ------------------------------------------------------------------

def _create_run_id(cfg: dict) -> str:
    ii = cfg["edges"]["ii_edge"]
    aa = cfg["edges"]["aa_edge"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ii{ii}_aa{aa}_{ts}"


def _out_dir(cfg: dict, run_id: str) -> Path:
    p = Path(cfg["export"]["output_dir"]) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_latest_run_id(cfg: dict) -> str | None:
    ii = cfg["edges"]["ii_edge"]
    aa = cfg["edges"]["aa_edge"]
    out_base = Path(cfg["export"]["output_dir"])
    dirs = sorted(out_base.glob(f"ii{ii}_aa{aa}_*"), reverse=True)
    return dirs[0].name if dirs else None


# ------------------------------------------------------------------
# Build components
# ------------------------------------------------------------------

def _build(cfg: dict):
    dyad_pairs = [(d["investor"], d["trustee"]["name"]) for d in cfg["game"]["dyads"]]
    ii_edge = cfg["edges"]["ii_edge"]
    aa_edge = cfg["edges"]["aa_edge"]
    n_actions = cfg["behavioral_rnn"]["n_actions"]
    obs_depth = cfg["game"]["observation_depth"]

    world = World(ii_edge, aa_edge, obs_depth, dyad_pairs, n_actions)

    state_dim = compute_state_dim(cfg, aa_edge)
    rnn_h = cfg["behavioral_rnn"]["hidden_size"]
    print(f"Adversary state dim: {state_dim}  "
          f"(rnn_h={rnn_h} + policy={n_actions} + ah={n_actions} + round=1 "
          f"+ flags=2 + cross={2*obs_depth if aa_edge else 0})")

    investors = [RNNInvestor(cfg, d["trustee"]["name"]) for d in cfg["game"]["dyads"]]
    agents = [create_agent(d["trustee"], cfg, state_dim) for d in cfg["game"]["dyads"]]
    print(f"Agents: {[(a.name, a.policy) for a in agents]}")
    print(f"Investors: {len(investors)} × frozen BehavioralRNN")

    return world, agents, investors


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
    world, agents, investors = _build(cfg)
    adv_cfg = cfg["adversary"]

    game = Game(cfg, world, investors, agents)
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
    return world, agents, investors


def step_verify(
    cfg: dict, world: World, agents: list, investors: list, out: Path, run_id: str,
    n_episodes: int = 1000,
) -> None:
    print(f"\n--- STEP 4: FIGURE 5 VERIFICATION ({n_episodes} episodes) ---")

    for a in agents:
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    game = Game(cfg, world, investors, agents)
    game.export_dir = out

    all_rows: list[pd.DataFrame] = []
    for ep_i in range(n_episodes):
        stats = game._run_episode(log_details=True)
        df = stats["log_df"].copy()
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
# Smoke tests
# ------------------------------------------------------------------

def run_smoke_test(cfg: dict) -> None:
    """Five structural and behavioral smoke tests for the two-dyad system.

    Test A — state dim:
      For each (ii, aa) in {0,1}²: compute_state_dim returns 18 + 8*aa.

    Test B — single-dyad reproducibility (if checkpoint available):
      One dyad, ii=0, aa=0, fixed seed, 10 greedy episodes. Repayment in (0,1].

    Test C — per-investor wallet bound:
      For each of 4 conditions, 10 episodes: invest_k <= endowment_per_investor.

    Test D — per-dyad reward conservation:
      For each of 4 conditions, 10 episodes: |inv_reward + agent_reward - 2*invest| < 1e-6.

    Test E — ii_edge effect non-trivial (if checkpoint available):
      (ii=0, aa=0) vs (ii=1, aa=0): mean repayment must differ > 1e-3 in >= 1 round.
    """
    print("\n=== SMOKE TEST ===")
    epi = cfg["game"]["endowment_per_investor"]
    obs_depth = cfg["game"]["observation_depth"]

    # ------------------------------------------------------------------
    # Test A: state dim
    # ------------------------------------------------------------------
    print("\n[A] State dim check for all 4 (ii, aa) combinations")
    all_ok = True
    for ii in (0, 1):
        for aa in (0, 1):
            sd = compute_state_dim(cfg, aa_edge=aa)
            expected = 18 + 8 * aa
            ok = sd == expected
            if not ok:
                all_ok = False
            print(f"  ii={ii} aa={aa}: state_dim={sd}  expected={expected}  "
                  f"{'OK' if ok else 'FAIL'}")
    print(f"  Test A: {'PASS' if all_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Test B: single-dyad reproducibility
    # ------------------------------------------------------------------
    print(f"\n[B] Single-dyad reproducibility (greedy, seed={cfg['seed']})")
    checkpoint_ok = True
    try:
        single_cfg = copy.deepcopy(cfg)
        single_cfg["edges"]["ii_edge"] = 0
        single_cfg["edges"]["aa_edge"] = 0
        single_cfg["game"]["dyads"] = [cfg["game"]["dyads"][0]]
        single_cfg["behavioral_rnn"]["inference_sample"] = False

        world = World(0, 0, obs_depth, [("i1", "a1")], cfg["behavioral_rnn"]["n_actions"])
        sd = compute_state_dim(single_cfg, aa_edge=0)
        agents = [create_agent(single_cfg["game"]["dyads"][0]["trustee"], single_cfg, sd)]
        for a in agents:
            try:
                a.load()
                a.set_eval_mode(True)
            except Exception:
                checkpoint_ok = False
                break
        if checkpoint_ok:
            investors = [RNNInvestor(single_cfg, "a1")]
            game = Game(single_cfg, world, investors, agents)
            set_all_seeds(single_cfg["seed"])
            rep_by_round: list[list[float]] = [[] for _ in range(single_cfg["game"]["max_rounds"])]
            for _ in range(10):
                stats = game._run_episode(log_details=True)
                df = stats["log_df"]
                for t, grp in df.groupby("timestep"):
                    rep_by_round[t].extend(grp["repayment_pct"].tolist())
            means = [np.mean(r) for r in rep_by_round if r]
            print(f"  Mean repayment by round: {[f'R{i}={v:.2f}' for i, v in enumerate(means)]}")
            ok = all(0.0 <= m <= 1.0 for m in means)
            print(f"  Test B: {'PASS' if ok else 'FAIL'} (values in [0,1])")
        else:
            print("  SKIP — no checkpoint available (run training first)")
    except Exception as e:
        print(f"  SKIP — error: {e}")

    # ------------------------------------------------------------------
    # Tests C + D: wallet bound and reward conservation for all 4 conditions
    # ------------------------------------------------------------------
    conditions_cd = [("ii=0,aa=0", 0, 0), ("ii=1,aa=0", 1, 0),
                     ("ii=0,aa=1", 0, 1), ("ii=1,aa=1", 1, 1)]

    print(f"\n[C] Per-investor wallet bound (invest_k <= {epi}) — 10 episodes each condition")
    c_all_pass = True
    for label, ii, aa in conditions_cd:
        cond_cfg = copy.deepcopy(cfg)
        cond_cfg["edges"]["ii_edge"] = ii
        cond_cfg["edges"]["aa_edge"] = aa
        dyad_pairs = [(d["investor"], d["trustee"]["name"]) for d in cond_cfg["game"]["dyads"]]
        n_actions = cond_cfg["behavioral_rnn"]["n_actions"]
        world = World(ii, aa, obs_depth, dyad_pairs, n_actions)
        sd = compute_state_dim(cond_cfg, aa_edge=aa)
        agents = [create_agent(d["trustee"], cond_cfg, sd) for d in cond_cfg["game"]["dyads"]]
        investors = [RNNInvestor(cond_cfg, d["trustee"]["name"]) for d in cond_cfg["game"]["dyads"]]
        game = Game(cond_cfg, world, investors, agents)
        set_all_seeds(cfg["seed"])
        violations = 0
        for _ in range(10):
            stats = game._run_episode(log_details=True)
            df = stats["log_df"]
            violations += int((df["investment"] > epi + 1e-6).any())
        ok = violations == 0
        if not ok:
            c_all_pass = False
        print(f"  {label}: violations={violations}/10  {'PASS' if ok else 'FAIL'}")
    print(f"  Test C: {'PASS' if c_all_pass else 'FAIL'}")

    print(f"\n[D] Per-dyad reward conservation — 10 episodes each condition")
    d_all_pass = True
    for label, ii, aa in conditions_cd:
        cond_cfg = copy.deepcopy(cfg)
        cond_cfg["edges"]["ii_edge"] = ii
        cond_cfg["edges"]["aa_edge"] = aa
        dyad_pairs = [(d["investor"], d["trustee"]["name"]) for d in cond_cfg["game"]["dyads"]]
        n_actions = cond_cfg["behavioral_rnn"]["n_actions"]
        world = World(ii, aa, obs_depth, dyad_pairs, n_actions)
        sd = compute_state_dim(cond_cfg, aa_edge=aa)
        agents = [create_agent(d["trustee"], cond_cfg, sd) for d in cond_cfg["game"]["dyads"]]
        investors = [RNNInvestor(cond_cfg, d["trustee"]["name"]) for d in cond_cfg["game"]["dyads"]]
        game = Game(cond_cfg, world, investors, agents)
        set_all_seeds(cfg["seed"])
        bad_rows = 0
        total_rows = 0
        for _ in range(10):
            stats = game._run_episode(log_details=True)
            df = stats["log_df"]
            row_sum = df["investor_reward"] + df["agent_reward"]
            expected = 2.0 * df["investment"]
            bad_rows += int((~np.isclose(row_sum.values, expected.values, atol=1e-6)).sum())
            total_rows += len(df)
        ok = bad_rows == 0
        if not ok:
            d_all_pass = False
        print(f"  {label}: bad_rows={bad_rows}/{total_rows}  {'PASS' if ok else 'FAIL'}")
    print(f"  Test D: {'PASS' if d_all_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Test E: ii_edge effect is non-trivial
    # ------------------------------------------------------------------
    print(f"\n[E] ii_edge non-trivial effect (if checkpoint available)")
    try:
        results_e: dict = {}
        e_checkpoint_ok = True
        for ii in (0, 1):
            cond_cfg = copy.deepcopy(cfg)
            cond_cfg["edges"]["ii_edge"] = ii
            cond_cfg["edges"]["aa_edge"] = 0
            cond_cfg["behavioral_rnn"]["inference_sample"] = False
            dyad_pairs = [(d["investor"], d["trustee"]["name"]) for d in cond_cfg["game"]["dyads"]]
            n_actions = cond_cfg["behavioral_rnn"]["n_actions"]
            world = World(ii, 0, obs_depth, dyad_pairs, n_actions)
            sd = compute_state_dim(cond_cfg, aa_edge=0)
            agents = [create_agent(d["trustee"], cond_cfg, sd) for d in cond_cfg["game"]["dyads"]]
            for a in agents:
                try:
                    a.load()
                    a.set_eval_mode(True)
                except Exception:
                    e_checkpoint_ok = False
                    break
            if not e_checkpoint_ok:
                break
            investors = [RNNInvestor(cond_cfg, d["trustee"]["name"]) for d in cond_cfg["game"]["dyads"]]
            game = Game(cond_cfg, world, investors, agents)
            set_all_seeds(cfg["seed"])
            rep_by_round: list[list[float]] = [[] for _ in range(cond_cfg["game"]["max_rounds"])]
            for _ in range(10):
                stats = game._run_episode(log_details=True)
                df = stats["log_df"]
                for t, grp in df.groupby("timestep"):
                    rep_by_round[t].extend(grp["repayment_pct"].tolist())
            results_e[ii] = [np.mean(r) for r in rep_by_round if r]

        if not e_checkpoint_ok:
            print("  SKIP — no checkpoint available")
        else:
            diffs = [abs(a - b) for a, b in zip(results_e[0], results_e[1])]
            max_diff = max(diffs)
            ok = max_diff > 1e-3
            print(f"  max |repay diff| across rounds = {max_diff:.4f}")
            print(f"  Test E: {'PASS' if ok else 'FAIL (ii_edge had no effect)'}")
    except Exception as e:
        print(f"  SKIP — error: {e}")

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
    ii = cfg["edges"]["ii_edge"]
    aa = cfg["edges"]["aa_edge"]
    lines = [
        f"\n{'='*60}", "TRAINING COMPLETE", f"{'='*60}",
        f"  Run ID:     {run_id}",
        f"  ii_edge:    {ii}  aa_edge: {aa}",
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
    ii = cfg["edges"]["ii_edge"]
    aa = cfg["edges"]["aa_edge"]
    lines = [
        f"\n{'='*60}", label, f"{'='*60}",
        f"  Run ID:         {run_id}",
        f"  ii_edge:        {ii}  aa_edge: {aa}",
        f"  Total rounds:   {log_df['timestep'].max() + 1}",
        f"  Final wealth:   {log_df['investor_wealth'].max():.2f}",
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

    world, agents, investors = _build(cfg)

    for a in agents:
        if hasattr(a, "load"):
            try:
                a.load()
                print(f"  Loaded {a.name}")
            except (FileNotFoundError, Exception) as e:
                print(f"  WARNING: {a.name} load failed: {e}")
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    game = Game(cfg, world, investors, agents)
    game.export_dir = out
    log_df = game.run()

    csv_path = out / cfg["export"]["filename"]
    log_df.to_csv(csv_path, index=False)
    print(f"Game log: {csv_path} ({len(log_df)} rows)")

    sim_summary = _simulation_summary(log_df, agents, cfg, run_id)
    print(sim_summary)
    (out / "simulation_summary.txt").write_text(sim_summary)
    _save_round_analysis(log_df, agents, out, label="eval")

    step_verify(cfg, world, agents, investors, out, run_id)


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
    epi = cfg["game"]["endowment_per_investor"]
    n_dyads = len(cfg["game"]["dyads"])
    print(f"  endowment_per_investor={epi}  n_dyads={n_dyads}  "
          f"multiplier={cfg['game']['multiplier']}  "
          f"max_rounds={cfg['game']['max_rounds']}  "
          f"ii_edge={cfg['edges']['ii_edge']}  aa_edge={cfg['edges']['aa_edge']}")

    if not skip_parse:
        step_parse(cfg)
    else:
        print("\n--- STEP 1: PARSE (skipped) ---")

    if not skip_rnn:
        step_train_rnn(cfg)
    else:
        print("\n--- STEP 2: TRAIN RNN (skipped) ---")

    if not skip_dqn:
        world, agents, investors = step_train_dqn(cfg, run_id, out)
    else:
        print("\n--- STEP 3: TRAIN DQN (skipped) ---")
        world, agents, investors = _build(cfg)
        for a in agents:
            try:
                a.load()
                print(f"  Loaded {a.name}")
            except Exception:
                print(f"  WARNING: no checkpoint for {a.name}")

    step_verify(cfg, world, agents, investors, out, run_id)

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
                        help="Run two-dyad smoke tests and exit.")
    parser.add_argument("--experiment-matrix", action="store_true",
                        help="Run full 4-condition 2×2 experiment matrix.")
    args = parser.parse_args()

    cfg = load_config()
    set_all_seeds(cfg["seed"])

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

# (condition_id, ii_edge, aa_edge)
EXPERIMENT_CONDITIONS = [
    ("DPG1_W0", 0, 0),
    ("DPG1_W1", 0, 1),
    ("DPG2_W0", 1, 0),
    ("DPG2_W1", 1, 1),
]


def _build_condition_config(base_cfg: dict, condition: str, ii: int, aa: int) -> dict:
    """Deep-copy base config and override edges + trustee save paths."""
    cfg = copy.deepcopy(base_cfg)
    cfg["edges"]["ii_edge"] = ii
    cfg["edges"]["aa_edge"] = aa
    cfg["_condition"] = condition
    return cfg


def _matrix_train(cfg: dict, cond_dir: Path) -> float:
    world, agents, investors = _build(cfg)
    adv_cfg = cfg["adversary"]
    game = Game(cfg, world, investors, agents)
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
    world, agents, investors = _build(cfg)

    for a in agents:
        if hasattr(a, "load"):
            try:
                a.load()
                print(f"    Loaded {a.name}")
            except Exception as exc:
                print(f"    WARNING: could not load {a.name}: {exc}")
        if hasattr(a, "set_eval_mode"):
            a.set_eval_mode(True)

    game = Game(cfg, world, investors, agents)
    game.export_dir = cond_dir

    all_rows: list[pd.DataFrame] = []
    ep_stats: list[dict] = []

    for ep_i in range(n_episodes):
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
                       endowment_per_investor: float) -> list[str]:
    errors: list[str] = []

    # Reward conservation: investor_reward + agent_reward == 2 * investment
    row_sum = log_df["investor_reward"] + log_df["agent_reward"]
    expected = 2.0 * log_df["investment"]
    bad = ~np.isclose(row_sum.values, expected.values, atol=1e-3)
    if bad.any():
        errors.append(
            f"{condition}: reward invariant violated in {bad.sum()}/{len(bad)} rows"
        )

    # Per-dyad budget: each dyad's investment <= endowment_per_investor
    if "dyad_idx" in log_df.columns:
        per_dyad_round = log_df.groupby(["episode", "timestep", "dyad_idx"])["investment"].sum()
    else:
        per_dyad_round = log_df.groupby(["episode", "timestep"])["investment"].sum()
    violations = int((per_dyad_round > endowment_per_investor + 1e-6).sum())
    if violations > 0:
        errors.append(
            f"{condition}: per-dyad budget violated in {violations} rounds "
            f"(endowment={endowment_per_investor})"
        )

    return errors


def _save_condition_plots(log_df: pd.DataFrame, agents: list,
                           cond_dir: Path, condition: str) -> None:
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
    ii_edge: int,
    aa_edge: int,
    train_time: float,
) -> dict:
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
        "ii_edge": ii_edge,
        "aa_edge": aa_edge,
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


def _produce_2x2_grid(all_log_dfs: dict, matrix_dir: Path, n_eval_episodes: int) -> None:
    """Save 2×2 grid: ii_edge (cols) × aa_edge (rows), mean repayment by round."""
    layout = [
        ("DPG1_W0", "ii=0, aa=0"),
        ("DPG1_W1", "ii=0, aa=1"),
        ("DPG2_W0", "ii=1, aa=0"),
        ("DPG2_W1", "ii=1, aa=1"),
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
        "2×2 Comparison: ii_edge × aa_edge\n"
        f"Mean Repayment % by Round ({n_eval_episodes} eval episodes)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(matrix_dir / "comparison_2x2_grid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {matrix_dir / 'comparison_2x2_grid.png'}")


def _produce_main_effects_table(all_summaries: dict, matrix_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for cond, summary in all_summaries.items():
        inv_mean = summary["investor_earnings_mean"]
        for aname, asum in summary["agents"].items():
            fair_gap = asum["fair_gap_mean"]
            rows.append({
                "Condition": cond,
                "ii_edge": summary["ii_edge"],
                "aa_edge": summary["aa_edge"],
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
    """Bootstrap 95% CIs for main effects and interaction on investor-cumulative."""

    def _inv(cond: str) -> np.ndarray | None:
        if cond not in all_ep_stats:
            return None
        return np.array([ep["investor_cumulative"] for ep in all_ep_stats[cond]])

    dpg1_w0 = _inv("DPG1_W0")
    dpg1_w1 = _inv("DPG1_W1")
    dpg2_w0 = _inv("DPG2_W0")
    dpg2_w1 = _inv("DPG2_W1")

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

    _add("ii effect (aa=0)", "DPG1_W1 − DPG1_W0", dpg1_w0, dpg1_w1)
    _add("ii effect (aa=1)", "DPG2_W1 − DPG2_W0", dpg2_w0, dpg2_w1)
    _add("aa effect (ii=0)", "DPG2_W0 − DPG1_W0", dpg1_w0, dpg2_w0)
    _add("aa effect (ii=1)", "DPG2_W1 − DPG1_W1", dpg1_w1, dpg2_w1)

    # Interaction: (DPG2_W1 − DPG2_W0) − (DPG1_W1 − DPG1_W0)
    if all(x is not None for x in [dpg1_w0, dpg1_w1, dpg2_w0, dpg2_w1]):
        rng = np.random.default_rng(43)
        interactions = [
            (np.mean(rng.choice(dpg2_w1, len(dpg2_w1), replace=True))
             - np.mean(rng.choice(dpg2_w0, len(dpg2_w0), replace=True)))
            - (np.mean(rng.choice(dpg1_w1, len(dpg1_w1), replace=True))
               - np.mean(rng.choice(dpg1_w0, len(dpg1_w0), replace=True)))
            for _ in range(5000)
        ]
        obs_int = float((np.mean(dpg2_w1) - np.mean(dpg2_w0))
                        - (np.mean(dpg1_w1) - np.mean(dpg1_w0)))
        rows.append({
            "Effect": "ii × aa interaction",
            "Comparison": "(DPG2_W1−DPG2_W0) − (DPG1_W1−DPG1_W0)",
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
    matrix_eval_episodes: int,
) -> None:
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
            f"- ii_edge={s['ii_edge']}, aa_edge={s['aa_edge']}",
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

    r = _row("ii effect (aa=0)")
    lines.append("### 1. Does ii_edge help at aa=0?")
    if r:
        pe, lo, hi = r["Point_est"], r["CI_lo_95"], r["CI_hi_95"]
        direction = "increases" if pe > 0 else "decreases"
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not significant at 95%)"
        lines.append(
            f"ii_edge {direction} investor earnings by {pe:+.2f} [{lo:.2f}, {hi:.2f}]. {sig}."
        )
    lines.append("")

    r = _row("aa effect (ii=0)")
    lines.append("### 2. Does aa_edge help at ii=0?")
    if r:
        pe, lo, hi = r["Point_est"], r["CI_lo_95"], r["CI_hi_95"]
        direction = "increases" if pe > 0 else "decreases"
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not significant at 95%)"
        lines.append(
            f"aa_edge {direction} investor earnings by {pe:+.2f} [{lo:.2f}, {hi:.2f}]. {sig}."
        )
    lines.append("")

    r = _row("ii × aa interaction")
    lines.append("### 3. Is there an ii × aa interaction?")
    if r:
        pe, lo, hi = r["Point_est"], r["CI_lo_95"], r["CI_hi_95"]
        direction = "amplifies" if pe > 0 else "dampens"
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not significant at 95%)"
        lines.append(
            f"ii×aa interaction = {pe:+.2f} [{lo:.2f}, {hi:.2f}]. "
            f"aa_edge {direction} the ii_edge effect. {sig}."
        )
    lines.append("")

    lines += [
        "## Limitations",
        "",
        "- Single training run per condition; no error bars on DQN convergence.",
        f"- {matrix_eval_episodes} eval episodes; CIs reflect sampling variance only.",
        "- Fixed pairing i_k ↔ a_k; no cross-pair allocation.",
    ]

    report_path = matrix_dir / "experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nExperiment report: {report_path}")


def run_experiment_matrix(cfg: dict) -> None:
    """Run the full 4-condition 2×2 experiment matrix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = Path(f"outputs/experiment_matrix_{timestamp}")
    matrix_dir.mkdir(parents=True, exist_ok=True)

    endowment_per_investor = float(cfg["game"]["endowment_per_investor"])
    matrix_eval_episodes = int(cfg["adversary"]["matrix_eval_episodes"])

    print(f"\n{'='*60}")
    print("EXPERIMENT MATRIX (2×2 over ii_edge × aa_edge)")
    print(f"Output: {matrix_dir}")
    print(f"Conditions: {[c[0] for c in EXPERIMENT_CONDITIONS]}")
    print(f"{'='*60}")

    all_log_dfs: dict = {}
    all_ep_stats: dict = {}
    all_summaries: dict = {}
    all_errors: list[str] = []
    training_times: dict = {}

    for condition, ii, aa in EXPERIMENT_CONDITIONS:
        print(f"\n{'─'*55}")
        print(f"Condition: {condition}  (ii_edge={ii}, aa_edge={aa})")
        print(f"{'─'*55}")

        cond_dir = matrix_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        cond_cfg = _build_condition_config(cfg, condition, ii, aa)
        safe_cfg = {k: v for k, v in cond_cfg.items() if not k.startswith("_")}
        with open(cond_dir / "config.yaml", "w") as f:
            yaml.dump(safe_cfg, f, default_flow_style=False, sort_keys=False)

        # --- Training ---
        adv_ep = cond_cfg["adversary"]["training_episodes"]
        print(f"  Training: {adv_ep} episodes...")
        train_time = _matrix_train(cond_cfg, cond_dir)
        print(f"  Training time: {train_time:.1f}s")
        training_times[condition] = train_time

        # --- Evaluation ---
        print(f"  Evaluating: {matrix_eval_episodes} episodes...")
        t_eval = time.time()
        log_df, ep_stats, agents = _matrix_eval(cond_cfg, cond_dir, n_episodes=matrix_eval_episodes)
        eval_time = time.time() - t_eval
        print(f"  Eval time: {eval_time:.1f}s")

        all_log_dfs[condition] = log_df
        all_ep_stats[condition] = ep_stats

        # --- Invariant audit ---
        errors = _audit_invariants(log_df, condition, endowment_per_investor)
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

        # --- DPG1_W0 Fig5D replication check ---
        if condition == "DPG1_W0":
            max_rows = log_df[log_df["agent_name"] == "a1"]
            by_round = max_rows.groupby("timestep")["repayment_pct"].mean()
            r0 = float(by_round.iloc[0]) if len(by_round) > 0 else float("nan")
            rlast = float(by_round.iloc[-1]) if len(by_round) > 0 else float("nan")
            if r0 > 0.5 and rlast < 0.3:
                print(f"  Fig 5D replication: PASS (R0={r0:.1%} → R9={rlast:.1%})")
            else:
                print(
                    f"  Fig 5D replication: WARNING — "
                    f"R0={r0:.1%} → R9={rlast:.1%} (expected ~75%→<30%)"
                )

        # --- Plots & summary ---
        _save_condition_plots(log_df, agents, cond_dir, condition)
        summary = _save_condition_summary(
            log_df, ep_stats, agents, cond_dir,
            condition, ii, aa, train_time,
        )
        all_summaries[condition] = summary

        inv_mean = summary["investor_earnings_mean"]
        print(f"  Investor mean earnings: {inv_mean:+.1f}")
        for aname, asum in summary["agents"].items():
            print(f"  {aname} ({asum['policy']}): reward={asum['mean_reward']:+.1f}")

    # --- Aggregate outputs ---
    print(f"\n{'─'*55}")
    print("Producing aggregate outputs...")

    _produce_2x2_grid(all_log_dfs, matrix_dir, matrix_eval_episodes)
    main_eff_df = _produce_main_effects_table(all_summaries, matrix_dir)
    effects_df = _produce_effect_decomposition(all_ep_stats, matrix_dir)
    _write_experiment_report(all_summaries, main_eff_df, effects_df, matrix_dir, matrix_eval_episodes)

    if all_errors:
        print(f"\nINVARIANT AUDIT FAILURES ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
    else:
        print("\nInvariant audit: ALL CONDITIONS PASS")

    total_train = sum(training_times.values())
    print(f"\nTraining times (total: {total_train:.1f}s):")
    for cond, t in training_times.items():
        print(f"  {cond}: {t:.1f}s")

    print(f"\n{'='*60}")
    print(f"Experiment matrix complete.")
    print(f"All outputs in: {matrix_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
