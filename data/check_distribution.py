"""Diagnostic script: inspect action distributions, transitions, and entropy."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import action_to_idx, load_config


def main() -> None:
    cfg = load_config()
    actions: list[int] = cfg["behavioral_rnn"]["actions"]
    n_actions = len(actions)
    n_rounds: int = cfg["data"]["original_rounds"]
    out_dir = Path(cfg["export"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["data"]["output_path"])
    df["action_idx"] = df["investment"].apply(lambda v: action_to_idx(v, actions))

    # --- A) Overall action distribution ---
    counts = np.bincount(df["action_idx"], minlength=n_actions)
    pcts = counts / counts.sum() * 100
    print("=== Overall Action Distribution ===")
    for i, a in enumerate(actions):
        print(f"  {a:>2}: {counts[i]:5d}  ({pcts[i]:5.1f}%)")

    plt.figure(figsize=(6, 4))
    plt.bar([str(a) for a in actions], counts)
    plt.xlabel("Investment Bucket")
    plt.ylabel("Count")
    plt.title("Overall Action Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "action_distribution.png", dpi=150)
    plt.close()

    # --- B) Action distribution by round ---
    table = np.zeros((n_rounds, n_actions), dtype=int)
    for r in range(n_rounds):
        rnd_data = df[df["round"] == r]["action_idx"]
        table[r] = np.bincount(rnd_data, minlength=n_actions)

    print("\n=== Action Counts by Round ===")
    header = "Round " + " ".join(f"{a:>5}" for a in actions)
    print(header)
    for r in range(n_rounds):
        row = " ".join(f"{table[r, i]:>5}" for i in range(n_actions))
        print(f"  {r:>2}   {row}")

    plt.figure(figsize=(7, 5))
    plt.imshow(table, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Count")
    plt.xticks(range(n_actions), actions)
    plt.yticks(range(n_rounds))
    plt.xlabel("Action Bucket")
    plt.ylabel("Round")
    plt.title("Action Distribution by Round")
    plt.tight_layout()
    plt.savefig(out_dir / "action_by_round_heatmap.png", dpi=150)
    plt.close()

    # --- C) Transition: prev repay_prop -> next action ---
    repay_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]
    trans = np.zeros((len(bin_labels), n_actions), dtype=int)

    for eid in df["episode_id"].unique():
        ep = df[df["episode_id"] == eid].sort_values("round").reset_index(drop=True)
        for t in range(1, len(ep)):
            rp = ep.iloc[t - 1]["repay_prop"]
            b = min(int(rp / 0.2), len(bin_labels) - 1)
            trans[b, ep.iloc[t]["action_idx"]] += 1

    print("\n=== Repay Proportion -> Next Action ===")
    header = "RepayBin " + " ".join(f"{a:>5}" for a in actions)
    print(header)
    for b, lbl in enumerate(bin_labels):
        row_sum = trans[b].sum()
        row_pct = trans[b] / max(row_sum, 1) * 100
        row = " ".join(f"{row_pct[i]:>4.0f}%" for i in range(n_actions))
        print(f"  {lbl:>5}  {row}  (n={row_sum})")

    plt.figure(figsize=(7, 5))
    row_sums = trans.sum(axis=1, keepdims=True).clip(1)
    plt.imshow(trans / row_sums, aspect="auto", cmap="Blues")
    plt.colorbar(label="P(next action | repay bin)")
    plt.xticks(range(n_actions), actions)
    plt.yticks(range(len(bin_labels)), bin_labels)
    plt.xlabel("Next Action Bucket")
    plt.ylabel("Previous Repay Proportion")
    plt.title("Transition: Repayment → Next Investment")
    plt.tight_layout()
    plt.savefig(out_dir / "repay_to_action_heatmap.png", dpi=150)
    plt.close()

    # --- D) Per-round entropy ---
    print("\n=== Per-Round Entropy ===")
    entropies = []
    for r in range(n_rounds):
        dist = table[r].astype(float)
        dist = dist / dist.sum()
        dist = dist[dist > 0]
        h = -np.sum(dist * np.log(dist))
        entropies.append(h)
        print(f"  Round {r}: {h:.3f}")
    avg_h = np.mean(entropies)
    max_h = np.log(n_actions)
    print(f"  Average entropy: {avg_h:.3f}  (max = ln({n_actions}) = {max_h:.3f})")
    if avg_h > 0.9 * max_h:
        print("  -> Near-maximum entropy: human behavior is close to random.")

    # --- E) Majority-class baseline ---
    print("\n=== Majority-Class Baseline ===")
    maj_accs = []
    for r in range(n_rounds):
        maj_pct = table[r].max() / table[r].sum() * 100
        maj_accs.append(maj_pct)
        print(f"  Round {r}: {maj_pct:.1f}%")
    avg_maj = np.mean(maj_accs)
    print(f"  Average majority-class baseline: {avg_maj:.1f}%")


if __name__ == "__main__":
    main()
