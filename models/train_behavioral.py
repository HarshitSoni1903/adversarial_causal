"""Phase 2: Train BehavioralRNN on parsed human behavioral data.

Dezfouli-faithful training format:
  - One (sequence, targets) example per episode.
  - Sequence length = n_rounds (including prepended dummy at position 0).
  - Targets = bucket indices for all n_rounds actions.
  - Loss = cross-entropy summed over all n_rounds timesteps (not just last).

Bucketing rule is configurable (paper_range | nearest_neighbor); see utils.bucket_investment.

Supports batch mode (default, Dezfouli-faithful) and iterative mode (extension).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.behavioral_rnn import BehavioralRNN, load_behavioral_rnn
from utils import bucket_investment, load_config, set_all_seeds


# ---------------------------------------------------------------------------
# Dataset construction (Dezfouli format: one example per episode, per-step)
# ---------------------------------------------------------------------------

def encode_episodes(
    df: pd.DataFrame,
    episode_ids: list[str],
    action_values: list[int],
    bucketing_rule: str,
    n_rounds: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build per-episode (sequence, targets) pairs.

    Each episode produces ONE training example:
      sequence: shape (n_rounds, 6) where
        - row 0 = zero dummy (no prior information before round 0)
        - row t (t>=1) = [prev_action_onehot(5), prev_repay_prop(1)]
      targets:  shape (n_rounds,) of int bucket indices [0..4]

    All n_rounds timesteps contribute to the loss (Dezfouli's
    "loss sums cross-entropy over ALL timesteps" convention).
    """
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for eid in episode_ids:
        ep = df[df["episode_id"] == eid].sort_values("round").reset_index(drop=True)
        if len(ep) != n_rounds:
            continue  # skip incomplete episodes

        inp = np.zeros((n_rounds, 6), dtype=np.float32)
        tgt = np.zeros(n_rounds, dtype=np.int64)

        for t in range(n_rounds):
            # Row t of sequence = features from round t-1 (dummy at t=0)
            if t > 0:
                prev = ep.iloc[t - 1]
                prev_idx = bucket_investment(int(prev["investment"]), bucketing_rule)
                onehot = np.zeros(5, dtype=np.float32)
                onehot[prev_idx] = 1.0
                inp[t, :5] = onehot
                inp[t, 5] = float(prev["repay_prop"])
            # else: row 0 stays all zeros (dummy)

            # Target = bucket of round t's investment
            tgt[t] = bucket_investment(int(ep.iloc[t]["investment"]), bucketing_rule)

        sequences.append(inp)
        targets.append(tgt)

    return sequences, targets


def split_episodes(
    df: pd.DataFrame, train_frac: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Split episode_ids into train/test lists reproducibly."""
    episode_ids = df["episode_id"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(episode_ids)
    split = int(len(episode_ids) * train_frac)
    return episode_ids[:split], episode_ids[split:]


class BehavioralDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], targets: list[np.ndarray]) -> None:
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.sequences[idx], self.targets[idx]


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack fixed-length sequences into batch tensors."""
    seqs, tgts = zip(*batch)
    return (
        torch.tensor(np.stack(seqs), dtype=torch.float32),   # (B, T, 6)
        torch.tensor(np.stack(tgts), dtype=torch.long),      # (B, T)
    )


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(targets: list[np.ndarray], n_actions: int) -> torch.Tensor:
    flat = np.concatenate(targets)
    counts = np.bincount(flat, minlength=n_actions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(flat) / counts
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: BehavioralRNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, tgts in loader:
        seqs, tgts = seqs.to(device), tgts.to(device)       # (B, T, 6), (B, T)
        logits = model(seqs)                                  # (B, T, n_actions)
        B, T, A = logits.shape
        loss = criterion(logits.reshape(B * T, A), tgts.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * (B * T)
        correct += (logits.argmax(-1) == tgts).sum().item()
        total += B * T

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: BehavioralRNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, tgts in loader:
        seqs, tgts = seqs.to(device), tgts.to(device)
        logits = model(seqs)
        B, T, A = logits.shape
        loss = criterion(logits.reshape(B * T, A), tgts.reshape(B * T))
        total_loss += loss.item() * (B * T)
        correct += (logits.argmax(-1) == tgts).sum().item()
        total += B * T
    return total_loss / total, correct / total


@torch.no_grad()
def print_full_metrics(
    model: BehavioralRNN,
    loader: DataLoader,
    n_actions: int,
    action_values: list[int],
    device: torch.device,
) -> None:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    total_nll = 0.0

    for seqs, tgts in loader:
        probs = model.predict_probs(seqs.to(device))         # (B, T, n_actions)
        preds = probs.argmax(-1)                              # (B, T)
        B, T, A = probs.shape
        all_preds.extend(preds.reshape(-1).cpu().tolist())
        all_targets.extend(tgts.reshape(-1).tolist())
        for b in range(B):
            for t in range(T):
                total_nll -= np.log(max(probs[b, t, tgts[b, t]].item(), 1e-12))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    n = len(y_true)
    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== Test Metrics ===")
    print(f"  Accuracy:    {acc:.3f}")
    print(f"  Macro F1:    {macro_f1:.3f}")
    print(f"  Avg NLL/step:{total_nll / n:.4f}")
    prec, rec, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_actions)), zero_division=0,
    )
    print(f"  {'Bucket':>6}  {'Support':>8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    for i, a in enumerate(action_values):
        print(f"    {a:>3}   {support[i]:>8}  {prec[i]:>6.3f}  {rec[i]:>6.3f}  {f1s[i]:>6.3f}")


def save_curves(train_losses, test_losses, train_accs, test_accs, save_dir: Path) -> None:
    epochs = range(1, len(train_losses) + 1)
    for name, tr, te in [("loss", train_losses, test_losses),
                          ("acc", train_accs, test_accs)]:
        plt.figure()
        plt.plot(epochs, tr, label="train")
        plt.plot(epochs, te, label="test")
        plt.xlabel("Epoch"); plt.ylabel(name); plt.legend(); plt.tight_layout()
        plt.savefig(save_dir / f"behavioral_rnn_{name}.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Iterative training (extension — NOT Dezfouli)
# ---------------------------------------------------------------------------

def train_iterative(
    model: BehavioralRNN,
    train_seqs: list[np.ndarray],
    train_tgts: list[np.ndarray],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    rnn_cfg: dict,
    training_cfg: dict,
    device: torch.device,
) -> float:
    """Online iterative training over episodes with optional multi-pass.

    This is a departure from Dezfouli's batch training. It trains the model
    by iterating episode-by-episode, optionally making multiple passes.
    Grad accumulation is supported for effective larger batch sizes.
    """
    it_cfg = training_cfg.get("iterative", {})
    passes = it_cfg.get("passes", 3)
    shuffle = it_cfg.get("shuffle_each_pass", True)
    accum = it_cfg.get("grad_accum_steps", 1)
    grad_clip = rnn_cfg.get("grad_clip", 1.0)

    model.train()
    total_loss = 0.0
    n_examples = len(train_seqs)

    for _ in range(passes):
        indices = list(range(n_examples))
        if shuffle:
            np.random.shuffle(indices)

        optimizer.zero_grad()
        step_loss = 0.0
        for step_i, idx in enumerate(indices):
            seq = torch.tensor(train_seqs[idx], dtype=torch.float32, device=device).unsqueeze(0)
            tgt = torch.tensor(train_tgts[idx], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(seq)
            B, T, A = logits.shape
            loss = criterion(logits.reshape(B * T, A), tgt.reshape(B * T)) / accum
            loss.backward()
            step_loss += loss.item() * accum

            if (step_i + 1) % accum == 0 or step_i == n_examples - 1:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += step_loss / n_examples

    return total_loss / passes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    set_all_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    rnn_cfg = cfg["behavioral_rnn"]
    training_cfg = cfg.get("training", {"mode": "batch"})
    action_values: list[int] = rnn_cfg["action_values"]
    n_actions: int = rnn_cfg["n_actions"]
    n_rounds: int = data_cfg["original_rounds"]
    bucketing_rule: str = data_cfg["bucketing_rule"]

    df = pd.read_csv(data_cfg["output_path"])
    train_ids, test_ids = split_episodes(df, rnn_cfg["train_frac"], cfg["seed"])
    print(f"Episodes: {len(train_ids)+len(test_ids)} total, "
          f"{len(train_ids)} train, {len(test_ids)} test")
    print(f"Bucketing rule: {bucketing_rule}")

    train_seqs, train_tgts = encode_episodes(df, train_ids, action_values, bucketing_rule, n_rounds)
    test_seqs, test_tgts = encode_episodes(df, test_ids, action_values, bucketing_rule, n_rounds)
    print(f"Episodes encoded: {len(train_seqs)} train, {len(test_seqs)} test  "
          f"(each: {n_rounds} steps × 6-dim input)")

    test_loader = DataLoader(
        BehavioralDataset(test_seqs, test_tgts),
        batch_size=rnn_cfg["batch_size"], shuffle=False, collate_fn=collate_fn,
    )

    model = BehavioralRNN(
        input_size=rnn_cfg["input_size"],
        hidden_size=rnn_cfg["hidden_size"],
        n_actions=n_actions,
        dropout=rnn_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=rnn_cfg["lr"])
    weights = compute_class_weights(train_tgts, n_actions).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Using class-weighted CrossEntropyLoss")

    save_path = Path(rnn_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_test_loss = float("inf")
    best_state: dict = {}
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    training_mode = training_cfg.get("mode", "batch")

    for epoch in range(1, rnn_cfg["epochs"] + 1):
        if training_mode == "batch":
            train_loader = DataLoader(
                BehavioralDataset(train_seqs, train_tgts),
                batch_size=rnn_cfg["batch_size"], shuffle=True, collate_fn=collate_fn,
            )
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                          rnn_cfg["grad_clip"], device)
        else:
            tr_loss = train_iterative(model, train_seqs, train_tgts, optimizer, criterion,
                                      rnn_cfg, training_cfg, device)
            tr_acc = 0.0  # not computed in iterative mode per-epoch

        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
        train_losses.append(tr_loss); test_losses.append(te_loss)
        train_accs.append(tr_acc);  test_accs.append(te_acc)

        if te_loss < best_test_loss:
            best_test_loss = te_loss
            best_state = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "input_size": rnn_cfg["input_size"],
                "hidden_size": rnn_cfg["hidden_size"],
                "n_actions": n_actions,
                "dropout": rnn_cfg["dropout"],
                "action_values": action_values,
                "bucketing_rule": bucketing_rule,
            }

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{rnn_cfg['epochs']}  "
                  f"train_loss={tr_loss:.4f}  acc={tr_acc:.3f}  "
                  f"test_loss={te_loss:.4f}  test_acc={te_acc:.3f}")

    torch.save(best_state, save_path)
    print(f"\nCheckpoint saved → {save_path}  (best test_loss={best_test_loss:.4f})")

    ckpt_dir = save_path.parent
    save_curves(train_losses, test_losses, train_accs, test_accs, ckpt_dir)

    final_model = load_behavioral_rnn(str(save_path), device=str(device))
    final_loss, final_acc = eval_epoch(final_model, test_loader, criterion, device)
    print(f"Final eval — loss={final_loss:.4f}  acc={final_acc:.3f}")
    print_full_metrics(final_model, test_loader, n_actions, action_values, device)


if __name__ == "__main__":
    main()
