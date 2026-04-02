"""Phase 2: Train BehavioralRNN on parsed human behavioral data.

Loads the CSV from Phase 1, builds sequential (history -> next action) examples,
trains a GRU classifier with class-weighted cross-entropy, and saves the best
checkpoint by test loss. All hyperparameters come from config.yaml.

Functions in this module are public so tune_behavioral.py can reuse them.
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
from utils import action_to_idx, load_config, set_all_seeds


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def encode_episodes(
    df: pd.DataFrame,
    episode_ids: list[str],
    actions: list[int],
    endowment: int,
    n_rounds: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Build (sequence, target) pairs from episode data.

    For each episode, for each round t >= 1 the input sequence is the
    encoded steps 0..t-1 and the target is the discretized action at t.
    """
    round_scale = n_rounds - 1
    sequences: list[np.ndarray] = []
    targets: list[int] = []

    for eid in episode_ids:
        ep = df[df["episode_id"] == eid].sort_values("round").reset_index(drop=True)

        encoded: list[list[float]] = []
        for t in range(len(ep)):
            row = ep.iloc[t]
            if t == 0:
                enc = [row["round"] / round_scale, 0.0, 0.0, 0.0]
            else:
                prev = ep.iloc[t - 1]
                enc = [
                    row["round"] / round_scale,
                    prev["repay_prop"],
                    prev["investment"] / endowment,
                    prev["reward"] / endowment,
                ]
            encoded.append(enc)

            if t >= 1:
                seq = np.array(encoded[:t], dtype=np.float32)
                target = action_to_idx(row["investment"], actions)
                sequences.append(seq)
                targets.append(target)

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
    def __init__(self, sequences: list[np.ndarray], targets: list[int]) -> None:
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.sequences[idx], self.targets[idx]


def collate_fn(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad sequences to the max length in the batch."""
    seqs, tgts = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    input_size = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, input_size)
    for i, s in enumerate(seqs):
        offset = max_len - len(s)
        padded[i, offset:] = torch.from_numpy(s)

    return padded, lengths, torch.tensor(tgts, dtype=torch.long)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(targets: list[int], n_actions: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=n_actions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(targets) / counts
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / eval loops
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
    for padded, lengths, tgts in loader:
        padded, lengths, tgts = padded.to(device), lengths, tgts.to(device)
        logits = model(padded, lengths)
        loss = criterion(logits, tgts)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(tgts)
        correct += (logits.argmax(1) == tgts).sum().item()
        total += len(tgts)

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
    for padded, lengths, tgts in loader:
        padded, lengths, tgts = padded.to(device), lengths, tgts.to(device)
        logits = model(padded, lengths)
        loss = criterion(logits, tgts)
        total_loss += loss.item() * len(tgts)
        correct += (logits.argmax(1) == tgts).sum().item()
        total += len(tgts)

    return total_loss / total, correct / total


@torch.no_grad()
def compute_test_nll(
    model: BehavioralRNN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return average negative log-likelihood per sample on the test set."""
    model.eval()
    total_nll, n_samples = 0.0, 0
    for padded, lengths, tgts in loader:
        probs = model.predict_probs(padded.to(device), lengths).cpu()
        for i, t in enumerate(tgts.tolist()):
            total_nll -= np.log(max(probs[i, t].item(), 1e-12))
        n_samples += len(tgts)
    return total_nll / n_samples


# ---------------------------------------------------------------------------
# Post-training diagnostics
# ---------------------------------------------------------------------------

def save_curves(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    save_dir: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "behavioral_rnn_training.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="train")
    plt.plot(epochs, test_accs, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "behavioral_rnn_accuracy.png", dpi=150)
    plt.close()


@torch.no_grad()
def save_predictions(
    model: BehavioralRNN,
    loader: DataLoader,
    actions: list[int],
    device: torch.device,
    save_dir: Path,
) -> None:
    model.eval()
    all_preds, all_targets = [], []
    for padded, lengths, tgts in loader:
        logits = model(padded.to(device), lengths)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_targets.extend(tgts.tolist())

    pred_actions = [actions[i] for i in all_preds]
    true_actions = [actions[i] for i in all_targets]
    pd.DataFrame({"true": true_actions, "predicted": pred_actions}).to_csv(
        save_dir / "behavioral_rnn_predictions.csv", index=False,
    )

    n = len(actions)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1
    cm_df = pd.DataFrame(cm, index=actions, columns=actions)
    cm_df.to_csv(save_dir / "behavioral_rnn_confusion.csv")


@torch.no_grad()
def print_full_metrics(
    model: BehavioralRNN,
    loader: DataLoader,
    actions: list[int],
    device: torch.device,
) -> None:
    """Print comprehensive evaluation metrics on the test set."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_true_probs: list[float] = []
    total_nll = 0.0

    for padded, lengths, tgts in loader:
        probs = model.predict_probs(padded.to(device), lengths).cpu()
        preds = probs.argmax(1)
        all_preds.extend(preds.tolist())
        all_targets.extend(tgts.tolist())

        for i, t in enumerate(tgts.tolist()):
            p_true = probs[i, t].item()
            all_true_probs.append(p_true)
            total_nll -= np.log(max(p_true, 1e-12))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    n_samples = len(y_true)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = (y_true == y_pred).mean()

    print("\n=== Summary Metrics ===")
    print(f"  Accuracy:     {accuracy:.3f}")
    print(f"  Macro F1:     {macro_f1:.3f}")
    print(f"  Weighted F1:  {weighted_f1:.3f}")

    avg_nll = total_nll / n_samples
    avg_true_prob = np.mean(all_true_probs)
    print(f"  Avg NLL/sample:         {avg_nll:.4f}")
    print(f"  Avg P(correct action):  {avg_true_prob:.3f}  "
          f"({avg_true_prob*100:.1f}% probability assigned to what human chose)")

    labels = list(range(len(actions)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )

    print(f"\n{'Bucket':>8}  {'Support':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    for i, a in enumerate(actions):
        print(f"  {a:>5}   {support[i]:>8}  {prec[i]:>10.3f}  {rec[i]:>8.3f}  {f1[i]:>6.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    set_all_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    rnn_cfg = cfg["behavioral_rnn"]
    actions: list[int] = rnn_cfg["actions"]
    endowment: int = data_cfg["original_endowment"]
    n_rounds: int = data_cfg["original_rounds"]

    df = pd.read_csv(data_cfg["output_path"])
    train_ids, test_ids = split_episodes(df, rnn_cfg["train_frac"], cfg["seed"])
    print(f"Episodes: {len(train_ids) + len(test_ids)} total, "
          f"{len(train_ids)} train, {len(test_ids)} test")

    train_seqs, train_tgts = encode_episodes(df, train_ids, actions, endowment, n_rounds)
    test_seqs, test_tgts = encode_episodes(df, test_ids, actions, endowment, n_rounds)
    print(f"Examples: {len(train_tgts)} train, {len(test_tgts)} test")

    train_loader = DataLoader(
        BehavioralDataset(train_seqs, train_tgts),
        batch_size=rnn_cfg["batch_size"], shuffle=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        BehavioralDataset(test_seqs, test_tgts),
        batch_size=rnn_cfg["batch_size"], shuffle=False, collate_fn=collate_fn,
    )

    model = BehavioralRNN(
        input_size=rnn_cfg["input_size"],
        hidden_size=rnn_cfg["hidden_size"],
        n_actions=rnn_cfg["n_actions"],
        dropout=rnn_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=rnn_cfg["lr"])
    if rnn_cfg.get("use_class_weights", True):
        weights = compute_class_weights(train_tgts, rnn_cfg["n_actions"]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using class-weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (no class weights)")

    save_path = Path(rnn_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_test_loss = float("inf")
    best_state: dict = {}
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, rnn_cfg["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, rnn_cfg["grad_clip"], device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        if te_loss < best_test_loss:
            best_test_loss = te_loss
            best_state = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "actions": actions,
                "input_size": rnn_cfg["input_size"],
                "hidden_size": rnn_cfg["hidden_size"],
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{rnn_cfg['epochs']}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                f"test_loss={te_loss:.4f}  test_acc={te_acc:.3f}"
            )

    torch.save(best_state, save_path)
    print(f"\nBest checkpoint saved to {save_path}  (test_loss={best_test_loss:.4f})")

    ckpt_dir = save_path.parent
    save_curves(train_losses, test_losses, train_accs, test_accs, ckpt_dir)

    final_model = load_behavioral_rnn(str(save_path), device=str(device))
    final_loss, final_acc = eval_epoch(final_model, test_loader, criterion, device)
    print(f"Final test evaluation — loss={final_loss:.4f}  acc={final_acc:.3f}")

    save_predictions(final_model, test_loader, actions, device, ckpt_dir)
    print(f"Predictions and confusion matrix saved to {ckpt_dir}/")

    print_full_metrics(final_model, test_loader, actions, device)


if __name__ == "__main__":
    main()
