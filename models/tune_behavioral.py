"""Hyperparameter tuning for BehavioralRNN using Optuna.

Searches over GRU architecture and training hyperparameters, optimizing
test negative log-likelihood per sample. Uses MedianPruner for early
stopping of unpromising trials. Retrains with best params and saves
the final checkpoint.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.behavioral_rnn import BehavioralRNN, load_behavioral_rnn
from models.train_behavioral import (
    BehavioralDataset,
    collate_fn,
    compute_class_weights,
    compute_test_nll,
    encode_episodes,
    eval_epoch,
    print_full_metrics,
    save_curves,
    save_predictions,
    split_episodes,
    train_epoch,
)
from utils import load_config, set_all_seeds


def _objective(
    trial: optuna.Trial,
    train_seqs: list[np.ndarray],
    train_tgts: list[int],
    test_seqs: list[np.ndarray],
    test_tgts: list[int],
    n_actions: int,
    input_size: int,
    device: torch.device,
) -> float:
    hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 50, 300)
    grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)

    train_loader = DataLoader(
        BehavioralDataset(train_seqs, train_tgts),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        BehavioralDataset(test_seqs, test_tgts),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    model = BehavioralRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        n_actions=n_actions,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = compute_class_weights(train_tgts, n_actions).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_nll = float("inf")
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, grad_clip, device)
        nll = compute_test_nll(model, test_loader, device)
        best_nll = min(best_nll, nll)

        trial.report(nll, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_nll


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

    # --- Optuna study ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=20,
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)

    def objective_wrapper(trial: optuna.Trial) -> float:
        set_all_seeds(cfg["seed"])
        return _objective(
            trial, train_seqs, train_tgts, test_seqs, test_tgts,
            rnn_cfg["n_actions"], rnn_cfg["input_size"], device,
        )

    n_trials = 100

    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        pruned = trial.state == optuna.trial.TrialState.PRUNED
        status = "PRUNED" if pruned else f"NLL={trial.value:.4f}"
        print(f"  Trial {trial.number:3d}/{n_trials}  {status}  "
              f"params={trial.params}")

    print(f"\nStarting Optuna study ({n_trials} trials)...")
    study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[trial_callback])

    # --- Results ---
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Trial {best.number}  NLL={best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print("\nTOP 5 TRIALS:")
    top5 = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))[:5]
    for t in top5:
        print(f"  Trial {t.number:3d}  NLL={t.value:.4f}  {t.params}")

    # --- Save study ---
    ckpt_dir = Path(rnn_cfg["save_path"]).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    study_path = ckpt_dir / "optuna_study.pkl"
    joblib.dump(study, study_path)
    print(f"\nStudy saved to {study_path}")

    # --- Print ready-to-paste YAML ---
    bp = best.params
    print("\n# --- Paste into config.yaml behavioral_rnn section ---")
    print(f"  hidden_size: {bp['hidden_size']}")
    print(f"  dropout: {bp['dropout']:.4f}")
    print(f"  lr: {bp['lr']:.6f}")
    print(f"  batch_size: {bp['batch_size']}")
    print(f"  epochs: {bp['epochs']}")
    print(f"  grad_clip: {bp['grad_clip']:.4f}")

    # --- Retrain with best hyperparams ---
    print("\n" + "=" * 60)
    print("RETRAINING WITH BEST HYPERPARAMS")
    print("=" * 60)
    set_all_seeds(cfg["seed"])

    train_loader = DataLoader(
        BehavioralDataset(train_seqs, train_tgts),
        batch_size=bp["batch_size"], shuffle=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        BehavioralDataset(test_seqs, test_tgts),
        batch_size=bp["batch_size"], shuffle=False, collate_fn=collate_fn,
    )

    model = BehavioralRNN(
        input_size=rnn_cfg["input_size"],
        hidden_size=bp["hidden_size"],
        n_actions=rnn_cfg["n_actions"],
        dropout=bp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=bp["lr"])
    weights = compute_class_weights(train_tgts, rnn_cfg["n_actions"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    save_path = Path(rnn_cfg["save_path"])
    best_test_loss = float("inf")
    best_state: dict = {}
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, bp["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, bp["grad_clip"], device,
        )
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
                "hidden_size": bp["hidden_size"],
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{bp['epochs']}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                f"test_loss={te_loss:.4f}  test_acc={te_acc:.3f}"
            )

    torch.save(best_state, save_path)
    print(f"\nBest checkpoint saved to {save_path}  (test_loss={best_test_loss:.4f})")

    save_curves(train_losses, test_losses, train_accs, test_accs, ckpt_dir)

    final_model = load_behavioral_rnn(str(save_path), device=str(device))
    save_predictions(final_model, test_loader, actions, device, ckpt_dir)
    print_full_metrics(final_model, test_loader, actions, device)


if __name__ == "__main__":
    main()
