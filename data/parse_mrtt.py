"""Phase 1: Parse raw Dezfouli MRTT CSVs into a flat behavioral dataset.

Walks data/mrtt/<condition>/<subject_id>/data/output.csv files,
cleans and computes derived columns, and saves a single CSV for
training the BehavioralRNN learner model.

Output columns: episode_id, round, investment, repay_prop.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_config


def _clean_bracket(val: str) -> float:
    """Strip bracket notation like '[21.]' or '[0.]' to a float."""
    return float(str(val).strip("[] "))


def parse_mrtt(cfg: dict) -> pd.DataFrame:
    """Parse raw MRTT CSVs for a given condition into a single DataFrame."""
    data_cfg = cfg["data"]
    condition = data_cfg["train_condition"]
    mrtt_dir = Path(data_cfg["mrtt_dir"])
    multiplier = data_cfg["multiplier"]

    condition_dir = mrtt_dir / condition
    csv_paths = sorted(condition_dir.glob("*/data/output.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No output.csv files found in {condition_dir}")

    rows: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        episode_id = csv_path.parent.parent.name
        df = pd.read_csv(csv_path)

        df["investment"] = df["investment"].astype(float)
        df["repay"] = df["repay"].apply(_clean_bracket)
        df["repay_prop"] = df.apply(
            lambda r: r["repay"] / (multiplier * r["investment"])
            if r["investment"] > 0
            else 0.0,
            axis=1,
        )
        df["round"] = range(len(df))
        df["episode_id"] = episode_id

        rows.append(df[["episode_id", "round", "investment", "repay_prop"]])

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    cfg = load_config()
    dataset = parse_mrtt(cfg)
    output_path = Path(cfg["data"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved {output_path}  shape={dataset.shape}  columns={list(dataset.columns)}")


if __name__ == "__main__":
    main()
