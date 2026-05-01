"""Sweep γ ∈ GAMMAS over W0 and W1 by patching config.yaml and running main.py."""

import copy
import subprocess
import sys

import yaml

from utils import load_config

GAMMAS = [0.3, 0.5, 0.70, 0.80, 0.90]
WORLDS = [0, 1]
SEEDS = [42]

base = load_config()
for s in SEEDS:
    for w in WORLDS:
        for g in GAMMAS:
            cfg = copy.deepcopy(base)
            cfg["seed"] = s
            cfg["behavioral_rnn"]["trust_decay"] = g
            cfg["world"]["mode"] = w
            for a in cfg["game"]["agents"]:
                a["save_path"] = f"checkpoints/sweep_g{int(g*100):03d}_w{w}_{a['name']}.pt"
            with open("config.yaml", "w") as f:
                yaml.dump(cfg, f, sort_keys=False)
            subprocess.run(
                [sys.executable, "main.py", "--skip-parse", "--skip-rnn"], check=True,
            )
