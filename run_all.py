"""Run all 4 two-dyad conditions sequentially.

Conditions:
  DPG1_W0  ii=0 aa=0  baseline
  DPG1_W1  ii=0 aa=1  trustees talk
  DPG2_W0  ii=1 aa=0  investors talk
  DPG2_W1  ii=1 aa=1  both talk

Usage:
    python run_all.py
"""

import copy

from agents import compute_state_dim, create_agent
from agents.investor import RNNInvestor
from game import Game
from utils import load_config, set_all_seeds
from world import World

CONDITIONS = [
    ("DPG1_W0", 0, 0),
    ("DPG1_W1", 0, 1),
    ("DPG2_W0", 1, 0),
    ("DPG2_W1", 1, 1),
]

cfg = load_config()

for condition, ii, aa in CONDITIONS:
    print(f"\n{'='*50}")
    print(f"  {condition}  (ii_edge={ii}, aa_edge={aa})")
    print(f"{'='*50}")

    c = copy.deepcopy(cfg)
    c["edges"]["ii_edge"] = ii
    c["edges"]["aa_edge"] = aa
    c["_condition"] = condition

    set_all_seeds(c["seed"])

    dyad_pairs = [(d["investor"], d["trustee"]["name"]) for d in c["game"]["dyads"]]
    world = World(ii, aa, c["game"]["observation_depth"], dyad_pairs, c["behavioral_rnn"]["n_actions"])
    state_dim = compute_state_dim(c, aa_edge=aa)
    investors = [RNNInvestor(c, d["trustee"]["name"]) for d in c["game"]["dyads"]]
    agents = [create_agent(d["trustee"], c, state_dim) for d in c["game"]["dyads"]]

    game = Game(c, world, investors, agents)
    game.run_training(
        num_episodes=c["adversary"]["training_episodes"],
        eval_interval=c["adversary"]["eval_interval"],
        eval_episodes=c["adversary"]["eval_episodes"],
    )

print("\nAll 4 conditions done.")
