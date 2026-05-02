# Two-Dyad MRTT — Implementation Plan

Self-contained plan to extend the current single-investor / N-trustee codebase to **two parallel dyads** (i1↔a1, i2↔a2) with two independent edge flags controlling cross-pair information flow. Hand to Sonnet as-is.

---

## 0. Scope

- Two investors (i1, i2), two trustees (a1, a2). Fixed pairing i_k ↔ a_k.
- One trained `BehavioralRNN` checkpoint, reused as a frozen function. **No retraining.**
- Two independent edge flags:
  - `ii_edge` — investors observe each other (cross-update on investor hidden state via the trained GRU).
  - `aa_edge` — trustees observe each other (cross-pair window appended to DQN state).
- Spillover is **deleted everywhere** (no α, no `_apply_spillover`).
- Per-investor wallet of 20 units; each wallet only funds its paired trustee. No cross-pair allocation, no trust-ranked sort.
- Trust decay `γ` (existing `behavioral_rnn.trust_decay`) is **preserved**, applied once per round per investor.

The 4 experimental conditions are the 2×2 over the edge flags:

| Condition  | ii_edge | aa_edge | Active edges                              |
|------------|---------|---------|-------------------------------------------|
| DPG1_W0    | 0       | 0       | a-i only                                  |
| DPG1_W1    | 0       | 1       | a-i, **a-a**  (trustees talk)             |
| DPG2_W0    | 1       | 0       | a-i, **i-i**  (investors talk)            |
| DPG2_W1    | 1       | 1       | a-i, **i-i**, **a-a**                     |

---

## 1. Update equations (compressed)

Per round t = 0,…,T−1, dyads k ∈ {1, 2}:

**Step A — decay (always, both investors):**
```
h_k ← γ · h_k
```

**Step B — cross-investor update (only if ii_edge = 1):**
Snapshot first, then symmetric apply:
```
h_1' = f_h(h_1, â_2^{t-1}, ρ_2^{t-1})
h_2' = f_h(h_2, â_1^{t-1}, ρ_1^{t-1})
h_1, h_2 ← h_1', h_2'
```
At t=0, prev-round values are zeros (same convention as the existing reset).
`f_h` = trained GRU step returning only the new hidden state.

**Step C — self update + decision (each investor independently):**
```
(h_k, π_k) = f(h_k, â_k^{t-1}, ρ_k^{t-1})
action_k    = sample(π_k) if inference_sample else argmax(π_k)
desired_k   = action_values[action_k]
```

**Step D — per-dyad allocation:**
```
invest_k = floor_to_bucket(min(desired_k, 20))
```
Per-dyad, no shared budget.

**Step E — trustee state, decision, reward (per dyad):**
```
s_k = [h_k(5), π_k(5), one_hot(action_k)(5), t/(T-1)(1),
       ii_edge(1), aa_edge(1),
       cross_window_k(8 if aa_edge else 0)]

repay_pct_k    = a_k.act(s_k) ∈ [0, 1]
repayment_k    = repay_pct_k · invest_k · multiplier
investor_reward_k = repayment_k − invest_k
agent_reward_k    = invest_k · multiplier − repayment_k
```
`cross_window_k` = last 4 (invest, repay) pairs from the *other* dyad, zero-padded at episode start. Same semantics as today's `World.get_others_observation` with d=4.

**Step F — bookkeeping (used by next round):**
```
â_k^t = one_hot(bucket_index(invest_k))
ρ_k^t = repayment_k / (invest_k · multiplier),   0 if invest_k = 0
```
Both stored on the investor (`_prev_ah`, `_prev_rp`) and on the World (per-dyad history).

---

## 2. Invariants (these are the smoke-test gates)

1. `0 ≤ invest_k ≤ 20` per round, per dyad.
2. `investor_reward_k + agent_reward_k = 2 · invest_k` per round, per dyad.
3. With `ii_edge = aa_edge = 0`, each dyad must be **bit-identical** to a standalone N=1 Dezfouli run with the same seed and same checkpoint — Fig 5D pattern must reproduce per dyad.
4. State dim: 18 when `aa_edge=0`, 26 when `aa_edge=1` (regardless of `ii_edge`).

---

## 3. Config schema (replace current `config.yaml` sections)

Replace:

```yaml
game:
  agents: [...]                 # current single-list
world:
  mode: 1                       # current binary global
behavioral_rnn:
  spillover_alpha: 0.0          # DELETE
```

With:

```yaml
game:
  dyads:
    - investor: i1
      trustee:  {name: a1, type: max,  save_path: checkpoints/{condition}_a1.pt}
    - investor: i2
      trustee:  {name: a2, type: fair, save_path: checkpoints/{condition}_a2.pt}
  endowment_per_investor: 20
  multiplier: 3
  max_rounds: 10
  observation_depth: 4          # for aa_edge cross-pair window
edges:
  ii_edge: 0                    # 0 or 1
  aa_edge: 0                    # 0 or 1
behavioral_rnn:
  # spillover_alpha removed
  # everything else unchanged (hidden_size, dropout, lr, trust_decay, save_path, ...)
```

The condition-builder in `main.py` overrides `edges.ii_edge` and `edges.aa_edge` per condition.

---

## 4. File-by-file changes

### 4.1 `world.py` — replace `mode` with explicit edge flags + dyad indexing

- Constructor: `__init__(self, ii_edge: int, aa_edge: int, observation_depth: int, dyad_pairs: list[tuple[str, str]])`
  where `dyad_pairs[k] = (investor_name_k, trustee_name_k)`.
- Drop `mode`, `communication`. Drop `agent_names`-only history; replace with per-dyad history `self.history[k] = [(invest_t, repay_t), …]`.
- Methods:
  - `record_dyad_step(k: int, invest: float, repay: float)` — append to dyad k's history.
  - `get_other_pair_window(k: int) -> np.ndarray` — last `observation_depth` (invest, repay) pairs from the *other* dyad, zero-padded. Always returns shape `(2 · observation_depth,)`. **Caller decides whether to use it** (gated by `aa_edge`).
  - `get_other_pair_last_action_repay(k: int) -> tuple[np.ndarray, float]` — returns `(â_other^{t-1}, ρ_other^{t-1})` for the cross-investor update. At t=0, returns `(zeros(n_actions), 0.0)`.
- Delete `get_others_observation`, `others_obs_dim`, `total_obs_dim`.

### 4.2 `agents/investor.py` — extract decay, add `cross_step`, simplify per-investor

- `RNNInvestor` now manages **exactly one** trustee. Constructor takes `(config, trustee_name, device)`. Drop the per-agent dict structure — single `_h`, single `_prev_ah`, single `_prev_rp`, single cached `_h_predecision`/`_policy_vec`/`_action_onehot`.
- Move trust-decay out of `act()` into a separate method:
  ```python
  def decay(self) -> None:
      if self.trust_decay < 1.0:
          self._h *= self.trust_decay
  ```
- Add cross-step method (no sampling, no caching, just hidden-state update):
  ```python
  def cross_step(self, other_action_oh: np.ndarray, other_repay_prop: float) -> None:
      ah = torch.tensor(other_action_oh, dtype=torch.float32, device=self.device)
      h_new, _ = self.model.step_forward(self._h, ah, float(other_repay_prop))
      self._h = h_new
  ```
- `act()` no longer applies decay. Otherwise unchanged (self GRU step, sample/argmax, cache for `get_rnn_info`).
- `observe_outcome(actual_investment, repay_prop)` — drop the `agent_name` arg.
- `get_rnn_info()` and `get_last_action_onehot_and_repay()` for the cross-step lookup.
- **Delete:** `get_hidden_state`, `set_hidden_state` (spillover artifacts).

### 4.3 `game.py` — multi-investor orchestration

- Constructor signature: `Game(config, world, investors: list[RNNInvestor], agents: list[Adversary])`.
  Maintain explicit dyad index k for each pair: `dyads = list(zip(investors, agents))`.
- Add `_pair_index: dict[str, int]` mapping trustee name → dyad index, used everywhere allocation/state needs the right investor.
- `_allocate(desired)` — replace trust-ranked sort with per-dyad floor:
  ```python
  return {name: self._floor_to_bucket(min(desired[name], self.endowment_per_investor))
          for name in desired}
  ```
- Per-round loop in `_run_episode`:
  1. **Decay** all investors: `for inv in investors: inv.decay()`.
  2. **Cross-investor update** if `ii_edge`:
     - Snapshot: `prev = [(world.get_other_pair_last_action_repay(k)) for k in range(2)]`
       — **important:** these come from the *previous* round's recorded outcomes, fetched from World, *not* from the investors' caches (so the snapshot/apply ordering is naturally safe).
     - Apply: `investors[0].cross_step(*prev_for_0)`, `investors[1].cross_step(*prev_for_1)`.
  3. **Self step + decision:** `desired[a_k] = investors[k].act()` for each k.
  4. **Allocate per-dyad** (Step 4 above).
  5. **Per-dyad trustee phase:** for each k, build state `s_k` (see 4.4), call `agent.act(s_k)`, compute repayment, accumulate rewards, call `investor.observe_outcome(invest, repay_prop)`, `world.record_dyad_step(k, invest, repayment)`, `agent.observe(reward, done)`, `agent.accumulate_investor_reward(investor_reward)`.
- **Delete** `_apply_spillover`.

### 4.4 `agents/__init__.py` — new state dim + state builder

- `compute_state_dim(config)`:
  ```python
  base = rnn_hidden(5) + n_actions(5) + n_actions(5) + 1   # = 16
  flags = 2                                                  # ii_edge, aa_edge always present
  cross = 2 * observation_depth if aa_edge else 0           # 8 or 0
  return base + flags + cross
  ```
  Drop the `world` argument; take `aa_edge` and `observation_depth` from config directly.
- The state builder lives in `Game._build_adversary_state(k, t, n_rounds)`:
  ```python
  h, policy, ah = investors[k].get_rnn_info()
  round_norm = t / max(n_rounds - 1, 1)
  flags = [edges.ii_edge, edges.aa_edge]
  parts = [h, policy, ah, [round_norm], flags]
  if edges.aa_edge:
      parts.append(world.get_other_pair_window(k))
  return np.concatenate(parts).astype(np.float32)
  ```

### 4.5 `main.py` — config plumbing, experiment matrix, smoke tests

- Strip α validation in `main()`.
- `_build(cfg)`:
  - `dyad_pairs = [(d['investor'], d['trustee']['name']) for d in cfg['game']['dyads']]`
  - `world = World(cfg['edges']['ii_edge'], cfg['edges']['aa_edge'], cfg['game']['observation_depth'], dyad_pairs)`
  - `investors = [RNNInvestor(cfg, trustee_name) for (_, trustee_name) in dyad_pairs]`
  - `agents = [create_agent(d['trustee'], cfg, state_dim) for d in cfg['game']['dyads']]`
- Replace `EXPERIMENT_CONDITIONS` with the 2×2 over edge flags:
  ```python
  EXPERIMENT_CONDITIONS = [
      ("DPG1_W0", 0, 0),
      ("DPG1_W1", 1, 0),
      ("DPG2_W0", 0, 1),
      ("DPG2_W1", 1, 1),
  ]
  ```
- `_build_condition_config(base_cfg, condition, ii, aa)` — deep-copy and override `edges.ii_edge`, `edges.aa_edge`, and the trustee `save_path` keys with the condition prefix.
- Smoke tests (replace existing A–D with):
  - **Test A — state dim:** for each (ii, aa) ∈ {0,1}²: `compute_state_dim` returns `18 + 8·aa`.
  - **Test B — single-dyad N=1 reproducibility:** with one dyad only, `ii=0`, `aa=0`, fixed seed, run 10 greedy episodes against the existing N=1 checkpoint. Mean per-round repayment must equal the pre-existing N=1 result to ≤ 1e-6.
  - **Test C — per-investor wallet bound:** for each of the 4 conditions, run 10 episodes, assert `invest_k ≤ 20` for every (round, dyad).
  - **Test D — per-dyad reward conservation:** for each of the 4 conditions, run 10 episodes, assert `|investor_reward_k + agent_reward_k − 2 · invest_k| < 1e-6` for every row.
  - **Test E — ii_edge effect is non-trivial:** run 10 greedy episodes with `(ii=0, aa=0)` vs `(ii=1, aa=0)` against the same DQN checkpoints (use a checkpoint trained on either condition — the *behavior* must differ because the investor hidden states diverge). Assert mean repayment differs by > 1e-3 in at least one round.
- Strip the 2×2 grid plot's `α` axis. Replace with a 2×2 over (ii, aa).
- Strip the bootstrap `α`-decomposition; replace with bootstrap CIs for:
  - `ii_effect_at_aa0 = mean(DPG1_W1) − mean(DPG1_W0)`
  - `ii_effect_at_aa1 = mean(DPG2_W1) − mean(DPG2_W0)`
  - `aa_effect_at_ii0 = mean(DPG2_W0) − mean(DPG1_W0)`
  - `aa_effect_at_ii1 = mean(DPG2_W1) − mean(DPG1_W1)`
  - `interaction = (DPG2_W1 − DPG2_W0) − (DPG1_W1 − DPG1_W0)`
  All on the investor-cumulative metric, total over both investors.

### 4.6 Files NOT changed

- `models/behavioral_rnn.py` — frozen architecture.
- `models/train_behavioral.py` — training data unchanged.
- `models/q_learner.py` — DQN code unchanged.
- `agents/adversary.py` — DQN adversary unchanged (state dim is passed in).
- `agents/base.py`, `agents/random_agent.py` — unchanged.
- `data/parse_mrtt.py` — unchanged.
- The existing N=1 checkpoint stays in place for Test B.

---

## 5. Implementation order

1. **Strip spillover.** Delete `_apply_spillover`, `spillover_alpha` from config + validation in `main.py`, all references in the matrix runner. Verify N=1 baseline still runs end-to-end with no behavior change (α was 0.0 in current default).
2. **Refactor `world.py`** to the new edge-flag interface. Update any imports.
3. **Refactor `RNNInvestor`** to single-trustee scope; extract `decay()`; add `cross_step()`. Update direct callers.
4. **Refactor `Game`** to take a list of investors, run the per-round flow above, and use per-dyad allocation.
5. **Update `compute_state_dim` and the state builder.** Move the builder fully into `Game` (or keep in `agents/__init__.py` if cleaner — Sonnet's call).
6. **Wire `main.py`:** new `_build`, new condition matrix, new condition-config builder, new smoke tests.
7. **Run smoke tests** (`python main.py --smoke-test`). All five must pass before training.
8. **Run the experiment matrix** (`python main.py --experiment-matrix`). 4 conditions × 50k episodes.
9. **Inspect aggregate plots and effect decomposition** in the matrix output dir.

After each numbered step, the code must still import and `python main.py --smoke-test` must execute without crashing (tests may fail until step 6).

---

## 6. Run plan

```bash
# Smoke
python main.py --smoke-test

# Full matrix (4 conditions, ~40 min compute)
python main.py --experiment-matrix
```

Outputs in `outputs/experiment_matrix_<timestamp>/`:
- `DPG1_W0/`, `DPG1_W1/`, `DPG2_W0/`, `DPG2_W1/` — per-condition: `config.yaml`, `training_returns.csv`, `training_curves.png`, `repayment_rates.png`, `game_log.csv`, `figure5c_mean_repayment.png`, `figure5c_mean_investment.png`, `summary.json`.
- `comparison_2x2_grid.png` — mean repayment by round, 2×2 over (ii_edge, aa_edge).
- `main_effects_table.csv` — per-condition investor + per-trustee earnings.
- `effect_decomposition.csv` — bootstrap 95% CI for the 5 effects above.
- `experiment_report.md` — text summary.

---

## 7. Acceptance criteria

A run is considered correct iff **all** hold:

1. All 5 smoke tests pass.
2. Invariant audit passes for all 4 conditions (already implemented in `_audit_invariants`; just update the wallet to per-investor).
3. In `DPG1_W0`, both dyads' Fig 5D-style per-round repayment curves match the existing N=1 baseline (R0 ≈ 75%, R9 ≈ 28% for MAX trustee; FAIR trustee curves match the N=2 baseline you've already validated).
4. The state dim printed at startup matches the table:
   - DPG1_W0, DPG1_W1: 18
   - DPG2_W0, DPG2_W1: 26
5. Per-investor wallet never exceeds 20 in any logged row.
6. Per-dyad reward conservation holds in every logged row.

---

## 8. Notes / things to be careful about

- **Cross-step inputs come from World, not from the other investor's cache.** This is the snapshot-safe pattern — World holds the previous-round outcome regardless of update order.
- **Decay placement:** decay must happen exactly once per round per investor, *before* both the cross-step and the self-step. Don't accidentally call it inside `act()` after extraction.
- **Bucketing for `â_k^{t-1}`:** the cross-step's action one-hot must use the same `bucket_investment` rule as `observe_outcome` so the input distribution matches what the RNN was trained on.
- **t=0 cross-step:** previous-round outcomes are zeros by convention (matches the dummy-zero prepend used in training).
- **Flag-bit dimensions:** even though `ii_edge` and `aa_edge` are constant within a single training run, they're kept in the state (2 dims) for code uniformity. Negligible cost; do not optimize them away.
- **Two trustee policies per dyad:** keep one MAX and one FAIR (matches the existing N=2 setup) so the FAIR-gap metric remains comparable.
