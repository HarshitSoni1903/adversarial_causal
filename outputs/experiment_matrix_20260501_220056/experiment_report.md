# Experiment Matrix Report

Generated: 2026-05-01 22:10:49

## Condition Summaries

### DPG1_W0
- ii_edge=0, aa_edge=0
- Investor earnings: -20.09 ± 57.36  (n=1000)
- a1 (max): reward=188.15 ± 113.66
- a2 (fair): reward=115.39 ± 60.13, FAIR gap=82.29
- Training time: 128.5s

### DPG1_W1
- ii_edge=0, aa_edge=1
- Investor earnings: -2.28 ± 62.78  (n=1000)
- a1 (max): reward=187.37 ± 115.08
- a2 (fair): reward=102.97 ± 61.16, FAIR gap=57.64
- Training time: 122.6s

### DPG2_W0
- ii_edge=1, aa_edge=0
- Investor earnings: 16.35 ± 76.81  (n=1000)
- a1 (max): reward=191.50 ± 138.26
- a2 (fair): reward=97.10 ± 59.93, FAIR gap=53.59
- Training time: 162.2s

### DPG2_W1
- ii_edge=1, aa_edge=1
- Investor earnings: 32.55 ± 78.35  (n=1000)
- a1 (max): reward=184.03 ± 132.42
- a2 (fair): reward=89.07 ± 53.66, FAIR gap=33.02
- Training time: 166.9s

## Main Effects Table

Condition  ii_edge  aa_edge Type Agent  Investor_earnings  Agent_earnings FAIR_gap
  DPG1_W0        0        0  MAX    a1             -20.09          188.15        -
  DPG1_W0        0        0 FAIR    a2             -20.09          115.39    82.29
  DPG1_W1        0        1  MAX    a1              -2.28          187.37        -
  DPG1_W1        0        1 FAIR    a2              -2.28          102.97    57.64
  DPG2_W0        1        0  MAX    a1              16.35          191.50        -
  DPG2_W0        1        0 FAIR    a2              16.35           97.10    53.59
  DPG2_W1        1        1  MAX    a1              32.55          184.03        -
  DPG2_W1        1        1 FAIR    a2              32.55           89.07    33.02

## Effect Decomposition (95% bootstrap CI)

             Effect                            Comparison  Point_est  CI_lo_95  CI_hi_95
   ii effect (aa=0)                     DPG1_W1 − DPG1_W0     17.814    12.652    23.097
   ii effect (aa=1)                     DPG2_W1 − DPG2_W0     16.191     9.273    22.999
   aa effect (ii=0)                     DPG2_W0 − DPG1_W0     36.448    30.819    42.361
   aa effect (ii=1)                     DPG2_W1 − DPG1_W1     34.825    28.625    41.216
ii × aa interaction (DPG2_W1−DPG2_W0) − (DPG1_W1−DPG1_W0)     -1.622   -10.234     6.666

## Observations

### 1. Does ii_edge help at aa=0?
ii_edge increases investor earnings by +17.81 [12.65, 23.10]. CI excludes 0.

### 2. Does aa_edge help at ii=0?
aa_edge increases investor earnings by +36.45 [30.82, 42.36]. CI excludes 0.

### 3. Is there an ii × aa interaction?
ii×aa interaction = -1.62 [-10.23, 6.67]. aa_edge dampens the ii_edge effect. CI includes 0 (not significant at 95%).

## Limitations

- Single training run per condition; no error bars on DQN convergence.
- 1000 eval episodes; CIs reflect sampling variance only.
- Fixed pairing i_k ↔ a_k; no cross-pair allocation.
