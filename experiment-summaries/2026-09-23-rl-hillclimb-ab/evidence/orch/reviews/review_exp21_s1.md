# Review exp21_s1 (exp21 recipe, seed 1; replicate)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp21_hot_focus_s1` (tools/recount.py)

- status `ok`, seed 1, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:32:05Z, rl_secs 572.6, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 318, by_len {'7': 98, '8': 94, '9': 126}, heldout_greedy 0.898
- score.json: solved 318, by_len {'7': 98, '8': 94, '9': 126}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 318
- metrics.jsonl: 30 records {'sample': 4, 'train': 24, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4]; sum(samples) 200200; last targets_solved_cum 730
  - vs rl_stats: rounds 4 (=), total_samples 200200 (=), targets_solved_cum 730 (=), total_train_steps 1200 (max step 1201)
  - harness eval heldout_greedy 0.898 == summary: True
  - r1: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.0119 solved_round 99 cum 99 by_len {'7': 98, '8': 1, '9': 0} new 109
  - r2: n_prompts 1336 k 16 T 0.8 samples 45304 accept 0.0483 solved_round 271 cum 273 by_len {'7': 252, '8': 21, '9': 0} new 185
  - r3: n_prompts 1336 k 16 T 0.8 samples 44928 accept 0.1288 solved_round 525 cum 574 by_len {'7': 437, '8': 126, '9': 11} new 336
  - r4: n_prompts 1289 k 16 T 0.8 samples 62032 accept 0.2116 solved_round 570 cum 730 by_len {'7': 523, '8': 167, '9': 40} new 176
  - train records 24: loss first 0.0635 last 0.0434; n_records last 11224
- found_targets.jsonl: 806 rows, 806 raw-distinct, 806 distinct by my normaliser; 730 targets solved, by L_true {7: 523, 8: 167, 9: 40}; names not in target pools: 0; thm field matches pool: 806/806
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt']
textbook solved cum by round [2, 49, 113, 159] schemas [('contraposition', 109), ('export', 50)]

source/rl.py byte-identical to exp21_hot_focus_s0. Per-round secs / sample / ft / steps: 134/92/38/300, 130/87/40/300, 135/88/44/300, 172/122/45/300 (focus); 4 rounds. Early ignition (2 by r1, 49 by r2).

**Hard constraints:** identical to exp21. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp21_s1.md, numbers.md, log.md)

Gate 0: prereg commit 328b56b at 2026-09-23T03:26:40Z; started_utc 03:32:05Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 318, 98/94/126; heldout 0.898 | 318, 98/94/126; 0.898 | reproduces |
| textbook cum 159 at r4; 4 rounds; focus round 122 s | 2/49/113/159; 4 rounds; r4 sampling 122 s | reproduces |
| prediction (325; falsified if no ignition by r3 or ≤ 290): not falsified, below the point estimate | ignition by r2; 318 | reproduces; correctly scored. L9 ≥ 120 met (126) |
| "exp21 two-seed mean 320.5 (323/318), the tightest pair so far" | 320.5; spread 5 vs exp19 27, exp13 55, exp17 95 | reproduces |

Early ignition is now 4 of 4 runs of the T 1.2 + round-1 code (exp19 s0/s1, exp21 s0/s1). That is the most consistent result of the session. exp21's two-seed mean (320.5) is the highest recipe mean, but it is not established over exp19 (314.5) with n = 2 each.

## Verdict

**REJECT** as a checkpoint (318 < 324). The incumbent checkpoint remains exp13_s0 (324), and the recipe that would most reliably reproduce ~320 is exp21 (323/318). If a single checkpoint must be chosen for the sealed L10+ evaluation, the incumbent and exp21_s0 (323, heldout 0.908) are equivalent within noise.

## Addendum (04:00 UTC)

exp19_s2 (same code as exp19/exp21 through round 3) did not ignite by round 3. Early ignition of the T 1.2 + round-1 code is **4 of 5**, not 4 of 4.
