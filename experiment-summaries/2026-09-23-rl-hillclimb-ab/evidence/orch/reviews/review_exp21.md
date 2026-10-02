# Review exp21 (exp19 + gen-stream focus from round 4; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp21_hot_focus_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:20:23Z, rl_secs 552.7, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 323, by_len {'7': 107, '8': 93, '9': 123}, heldout_greedy 0.908
- score.json: solved 323, by_len {'7': 107, '8': 93, '9': 123}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 323
- metrics.jsonl: 30 records {'sample': 4, 'train': 24, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4]; sum(samples) 194280; last targets_solved_cum 759
  - vs rl_stats: rounds 4 (=), total_samples 194280 (=), targets_solved_cum 759 (=), total_train_steps 1200 (max step 1201)
  - harness eval heldout_greedy 0.908 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 47696 accept 0.0113 solved_round 99 cum 99 by_len {'7': 97, '8': 2, '9': 0} new 111
  - r2: n_prompts 1320 k 16 T 0.8 samples 42792 accept 0.0653 solved_round 326 cum 329 by_len {'7': 274, '8': 55, '9': 0} new 255
  - r3: n_prompts 1319 k 16 T 0.8 samples 42576 accept 0.1539 solved_round 592 cum 621 by_len {'7': 465, '8': 140, '9': 16} new 331
  - r4: n_prompts 1265 k 16 T 0.8 samples 61216 accept 0.2171 solved_round 584 cum 759 by_len {'7': 547, '8': 174, '9': 38} new 175
  - train records 24: loss first 0.0651 last 0.0475; n_records last 11484
- found_targets.jsonl: 872 rows, 872 raw-distinct, 872 distinct by my normaliser; 759 targets solved, by L_true {7: 547, 8: 174, 9: 38}; names not in target pools: 0; thm field matches pool: 872/872
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt']
textbook solved cum by round [2, 27, 119, 165] schemas [('contraposition', 109), ('export', 55), ('demorgan_nand_to_or', 1)]

Per-round secs / sample / ft / steps: 135/94/37/300, 121/76/42/300, 127/79/45/300, 168/117/46/300 (focus). Only 4 rounds: rl_secs 552.7, and no 5th (shrunk) round ran. The 168 s focus round left ~27 s, under the 25 % tail-fill threshold. Round 1 identical to exp19 (99 solved, accept 0.0113); round 2 diverges (326 vs 299) with identical code: GPU training nondeterminism again.

### rl.py static checks (/work/exec/runs/exp21_hot_focus_s0/source/rl.py, 373 lines, diff vs frozen seed: 150 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** exp19 + exp18's focus code. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp21.md, numbers.md, log.md)

Gate 0: prereg commit 20d70e0 at 2026-09-23T03:20:22Z; started_utc 03:20:23Z. **Pass** (1 s).

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 323, 107/93/123; heldout 0.908 | 323, 107/93/123; 0.908 | reproduces |
| textbook 2/27/119/165 by round | 2/27/119/165; schemas contraposition 109, export 55, demorgan_nand_to_or 1 | reproduces |
| prediction (335; falsified if ≤ 324 or no ignition by r3): MISSED, falsified | 323 ≤ 324 | reproduces; honest. The ignition sub-prediction (≥ 20 by r3, ≥ 100 at the end) was met |
| "focus round cost 117 s, so only 4 rounds and no L9 gain" | r4 sampling 117 s; 4 rounds; L9 123 vs exp19's 122 | reproduces |
| (log 03:35) per-run textbook solves by schema for 9 runs | matches my tbcount.py output for every run I counted (exp13 s0/s1, 15, 16, 17 s0/s1, 18, 19, 21) | reproduces |
| "third schemas appear in the last round with one instance, too late to spread" | exp17_s1 r4, exp18 r3, exp19 r5, exp21 (demorgan) | partly: exp18's came in round 3 of 5 and exp17_s1's in round 4 of 5, and neither spread. "late and never spread" is accurate; "in the last round" is not for all of them |

## Verdict

**REJECT** (323 vs 324; equal within noise, heldout 0.908 vs 0.896 not clearly better). Ignition happened early again (2 by r1, 27 by r2), as in exp19: two of two T 1.2 + round-1 runs ignited by round 2.

## Seed-1 rescore (04:11–04:13 UTC, GPU1, after the executor released it)

319/1610 (108/91/120) vs 323 at seed 0 (−4). The REJECT stands.
