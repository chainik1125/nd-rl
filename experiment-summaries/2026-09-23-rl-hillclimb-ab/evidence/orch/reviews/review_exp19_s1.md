# Review exp19_s1 (exp19 recipe, seed 1; replicate)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp19_tbhot_s1` (tools/recount.py)

- status `ok`, seed 1, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:26:33Z, rl_secs 570.8, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 301, by_len {'7': 98, '8': 89, '9': 114}, heldout_greedy 0.906
- score.json: solved 301, by_len {'7': 98, '8': 89, '9': 114}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 301
- metrics.jsonl: 35 records {'sample': 5, 'train': 28, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 208940; last targets_solved_cum 705
  - vs rl_stats: rounds 5 (=), total_samples 208940 (=), targets_solved_cum 705 (=), total_train_steps 1390 (max step 1391)
  - harness eval heldout_greedy 0.906 == summary: True
  - r1: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.0119 solved_round 99 cum 99 by_len {'7': 98, '8': 1, '9': 0} new 109
  - r2: n_prompts 1336 k 16 T 0.8 samples 45304 accept 0.0488 solved_round 271 cum 273 by_len {'7': 255, '8': 18, '9': 0} new 182
  - r3: n_prompts 1336 k 16 T 0.8 samples 44944 accept 0.1225 solved_round 491 cum 538 by_len {'7': 415, '8': 118, '9': 5} new 288
  - r4: n_prompts 1334 k 16 T 0.8 samples 44456 accept 0.1586 solved_round 548 cum 654 by_len {'7': 470, '8': 152, '9': 32} new 130
  - r5: n_prompts 1332 k 10 T 0.8 samples 26300 accept 0.1946 solved_round 586 cum 705 by_len {'7': 504, '8': 165, '9': 36} new 60
  - train records 28: loss first 0.0635 last 0.0428; n_records last 11072
- found_targets.jsonl: 769 rows, 769 raw-distinct, 769 distinct by my normaliser; 705 targets solved, by L_true {7: 504, 8: 165, 9: 36}; names not in target pools: 0; thm field matches pool: 769/769
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [2, 47, 104, 115, 131] schemas [('contraposition', 109), ('export', 22)]

source/rl.py byte-identical to exp19_tbhot_s0. Per-round secs / sample / ft / steps: 127/91/33/300, 120/83/34/300, 124/84/37/300, 122/81/38/300, 77/50/25/190 (r5 shrunk). Early ignition again (2 by r1, 47 by r2) but it plateaued (104 → 115 → 131); 2 schemas.

**Hard constraints:** identical to exp19. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp19_s1.md, numbers.md, log.md)

Gate 0: prereg commit 293bcc5 at 2026-09-23T03:22:16Z; started_utc 03:26:33Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 301, 98/89/114; heldout 0.906 | 301, 98/89/114; 0.906 | reproduces |
| textbook 2/47/104/…/131 (contraposition 109, export 22) | 2/47/104/115/131; contraposition 109, export 22 | reproduces |
| prediction (320; falsified if no ignition by r3 or ≤ 290): not falsified, below the point estimate | ignition by r2; 301 > 290 | reproduces; correctly scored |
| two-seed means: exp19 314.5, exp13 296.5, exp17 283.5 | 314.5 / 296.5 / 283.5 | arithmetic reproduces. Each mean rests on n = 2 with spreads of 27 / 55 / 95. The +18 between exp19 and exp13 is **not a demonstrated difference**. It should not be read as ranking the recipes |
| "exp19 is the only textbook recipe whose ignition came by round 2 on both seeds" | exp19 s0 27 by r2, s1 47 by r2; exp13 1/0, exp17 1/1 by r2 | reproduces among the 2-seed recipes. exp21 (same code as exp19 through round 3) also ignited by r2 on s0 |

## Verdict

**REJECT** (301 < 324). As a replicate it supports "T 1.2 + round-1 start ignites early" (2 of 2 seeds, plus exp21_s0 on the same code: 3 of 3 runs). Early ignition did not guarantee a high score (131 solves → 301).
