# Review exp19_s2 (exp19 recipe, seed 2; third seed)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp19_tbhot_s2` (tools/recount.py)

- status `ok`, seed 2, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:44:13Z, rl_secs 575.5, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 309, by_len {'7': 98, '8': 90, '9': 121}, heldout_greedy 0.889
- score.json: solved 309, by_len {'7': 98, '8': 90, '9': 121}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 309
- metrics.jsonl: 33 records {'sample': 5, 'train': 26, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 205493; last targets_solved_cum 664
  - vs rl_stats: rounds 5 (=), total_samples 205493 (=), targets_solved_cum 664 (=), total_train_steps 1294 (max step 1295)
  - harness eval heldout_greedy 0.889 == summary: True
  - r1: n_prompts 1343 k 16 T 0.8 samples 48048 accept 0.0124 solved_round 103 cum 103 by_len {'7': 102, '8': 1, '9': 0} new 118
  - r2: n_prompts 1343 k 16 T 0.8 samples 48048 accept 0.0705 solved_round 364 cum 364 by_len {'7': 279, '8': 84, '9': 1} new 282
  - r3: n_prompts 1343 k 16 T 0.8 samples 48048 accept 0.1387 solved_round 535 cum 535 by_len {'7': 347, '8': 163, '9': 25} new 209
  - r4: n_prompts 1340 k 16 T 0.8 samples 48000 accept 0.1686 solved_round 592 cum 595 by_len {'7': 387, '8': 175, '9': 33} new 74
  - r5: n_prompts 1339 k 5 T 0.8 samples 13349 accept 0.2098 solved_round 646 cum 664 by_len {'7': 449, '8': 181, '9': 34} new 75
  - train records 26: loss first 0.0624 last 0.0452; n_records last 11020
- found_targets.jsonl: 758 rows, 758 raw-distinct, 758 distinct by my normaliser; 664 targets solved, by L_true {7: 449, 8: 181, 9: 34}; names not in target pools: 0; thm field matches pool: 758/758
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [0, 0, 0, 11, 66] schemas [('contraposition', 40), ('export', 26)]

source/rl.py byte-identical to exp19_tbhot_s0. Ignition only in round 4 (0/0/0/11/66).

**Hard constraints:** identical to exp19. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp19_s2.md, numbers.md, log.md)

Gate 0: prereg commit 8632e58 at 2026-09-23T03:38:42Z; started_utc 03:44:13Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 309 (98/90/121), heldout 0.889 (numbers.md) | 309, 98/90/121; 0.889 | reproduces |
| textbook cum 0/0/0/11/66; no ignition by r3, so FALSIFIED despite dev 309 > 285 | 0/0/0/11/66 | reproduces; correctly scored, and honest |
| "ignited by r2 on 2 of 3 seeds; ignition still random" | exp19 s0 27, s1 47, s2 0 by r2 | reproduces |

Correction to my review_exp21_s1 remark ("early ignition 4 of 4 runs of the T 1.2 + round-1 code"): with this run it is **4 of 5**.

exp19 over three seeds: 328 / 301 / 309, mean 312.7, SD ≈ 14.

## Verdict

**REJECT** (309 < 324). It narrows the recipe estimate: exp19 ≈ 313 ± 14 (n = 3), so the incumbent's 324 is at the top of what this family produces in 600 s.
