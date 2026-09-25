# Review exp23_s1 (exp23 recipe, seed 1; replicate)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp23_cheapfocus_s1` (tools/recount.py)

- status `ok`, seed 1, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T04:02:17Z, rl_secs 567.9, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 323, by_len {'7': 105, '8': 95, '9': 123}, heldout_greedy 0.902
- score.json: solved 323, by_len {'7': 105, '8': 95, '9': 123}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 323
- metrics.jsonl: 35 records {'sample': 5, 'train': 28, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 201768; last targets_solved_cum 733
  - vs rl_stats: rounds 5 (=), total_samples 201768 (=), targets_solved_cum 733 (=), total_train_steps 1367 (max step 1368)
  - harness eval heldout_greedy 0.902 == summary: True
  - r1: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.0119 solved_round 99 cum 99 by_len {'7': 98, '8': 1, '9': 0} new 109
  - r2: n_prompts 1336 k 16 T 0.8 samples 45304 accept 0.0517 solved_round 292 cum 294 by_len {'7': 274, '8': 20, '9': 0} new 205
  - r3: n_prompts 1336 k 16 T 0.8 samples 44872 accept 0.1227 solved_round 497 cum 553 by_len {'7': 433, '8': 112, '9': 8} new 284
  - r4: n_prompts 1296 k 16 T 0.8 samples 43336 accept 0.1495 solved_round 516 cum 661 by_len {'7': 479, '8': 151, '9': 31} new 127
  - r5: n_prompts 1278 k 8 T 0.8 samples 20320 accept 0.1776 solved_round 551 cum 733 by_len {'7': 530, '8': 165, '9': 38} new 76
  - train records 28: loss first 0.0635 last 0.0515; n_records last 11204
- found_targets.jsonl: 801 rows, 801 raw-distinct, 801 distinct by my normaliser; 733 targets solved, by L_true {7: 530, 8: 165, 9: 38}; names not in target pools: 0; thm field matches pool: 801/801
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [2, 56, 105, 124, 165] schemas [('contraposition', 109), ('export', 55), ('demorgan_nand_to_or', 1)]

source/rl.py byte-identical to exp23_cheapfocus_s0. **Hard constraints:** pass. **Gate 1:** pass.

## Compare

Gate 0: prereg commit ae4cb58 at 2026-09-23T03:56:39Z; started_utc 04:02:17Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 323 (105/95/123), heldout 0.902; 5 rounds; textbook 2/56/105/124/165 | 323, 105/95/123; 0.902; 5 rounds; 2/56/105/124/165 (contraposition 109, export 55, demorgan_nand_to_or 1) | reproduces |
| prediction (320; falsified if ≤ 300 or 4 rounds): HIT | 323, 5 rounds | reproduces |
| exp23 two-seed mean 331 (339/323); both seeds ignited by round 2 | 331; 32 and 56 by r2 | reproduces |

## Verdict

**REJECT** as a checkpoint (323 vs 324). As a replicate it makes exp23 the best-supported recipe: two seeds at 339/323, both ignited by round 2, spread 16.
