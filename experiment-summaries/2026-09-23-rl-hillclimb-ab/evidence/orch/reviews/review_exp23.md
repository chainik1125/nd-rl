# Review exp23 (exp21 with focus_k 16; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp23_cheapfocus_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:56:32Z, rl_secs 575.0, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 339, by_len {'7': 112, '8': 95, '9': 132}, heldout_greedy 0.901
- score.json: solved 339, by_len {'7': 112, '8': 95, '9': 132}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 339
- metrics.jsonl: 34 records {'sample': 5, 'train': 27, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 189191; last targets_solved_cum 780
  - vs rl_stats: rounds 5 (=), total_samples 189191 (=), targets_solved_cum 780 (=), total_train_steps 1313 (max step 1314)
  - harness eval heldout_greedy 0.901 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 47696 accept 0.0113 solved_round 99 cum 99 by_len {'7': 97, '8': 2, '9': 0} new 111
  - r2: n_prompts 1320 k 16 T 0.8 samples 42792 accept 0.0669 solved_round 318 cum 321 by_len {'7': 271, '8': 50, '9': 0} new 251
  - r3: n_prompts 1318 k 16 T 0.8 samples 42520 accept 0.1459 solved_round 566 cum 601 by_len {'7': 443, '8': 147, '9': 11} new 308
  - r4: n_prompts 1266 k 16 T 0.8 samples 41024 accept 0.1616 solved_round 558 cum 728 by_len {'7': 523, '8': 177, '9': 28} new 146
  - r5: n_prompts 1248 k 6 T 0.8 samples 15159 accept 0.1841 solved_round 551 cum 780 by_len {'7': 556, '8': 182, '9': 42} new 54
  - train records 27: loss first 0.0654 last 0.0460; n_records last 11476
- found_targets.jsonl: 870 rows, 870 raw-distinct, 870 distinct by my normaliser; 780 targets solved, by L_true {7: 556, 8: 182, 9: 42}; names not in target pools: 0; thm field matches pool: 870/870
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [2, 32, 115, 154, 172] schemas [('contraposition', 109), ('export', 63)]

Per-round secs / sample / ft / steps: 137/96/38/300, 125/80/43/300, 133/84/45/300, 130/81/46/300 (focus, k 16), 49/30/17/113 (r5 shrunk). 5 rounds (exp21: 4).

### rl.py static checks (/work/exec/runs/exp23_cheapfocus_s0/source/rl.py, 373 lines, diff vs frozen seed: 150 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** one knob on exp21. Pass. **Gate 1:** pass (GPU1 03:56–04:09; no reviewer rescore overlapped: the exp13 rescore started 04:08:45 on GPU1. exp23's RL phase ended at ≈ 04:06:07 (started 03:56:32 + rl_secs 575), so training was unaffected; at most its seed-fixed harness scoring shared the GPU briefly, which changes wall time only).

## Compare (after reading preregistration/exp23.md, numbers.md, log.md)

Gate 0: prereg commit 03b0639 at 2026-09-23T03:50:15Z; started_utc 03:56:32Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 339 (112/95/132), heldout 0.901 | 339, 112/95/132; 0.901 | reproduces |
| 5 rounds; textbook cum 2/32/115/154/172 | 5 rounds; 2/32/115/154/172 (contraposition 109, export 63) | reproduces |
| prediction (5 rounds, 325; falsified if ≤ 320.5 or 4 rounds): HIT | 5 rounds, 339 | reproduces |
| "best single run, best L9 (132)" | 339 is the top dev score of all 28 runs; L9 132 vs the next best 131 (exp14) | reproduces. The L9 lead is noise-level |
| "one seed so far; s1 running" | exp23_s1 on GPU0 | pending |

## Decision rule for this verdict (written 04:16 UTC, before exp23's seed-1 rescore returned)

Incumbent exp13_s0: 324 (s0) and 328 (s1), mean 326. exp23_s0: 339 (s0), s1 pending. Replicate exp23_s1: 323.
**ADOPT exp23_s0 only if its seed-1 rescore is ≥ 344**, so that its two-sampling-seed mean beats 326 by more than 15. Otherwise REJECT: it is equivalent to the incumbent within noise, and adopting it would be picking the best of 28 dev-selected runs on a threshold-sized gap.

## Seed-1 rescore and verdict

Reviewer seed-1 rescore (GPU0, 04:14–04:16 UTC, after the executor finished): **338/1610** (L7 114, L8 91, L9 133). Two-sampling-seed mean 338.5 vs the incumbent's 326: **+12.5, under the committed rule (≥ 344 needed)**. The training-seed replicate (exp23_s1, 323) does not repeat the gain over the incumbent either.

**REJECT.** Sound (gates 0–2 pass), and the highest-scoring checkpoint of the session, but not better than the incumbent by more than noise under the pre-committed rule. exp23 is the best-supported *recipe* (339/323, both seeds ignited by round 2). If the monitor wants the single highest dev checkpoint for the sealed evaluation, exp23_s0 is the one. The difference from exp13_s0 is not established.
