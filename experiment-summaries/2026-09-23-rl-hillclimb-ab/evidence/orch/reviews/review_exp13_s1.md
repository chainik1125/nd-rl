# Review exp13_s1 (exp13 recipe, training seed 1; replicate)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp13_textbook_s1` (tools/recount.py)

- status `ok`, seed 1, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:38:58Z, rl_secs 566.9, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 269, by_len {'7': 67, '8': 85, '9': 117}, heldout_greedy 0.91
- score.json: solved 269, by_len {'7': 67, '8': 85, '9': 117}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 269
- metrics.jsonl: 36 records {'sample': 5, 'train': 29, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 201124; last targets_solved_cum 588
  - vs rl_stats: rounds 5 (=), total_samples 201124 (=), targets_solved_cum 588 (=), total_train_steps 1438 (max step 1439)
  - harness eval heldout_greedy 0.91 == summary: True
  - r1: n_prompts 1336 k 16 T 0.8 samples 21376 accept 0.0266 solved_round 97 cum 97 by_len {'7': 96, '8': 1, '9': 0} new 107
  - r2: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.0435 solved_round 228 cum 229 by_len {'7': 200, '8': 29, '9': 0} new 139
  - r3: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.1087 solved_round 454 cum 454 by_len {'7': 319, '8': 122, '9': 13} new 257
  - r4: n_prompts 1336 k 16 T 0.8 samples 47936 accept 0.1491 solved_round 536 cum 537 by_len {'7': 356, '8': 150, '9': 31} new 100
  - r5: n_prompts 1335 k 12 T 0.8 samples 35940 accept 0.1727 solved_round 586 cum 588 by_len {'7': 384, '8': 165, '9': 39} new 60
  - train records 29: loss first 0.0608 last 0.0446; n_records last 10652
- found_targets.jsonl: 663 rows, 663 raw-distinct, 663 distinct by my normaliser; 588 targets solved, by L_true {7: 384, 8: 165, 9: 39}; names not in target pools: 0; thm field matches pool: 663/663
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

source/rl.py is byte-identical to exp13_textbook_s0's. Per-round secs / sample / ft / steps: 78/44/32/300, 123/85/35/300, 130/89/38/300, 131/89/39/300, 103/69/31/238 (r5 shrunk).

**Textbook-source targets solved (own count), cumulative by first-solve round: 0/0/0/1/13** (vs exp13_s0 0/1/16/84/145). Schemas: export 11, contraposition 2. The textbook ignition happened a round later and 10× smaller.

Dev: 269 (67/85/117) vs exp13_s0 324 (109/86/129); heldout 0.910 vs 0.896.

**Hard constraints:** identical code to exp13 (reviewed clean). Pass. **Gate 1:** pass (GPU0 02:39–02:51; no reviewer rescore ran).

## Compare (after reading preregistration/exp13_s1.md, numbers.md, log.md)

Gate 0: prereg commit 8fec468 at 2026-09-23T02:38:57Z; started_utc 02:38:58Z. **Pass** (1 s).

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 269, 67/85/117; heldout 0.910 | 269, 67/85/117; 0.91 | reproduces |
| prediction (310, textbook ≥ 80; falsified if ≤ 270 or textbook < 20): MISSED, falsified on both | 269; 13 | reproduces; honest miss by the executor's own rule |
| textbook cum 0/0/1/13 by round (log lists 4 values for 5 rounds) | 0/0/0/1/13 (rounds 1–5) | reproduces (the log omits round 1) |
| exp13 s0 textbook by schema: contraposition 108/109, export 37/94, 17 others 0 | contraposition 108, export 37 (own count). Denominators not re-derived | reproduces (numerators) |
| "two-seed mean 296.5 vs control-like 219–249; both seeds ≥ 249; size of the gain depends on when the first textbook schema cracks" | 324 and 269: both above every non-textbook run (max 269, exp14; 255 excluding exp14). The dev L7 gain tracks textbook solves: s0 145 → L7 109; s1 13 → L7 67 | supported. Two seeds, both above the control band, so it passes the two-seed standard. The size (+20 to +75) is seed-dependent |

## Verdict

**REJECT** as a checkpoint (269 < incumbent 324). As a replicate it **supports** exp13's recipe over the seed control on two seeds (269, 324 vs 219–249), and exp13_s0 stays the incumbent (see the addendum in review_exp13.md). The s0 result should be described as the upper end of a seed-dependent effect, not a typical outcome.
