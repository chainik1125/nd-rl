# Review exp19 (textbook stream from round 1, schema-aware: uncracked schemas at k 32, T 1.2; cracked unsolved at k 8, T 1.0; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp19_tbhot_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:14:39Z, rl_secs 568.4, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 328, by_len {'7': 115, '8': 91, '9': 122}, heldout_greedy 0.897
- score.json: solved 328, by_len {'7': 115, '8': 91, '9': 122}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 328
- metrics.jsonl: 36 records {'sample': 5, 'train': 29, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 203494; last targets_solved_cum 774
  - vs rl_stats: rounds 5 (=), total_samples 203494 (=), targets_solved_cum 774 (=), total_train_steps 1409 (max step 1410)
  - harness eval heldout_greedy 0.897 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 47696 accept 0.0113 solved_round 99 cum 99 by_len {'7': 97, '8': 2, '9': 0} new 111
  - r2: n_prompts 1320 k 16 T 0.8 samples 42792 accept 0.0611 solved_round 299 cum 302 by_len {'7': 258, '8': 44, '9': 0} new 228
  - r3: n_prompts 1318 k 16 T 0.8 samples 42560 accept 0.1389 solved_round 533 cum 563 by_len {'7': 430, '8': 125, '9': 8} new 296
  - r4: n_prompts 1317 k 16 T 0.8 samples 41928 accept 0.1754 solved_round 594 cum 702 by_len {'7': 516, '8': 162, '9': 24} new 157
  - r5: n_prompts 1314 k 11 T 0.8 samples 28518 accept 0.201 solved_round 618 cum 774 by_len {'7': 556, '8': 183, '9': 35} new 78
  - train records 29: loss first 0.0652 last 0.0454; n_records last 11464
- found_targets.jsonl: 870 rows, 870 raw-distinct, 870 distinct by my normaliser; 774 targets solved, by L_true {7: 556, 8: 183, 9: 35}; names not in target pools: 0; thm field matches pool: 870/870
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [2, 27, 104, 149, 177] schemas [('contraposition', 109), ('export', 67), ('disjunctive_syllogism', 1)]

Per-round secs / sample / ft / steps: 128/92/32/300, 115/78/35/300, 119/78/37/300, 122/81/38/300, 83/53/27/209 (r5 shrunk). Ignition in round 1 (2 textbook solves), the earliest of any run.

### rl.py static checks (/work/exec/runs/exp19_tbhot_s0/source/rl.py, 364 lines, diff vs frozen seed: 141 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** exp16's allocation + exp17's start round + T 1.2 for uncracked schemas. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp19.md, numbers.md, log.md)

Gate 0: prereg commit 4bf660c at 2026-09-23T03:08:11Z; started_utc 03:14:39Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 328, 115/91/122; heldout 0.897 | 328, 115/91/122; 0.897 | reproduces |
| textbook cum 27 by r2, 104 r3, 149 r4 | 2 / 27 / 104 / 149 / 177 (r1–r5) | reproduces; r1 = 2 and r5 = 177 omitted |
| "still only contraposition + export" | **3 schemas**: contraposition 109, export 67, **disjunctive_syllogism 1** (la_reserve_510, round 5, re-verified True, 7 lines). exp17_s1 solved the same target at T 1.0; exp18 solved one demorgan_nand_to_or target | **differs**: the executor missed the third schema |
| prediction (≥ 3 schemas, 330; falsified if 2 schemas or dev ≤ 315): "MISSED on schema count, falsified" | by its own criterion **not falsified**: 3 schemas, 328 > 315. Substantively, the third schema is one instance of one target, also reached at T 1.0 (exp17_s1), so T 1.2 is not shown to be the cause | reword: "formally met (3 schemas, 328); the third crack is a single instance, not attributable to T 1.2" |
| "the early ignition it gives is what the score needs" | ignition in r1 (2 solves) → 328. Across the family, dev tracks textbook solves (see review_exp17_s0). One run | plausible; single seed (the s1 replicate is running) |

## Verdict

**REJECT** as incumbent (a single-seed difference too small to count: +4 over 324; heldout 0.897 vs 0.896, equal). Along with exp17_s1 (331) and exp16 (324), this is the fourth textbook-family run at 324–331. The incumbent is not unique, and the family's ceiling at this budget looks like ~330 when ignition happens early. exp19_s1 will show whether early ignition is reliable with T 1.2.

## Addendum (03:36 UTC)

The executor's log (03:35 entry) corrected its exp19 schema count to 3 and re-scored the prediction as HIT. That matches the reviewer's count above. The substantive caveat stands: the third schema is one instance and was also reached at T 1.0.
