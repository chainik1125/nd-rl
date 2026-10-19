# Review exp15 (exp13 recipe + per-target weighting 12; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp15_tb_pertarget_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:43:05Z, rl_secs 570.2, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 303, by_len {'7': 104, '8': 89, '9': 110}, heldout_greedy 0.897
- score.json: solved 303, by_len {'7': 104, '8': 89, '9': 110}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 303
- metrics.jsonl: 35 records {'sample': 5, 'train': 28, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 187952; last targets_solved_cum 741
  - vs rl_stats: rounds 5 (=), total_samples 187952 (=), targets_solved_cum 741 (=), total_train_steps 1361 (max step 1362)
  - harness eval heldout_greedy 0.897 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 21136 accept 0.0253 solved_round 97 cum 97 by_len {'7': 95, '8': 2, '9': 0} new 109
  - r2: n_prompts 1320 k 16 T 0.8 samples 47680 accept 0.0746 solved_round 345 cum 346 by_len {'7': 271, '8': 75, '9': 0} new 274
  - r3: n_prompts 1319 k 16 T 0.8 samples 47664 accept 0.1365 solved_round 509 cum 511 by_len {'7': 355, '8': 137, '9': 19} new 176
  - r4: n_prompts 1319 k 16 T 0.8 samples 47664 accept 0.2082 solved_round 664 cum 666 by_len {'7': 468, '8': 165, '9': 33} new 170
  - r5: n_prompts 1316 k 8 T 0.8 samples 23808 accept 0.2718 solved_round 736 cum 741 by_len {'7': 533, '8': 172, '9': 36} new 78
  - train records 28: loss first 0.0550 last 0.0341; n_records last 16892
- found_targets.jsonl: 807 rows, 807 raw-distinct, 807 distinct by my normaliser; 741 targets solved, by L_true {7: 533, 8: 172, 9: 36}; names not in target pools: 0; thm field matches pool: 807/807
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

Per-round secs / sample / ft / steps: 89/46/41/300, 131/84/44/300, 134/84/46/300, 141/90/47/300, 75/47/25/161 (r5 shrunk). Textbook solved cum (own count) 0/1/12/95/150; schemas contraposition 106, export 44 (the same two as exp13).

### rl.py static checks (/work/exec/runs/exp15_tb_pertarget_s0/source/rl.py, 356 lines, diff vs frozen seed: 137 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** exp13 code + exp12's build_mix weighting (both reviewed clean). Pass. **Gate 1:** pass.

Seed-1 rescore: not queued. 303 is below the incumbent's 324.

## Compare (after reading preregistration/exp15.md, numbers.md, log.md)

Gate 0: prereg commit 838368e at 2026-09-23T02:39:13Z; started_utc 02:43:05Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 303, 104/89/110; heldout 0.897 | 303, 104/89/110; 0.897 | reproduces |
| textbook solved 0/12/95/150 | 0/1/12/95/150 (rounds 1–5; the log omits round 1's 0 or round 2's 1) | reproduces |
| prediction (340, textbook ≥ 180; falsified ≤ 302): MISSED, falsified | 303 is 1 above the 302 line, so the dev criterion is formally *not* falsified. Textbook 150 < 180 misses the other expectation, which was stated as an expectation, not a falsifier | the executor is stricter than its own rule. Reporting it as a miss is fine; to be exact, "dev passed the falsification line by 1; textbook expectation missed" |
| "per-target weighting adds nothing on top of the textbook stream (324 for exp13 on the same seed)" | −21 vs exp13_s0 on one seed. The exp13 recipe's own seed spread is 269–324 (55), so one paired run cannot show "nothing" or "harm" | **reword**: "no gain detected (one seed, −21, inside exp13's seed spread)" |

## Verdict

**REJECT.** Sound, not better than the incumbent (303 vs 324). It is a third run of the textbook stream that ignited (150 solves), which strengthens the reading that the stream is what moves dev L7 (104 here, 109 in exp13_s0, 67 in exp13_s1 with 13 solves).
