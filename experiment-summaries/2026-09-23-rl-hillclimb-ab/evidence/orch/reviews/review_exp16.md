# Review exp16 (exp13 + schema-aware textbook allocation: k 32 on uncracked schemas, k 8 on unsolved targets of cracked schemas; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp16_schema_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:50:56Z, rl_secs 558.8, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 324, by_len {'7': 109, '8': 90, '9': 125}, heldout_greedy 0.907
- score.json: solved 324, by_len {'7': 109, '8': 90, '9': 125}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 324
- metrics.jsonl: 37 records {'sample': 5, 'train': 30, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 193139; last targets_solved_cum 732
  - vs rl_stats: rounds 5 (=), total_samples 193139 (=), targets_solved_cum 732 (=), total_train_steps 1451 (max step 1452)
  - harness eval heldout_greedy 0.907 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 21136 accept 0.0253 solved_round 97 cum 97 by_len {'7': 95, '8': 2, '9': 0} new 109
  - r2: n_prompts 1320 k 16 T 0.8 samples 47680 accept 0.0516 solved_round 261 cum 262 by_len {'7': 215, '8': 47, '9': 0} new 184
  - r3: n_prompts 1320 k 16 T 0.8 samples 45056 accept 0.1229 solved_round 473 cum 475 by_len {'7': 331, '8': 130, '9': 14} new 253
  - r4: n_prompts 1318 k 16 T 0.8 samples 45000 accept 0.1699 solved_round 617 cum 624 by_len {'7': 425, '8': 168, '9': 31} new 171
  - r5: n_prompts 1315 k 13 T 0.8 samples 34267 accept 0.2056 solved_round 667 cum 732 by_len {'7': 513, '8': 181, '9': 38} new 122
  - train records 30: loss first 0.0684 last 0.0422; n_records last 11348
- found_targets.jsonl: 839 rows, 839 raw-distinct, 839 distinct by my normaliser; 732 targets solved, by L_true {7: 513, 8: 181, 9: 38}; names not in target pools: 0; thm field matches pool: 839/839
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

Per-round secs / sample / ft / steps: 80/46/32/300, 123/84/36/300, 126/85/39/300, 129/86/39/300, 99/63/33/251 (r5 shrunk). Textbook solved cum (own count) 0/1/4/58/130; schemas contraposition 100, export 30. **No new schema cracked** (the same 2 of 19 as exp13/exp15).

Dev 324 (109/90/125) = incumbent 324 (109/86/129). heldout 0.907 vs 0.896. Different checkpoint (score.json sha matches its own final.pt).

### rl.py static checks (/work/exec/runs/exp16_schema_s0/source/rl.py, 362 lines, diff vs frozen seed: 139 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** uses the targets' `schema` metadata field to allocate samples (target metadata, no proofs). Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp16.md, numbers.md, log.md)

Gate 0: prereg commit f95b840 at 2026-09-23T02:43:58Z; started_utc 02:50:56Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 324, 109/90/125; heldout 0.907 | 324, 109/90/125; 0.907 | reproduces |
| textbook solved cum 130; only contraposition + export cracked | 130 (contraposition 100, export 30) | reproduces |
| prediction (≥ 4 schemas, 340; falsified if ≤ 2 schemas or dev ≤ 302): MISSED, falsified (2 schemas) | 2 schemas; 324 > 302 | reproduces; falsified on the schema criterion, as stated |
| "dev equals exp13 s0's (same seed)" | 324 = 324; by_len differs (L8 +4, L9 −4); a different checkpoint | reproduces. The equal total is a coincidence of the per-length trade |

Process note: the executor smoke-ran exp18's variant with a 20 s budget on GPU0 at 02:55 while this run was sampling there. That is longer than earlier smokes; it probably cost this run a few seconds of sampling. It did not decide the round count (5 rounds, r5 shrunk, same as exp13).

## Verdict

**REJECT** (tie). Dev 324 equals the incumbent's. The tie-break needs "clearly better heldout_greedy": 0.907 vs 0.896 is +0.011 on 1,000 records (SE of the difference ≈ 0.013), not clearly better. The schema-aware allocation did not crack a third schema, so the mechanism it targeted is untested beyond "k 32 at T 1.0 on 17 uncracked schemas for 4 rounds is not enough".
