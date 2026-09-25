# Review exp18 (exp13 + gen-stream frontier focus from round 4: < 2-proof gen targets, k 32, T 1.0; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp18_tb_focus_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:02:40Z, rl_secs 566.3, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 300, by_len {'7': 76, '8': 94, '9': 130}, heldout_greedy 0.899
- score.json: solved 300, by_len {'7': 76, '8': 94, '9': 130}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 300
- metrics.jsonl: 34 records {'sample': 5, 'train': 27, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 212640; last targets_solved_cum 687
  - vs rl_stats: rounds 5 (=), total_samples 212640 (=), targets_solved_cum 687 (=), total_train_steps 1348 (max step 1349)
  - harness eval heldout_greedy 0.899 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 21136 accept 0.0253 solved_round 97 cum 97 by_len {'7': 95, '8': 2, '9': 0} new 109
  - r2: n_prompts 1320 k 16 T 0.8 samples 47680 accept 0.0539 solved_round 278 cum 279 by_len {'7': 229, '8': 50, '9': 0} new 206
  - r3: n_prompts 1318 k 16 T 0.8 samples 47648 accept 0.1208 solved_round 498 cum 501 by_len {'7': 338, '8': 145, '9': 18} new 250
  - r4: n_prompts 1269 k 16 T 0.8 samples 67168 accept 0.2012 solved_round 553 cum 605 by_len {'7': 388, '8': 180, '9': 37} new 132
  - r5: n_prompts 1242 k 7 T 0.8 samples 29008 accept 0.2313 solved_round 608 cum 687 by_len {'7': 454, '8': 188, '9': 45} new 96
  - train records 27: loss first 0.0697 last 0.0513; n_records last 11168
- found_targets.jsonl: 793 rows, 793 raw-distinct, 793 distinct by my normaliser; 687 targets solved, by L_true {7: 454, 8: 188, 9: 45}; names not in target pools: 0; thm field matches pool: 793/793
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [0, 0, 2, 5, 53] schemas [('contraposition', 52), ('demorgan_nand_to_or', 1)]

Per-round secs / sample / ft / steps: 79/44/33/300, 119/80/36/300, 124/82/39/300, 165/120/40/300 (focus), 77/55/20/148 (r5 shrunk).

**Bookkeeping defect (minor, no score effect):** source line 322–323 logs `k=rcfg['k']` and `temperature=cfg['temperature']`, so the r4–r5 sample records show k 16 / T 0.8 while the gen stream actually ran at focus k 32 / T 1.0 (and the textbook stream at T 1.0 in every exp13-family run). The `samples` counts are correct (they are counted from the returned samples).

### rl.py static checks (/work/exec/runs/exp18_tb_focus_s0/source/rl.py, 362 lines, diff vs frozen seed: 139 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** exp13 + exp14's focus logic. Pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp18.md, numbers.md, log.md)

Gate 0: prereg commit de1ea05 at 2026-09-23T02:55:58Z; started_utc 03:02:40Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 300, 76/94/130; heldout 0.899 | 300, 76/94/130; 0.899 | reproduces |
| textbook solved cum 0/0/2/?/53 | 0/0/2/5/53 (contraposition 52, demorgan_nand_to_or 1) | reproduces; round 4 = 5 |
| prediction (315; falsified if ≤ 297 or L9 ≤ 117): not falsified, below the point estimate | 300 > 297; L9 130 > 117 | reproduces; correctly scored |
| (log 03:09) same code and seed as exp13 through round 3, yet r3 textbook 2 vs 16 → not bit-reproducible; same-seed comparisons are not paired | own check: round-1 sample stats are identical across exp13/15/16/18 seed 0 (97 solved, accept 0.0253, 109 new). exp13 and exp18 diverge at round 2 (330 vs 279 cum) with identical code up to round 4. So the divergence comes after round-1 fine-tuning: training is nondeterministic on GPU | **confirmed**. It also retroactively weakens "seed-matched" readings in exp15/16/17 (the executor made the same point) |
| "focus gives the best L9 band again (130–131 vs 110–129 without)" | L9: exp14 131, exp18 130, vs exp13_s0 129, exp08 126, exp16 125, exp10 125 | **reword**: +1 to +2 over the best non-focus run is noise. "Focus runs have L9 at the top of the range" is fine; "a band" is not |
| logging: sample records' k / T in focus rounds | see recount: they log rcfg k and cfg T | bookkeeping defect, not mentioned; no effect on the score |

## Verdict

**REJECT.** Sound, not better (300 vs 324). The textbook stream ignited late and small (53 solves), which, like exp13_s1, shows ignition is stochastic per run.
