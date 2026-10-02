# Review exp07 (length curriculum 7→8→9 over all 3,804 targets, 1,500 prompts/round, skip saturated; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp07_curriculum_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:51:58Z, rl_secs 564.8, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 217, by_len {'7': 33, '8': 93, '9': 91}, heldout_greedy 0.915
- score.json: solved 217, by_len {'7': 33, '8': 93, '9': 91}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 217
- metrics.jsonl: 41 records {'sample': 3, 'train': 36, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3]; sum(samples) 144000; last targets_solved_cum 866
  - vs rl_stats: rounds 3 (=), total_samples 144000 (=), targets_solved_cum 866 (=), total_train_steps 1800 (max step 1801)
  - harness eval heldout_greedy 0.915 == summary: True
  - r1: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.0502 solved_round 230 cum 230 by_len {'7': 230, '8': 0, '9': 0} new 260
  - r2: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.1693 solved_round 463 cum 558 by_len {'7': 492, '8': 66, '9': 0} new 386
  - r3: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.2515 solved_round 531 cum 866 by_len {'7': 674, '8': 171, '9': 21} new 357
  - train records 36: loss first 0.0618 last 0.0514; n_records last 23992
- found_targets.jsonl: 1003 rows, 1003 raw-distinct, 1003 distinct by my normaliser; 866 targets solved, by L_true {7: 674, 8: 171, 9: 21}; names not in target pools: 0; thm field matches pool: 1003/1003
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt']

Per-round secs / sample / ft: r1 180/111/65, r2 190/115/71, r3 193/114/74; RL records 1,040 / 2,580 / 3,992. Sampling 431/417/420 samples/s (GPU0, uncontended). rl_secs 564.8; the tail-fill code path did not fire (3 full rounds).

### rl.py static checks (/work/exec/runs/exp07_curriculum_s0/source/rl.py, 272 lines, diff vs frozen seed: 37 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** the curriculum filters on the targets' own `L_true` field (target metadata supplied by the harness, no proofs). Otherwise exp06's code. Pass. **Gate 1:** pass.

Seed-1 rescore: not run. 217 is below the incumbent's 249.

(Recount §static-check block re-inserted after the first commit: a shell variable was unset when it was generated. No numbers changed.)

## Compare (after reading preregistration/exp07.md, numbers.md, log.md)

Gate 0: prereg commit e31091b at 2026-09-23T01:43:59Z; started_utc 01:51:58Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 217, 33/93/91; heldout 0.915 | 217, 33/93/91; 0.915 | reproduces |
| cum solved 230/558/866 vs exp01 111/351/537 | 230/558/866 (found_targets 866, 1,003 distinct proofs, 150/150 re-verify); exp01 111/351/537 | reproduces. Pools differ (3,804 vs 1,500), so as fractions: 23 % vs 36 % |
| solved by length 674 L7 / 171 L8 / 21 L9 | 674 / 171 / 21 | reproduces |
| prediction (290, L9 ≥ 125; falsified if ≤ exp06 + 10 = 244 or L9 ≤ 110): MISSED | 217, L9 91 | reproduces; honest miss on both criteria |
| "more solved training targets is not the same as a higher dev score" | true of this run: most of the extra solves are L7 (674 vs 363), dev L7 fell to 33 (lowest so far) and L9 to 91 | fair as a statement about this run (n = 1) |
| bench_gen.py on GPU0 during exp07's scoring (02:01–02:04) | not visible in run files. The score is seed-fixed; contention can change only wall time, unless the sampler is not bit-deterministic under contention, which I did not test | accepted |

## Verdict

**REJECT.** Sound, not better: 217 vs 249, a −32 that is at the edge of the near-control spread (219–249). The curriculum spent its samples on L7 targets and did not bring the L9 frontier forward. The dev L7 decline across exp03–exp07 (33–41 vs 49) is worth watching, but it sits inside single-seed noise.
