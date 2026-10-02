# Review exp09 (custom KV-cached sampler with finished-row compaction; seed knobs otherwise; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp09_fastgen_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:04:13Z, rl_secs 494.6, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 235, by_len {'7': 34, '8': 91, '9': 110}, heldout_greedy 0.924
- score.json: solved 235, by_len {'7': 34, '8': 91, '9': 110}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 235
- metrics.jsonl: 41 records {'sample': 3, 'train': 36, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3]; sum(samples) 144000; last targets_solved_cum 530
  - vs rl_stats: rounds 3 (=), total_samples 144000 (=), targets_solved_cum 530 (=), total_train_steps 1800 (max step 1801)
  - harness eval heldout_greedy 0.924 == summary: True
  - r1: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.023 solved_round 112 cum 112 by_len {'7': 110, '8': 2, '9': 0} new 131
  - r2: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.1217 solved_round 348 cum 348 by_len {'7': 285, '8': 63, '9': 0} new 284
  - r3: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.261 solved_round 530 cum 530 by_len {'7': 364, '8': 147, '9': 19} new 232
  - train records 36: loss first 0.0678 last 0.0520; n_records last 22520
- found_targets.jsonl: 647 rows, 647 raw-distinct, 647 distinct by my normaliser; 530 targets solved, by L_true {7: 364, 8: 147, 9: 19}; names not in target pools: 0; thm field matches pool: 647/647
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt']

Per-round secs / sample / ft: 162/95/64, 164/91/69, 167/92/71. Sampling 507/525/524 samples/s (vs 422–466 with lib.sample.generate on GPU0), about +17 %. Still 3 rounds; rl_secs 494.6, so ~85 s was unused (the next round's estimate was 167 s).

**Sampler fidelity check (own):** the per-round accept rates are 2.30 / 12.17 / 26.10 % vs the seed control exp01's 2.46 / 11.79 / 26.49 % and exp05's 2.29 / 10.95 / 25.74 %; qed_rate 1.0. No sign that the custom sampler changes the sample distribution. Every training proof is still nd_verify-accepted (150/150 re-verified), and the dev score comes from the frozen scorer, so the sampler cannot inflate the score.

### rl.py static checks (/work/exec/runs/exp09_fastgen_s0/source/rl.py, 327 lines, diff vs frozen seed: 78 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
```

**Hard constraints:** adds fast_generate (imports model.rope_cache from frozen lib, read-only use). No new reads. Pass. **Gate 1:** pass.

Seed-1 rescore: not run. 235 is below the incumbent's 249.

## Compare (after reading preregistration/exp09.md, numbers.md, log.md)

Gate 0: prereg commit e66e60a at 2026-09-23T02:04:12Z; started_utc 02:04:13Z. Precedes by 1 s. **Pass.** (Pattern: the executor commits each prereg 1–2 s before launch. That is compliant, but it leaves no room for a pre-launch gate-0 check.)

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 235, 34/91/110; heldout 0.924 | 235, 34/91/110; 0.924 | reproduces |
| round-1 sampling 95 s vs 109 s; 91–95 s per round | 95 / 91 / 92 s; exp01 r1 109 s | reproduces |
| "~1.15x" / "~1.2x faster in real runs" | 507–525 vs 422–466 samples/s on GPU0: 1.13–1.24x | reproduces |
| bench 1.68x (195 → 327) "same distribution" (prereg) | bench files not in a run dir, so not checked. Independent evidence from the run: accept rates match the controls (see recount) | bench not derivable; the "same distribution" conclusion is supported by run data |
| still 3 rounds; 85 s unused | rl_secs 494.6 → ~85 s | reproduces |
| prediction (≤ 75 s r1, 4 rounds, 275; falsified > 90 s or ≤ 250): MISSED on both | 95 s, 3 rounds, 235 | reproduces; honest miss |
| exp10 at ~490 samples/s on GPU1 (log) | exp10 not finished; checked in review_exp10 | pending |

## Verdict

**REJECT.** Sound, not better (235 vs 249). The sampler works (about 1.2x faster, same accept rates) but did not buy a round at seed knobs. It is plumbing whose value depends on the round structure. Since training, not sampling, is now ~40 % of each round, the stop rule decides whether the saved seconds turn into anything.
