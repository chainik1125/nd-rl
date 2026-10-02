# Review exp04 (breadth: all 3,804 targets, k 12, skip saturated; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp04_breadth_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:31:03Z, rl_secs 420.8, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 193, by_len {'7': 34, '8': 89, '9': 70}, heldout_greedy 0.913
- score.json: solved 193, by_len {'7': 34, '8': 89, '9': 70}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 193
- metrics.jsonl: 28 records {'sample': 2, 'train': 24, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2]; sum(samples) 91296; last targets_solved_cum 913
  - vs rl_stats: rounds 2 (=), total_samples 91296 (=), targets_solved_cum 913 (=), total_train_steps 1200 (max step 1201)
  - harness eval heldout_greedy 0.913 == summary: True
  - r1: n_prompts 3804 k 12 T 0.8 samples 45648 accept 0.0242 solved_round 233 cum 233 by_len {'7': 231, '8': 2, '9': 0} new 256
  - r2: n_prompts 3804 k 12 T 0.8 samples 45648 accept 0.1453 solved_round 912 cum 913 by_len {'7': 697, '8': 215, '9': 1} new 755
  - train records 24: loss first 0.0616 last 0.0537; n_records last 24036
- found_targets.jsonl: 1011 rows, 1011 raw-distinct, 1011 distinct by my normaliser; 913 targets solved, by L_true {7: 697, 8: 215, 9: 1}; names not in target pools: 0; thm field matches pool: 1011/1011
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt']

Per-round: r1 209 s (sample 129, ft 77, 1024 RL records); r2 210 s (sample 122, ft 84, 4036 RL records). Loop stopped at rl_secs 420.8: next-round estimate 210 s > 159 s left, so ~160 s of the budget went unused.

### rl.py static checks (/work/exec/runs/exp04_breadth_s0/source/rl.py, 258 lines, diff vs frozen seed: 15 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
-- file access (open / read_jsonl / paths[...] / os / subprocess / glob / torch.load):
65:def read_jsonl(fn, limit=None):
67:    with open(fn) as f:
195:    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
199:    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
233:        paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
236:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
248:    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
-- heldout usage:
16:  paths        dict: targets_extra, train_cap6, heldout_cap6, out_dir (files you may read / write under),
17:               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
23:premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
236:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
237:            metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te)
-- literal proof-like strings (': PR', 'QED', 'N1 '):
   none
-- lines longer than 300 chars:
```

**Hard constraints:** n_targets 100000 (i.e. all 3,804 of `targets`), k 12, skip_saturated (targets with ≥ max_per_thm proofs not resampled). No new reads. Pass. **Gate 1:** pass. GPU1; no overlap with a reviewer rescore.

Seed-1 rescore: not run. 193 is below the incumbent's 249.

## Compare (after reading preregistration/exp04.md, numbers.md, log.md)

Gate 0: prereg commit 006527d at 2026-09-23T01:29:52Z; started_utc 01:31:03Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 193, 34/89/70; heldout 0.913 | 193, 34/89/70; 0.913 | reproduces |
| 2 rounds; r1 209 s at 354 samples/s, ft 0.13 s/step; round 3 skipped (210 > 159 s left), 159 s unused | 2 rounds; r1 209 s, 355 samples/s, ft 77 s/600 = 0.128 s/step; rl_secs 420.8 → ~159 s unused | reproduces |
| 913 targets solved after 2 rounds vs 351 in exp01 | 913 (found_targets: 913, 1,011 distinct proofs, 150/150 re-verify) vs exp01 r2 cum 351 | reproduces; different target pools (3,804 vs 1,500), so the fraction solved is 24 % vs 23 % |
| 4,036 vs 1,568 RL records | 4,036 vs 1,568 (per_round mix_rl_records) | reproduces |
| prediction 275 (falsified ≤ 250): MISSED, "falsified as run" | 193 | reproduces; honest miss |
| "Lesson: the round count dominates" | n = 1 run and the variable is confounded (breadth and round count changed together). Consistent with it: L9 on the RL targets reached only 1 after 2 rounds (vs 28 in exp01 after 3), and dev L9 70 vs 109. That is consistent with the frontier needing rounds, not a measurement of it | **reword**: "suggests" / hypothesis; needs a matched-rounds comparison |
| "the seed's all-or-nothing last round wastes the tail" | exp04 159 s, exp03 76 s, exp01 60 s unused of 600 (rl_secs 540, 504, 421) | reproduces as a fact about these runs |

## Verdict

**REJECT.** Sound, but 193 < 249 (−56, beyond noise in the wrong direction). As run, the breadth idea is confounded with losing a round. The −56 is more likely due to 2 vs 3 rounds than to breadth itself, but that is not shown either. The next measurement: breadth at a matched round count, e.g. a smaller k or fewer ft steps so that 3 rounds fit.
