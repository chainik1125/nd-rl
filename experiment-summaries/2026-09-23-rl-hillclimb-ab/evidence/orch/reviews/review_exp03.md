# Review exp03 (fast iteration: k16, 300 ft steps, retain 8000, up to 12 rounds; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp03_fastiter_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:29:19Z, rl_secs 503.7, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 238, by_len {'7': 37, '8': 90, '9': 111}, heldout_greedy 0.909
- score.json: solved 238, by_len {'7': 37, '8': 90, '9': 111}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 238
- metrics.jsonl: 30 records {'sample': 4, 'train': 24, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4]; sum(samples) 96000; last targets_solved_cum 571
  - vs rl_stats: rounds 4 (=), total_samples 96000 (=), targets_solved_cum 571 (=), total_train_steps 1200 (max step 1201)
  - harness eval heldout_greedy 0.909 == summary: True
  - r1: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.023 solved_round 100 cum 100 by_len {'7': 99, '8': 1, '9': 0} new 118
  - r2: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.1178 solved_round 316 cum 317 by_len {'7': 255, '8': 62, '9': 0} new 246
  - r3: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.2483 solved_round 498 cum 498 by_len {'7': 342, '8': 141, '9': 15} new 214
  - r4: n_prompts 1500 k 16 T 0.8 samples 24000 accept 0.3162 solved_round 571 cum 571 by_len {'7': 374, '8': 164, '9': 33} new 91
  - train records 24: loss first 0.0689 last 0.0443; n_records last 10652
- found_targets.jsonl: 669 rows, 669 raw-distinct, 669 distinct by my normaliser; 571 targets solved, by L_true {7: 374, 8: 164, 9: 33}; names not in target pools: 0; thm field matches pool: 669/669
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt']

Per-round (summary per_round): secs / sample_secs / ft_secs / ft_steps / RL records
```
r1 129 85 42 300 472   (sampling 284 samples/s)
r2 175 122 51 300 1452 (196 samples/s)
r3 102 62 38 300 2300  (386 samples/s)
r4 96 55 39 300 2652   (436 samples/s)
```

### rl.py static checks (/work/exec/runs/exp03_fastiter_s0/source/rl.py, 255 lines, diff vs frozen seed: 8 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
-- file access (open / read_jsonl / paths[...] / os / subprocess / glob / torch.load):
64:def read_jsonl(fn, limit=None):
66:    with open(fn) as f:
193:    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
197:    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
230:        paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
233:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
245:    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
-- heldout usage:
16:  paths        dict: targets_extra, train_cap6, heldout_cap6, out_dir (files you may read / write under),
17:               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
23:premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
233:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
234:            metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te)
-- literal proof-like strings (': PR', 'QED', 'N1 '):
   none
-- lines longer than 300 chars:
```

**Hard constraints:** knob changes only (rounds 12, k 16, retain 8000, ft_steps 300). Pass. **Gate 1:** pass.

**Contamination by the reviewer.** My exp01 seed-1 rescore ran on GPU0 from about 01:30 to 01:34 UTC. That is elapsed ≈ 40–300 s of this run, i.e. rounds 1–2. Round-2 sampling ran at 196 samples/s against 386–436 in rounds 3–4, and the monitor measured about a 35 % slowdown. The loop stopped after 4 rounds with 76 s left and a 96 s round estimate. An unshared GPU would likely have given it a 5th round (or more; roughly 60–90 s were lost). **The score 238 is probably biased low and cannot fairly be compared with the others.** That is my fault, not the executor's.

Seed-1 rescore: not run. 238 is below the incumbent's 249.

## Compare (after reading preregistration/exp03.md, numbers.md, log.md)

Gate 0: prereg commit 47fd366 at 2026-09-23T01:21:57Z; started_utc 01:29:19Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 238, 37/90/111; heldout 0.909 | 238, 37/90/111; 0.909 | reproduces |
| only 4 rounds completed, not ~6 | 4 sample rounds; rl_secs 503.7 (stopped early by the round-cost estimate) | reproduces |
| rounds 1–2 sampled at 282 and 197 samples/s; rounds 3–4 at ~400/s | 284 / 196 / 386 / 436 | reproduces (±2) |
| cause: reviewer re-score on GPU0 at the same time (279 s) | my rescore output: secs 279 on GPU0, 01:30–01:34 | reproduces; confirmed from my side |
| prediction 300 (falsified if ≤ 250): MISSED, "falsified as run" | 238 ≤ 250 | reproduces. The executor's own qualifier holds: the intended condition (~6 rounds) was not reached, so this does not falsify the idea |
| (prereg) "total samples per target about equal (~96)" | 64 per target reached (4 × 16) vs 96 in exp01 (3 × 32) | the run used ⅔ of the control's samples. Another reason it is not a fair comparison |
| (prereg) "L9 gains most" | L9 111 vs incumbent 109 (+2, noise); L7 37 vs 49 | not supported, confounded |
| (log) round-1 fine-tune 0.14 s/step vs 0.095 | ft r1 42 s / 300 = 0.14 s/step | reproduces |

Of note, though still inside noise: with ⅔ of the samples and a slowed start it still reached 571 cumulative targets on the RL set (vs 537), and 238 on dev.

## Verdict

**REJECT** as incumbent (238 < 249, a single seed). The run is **confounded by the reviewer's re-score**, so this is not evidence against faster iteration. A clean rerun at seed 0 on an unshared GPU is justified. It should be read against the 219–249 control spread.
