# Review exp13 (textbook-source exploration: all 830 textbook-source targets from targets + targets_extra, T 1.0 from round 2; on exp10 code; seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp13_textbook_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T02:26:59Z, rl_secs 562.8, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 324, by_len {'7': 109, '8': 86, '9': 129}, heldout_greedy 0.896
- score.json: solved 324, by_len {'7': 109, '8': 86, '9': 129}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 324
- metrics.jsonl: 36 records {'sample': 5, 'train': 29, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 199836; last targets_solved_cum 774
  - vs rl_stats: rounds 5 (=), total_samples 199836 (=), targets_solved_cum 774 (=), total_train_steps 1443 (max step 1444)
  - harness eval heldout_greedy 0.896 == summary: True
  - r1: n_prompts 1321 k 16 T 0.8 samples 21136 accept 0.0253 solved_round 97 cum 97 by_len {'7': 95, '8': 2, '9': 0} new 109
  - r2: n_prompts 1320 k 16 T 0.8 samples 47680 accept 0.0621 solved_round 329 cum 330 by_len {'7': 259, '8': 71, '9': 0} new 259
  - r3: n_prompts 1319 k 16 T 0.8 samples 47664 accept 0.1311 solved_round 522 cum 524 by_len {'7': 352, '8': 149, '9': 23} new 221
  - r4: n_prompts 1317 k 16 T 0.8 samples 47632 accept 0.1987 solved_round 677 cum 681 by_len {'7': 462, '8': 181, '9': 38} new 169
  - r5: n_prompts 1317 k 12 T 0.8 samples 35724 accept 0.2619 solved_round 770 cum 774 by_len {'7': 539, '8': 188, '9': 47} new 99
  - train records 29: loss first 0.0695 last 0.0416; n_records last 11428
- found_targets.jsonl: 857 rows, 857 raw-distinct, 857 distinct by my normaliser; 774 targets solved, by L_true {7: 539, 8: 188, 9: 47}; names not in target pools: 0; thm field matches pool: 857/857
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']

Per-round secs / sample / ft / steps / RL records / mix: 82/47/33/300/436/8,436; 124/85/36/300/1,472/9,472; 126/85/38/300/2,356/10,356; 130/87/40/300/3,032/11,032; r5 shrunk (k 12) 100/65/32/243/3,428/11,428. 5 rounds, 199,836 samples (the metrics n_prompts counts only the 1,321 gen targets; samples include the textbook pass).

Solved training targets by source × length (own count from found_targets.jsonl joined to the pool files): gen L7 394, gen L8 188, gen L9 47, **textbook L7 145** (90 of them from targets_extra). No textbook L8/L9 solved.

Dev per length: **L7 109** (previous max 71), L8 86, L9 129 (previous max 126). heldout_greedy 0.896.

### rl.py static checks (/work/exec/runs/exp13_textbook_s0/source/rl.py, 353 lines, diff vs frozen seed: 130 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
-- file access (open / read_jsonl / paths[...] / os / subprocess / glob / torch.load):
70:def read_jsonl(fn, limit=None):
72:    with open(fn) as f:
271:    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
277:    tb = [t for t in pool if t['source'] == 'textbook'] + [t for t in read_jsonl(paths['targets_extra']) if t['source'] == 'textbook']
278:    targets = targets + (tb[:4] if paths.get('smoke') else tb)
280:    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
327:        paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
330:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
343:    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
-- heldout usage:
16:  paths        dict: targets_extra, train_cap6, heldout_cap6, out_dir (files you may read / write under),
17:               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
23:premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
330:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
331:            metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te)
-- literal proof-like strings (': PR', 'QED', 'N1 '):
   none
-- lines longer than 300 chars:
```

**Hard constraints (read by hand):** the new reads are `paths['targets_extra']`, filtered to `source == 'textbook'` (allowed file; the targets carry no proofs). The textbook targets are sampled by the run's own model; only nd_verify-accepted proofs enter found (same judge_samples path; 150/150 of the found proofs re-verify, all names resolve to the two allowed pool files). No proof strings in code, no sealed/other paths. SPEC: "All four pools and the sealed sets are pairwise disjoint by atom-renaming class", so training on pool textbook targets is not training on dev theorems. I cannot check that disjointness myself (the sealed set is invisible to me by rule). **Pass.** **Gate 1:** pass. GPU0 02:27–02:39; no reviewer rescore overlapped (none ran: both GPUs listed busy).

Seed-1 rescore: queued first.

## Compare (after reading preregistration/exp13.md, numbers.md, log.md)

Gate 0: prereg commit e84e389 at 2026-09-23T02:19:38Z; started_utc 02:26:59Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 324, 109/86/129; heldout 0.896 | 324, 109/86/129; 0.896 | reproduces |
| textbook solved cum 1/16/84/145 by round (rounds 2–5) | first-solve round from found_targets.jsonl: 0/1/16/84/145 for rounds 1–5 | reproduces |
| (prereg) exp08 solved 0 of 179 textbook targets | own reconstruction of exp08's target sample (Random(0).sample of the pool, 1,500): 179 textbook, 0 solved. exp10 0, exp01 2, exp12 5 | reproduces |
| (prereg) "19 classic schemas" | 19 distinct schema values among textbook-source targets | reproduces |
| predictions (textbook 5, range 0–40; dev 250, range 225–285): MISSED on the upside | 145; 324 | reproduces. Honest miss; first upside miss of the session |
| "belief that textbook schemas are out of reach in 600 s is falsified" | 145 textbook targets solved in 600 s | supported |
| "once a few are found at T 1.0, expert iteration spreads them across instances of the same schema" | solved textbook targets come from only **2 of 19 schemas**: contraposition 108, export 37 | supported for 2 schemas; reword to "within 2 of 19 schemas". 17 schemas are still at 0 |
| dev L7 109 vs 33–71 earlier | 109 vs 33–71 | reproduces. Consistent with the sealed L7 slice being rich in textbook-like schemas, which would also explain L7 < L8 in every earlier run. Inference only: the dev composition is invisible |
| heldout cost | 0.896 vs incumbent 0.920 (−0.024) | small; within the range of the other retain-8k runs (0.902–0.910) |

## Verdict

**ADOPT** — new incumbent. Gates 0–2 pass and the hard constraints are clean. +75 over the incumbent (249 → 324) is about 15× the measured sampling-seed noise (5, exp01) and more than twice the full near-control training spread (219–249). The gain is concentrated where the mechanism predicts (L7, where textbook targets were solved) and L9 also rose (+20).

Pending, to be appended: (1) reviewer seed-1 rescore (queued; both GPUs have been busy continuously); (2) the executor's training-seed replicate exp13_textbook_s1. **Revisit condition:** if the seed-1 rescore is below 264 (incumbent + 15), or the s1 replicate lands inside the old 219–255 band, I will downgrade this verdict.

## Addendum (02:55 UTC): training-seed replicate exp13_textbook_s1

269 (67/85/117), heldout 0.910, 13 textbook targets solved (vs 145). The revisit condition is not met (269 is outside the 219–255 band, above it). **ADOPT stands** for the checkpoint `/work/exec/runs/exp13_textbook_s0/final.pt`. Wording for any summary: the recipe beats the control on two seeds (+20 and +75); the size of the gain depends on whether and when textbook schemas ignite. Selecting the best of 16 runs on dev also inflates 324 somewhat (winner's curse), and the sealed L10+/textbook evaluation will tell.

## Seed-1 rescore (04:08–04:11 UTC, GPU1, after the executor released it)

**328/1610** (L7 106, L8 90, L9 132) vs seed 0's 324 (109/86/129). Sampling-seed spread +4, like exp01's +5. The incumbent's score is not a sampling fluke.
