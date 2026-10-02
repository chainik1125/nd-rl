# Review exp02 (hindsight relabelling, seed 0)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp02_relabel_s0` (tools/recount.py)

- status `ok`, seed 0, budget 600, gpu `1:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T01:18:51Z, rl_secs 579.5, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 219, by_len {'7': 38, '8': 89, '9': 92}, heldout_greedy 0.916
- score.json: solved 219, by_len {'7': 38, '8': 89, '9': 92}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 219
- metrics.jsonl: 41 records {'sample': 3, 'train': 36, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3]; sum(samples) 144000; last targets_solved_cum 527
  - vs rl_stats: rounds 3 (=), total_samples 144000 (=), targets_solved_cum 527 (=), total_train_steps 1800 (max step 1801)
  - harness eval heldout_greedy 0.916 == summary: True
  - r1: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.023 solved_round 114 cum 114 by_len {'7': 112, '8': 2, '9': 0} new 132
  - r2: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.1229 solved_round 347 cum 347 by_len {'7': 286, '8': 61, '9': 0} new 274
  - r3: n_prompts 1500 k 32 T 0.8 samples 48000 accept 0.2662 solved_round 527 cum 527 by_len {'7': 372, '8': 138, '9': 17} new 233
  - train records 36: loss first 0.0673 last 0.0519; n_records last 22506
- found_targets.jsonl: 639 rows, 639 raw-distinct, 639 distinct by my normaliser; 527 targets solved, by L_true {7: 372, 8: 138, 9: 17}; names not in target pools: 0; thm field matches pool: 639/639
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- found_relabelled.jsonl: 11 rows, 11 distinct prompts; premises equal to a target/extra target's premises: 11/11; re-verified 11/11; written-length hist {7: 8, 8: 3}
- other files: ['found_relabelled.jsonl', 'round_01.pt', 'round_02.pt', 'round_03.pt']

### rl.py static checks (/work/exec/runs/exp02_relabel_s0/source/rl.py, 298 lines, diff vs frozen seed: 55 changed lines)
```
-- forbidden paths / sealed / other workspaces:
   none
-- file access (open / read_jsonl / paths[...] / os / subprocess / glob / torch.load):
68:def read_jsonl(fn, limit=None):
70:    with open(fn) as f:
231:    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
235:    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
269:        paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
272:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
284:    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
289:        with open(f'{paths["out_dir"]}/found_relabelled.jsonl', 'w') as f:
-- heldout usage:
16:  paths        dict: targets_extra, train_cap6, heldout_cap6, out_dir (files you may read / write under),
17:               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
23:premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
272:            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
273:            metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te)
-- literal proof-like strings (': PR', 'QED', 'N1 '):
   none
-- lines longer than 300 chars:
```

**Hard constraints:** diff vs seed adds relabel(): applied only to samples that fail the prompted sequent, rebuilds the prompt from the target's own premises + the proof's last-line formula, and keeps it only if nd_verify accepts (verified by-product, allowed by SPEC rule 3). No new file reads. Pass.

**Gate 1:** pass (all checks above).

Seed-1 rescore: not run. 219 is below the incumbent's 249, so this is not a candidate.

## Compare (after reading preregistration/exp02.md, numbers.md, log.md)

Gate 0: prereg commit c68033d at 2026-09-23T01:18:49Z; started_utc 01:18:51Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 219, 38/89/92 (numbers.md) | 219, 38/89/92 | reproduces |
| heldout_greedy 0.916 | 0.916 | reproduces |
| only 11 relabelled by-products ≥ 7 lines, 3/10/11 cum (log.md) | relabelled_total 3/10/11; found_relabelled.jsonl 11 rows, 11/11 re-verify, premises all from target pools, lengths 7×8, 8×3 | reproduces |
| prediction 275 (falsified if ≤ 238): MISSED (log.md) | 219 ≤ 238: missed. Reported honestly as a miss | reproduces |
| "in effect a second control" (log.md) | relabelled records in the mix: 6 / 20 / 22 of 20,534 / 21,612 / 22,506 (≤ 0.1 %). Per-round wall 184/193/201 s vs exp01 169/181/188 s (+15/+12/+13 s; judge +1 s, most of the extra is fine-tune 64/75/82 s vs 57/66/69 s). Code path and RNG stream differ from the seed, so it is a near-control, not an exact one | reproduces, with that caveat |
| "~20 s more judge/ft time" | +39.5 s rl_secs overall (579.5 vs 540.0 s), mostly fine-tune time, not judging | differs: about 40 s, not 20 s |
| "219 vs 249 vs 238 gives a sense of seed noise (~±15)" | three near-control runs span 219–249 (30 theorems); SD ≈ 15. Sampling-seed noise alone (exp01 rescore) was 5. n = 3, and one of the three is a variant | acceptable as a rough estimate; "±15" is a spread from 3 runs, not an interval |

Note for later verdicts: if training-seed spread really is about 30 theorems (range), a single run has to beat the incumbent by more than about 30 before it is a clear gain. The SPEC's 15 is a lower bound.

## Verdict

**REJECT.** Sound (gates 0–2 pass, no constraint issue) but not better: 219 vs incumbent 249. It is also not a test of relabelling: with 11 by-products (≤ 0.1 % of the mix) the intervention never took effect. The pre-registered falsification ("relabelled data is not the bottleneck signal") does not follow from this run. What it shows is that with this sampler, failed samples almost never form complete valid proofs of anything, because they fail mid-proof on rule checks (qed_rate 1.0). To settle it, the next measurement would count by-products under a looser rule (relabel to any intermediate line, not just the last line) at round 3. Only if that count is large would relabelling be worth another run.
