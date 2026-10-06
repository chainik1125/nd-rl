# STATUS

Updated: 2026-09-15 07:52 UTC

## Plan
1. 03:25–04:30  gen.py (forward random generator with lazy premises + goal completion for ORE branches, dependency pruning, verify every proof), tokenizer.py (two modes: `rel` relative refs, `abs` absolute refs with random start offset), model.py (4L d256 RoPE decoder), train.py, sample.py (batched KV-cache sampling), prove.py. Unit tests vs verifier.
2. 04:30–06:00  Stage 1 on pod: 100k+ cap-6 proofs, held-out split by renaming class, train both tokenizer modes, pick by held-out greedy AND by frozen-model pass@k on 7–9 line transfer theorems (the length-generalisation question is the point). Eval by length with Wilson CIs.
3. 06:00–09:30  Stage 2 expert iteration (k=32–64, T=0.8) with frozen control, per-round stats + plots.
4. 09:30–13:00  Stage 2b (second seed or relabelling arm), validation-36 analysis, figures.
5. 13:00–14:30  Test set once (create artifacts/TEST_RUN_DONE first), writeup.md, numbers.md, log.md, README reproduction section.
6. 15:00        Final pull, commit, push. DONE.

## Done
- Read README.md, spec.md, verifier, validation set, pod check (A40, torch 2.8+cu128).
- gen.py (0 verifier rejects), tokenizer.py (rel/abs, round-trip 0 mismatches), model.py (3,210,240 params), train.py, sample.py (KV cache), eval_set.py, prune.py, expert_iter.py, plots.py, prove.py.
- data/: train 154,990 (31k per length 2-6), heldout 5,000 (1k per length), rl_targets 3,000 (300 per length 7-16), transfer 1,600. Disjoint by atom-renaming class; validation-36 classes removed (10 hits in cap-6 raw, 5 in long raw). Committed + pushed (train.jsonl.gz).

- Stage 1 trained + evaluated both modes (see log.md 03:55). Chosen: abs. ckpts/stage1_abs.pt, ckpts/stage1_rel.pt.
- Long pools regenerated in strict mode (v2); v1 kept as data/*_v1.jsonl.

- writeup.md sections 2 and 3.1 written; numbers.md Stage-1 part; README reproduction draft; figures/data_stats.png.
- EI round 2 (after one fine-tune): transfer pass@32 62.0% (r1 47.0%), greedy 45.1% (32.2%), held-out 94.5%; written-8 proofs 6 -> 98 per round.

## Running on pod
- nothing (all results pulled: artifacts/, ckpts/ (all per-round checkpoints local, only stage1_abs.pt + final.pt in git), data/)

## Done
- Stage 1: generator, splits, three tokenizer variants, evals. Stage 2: EI vs frozen (2 seeds), EI-long, 16-round continuation, pass@128 budget check. Stage 3: validation-36 per round, test set run ONCE at 07:38 UTC (artifacts/TEST_RUN_DONE).
- Deliverables: writeup.md (exec summary marked DRAFT for the human), numbers.md, log.md, README.md reproduction, figures/, data/, prove.py, requirements.txt, ckpts/stage1_abs.pt, ckpts/final.pt. All pushed to github.com/dan-pandori/nd-takehome main.

## Headline
- Robust frontier L = 9 (transfer) / 10 (RL targets) vs P = 8 for the frozen model with the same attempts; L - P = 1 / 2.
- Transfer: EI 86.1% cumulative / 66.8% greedy vs frozen 57.3% / 32.2% (n=1638, 2 seeds agree to 0.5pp). Test: short 73.0% -> 73.0%, long 9.8% -> 14.5%.

DONE 2026-09-15 07:52 UTC
