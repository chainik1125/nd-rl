---
written_on: 2026-09-29
written_by: agent:claude
---

# Where the combined pipeline's gains come from: a 2×2×2 factorial

Robbie (2026-09-29), for next week's blog post: score the most naive pipeline, naive + each best
piece, the best pipeline, and best minus each piece, on pass@k for held-out proofs ≥ 7 lines and
on the textbook set. That is the full 2³ factorial of three binary factors, 8 cells × 3 seeds.
Branch `factorial-post`, checkout `nd-rl-7`, off `combined-pipeline` @ `3f484b0`.

| factor | naive | best |
| --- | --- | --- |
| tok: data format | `abs` token format (token era) | Dan's `lean_seq` Lean format |
| pre: pretraining | autoresearch 012 baseline (`pretrain_naive.py`: fork GPT 4×256, RoPE, AdamW) | 186-claude incumbent (`pretrain_best.py`: 6×384 Peri-LN, ALiBi/NoPE, Muon, MTP) |
| rl: RL recipe | frozen harness expert iteration (4 rounds × 1,500 × 32) | c005: Leon's EI + verifier-guided sampling, 1,800 s deadline |

Everything else is one protocol for all 24 runs: fork `51604b3`, cap-6 `train.jsonl` (the same
records in both formats), 300 s pretraining, the same seeded 1,500 RL targets, 2,400 s pipeline
budget (the pod alarm is 4,200 s so pass@k sampling fits after it), one A40 or RTX A6000 per run.
Reward/judge: Lean ∧ nd_verify for lean_seq, nd_verify alone for abs (the fork gates only Lean
text; the two agree on > 99.9% of samples). Decisions (Robbie, 2026-09-29): 3 seeds fresh, 012 as
naive pretraining, 186's recipe with abs tokens for the abs × best cells, pass@k on a 250-problem
holdout subset up to k = 256.

## Measurements per run

- `metric`: the autoresearch dev metric (dev half of transfer, 1,108 theorems, 64 samples at
  T 0.8, solved at L_true ≥ 7) — the bar chart.
- `eval.passk.holdout250`: per-problem success counts from 256 samples on the 250 holdout problems
  nd-rl-2's `score_holdout.py --pool holdout250` uses (all L_true ≥ 7, never used for any
  selection) — pass@k curves, unbiased estimator.
- `eval.passk.textbook72`: the same on the 72 textbook problems (58 dev + 14 train; copied with
  its MANIFEST, hashes verified).
- `eval.pipeline_secs`: harness clock at the end of the dev eval, against the 2,400 s budget.

## Checks before launch (CPU)

All 8 cells ran `harness.py --smoke` to a summary with pass@k output. abs guided sampling on the
token-era 012 checkpoint: 0 false rejects on 1,500 training proofs; 77 vs 65 of 192 samples
accepted (guided vs unguided). Leon's abs index cap (starts ≤ N41, 256/256 renumbered proofs
still verify). The lean_seq guided checks are the c004 ones.

## Prediction (written before any pod ran)

3-seed dev metric means, rough: abs-naive-ei 120-220; abs-best-ei 200-400; lean-naive-ei 350-550;
lean-best-ei 650-720 (c001 687, 186 680); lean-best-leon 870-940 (c005 904); the RL swap adds
the most in every row, the Lean format second, the pretraining swap least; the RL gain is larger
on better starting models (a positive pre × rl interaction). Textbook72 pass@256 moves far less
than the transfer metric in every cell (the 184 pretraining runs barely moved it).

## Results (2026-09-29 01:00–01:58 UTC; all 24 runs ok)

Dev metric, 3-seed means (seeds): abs-naive-ei 142.3 (131/162/134); lean-naive-ei 192.7
(209/195/174); abs-best-ei 456.7 (493/436/441); abs-naive-leon 443.7 (409/465/457);
lean-best-leon **904.0** (916/895/901, reproducing c005's 904.3); abs-best-leon 782.3
(815/777/755); lean-naive-leon 413.7 (434/420/387); lean-best-ei 681.7 (694/677/674; c001 687.3).
Pipeline seconds ≤ 2,374 in every run. Main effects: pre +408, rl +268, tok +92; interactions
tok×pre +82, tok×rl −46, pre×rl +6, three-way −6.

Against the prediction: 3 of the 5 cells given ranges landed in them (abs-naive-ei,
lean-best-ei, lean-best-leon); abs-best-ei (457) was above its 200-400 and lean-naive-ei (193)
below its 350-550. The ordering was wrong: pretraining, not RL, has the largest effect, and the Lean format the smallest;
pre × rl is ~0, not positive. Textbook72 did move: 12.5% → 43.5% pass@256, mostly from RL on top
of the best pretraining (27.8% → 43.5%). Leon's RL lowers held-out pass@1 (43.8% → 30.1% on the
best pretraining) while raising pass@64 (58.9% → 79.1%).

The Mac crashed at 01:55 UTC with 6 runs finished on their pods but not pulled; their artifacts
were pulled by hand at 02:12-02:14 UTC and the pods terminated (all 24 summaries are complete).
Post: [experiment summary / blog draft](../../../../experiment-summaries/2026-09-28-combined-model/README.md).
