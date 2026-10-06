---
written_on: 2026-10-06
written_by: agent:claude
status: running
---

# Sprint kickoff: can the proof model select its own successful attempts and decide when to stop?

## Time

- Start: 2026-10-06 01:42 UTC (file `~/runs/proof-selection/SPRINT_START`).
- Deadline: 2026-10-06 11:42 UTC (10 h wall clock). Writing-only block: 10:40–11:42 UTC.
- Hourly check: `date -u` against the deadline; log elapsed/remaining in `research_log.md`.

## Context

Dan's best model, `state_env_october_r16` (9.56M ALiBi/NoPE GPT, proof-state interface `lean_staten`, 16 rounds of
expert iteration), solves 51/57/56 of textbook72 at pass@256 (T 0.8) but far fewer at pass@1: per-attempt validity on
textbook72 is 0.41–0.44, on holdout250 0.82–0.83. The gap between pass@1 and pass@k is the test-time-scaling headroom.
Today the *verifier* (Lean) closes it. The question is whether the *model* can.

Prior-evidence table from Dan's own k=256 pools (`prior_pool_headroom.txt`, read-only analysis of
`hf://buckets/dan-pandori/nd-rl/rl-continue/artifacts/rc/eval/`): the environment's "attempt finished" flag is a strong
recogniser on easy theorems (P(valid | finished) 0.99 when pass rate > 0.9) and a weak one on hard theorems
(0.00–0.30 when pass rate < 0.25; 0.53 on textbook theorems with ≥ 13 reference lines). So recognition has real
headroom beyond "did it finish", exactly on the theorems where extra attempts matter.

The design reference is `proposals/learned-compute-allocation/README.md` (experiment 1: select among eight;
experiments 2–3: stopping with/without an attempt cost). This sprint executes experiment 1 with a per-candidate
scorer and a *frozen-generator* version of experiment 2 (threshold stopping on learned and intrinsic scores). It does
**not** train `ACCEPT`/`CONTINUE` tokens by RL (experiments 2–3 as written) and does **not** do listwise 8-proof
selection in one context (8 proofs ≈ 2k tokens > the checkpoint's `max_len` 1024). Both are stated as scope-outs.

## Goal

Measure, on held-out theorems with the r16 generator frozen, how much of the gap between random single-attempt
accuracy and oracle pass@k a model-side selector recovers, and how an adaptive stopping rule driven by the same
score trades accuracy against mean attempts, compared with fixed-k and "stop at first finished proof".

## Hypotheses

| | Hypothesis | Prediction | Falsifier |
|---|---|---|---|
| H1 | A verifier head fine-tuned from r16 on (theorem, attempt) → P(valid) recognises valid proofs better than completion + likelihood | Selected-proof accuracy at k=8 on textbook72 exceeds the best non-oracle baseline, paired over theorems, in 3/3 selector seeds | No improvement, or improvement only on pools where completion alone already picks a valid proof |
| H2 | Adaptive stopping on the learned score beats fixed-k at matched mean attempts | At every mean attempt count in [1, 8], the adaptive curve is above the fixed-k curve with the same selector | Curves coincide, or "stop at first finished" matches the learned rule |
| H3 | Intrinsic signals (teacher-forced likelihood, entropy) carry less of the recognition signal than the trained head | AUROC(valid) of the head > AUROC of likelihood on finished candidates | Likelihood alone reaches the head's AUROC |

## Data, splits, protocol

- Generator: `la_T1_best12_s0_r16.pt` (primary), T 0.8, `max_action 512`, `max_steps 96` (Dan's read-out settings).
  Seeds s1/s2 only as replication if time allows.
- Theorem sets (all from Dan's `nd-takehome` clone at `~/runs/proof-selection/nd-takehome`, branch `dan` @ 4dcd0e4f):
  `data/ladder/rl_targets.jsonl` (4,495, the EI training targets), `data/bs/dev1108.jsonl`, `data/ladder/transfer_long_rr600.jsonl`
  (long proofs), `data/bs/holdout250.jsonl`, `data/bs/textbook72.jsonl` (**eval only, never tune**).
- Selector train: rl_targets + part of dev1108 + part of rr600. Selector dev: the other parts. **Test (frozen before any
  test read): holdout250 and textbook72.** Theorem-level, not attempt-level. Report overlap by exact prompt and by
  Dan's renaming-class key where available; a structural-family overlap audit is a stated limitation.
- Lean alone decides validity; verdicts live in a separate labels file; selectors see only raw attempt text/tokens.
- Uncertainty: paired theorem-level bootstrap (resample theorems with all their attempts) and 3 selector seeds.

## Team

- Worker A (Opus 5.5): collector + pools + baselines + verifier head. GPUs 5, 6.
- Worker B (Opus 5.5): teacher-forced scorer + stopping simulator + accuracy-vs-attempts figures. GPU 7.
- Reviewer (Codex astra): leakage/design audit of the collector and schema (~02:45), mid-results audit (~06:30),
  write-up red team (~10:40).
- Root (this session): coordination, research log, figures, `summary.md`.

## Compute and safety

- Local H200s 5–7 only (`~/GPU_CLAIMS.md`); GPU 0 belongs to another user; GPUs 1–4 are used by a parallel Codex sprint
  on the same task in the main working tree (`/data/users/dmitry/ndrl`, branch `dmitry/main`). This sprint lives in the
  worktree `/data/users/dmitry/ndrl-selection-sprint` on branch `dmitry/selection-sprint`. Never read, edit, or
  stop anything belonging to the other sprint.
- No paid compute. No pushes. Heavy artifacts in `~/runs/proof-selection/` (proposed durable home: the `ndrl` HF org).
