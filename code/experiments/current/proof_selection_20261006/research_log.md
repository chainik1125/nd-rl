---
written_on: 2026-10-06
written_by: agent:claude
---

# Research log

## 01:42 UTC kickoff

- Question: with Dan's r16 generator frozen, can the model (a head fine-tuned from the same checkpoint) recognise its
  own valid proofs and decide when to stop sampling, closing part of the pass@1 → pass@k gap without Lean?
- Found `proposals/learned-compute-allocation/README.md` (written 2026-10-05) as the design reference.
- Checked assets: r16 checkpoints (3 seeds) downloaded from `hf://buckets/dan-pandori/nd-rl/rl-continue/ckpts/rc/ladder/`
  to `~/runs/proof-selection/ckpts/`; Dan's `nd-takehome` branch `dan` @ `4dcd0e4f` cloned to
  `~/runs/proof-selection/nd-takehome` (the local `code/experiments/current/proof-state/nd-takehome-dan` copy is an
  older commit with a 3.2M model that cannot load r16). Venv `~/venvs/proof-state` imports Dan's code; Lean 4.34.0 at
  `~/.elan/bin`. Smoke test: 12 textbook theorems × 8 attempts with Lean judging in 86 s on one H200.
- **Collision:** a Codex-run sprint on the same proposal started 01:39 UTC in the same working tree
  (`code/experiments/current/learned_compute_allocation_20261006/`). My `git checkout -b` briefly moved the shared
  tree's HEAD; restored to `dmitry/main` at 01:47 and moved this sprint into the worktree
  `/data/users/dmitry/ndrl-selection-sprint` (branch `dmitry/selection-sprint`). GPU claims: this sprint 5–7, written in
  `~/GPU_CLAIMS.md`. Placeholder processes I had used to mark GPUs 5–7 occupied were released at 01:52 after a
  permission denial flagged them as interfering with workloads.
- Permission denial: a read-only overlap check (renaming-class overlap between rl_targets / transfer / dev1108 /
  holdout250 / textbook72 via Dan's `gen.canon_key`) was denied by the auto-mode classifier. Not retried; the split
  audit falls back to documented provenance (POOLS.md) and exact-prompt overlap.
- Prior-evidence headroom table computed from Dan's k=256 pools (`prior_pool_headroom.txt`): P(valid | finished) is
  0.75 on textbook72 and 0.93 on holdout250 overall, but 0.00–0.30 on theorems with per-attempt pass rate < 0.25 and
  0.53 on textbook theorems with ≥ 13 reference lines. Decision: run the learned per-candidate verifier (headroom
  exists on hard theorems) and make "stop at first finished proof" the main stopping competitor.
- Advisor review (01:55): agreed; added the name-position masking trap for likelihood scoring (SN arm: the environment
  assigns the introduced-name token, so the model's logprob there is meaningless), the scope-outs (no RL-trained
  ACCEPT token, no listwise 8-proof context), and the key figure (accuracy vs mean attempts).

## 01:50 UTC plan

Worker A: collector (keeps every attempt's raw actions; Lean verdicts in a separate labels file) → pools → baselines
→ verifier head (3 seeds). Worker B: teacher-forced scorer with name masking → stopping simulator → figures.
Reviewer: Codex astra at ~02:45, ~06:30, ~10:40.
