# Run brief: support-curves — does expert iteration expand the base model's support?

Run id `support-curves`. Repository `~/work/support-curves` on **VPS1** (branch `dan_support-curves`,
a worktree from the fork's `origin/dan`). Role: executor. `AGENT_POLICY.md` (nd-rl canonical copy)
governs; designs are suggestions, the question is what matters. Pod budget **$15**, ceiling **30
pod-hours**, hard stop 30 h. Balance floor **$100** (`rpbalance`); two sibling runs on the other host
hold up to $34 more, so do not extend past $15 without asking in `QUESTIONS.md`.

Read first: **proposal 12**, `~/nd-rl/docs/proposals/2026-09-27-elicitation-curve.md` (this run is its
experiment 1); `NOISE_FLOOR.md`; the `cap-horizon` and `noise-floor` summaries in
`~/nd-rl/experiment-summaries/`; `ladder_ei.py`, `coverage.py`, `sample.py`, `lean_gate.py`.

Siblings on agent2: `lean-judge` (making Lean the only judge — it edits `lean_gate.py`, adds
`lean_judge.py`, swaps the `nd_verify` import in `coverage.py`, `eval_set.py`, `expert_iter.py`,
`ladder_ei.py`, `eval_targets.py`, `grpo.py`) and `stage1-dynamics` (edits `train.py`). **Do not edit
any of those files.** Write your own scripts (e.g. `support.py`) on top of `sample.generate`.

## Question

On theorems neither model was trained on, does the expert-iteration (EI) model solve theorems the
base model essentially cannot — at k far beyond the attempts RL used — or does base pass@k catch up?
Equivalently: the first measured points of the elicitation curve, per-theorem log p_base against
log p_RL.

## Models (label every number with these)

- **Base:** the control Stage-1 checkpoints, 3.2 M-parameter from-scratch GPT, `lean_seq`, cap 6,
  trained on `data/p2/train_depth3_f0_a1.jsonl` — `ckpts/lf/stage1_a1_seq_s{0,1}.pt` from
  `hf://buckets/dan-pandori/nd-rl/lean-format/` (md5 `9bde44c0…` / `fc27e52d…` per `ds-generator`'s
  review; verify).
- **EI:** ladder rung T1 (plain EI, 8 rounds × k 32, T 0.8) from each base seed, **re-trained** in this
  run, because no Lean-format EI checkpoint was ever uploaded. Mirror `noise-floor`'s `la_T1_dsc_*`
  command exactly (its log `artifacts/nf/logs/la_T1_dsc_a1_s1.log` on `dan_noise-floor` shows it) and
  **upload `ckpts/ladder/` for both seeds, every round**. Sanity check: transfer solved should land
  near the on-file 856–965 for this base (seed-dependent; say where it lands).

## Design (a suggestion; pre-register what you actually run)

**Theorem set.** ≈ 400 theorems from `data/ladder/transfer.jsonl` (2,285, never trained on, labelled by
true minimal length `L_true` from `minlen.py`), stratified by `L_true` with the long bins
over-sampled — e.g. 25 each at `L_true` 4–6 as anchors, 50 each at 7–12, and every theorem ≥ 13 up to
50. Draw with a fixed seed and **commit the list before sampling** (gate-0 material). Note that under
Lean `L_true` is an upper bound on the minimal proof length.

**Stage 1 — both models, seed 0, T = 0.8, up to k = 4,000 per theorem.** Stop a theorem after 50
successes (record attempts n and successes c; p is then well estimated). This yields per-theorem
(n, c) for base and EI.

**Stage 2 — the crux sets, adaptive.**
- *Forward crux:* theorems with c_EI ≥ 1 and c_base = 0. Give the base up to **40,000 more attempts**
  each, split between **T = 0.8 and T = 1.0** (support at T = 0.8 is not the model's support), stopping
  at 5 successes per temperature. Give the EI model 4,000 attempts at T = 1.0 on the same theorems for
  symmetry.
- *Reverse crux:* theorems with c_base ≥ 1 and c_EI = 0 (Yue et al. find RL loses some coverage at
  large k). Give the EI model up to 20,000 more attempts on them at T = 0.8. Report how many exist.

**Stage 3 — replication, seed 1:** both models at k = 2,000, T = 0.8, on the same theorem set. Drop
first if the budget binds, and say so.

**Secondary (forward passes only):** the base model's teacher-forced log-probability of each distinct
EI-found proof on the forward crux, marginalised over the random name offset as the project's base-
reachability tool does. A lower bound on base's per-sample probability for that *proof*, not the
theorem (other proofs may exist) — report it as such.

**Throughput.** Measure samples per pod-hour on these theorems in the first stage-1 batch (they are
longer than the reductio pool; `noise-floor` measured ≈ 0.7 A40-hours per 600,000 samples on
7-line targets) and project the whole run. If the projection exceeds the ceiling: first cut stage 3,
then stage 1's k to 2,000, never the forward crux's base attempts, then the theorem count.

## Judging

**Lean alone decides** (Dan, 2026-09-27), and **both models must be judged identically** — at
1-in-40,000 rates the Lean-only acceptance class (≈ 67 per million at the gate) is not negligible. If
`lean_judge.py` is on `origin/dan` when you start sampling, use it. Otherwise the gate already records
every Lean verdict: count Lean-accepted = gate "both ok" ∪ the gate's `*.disagree.jsonl`
`nd_rej & lean_ok` rows, keep every Lean-accepted literal text, and re-judge with `lean_judge.py` when
it lands (no GPU). State which was used.

**Storage.** Do not store failing samples in bulk (this run draws ≈ 8–10 M samples; VPS1 has ≈ 8 GB
free): keep per-theorem (n, c, first-success index) per model × temperature × seed, every accepted
literal text, and a small random sample of failures. Upload to the bucket as you go.

## Analysis

- **The scatter:** per theorem, log10 p̂_base against log10 p̂_EI (Wilson intervals; for c = 0 plot
  the 95 % upper bound 3/n as an arrow), coloured by `L_true`, both temperatures for the crux.
- **Pass@k curves** per `L_true` stratum for both models (unbiased estimator, k ≤ n; any extrapolation
  beyond n is labelled as such), and the crossover k per stratum if one exists.
- **Counts:** forward-crux theorems still at 0 base successes after all attempts, at each temperature,
  with the EI model's p̂ on each; reverse-crux size and outcome.
- Seed 0 vs seed 1 agreement on stage 1; quote `NOISE_FLOOR.md` for anything cross-run.

## Pre-registration (`preregistration/support-curves.md`, before the first pod)

Proposal 12 predicts the amplifier: base pass@k on the forward crux rises toward the EI model's with
k. **Falsifier (support expansion):** ≥ 20 forward-crux theorems with 0 base successes in ≥ 40,000
attempts at both temperatures while the EI model solves each at p̂_EI ≥ 0.01. Add your own numbers:
forward-crux size, how many survive stage 2, the reverse-crux size, per-stratum crossover k.

## Deliverables

`preregistration/support-curves.md`; `run_support_curves.md` (≤ 400 words + the scatter and the
per-stratum pass@k figure); `numbers.md` § support-curves (every entry names its source file and the
model); `log.md` dated; `artifacts/sc/summary.json` (one row per theorem × model × temperature × seed);
`STATUS.md` line `SUPPORT-CURVES DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/support-curves/{ckpts,artifacts,data}`; pods deleted;
`touch ~/runs/support-curves/executor.done`. Questions for Dan in `QUESTIONS.md` with your default.
