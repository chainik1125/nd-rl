# Run brief: support-followups — pressure-testing the support-expansion result

Run id `support-followups`. Repository `~/work/support-followups` on VPS1 (branch
`dan_support-followups` from `origin/dan_support-curves`, made by `newrun`, so `support-curves`' code,
theorem list, crux lists and survivor list are all here). Role: executor. `AGENT_POLICY.md` (nd-rl
canonical) governs. Budget **$15**, ceiling **30 pod-hours**, registered. Authorised by Dan 2026-09-28.

Read first: the `support-curves` summary in `~/nd-rl/experiment-summaries/` and
`review_support-curves.md` (its §Verdict lists these follow-ups), `run_support_curves.md`,
`preregistration/support-curves.md`, proposal 12. Sibling run `state-env` is on agent2; nothing else
here.

## Why

`support-curves` fired its pre-registered falsifier: **29 transfer theorems with 0 base successes in
200,000 attempts at T = 0.8 and 200,000 at T = 1.0, which the base's own 8 × 32 EI descendant solves at
p̂ 0.022–0.9995** (base `stage1_a1_seq_s0.pt`, md5 `9bde44c0…`, 3.2 M, from scratch, `lean_seq`, cap 6;
EI `la_T1_sc_s0_r8.pt`, md5 `5cebd7ec…`). The reviewer reproduced it and named three follow-ups. A fourth
asks what *kind* of capability it is. Survivors by `L_true`: 7 → 1, 9 → 13, 10 → 9, 11 → 5, 12 → 1
(`data/sc/falsifier_survivors.txt`).

## The four measurements (order: D, A, B, C — cheapest and most interpretive first)

**D. Where is the improbability? (forward passes only.)** For every distinct EI proof of the 29
survivors, the base's teacher-forced log-probability **per token**, summed per `lean_seq` step (one
`have`/box-opening/`exact`), marginalised over the random name offset as `sc_secondary.py` does. Report
per proof: the worst single step's log p, the share of the total in the worst 1 and worst 2 steps, and
the tokens where the base's log p is lowest, classified as syntax (brackets, keywords, separators),
naming/citation (which hypothesis), or logic (the rule or formula chosen). The same for two controls:
EI proofs of forward-crux theorems the base *does* solve, and the base's own accepted proofs. The two
readings, to pre-register with thresholds of your choosing:
- **a new move** — a few steps carry most of the improbability (one or two steps far below the rest);
- **compounding reliability** — the improbability is spread over many steps, none individually
  implausible, so small per-step error rates multiply over a long proof.
The distinction matters for the SPAR question (proposal 12): the second is sharpening per step that
becomes expansion per proof.

**A. A seed-1 column on an uploaded checkpoint.** The seed-1 EI checkpoint behind `support-curves`' stage
3 (md5 `105be4f3…`) was lost. Re-run stage 3 — both models, the 383 theorems, k = 2,000, T = 0.8 — with
EI = `la_T1_sc_s1rerun_r8.pt` (md5 `12c13e61…`) and base `stage1_a1_seq_s1.pt` (md5 `fc27e52d…`).
Report it beside the lost-checkpoint numbers, labelled.

**B. Depth on the longest survivors.** The 6 survivors at `L_true` 11 (5) and 12 (1), where the EI
proof's base log p is ≤ −30: **10⁷ further base attempts in total** at T = 1.0 (≈ 1.67 M each, on top of
the 400,000 already spent), seed-0 base, stopping a theorem at 5 successes. Any success gives p̂; none
bounds p_base below ≈ 1.5 × 10⁻⁶ each. ≈ 3 pod-hours at the measured 3.6 M samples per pod-hour.

**C. A bigger base.** Does a larger model trained on the same data already reach these theorems? One
Stage-1 model at ≈ 8× the parameters (e.g. 8 layers, d 512 — report the exact count) on the control set
`data/p2/train_depth3_f0_a1.jsonl` with the control's schedule (6,000 steps × 128; `stage1-dynamics`
found the 3.2 M model not saturated at 6,000 steps, and a larger one may be less so — say so; this is
"more capacity, same data and steps", not a tuned larger model). Held-out greedy as a sanity check.
Then sample the **82 forward-crux theorems** (`data/sc/crux_forward.txt`) at 10,000 per theorem, T 0.8,
and the survivors it has not solved to up to 200,000 per temperature (T 0.8, 1.0), stopping at 5
successes per temperature. Question: how many of the 29 does the bigger base reach, and at what p̂?
Seed 0; seed 1 only if the budget allows. No EI on the bigger model in this run.

## Settings

Lean alone decides (`lean_judge.py`), both models judged identically; store every accepted literal
text. Sampler per the policy: largest batch that fits (record peak memory; the 25 M model needs its own
measurement), **`max_new` 512** — the review found ≈ 1.0 % of base samples truncated at 400 in
`L_true` 10 — with the truncation rate reported per stratum. Label every number with its model and md5.

## Pre-registration (`preregistration/support-followups.md`, before the first pod)

Numbers for each part: D's thresholds and which reading you expect; A's seed-1 survivor count and
agreement with seed 0; B's expected successes; C's expected number of the 29 the bigger base reaches.
Falsifier for "the expansion is a 3.2 M-capacity artefact": the bigger base reaches ≥ 15 of the 29
within 200,000 attempts per temperature.

## Budget and order

`podbudget support-followups` is registered at 30 h / $15. D needs only the saved models (a CPU pass
may do; a GPU is minutes). Then A (≈ 1 pod-hour), B (≈ 3), C (Stage-1 plus sampling — measure the
throughput of the larger model first and re-project). Dan is topping up the RunPod balance; check
`rpbalance` before C and keep the balance above $100 counting `state-env`'s remaining ceiling — if it
would not, finish D, A and B, write them up, and put C's projection in `QUESTIONS.md`. Drop order: C
seed 1, then C's T 1.0 depth, then B's depth halved.

## Deliverables

`preregistration/support-followups.md`; `run_support_followups.md` (≤ 400 words + two figures: D's
per-step log p profiles, survivors vs controls; C's survivor reach by `L_true`); `numbers.md` §
support-followups (every entry names its source file and model md5); `log.md` dated;
`artifacts/sf/summary.json`; `STATUS.md` line `SUPPORT-FOLLOWUPS DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/support-followups/{ckpts,artifacts}`; pods deleted;
`touch ~/runs/support-followups/executor.done`. Questions for Dan in `QUESTIONS.md` with your default.
