# Run brief: lean-prefilter — take Lean off the RL critical path (proposal 15, items 3 and 4)

Run id `lean-prefilter`. Repository `~/work/lean-prefilter` (branch `dan_lean-prefilter` from the fork's
`origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` (nd-rl canonical) governs. Budget
**$3**, ceiling **6 pod-hours**. Authorised by Dan 2026-09-28 (proposal 15's queue); Claude quota may
throttle it, which is accepted. Sibling `fast-stage1` edits `train.py` / `model.py` at the same time — do
not touch those. Read first: proposal 15 §§3–4, `lean_gate.py`, `lean_judge.py`, `lean_check.py`,
`lean_tok.py` (the strict `lean_seq` grammar), `sample.py`, `ladder_ei.py`, and the `lean-judge` and
`efficiency` summaries in `~/nd-rl/experiment-summaries/`.

## Measured starting point (2026-09-28)

Round 1 of a T1 ladder (`noise-floor`'s `la_T1_dsc_a1_s1.log`) took 1,932 s: sampling ≈ 60 % at batch 512,
**Lean ≈ 30 %** (353 s gating 143,840 target samples, 206 s the 73,120 transfer samples; round 2: 512 s +
233 s), fine-tuning ≈ 7 %. Lean ran on **3 workers**. Of 92,074 distinct texts checked in round 1,
**78,151 were rejected**. `efficiency` measured ≈ 2.0 s of fixed start-up per `lean` process, 36 % of gate
process time.

## What to build

1. **A sound, reject-only pre-filter** for `lean_seq` texts, in Python, run before Lean: parse (exists),
   every cited name in scope at its use, and whatever cheap declared-type bookkeeping is decidable without
   a type checker (the `state-env` run's renderer, if merged, does most of the scoping). It may only
   *reject*; Lean still decides every text the filter passes. **Soundness is the acceptance test**: on
   ≥ 1 M stored gate texts with Lean verdicts (gate logs and `*.disagree.jsonl` across recent runs in the
   bucket, re-checked in Lean where the verdict is not stored), the filter must reject **zero** texts Lean
   accepts — including `Not.elim` and the other `¬A ≡ A → False` unfoldings Lean allows. Report how much of
   Lean's reject load it removes (aim ≥ 70 %).
2. **Lean on more cores**: size the gate's worker pool to the machine instead of 3, and/or pipeline — the
   GPU samples the next batch while CPU workers check the last. Measure; don't assume.
3. **Script defaults (item 4)**: the loop scripts' `--batch` defaults to the largest batch that fits (the
   policy's standard), `max_new` with the truncation rate reported, as `AGENT_POLICY.md` now asks.

A persistent Lean server is out of scope (later, with the step-level environment).

## Acceptance test (pre-register it before the first pod)

(a) Soundness as above: 0 false rejects. (b) Identical accepted sets with and without the filter on a
fixed stored sample. (c) One T1 ladder round on a fixed checkpoint and seed, old path vs new: round
wall-clock (target ≤ 1/3 of the old), gate share of the round, and the accepted set (identical at fixed
batch).

## Deliverables

Merged into the fork's `dan` **after** (a)–(c) pass (the reviewer follows; quarantine means revert);
`LEAN_GATE.md` at the repo root (what runs where, measured speeds, the soundness table);
`preregistration/lean-prefilter.md`; `run_lean_prefilter.md` (≤ 300 words); `numbers.md` §
lean-prefilter; `log.md`; `STATUS.md` line `LEAN-PREFILTER DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/lean-prefilter/artifacts`; pods deleted;
`touch ~/runs/lean-prefilter/executor.done`.
