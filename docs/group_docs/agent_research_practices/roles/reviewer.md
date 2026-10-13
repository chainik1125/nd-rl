---
written_on: 2026-10-13
written_by: agent:claude
---

# Role brief: reviewer

You check one finished run. You are a different session from the executor and must stay independent of it.
[POLICY.md](../POLICY.md) governs.

## Phase 1: recount without the write-up

You start on a copy of the run's working folder **prepared by whoever launched you**. The executor's write-ups have
been removed from it: `run_*.md`, `numbers.md`, `log.md`, `STATUS.md`, `next_steps.md` and any summaries. Do not open
the original folder until phase 2. You have:
- the run brief and `preregistration.md`;
- the code and the data;
- the raw artefacts (sample dumps, per-round files, checkpoints).

**Using code you write yourself**, re-derive every quantity the pre-registration promised: per-arm counts, rates,
frontiers, acquisition and base reachability. Write your own normaliser, your own pattern predicates and your own
counters; do not import the executor's analysis code.

Also check:
- **Proofs.** Re-check ≥ 100 counted proofs per arm in Lean from their stored literal text. A counted proof Lean
  rejects is a finding. For free-form Lean, also run the allowlist and axiom check (POLICY § Judging proofs). Confirm
  no judge other than Lean was used.
- **Lengths.** Re-derive term sizes as well as line counts.
- **Splits.** Recompute disjointness by renaming class between every training file and every evaluation pool.
- **Constraints.** No evaluation file is read by training code; protected files are unchanged; the pre-registration
  predates the first job.

Commit your recount as `review_<run>.md` § Recount **before** phase 2.

## Phase 2: compare

Now read the full working folder: `run_<run>.md`, `numbers.md`, `log.md`, `STATUS.md`, `next_steps.md`.
- **Claim by claim:** one table row per claim, giving the claim, your independent value, and a verdict (reproduces /
  differs, by how much / not derivable).
- **Wording against n:**
  - a difference needs ≥ 2 seeds and must clear the noise floor;
  - "never", "wall" and "needs" need the evidence to say so;
  - misses must be reported as misses.
- **Labels:** every number names its model and, across judges, its checker. For an unlabelled number, say which model
  it actually applies to, or that this cannot be determined.

Then write § Verdict:
- what stands;
- what must be reworded, with exact replacement wording;
- what is not supported;
- the next measurement that would settle what is left open.

On a hard-constraint violation (for example, training on an evaluation set), write `QUARANTINE <UTC> <reason>` in the
run's `STATUS.md` and stop: nothing from the run goes further.

Commit `review_<run>.md` to the run's working folder. Do not edit the executor's files.
