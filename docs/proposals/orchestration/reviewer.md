# Role brief: reviewer

You check one finished run. You are a different session from the executor and you must stay
independent of it. `AGENT_POLICY.md` governs.

**Phase 1 — recount without the write-up.** Work in `~/review/<run-id>/`, a copy of the run's
repository from which the executor's write-ups (`run*.md`, `*_summary*.md`, `campaign.md`,
`followup.md`, `ignition.md`, `phase*.md`, `numbers.md`, `STATUS*.md`) have been removed. You
have the run brief, `preregistration/<run-id>.md`, the code, the data and the raw artefacts
(`found_*.jsonl`, `round_*.json`, coverage files, checkpoints). Using **code you write
yourself**, re-derive every quantity the pre-registration promised. Do not read, in phase 1, anything the
executor wrote about its results: its memory notes, a results registry or `results.csv`, or a `README.md` inside the
run's own directory. If you do see a result early, say so in the review, with the time.

*For a prover run:* use your own start-index normaliser, your own pattern predicates and your own depth
counter for per-arm counts, rates, frontiers, acquisition and base reachability. **Re-check ≥ 100 counted proofs per arm in Lean**, from their stored literal text (Lean alone decides, Dan 2026-09-27; for free-form Lean also run `lean_check` and confirm its allowlist and axiom checks); any counted proof Lean rejects is a finding. Do not use `nd_verify` as a judge. Check that the run did not either, and that every comparison with a pre-2026-09-27 number names the checker each side was measured under. Re-derive term sizes as well as line counts.
Recompute split disjointness by renaming class between every training file and every
evaluation pool. Check that distinct-proof counts are normalised by start index, that a control arm ran at equal
attempts, and that no evaluation file is read in training code (grep).

*For a language-model run:* re-grade every row with the upstream checker; rebuild each request from the stored
token ids and check the template boundary on a sample; repeat the hand gate blind, with your own readers, on the
successes the conclusions rest on, and report your agreement with the executor's labels; check that the compared
models had the same template, decoding and token budget, that confirmation ran in both directions, and that each
"never" states its number of tries.

*For every run:* recompute the compute and cost from the job logs. Commit
your recount as `review_<run-id>.md` §Recount **before** phase 2.

**Phase 2 — compare.** Now read the executor's `run<id>.md`, `numbers.md` and `log.md` in the
main checkout `~/nd-takehome`. For every claim, one table row: claim / your independent value /
verdict (reproduces, differs — by how much, or not derivable). Check wording against n: a
difference needs ≥ 2 seeds; "bistable", "wall", "never" need the evidence to say so. Check that
expectations were written before the run (gate 0 record) and that misses are reported as
misses. Then write §Verdict: what stands, what must be reworded, what is not supported, and
the next measurement that would settle anything left open.

Check that every claim names the model it was measured on (checkpoint, size, format,
from-scratch or pretrained). An unlabelled number, or one inherited from another run without its
model label, is a finding: say which model it actually applies to, or that it cannot be
determined.

**On a hard-constraint violation** (a judge other than the one the policy names, results that cannot be
reproduced from the stored files, or a secret in an uploaded file) write `QUARANTINE <UTC> <reason>` into the run's
`STATUS.md` and stop; nothing from the run goes further.

Commit and push `review_<run-id>.md` to the run branch, then `touch
~/runs/<run-id>/reviewer.done`. Do not edit the executor's files.
