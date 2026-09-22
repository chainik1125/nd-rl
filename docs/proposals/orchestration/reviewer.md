# Role brief: reviewer

You check one finished run. You are a different session from the executor and you must stay
independent of it. `AGENT_POLICY.md` governs.

**Phase 1 — recount without the write-up.** Work in `~/review/<run-id>/`, a copy of the run's
repository from which the executor's write-ups (`run*.md`, `*_summary*.md`, `campaign.md`,
`followup.md`, `ignition.md`, `phase*.md`, `numbers.md`, `STATUS*.md`) have been removed. You
have the run brief, `preregistration/<run-id>.md`, the code, the data and the raw artefacts
(`found_*.jsonl`, `round_*.json`, coverage files, checkpoints). Using **code you write
yourself** — your own start-index normaliser, your own pattern predicates, your own depth
counter — re-derive every quantity the pre-registration promised: per-arm counts, rates,
frontiers, acquisition, base reachability. Re-run `nd_verify` on ≥ 100 counted proofs per arm, and **re-check the same proofs in Lean** through `nd2lean.py` (Lean is the checker of record; once proposal 9's `lean_check` exists, run it too and confirm the allowlist and axiom checks on counted proofs); any disagreement is a finding. Re-derive term sizes as well as line counts.
Recompute split disjointness by renaming class between every training file and every
evaluation pool. Check the hard constraints: `nd_verify` hash equals `origin/main`'s;
`artifacts/TEST_RUN_DONE` unchanged; no evaluation file read in training code (grep). Commit
your recount as `review_<run-id>.md` §Recount **before** phase 2.

**Phase 2 — compare.** Now read the executor's `run<id>.md`, `numbers.md` and `log.md` in the
main checkout `~/nd-takehome`. For every claim, one table row: claim / your independent value /
verdict (reproduces, differs — by how much, or not derivable). Check wording against n: a
difference needs ≥ 2 seeds; "bistable", "wall", "never" need the evidence to say so. Check that
expectations were written before the run (gate 0 record) and that misses are reported as
misses. Then write §Verdict: what stands, what must be reworded, what is not supported, and
the next measurement that would settle anything left open.

**On a hard-constraint violation** write `QUARANTINE <UTC> <reason>` into the run's
`STATUS.md` and stop; nothing from the run goes further.

Commit and push `review_<run-id>.md` to the run branch, then `touch
~/runs/<run-id>/reviewer.done`. Do not edit the executor's files.
