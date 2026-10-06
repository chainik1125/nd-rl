# Standing policy for agent runs in this repository (from Dan, 2026-09-17)

Every brief in this repository is read together with this file. Where a brief and this file
disagree, this file wins.

## The one hard rule

**Do not contact or involve other humans.** No PR reviewers, no messages, nothing sent to
Dmitry or anyone else. Draft PRs with no reviewers are fine. Dan reads and shares things himself.

## Standards with reasons (keep them; say when and why you deviate)

- The take-home's own rules: cap 6 on supervised data, at most one test-file run per
  submission, `nd_verify` unmodified, no hand-written or LLM-written training proofs.
- Learned from real bugs, each of which produced a wrong number once: start-index-normalised
  distinct-proof counts; a frozen control at equal attempts; a base-model reachability number
  with every "RL solved X"; two seeds before calling a difference; expected results written
  down (and committed) before a run; splits disjoint by renaming class; every count
  reproducible from files that are pulled back.

- **Lean is the checker of record (Dan, 2026-09-20).** Every counted proof must translate with `nd2lean.py` and check in Lean; report agreement with `nd_verify`; a disagreement is a bug. Proofs in write-ups are shown in Lean. Pretrained-model experiments use Lean as the surface form. From-scratch training keeps the token format until proposal 8 (`~/nd-rl/docs/proposals/2026-09-20-lean-training-format.md`) reports. *Why:* the translator agrees with `nd_verify` on 181k/181k proofs and pretrained code models prove 2.7× more theorems in Lean.

## Everything else is a suggestion

Experiment designs, pod counts, GPU classes, deadlines, deliverable formats. Use judgment. If
a different or bigger experiment answers the project's question better, propose it or do it.
The question is the SPAR project's: to what extent does RL against a verifier produce
genuinely new capability rather than eliciting rare-but-known behaviour, and what limits it.

## Dan is reachable

He is not watching live but reads `STATUS.md` and answers questions. Put questions in
`QUESTIONS.md` at the repo root — dated, with the default you will follow if unanswered —
commit, and proceed on the default. Direction, scope, spend, and "what would you find most
useful" are all fair questions.

## Spend and compute

- **Per-run pod budget: $50** unless Dan authorises more for that run (ask in `QUESTIONS.md`;
  proceed within $50 meanwhile). The RunPod balance is the only hard cap. Check it with
  `rpbalance`; `~/ALERTS.md` gets a line when it falls below $50, and you should add one too
  if you see it first — Dan tops up.
- No limit on concurrent pods; no fixed GPU class. Choose per experiment (Stage-1 sweeps on a
  cheap card; bigger cards only when a brief needs them). Delete pods when their work is pulled.
- Claude usage: roles share Dan's Max 20x subscription. If a usage limit is slowing progress,
  write `LIMIT_SLOWING <UTC> <what it cost you>` in `STATUS.md`; Dan will move agents to API
  keys if it is flagged again.
- VPS: 4 GB / 2 vCPU. New VPSs can be created with `newvps <name>` (needs Dan's Claude and gh
  logins afterwards).

## nd-rl target branch

Dan's integration branch in `~/nd-rl` is **`dan`**. Librarians branch from `origin/dan` and merge back into `dan` after review; one standing draft PR `dan` → `main` exists. Never merge into `main`.

## Pause (2026-09-18)

No new experiment launches until Dan gives the go-ahead. Runs already started finish; their reviewer and librarian steps complete.

## Artifacts

At `DONE`, upload the run's `ckpts/`, `artifacts/` and `data/` to the public HF bucket:
`hf buckets sync <dir> hf://buckets/dan-pandori/nd-rl/<run-name>/<dir>` (the `hf` CLI is
logged in on the VPS; `podrun` passes `HF_TOKEN` to pods). Record the bucket paths in
`numbers.md`. The bucket is **public**: never upload `~/.config`, shell logs, or anything that
could contain a token, and never print tokens.
