---
written_on: 2026-09-20
written_by: agent:claude
---

# Decision: Lean 4 is the default proof representation and checker of record

Decided by Dan on 2026-09-20 after reading the round-2 run-1 results
([summary](../../experiment-summaries/2026-09-18-round2-run1-nd-to-lean/README.md)): a
deterministic ND → Lean 4 translation agrees with `nd_verify` on every pool proof (181,464 /
181,464 accepted, 0 / 4,000 mutated proofs accepted), and a pretrained code model
(Qwen3-Coder-30B, in context) proves 0.547 of theorems written in Lean against 0.201 in the
project's token format and 0.225 in plain-English rule names (paired Lean − tokens +0.347
[+0.300, +0.393], replicated independently at +0.31).

## What changes now

1. **Checker of record.** Every proof that is counted in a result must translate with
   `nd2lean.py` and be accepted by Lean (core, no Mathlib). `nd_verify` remains the fast reward
   inside RL loops; a disagreement between the two on any counted proof is a bug to be reported,
   not a data point.
2. **Reporting.** Proofs shown in write-ups and summaries are rendered in Lean, with the token
   form available on request.
3. **Novelty by scale** (the smallest pretrained model that proves a theorem) is measured in
   Lean, and is the cross-project novelty measure alongside base-model log-probability.
4. **Pretrained-model experiments** (in-context, fine-tuning, distillation) use Lean as the
   surface form by default; the token format is a control, not the default.

## Update 2026-09-22: Lean (`lean_seq` naming) is also the training format

Proposal 8 ran on 2026-09-21 (reviewed; summary in `experiment-summaries/`). Under its
pre-registered rule, the `lean_seq` naming scheme (hypotheses named in first-appearance order
plus a random offset) was **not worse on all three predictions** and better on the two RL
outcomes: depth-3 f = 0 acquisition 0.476 / 0.479 vs 0.335 / 0.364, ladder transfer `L*` 11 / 11
vs 10 / 10 (one Stage-1 model per scheme; a second seed is being run). `lean_rand` failed the
in-distribution prediction and is not used. **Dan adopted `lean_seq` as the training format on
2026-09-22.** The token format is retired for new runs; existing token-format results stay on
file with their format labelled. Two conditions attach: `nd2lean.py`'s `BOTE` rendering
(`na.elim`, which Lean resolves to `Not.elim` on negations and thereby accepted 460 non-proofs in
13.9 M) is fixed before Lean alone is the reward — until then the reward is Lean ∧ `nd_verify`;
and the `L*` = 11 result carries "one Stage-1 model" until the second seed reports.

A consequence for the project's question: Lean-format base models trained on zero depth-3
proofs already write depth-3 proofs for 13–28 % of targets and have frozen `L*` 9–10 (token
format: 0.5 % and 7); expert iteration's increment over the frozen control is no larger in Lean
than in tokens. Part of the campaign's "RL creates depth-3 from zero coverage" was the token
format's unary depth prefix; the surviving account — RL amplifies what the base model already
does — holds in both formats.

## Update 2026-09-22 (later): Lean is the only checker

Dan decided to retire `nd_verify` as reward and checker. Proposal 9 builds `lean_check` (elaborated-term allowlist: `And`/`Or`/`Not`/`False`/`Iff` constructors and eliminators plus `Classical.byContradiction`; no `sorryAx`; no automation) and `term_size`, relabels the pools in both units, and tests free-form un-automated Lean against the fragment. Until `lean_check` is validated, the reward stays Lean ∧ `nd_verify`. Lines stay the reported unit for now; term size is reported beside them.

## What did not change before proposal 8

The from-scratch model (Stage-1 pretraining and expert iteration) kept the token format
until proposal 8 had run: the Lean result is about models with a prior over Lean syntax, and
says nothing about a model trained from scratch. If proposal 8 shows Lean-format training is
not worse on the depth-3 dial and a ladder rung, Lean becomes the training format too and the
token format is retired.

## Where this is enforced

- `docs/proposals/orchestration/executor.md`, `reviewer.md`, `proposer.md`: updated with the
  rules above (reviewer re-checks counted proofs in Lean; executor reports in Lean; proposer
  designs pretrained-model work in Lean).
- The fork's `AGENT_POLICY.md` gains the same standard with its reason.
- Known semantic note from the review: Lean's `¬A` is `A → False`; the translation maps `NEGE`
  and `NEGI` accordingly and this is where any future disagreement is most likely to appear.
