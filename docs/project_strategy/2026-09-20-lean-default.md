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

## What does not change yet

The from-scratch model (Stage-1 pretraining and expert iteration) keeps the token format
until proposal 8 has run: the Lean result is about models with a prior over Lean syntax, and
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
