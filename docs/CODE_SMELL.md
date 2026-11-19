---
written_on: 2026-11-18
written_by: agent:claude
---

# Code smells

Known structural problems in the codebase that nobody has scheduled a fix for. One section each.
Write down what the problem is, where it lives, and what it costs, so the next person does not
have to rediscover it.

## Fitch natural deduction is still the internal proof representation

Raised by Robbie on 2026-11-18.

### Summary

Lean 4 became the checker of record on 2026-09-20 and the training format on 2026-09-22, when the
old token format was retired for new runs. Fitch did not go away. Every proof still exists as a
Fitch proof internally: the datasets store Fitch, the pipeline renders it to Lean to train, and it
parses every sample the model writes back into Fitch. The model's Lean is one fixed template per
Fitch rule, so the two forms are the same object in two notations, and the code maintains both.

Someone reading the datasets or the write-ups would reasonably conclude that the model writes
Fitch. It does not: it reads a Lean theorem statement and writes a chain of
`have … ; … ; exact …`.

The two dates and the `BOTE` bug below are from Dan's decision note,
`docs/project_strategy/2026-09-20-lean-default.md` on branch `dan` (not on `main`).

### Evidence

Observed in the whole-proof pipeline,
[`code/experiments/current/long_rl_20261014`](../code/experiments/current/long_rl_20261014/README.md),
at `main` `a4727230`:

- **Data on disk is Fitch.** The `prompt` and `proof` fields of the pretraining, RL, dev and
  holdout files are `THM … SEQ … PRF` and `N1 … : RULE refs ; … QED`.
- **The tokenizer converts both ways.** `overlay/lean_tok.py` renders a Fitch proof to Lean tokens
  in `encode_proof`, and `decode()` parses sampled tokens back to Fitch. Its docstring gives the
  reason: "so nd_verify, prune, normalize and every analysis script work unchanged".
- **Samples are decoded before anything else happens.** `harness.py` calls `tok.decode` on every
  sample and counts written and pruned body lines on the Fitch form (`body_lines`).
- **Found proofs are kept as Fitch.** `pretrain.py` stores them as Fitch text, dedupes them with the
  fork's `normalize.norm`, and re-renders them to Lean each time they are trained on.
- **A sample must parse as Fitch before Lean sees it.** In `overlay/lean_gate.py`, a sample outside
  the strict grammar is marked `LEANPARSE` and is not sent to Lean.
- **The retired verifier is still a dependency.** `nd_verify.verify` is imported for
  `parse_proof_tokens` and `parse_formula`. It parses; it does not judge.
- **The rule-to-template mapping is written three times:** `nd2lean.translate`, the tokenizer's
  encoder ("Mirrors nd2lean.translate"), and the tokenizer's inverse parser.

Not audited: the proof-state pipeline. `code/thinking/vendor/danstate` and
`code/ndbench/vendor/danstate` carry copies of the same `lean_tok.py`, `lean_gate.py` and
`nd2lean.py`, so the same structure is likely there.

### What it costs

Interpretation, not measured:

1. **The model's proof steps are capped by the Fitch rule list.** Valid Lean that is not one of the
   templates earns no credit, because it fails the parse before Lean is asked. The vocabulary has
   one classical token, `Classical.byContradiction`, used only in the double-negation template.
   There is no `Classical.em`. For a project about capabilities emerging under RL, a "new strategy"
   can only be a new arrangement of the existing rules, never a new kind of step.
2. **The translation layer is a place for bugs.** The 2026-09-22 `BOTE` rendering bug
   (`na.elim`, which Lean accepted on 460 non-proofs in 13.9 M) lived there. Three copies of the
   mapping have to stay in step.
3. **"Lean only" is hard to audit** while the old verifier package is on the import path of the
   tokenizer and the data builder.
4. **The metrics are tied to Fitch.** Body lines, pruned length, box depth and the proof-strategy
   taxonomy are all defined on Fitch lines and citations.

### Next steps

Nobody has scoped a removal. The two shapes it could take:

- Make Lean the stored form: keep the literal Lean text, and compute length, pruning and dedupe on
  a parse of that text.
- Keep an explicit proof tree as the internal form, but name it as that, with one renderer and one
  parser, and stop routing through the Fitch text format and `nd_verify`.

Either one lifts the cap in cost 1 only if the gate also stops requiring the strict grammar.
