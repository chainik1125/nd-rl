---
written_on: 2026-10-14
written_by: agent:claude
---

# Long proofs: certified-hard natural-deduction theorems

A generator of natural-deduction theorems whose minimal proof needs 20+ body lines (lines after the premise
block), and an exact certifier that proves the minimum. The data built with it (2026-10-13/14) is on Hugging Face,
`data/long-proofs-20261013/` in `robbiethompson2018/nd-rl-checkpoints`; see [DATASETS.md](../../../../docs/DATASETS.md).

| file | what |
| --- | --- |
| `approach_b_backward/gen_deep.py` | the generator: a top-down plan of case splits over fresh atoms and IMPE chains, with facts packed into 3–10 conjunction premises, so nothing can be hoisted or shared |
| `certify.py` | the validity filter (`check`) and the scoreboard (`--score`) |
| `certifier_exact/certify_fast.py`, `fastmin.c` | a C reimplementation of Dan's `minlen`, about 15,000× faster; compiled with clang on first use |
| `production/canon.py` | the canonical key used to deduplicate theorems (atom renaming, premise order) |
| `vendor/fork/` | Dan's `minlen.py`, `gen.py` and `nd_verify`, from the take-home fork (branch `dan/dan`) |

```bash
python approach_b_backward/gen_deep.py raw.jsonl --n 24000 --seed 2001 --lo 20 --hi 80
python certifier_exact/certify_fast.py raw.jsonl f20.jsonl --floor 20 --secs 15 --mem 1024 --proofs
python certify.py --score f20.jsonl
```

Short theorems with exact labels came from the same generator: `--seed 3001 --lo 2 --hi 16 --body 1-19`.

## What "certified" means

1. `certify.check`: the proof passes nd_verify (atoms P Q R S T U W X Y Z), and the theorem passes these checks:
   - no duplicate or uncited premises;
   - the conclusion is not a premise;
   - no F premise;
   - the premises are jointly satisfiable, and the sequent is valid (truth table).
2. `certify_fast`: `cert_ge=20` means no proof with fewer than 20 body lines exists in minlen's search space.
   `min_body` is the exact minimum, when there is one below the floor.

The search space is formulas built from the theorem's subformulas, their negations and double negations, using
minlen's rule shapes. That gives two known limits:
- Proofs that use formulas outside that space are not searched.
- minlen never tries a NEGI box whose only content is its hypothesis F. Goals that need `~F` are therefore
  over-counted by one line.

Validation: agrees with unmodified minlen on 521 labelled records, 15,500 fuzzed sequents and 300 production labels.

## Final counts (`certify.py --score`, all production files)

- 21,046 theorems certified to need ≥ 20 body lines, nearly all at written length 20–29.
- 21,882 theorems with an exact minimum of 2–19.
- 3–10 premises, up to 10 atoms.

Caveats:
- Most of the data comes from this one generator family, so the theorems are similar in shape.
- A certified theorem's exact minimum is in [20, written length]. Exact values above 20 exist only for 5,760
  records, at written 20–24.

## Not in this folder

These pieces built the rest of the published data:
- the other generators (compose, premise DAGs, theorem-first sampling, guarded 35–80-line proofs);
- the scoped lower bound for 35–80-line proofs;
- the stage-3 tightening;
- the agents' reports.

They're on branch `robbie/handoff-full-archive`, under this same path.
