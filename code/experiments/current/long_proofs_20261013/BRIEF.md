---
written_on: 2026-10-13
---

# Long-proofs hunt: shared brief for approach agents

## Goal (Robbie, 2026-10-13)

Primary, the win condition: **10,000 distinct natural-deduction theorems whose minimal proof is certified to
need at least 20 body lines** (lines after the premise block), each with a verified proof.
Secondary: certified proofs at 10–20 lines too, a roughly uniform spread of lengths from 10 up to 50–80, and
premise counts roughly uniform over 3–10. Training on the data is out of scope; ignore tokenizer and context limits.

## Scoreboard (use it, don't fork it)

`certify.py` in this folder: `check(text)` then `certify(prompt, n_prem, floor=20, secs=...)`.
- `check` verifies with the take-home `nd_verify` (atoms P Q R S T U W X Y Z), and rejects: invalid proofs,
  duplicate or uncited premises, conclusion-is-premise, F premise, and premises that are **jointly
  inconsistent** by truth table (ex falso makes any goal a 2-line proof).
- `certify` runs Dan's `minlen` (vendor/fork/minlen.py): an iterative-deepening, memoised search over every
  proof whose formulas come from S = subformulas of the theorem plus their negations and double negations.
  `cert_ge = 20` means it exhausted all proofs with < 20 body lines and found none. A found proof gives the
  exact minimum (`min_body`). A timeout certifies nothing.
- CLI: `python certify.py IN.jsonl OUT.jsonl --floor 20 --secs 60 --procs 2` (input records need `text`:
  `THM p1 , p2 SEQ c PRF N1 ... QED`), and `python certify.py --score OUT.jsonl ...`.
- Python: `~/repos/nd-takehome/.venv/bin/python` (3.13, stdlib only is enough).

## What we already know (Claude's yield test, 2026-10-13)

- Robbie's backward generator (`~/repos/nd-takehome/gen.py`) can't make long proofs: length is capped by goal
  formula size (max_fdepth 2-3) and premise cap, not by its budget; max 16 body lines by default, 27 with
  every knob raised. Of its 18+-line proofs, 58% have inconsistent premises; the rest shortcut to 5-11 lines.
- Dan's forward generator (`code/shared_code/proof_gen.py` on branch `dan_shared_proof_gen`, a copy is at
  `/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/dan/code/shared_code/`)
  writes 80-120-line proofs easily, but 100% of a 40+-line sample had inconsistent premises, and the
  directly-contradictory-free ones shortcut to 5-11 lines. Random forward steps derive F and pad with ∨E.
- minlen is cheap when a short proof exists (deepening stops early) and expensive when it doesn't: proving
  "nothing under 20" means exhausting every total up to the bound, and cost grows roughly exponentially with
  the size of S (theorem size) and the bound. At 30 s, 5 of 17 consistent backward-generator theorems timed out.
  So **certification throughput is probably the bottleneck**, not generation.

## Honesty rules

- Don't game minlen. Its blind spot is formulas outside S: if a generator's own proof needs formulas outside S,
  minlen cannot find even that proof, so "nothing under 20" would be vacuous. Prefer proofs whose formulas are
  in S, report the fraction that aren't, and flag any trick that might fool the search.
- Report measured numbers only; say which runs you actually did.
