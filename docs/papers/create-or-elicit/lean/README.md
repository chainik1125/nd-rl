# Lean files for the paper and blog

- `blog_example.lean`: the simplified σ proof shown in `paper/blog.md` (simplified by hand from the knockout s1 base's
  literal proof of `la_transfer_916`, which `paper/scripts/lean_examples.py` renders and checks).
  `lean blog_example.lean` → no error; `#print axioms t` = [propext, Classical.choice, Quot.sound].
- `blog_example_negctl.lean`: the same proof term for the negated goal; Lean must reject it (type mismatch).

Checked 2026-10-13 with Lean 4.35.0 on the VPS (the pods use 4.34.1).
