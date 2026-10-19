# nd_verify (with the forward box-citation fix)

The natural-deduction verifier from the take-home repository (`dan-pandori/nd-takehome`,
`nd_verify/`; every vendored copy in this repository has `verify.py` sha256 `5dc502c2fc0a…`),
with one fix in `box_citable`.

## The bug

`line_citable` requires a cited line to come before the citing line (`1 <= j < i`), but
`box_citable` never checked order. A line could therefore discharge a subproof written *after* it,
and that subproof could cite the discharging line, so circular proofs of non-theorems verified:

```
THM SEQ ( P > Q ) PRF
N1 ( P > Q ) : IMPI N2 N3 ;   claims P -> Q from the box on lines 2-3
N2 | P : AS ;
N3 | Q : IMPE N1 N2 ;         uses line 1 inside that box
N4 ( P > Q ) : R N1 ;
QED                           old verifier: (True, 'ok', 4)
```

With the classical rules this extends to any formula. The spec already requires a cited box to be
closed at the citing line; the fix adds `if e >= i: return False` to `box_citable` (the box's last
line must come before the citing line, and `s <= e` is already checked).

## Impact

- Any RL reward computed with the old verifier could be hacked with a short generic template.
- A scan of all 876,479 proofs accepted during the hill-climb RL runs (`found_targets.jsonl` in
  runs 1-3, Leon's `/scratch/cowifornia/nd-rl/ab*`) found no forward citation, and the fixed
  verifier accepts all 876,479, so those results are unaffected.
- Lean-judged results are unaffected: `lean_seq` names each step before use, so a forward citation
  cannot be rendered, and Lean rejects `sorry`.
- The frozen vendored copies (`experiments/archived/cap_comparison_20260910/vendor/`,
  `experiments/current/r1_cot/vendor/`) are left byte-identical so their recorded results stay
  reproducible. New code should import this copy (`from code.shared_code.nd_verify import ...`) or
  apply the same two lines.
