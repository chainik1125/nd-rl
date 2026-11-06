---
written_on: 2026-10-14
written_by: agent:claude (subagent report, saved by the coordinating session)
---

# Signature lower bound: certifying 35–80-line proofs

The certifier is sound and certifies all 3,220 `e_raw` records (approach E, written 35–80 body lines) at ≥ 30 body
lines. 469 of them are exact.

## Side-finding: nd_verify accepts circular proofs

`box_citable` never checks that a cited box ends before the line that cites it. So a line can cite a later box whose
lines cite it back:

```
THM P SEQ Q PRF N1 P : PR ; N2 ( P > Q ) : IMPI N3 N4 ; N3 | P : AS ; N4 | Q : IMPE N2 N3 ; N5 Q : IMPE N2 N1 ; QED
```

This invalid sequent is accepted by the vendored verifier and by `~/repos/nd-takehome` `main`, in 4 body lines.
Every sequent can be "proved" this way, so an RL reward that uses nd_verify alone can be gamed. The Lean path rejects
it: nd2lean raises `forward cite`.

## How the bound works

- **Reinterpret one subformula.** Pick a subformula X of the sequent and reinterpret it with a fresh atom z,
  everything else classical. For example, `I(A>B) := (¬A ∨ B) ∨ z`.
- **Only rules on X can break.** Each rule's soundness rests on the truth clause of one principal formula. Under the
  reinterpretation, every rule step stays sound except the ones whose principal formula is X.
- **So such a line is required.** If the reinterpreted sequent is invalid by truth table, every proof contains a line
  of that rule on X.
- **Distinct signatures are distinct lines.** A line has at most one (rule, principal formula) signature.
- **The bound** is the number of pairwise-disjoint required sets, plus the distinct hypothesis formulas forced by
  required ORE and IMPI lines.
- **Negation** is first rewritten `~A → (A > F)`, which is how Lean treats it.
- **Robbie's per-scope idea is subsumed.** Disjointness comes from distinct signatures rather than from a box-by-box
  argument. Any shortcut (shared lemma, ex falso, RAA, ORE merging) shows up in the truth tables and removes the
  requirement.

The full argument is in the `sigbound.py` docstring.

### Claimed space

1. Proofs in nd_verify's rule shapes in which every citation points backwards. This covers all genuine Fitch
   proofs and minlen's whole space.
2. Every proof the Lean gate accepts (nd2lean at 51604b3, then Lean).

Codex found the Lean gate looser than the first containment claim:

- NEGE becomes arbitrary function application;
- a cited box's parent context is not checked;
- unused boxes are dropped.

The Lean case was re-argued over the emitted Lean term instead. That re-argument has not been re-reviewed.

## Validation

There were zero violations in every check.

| check | cases | bound / exact |
| --- | --- | --- |
| production files vs written proofs | 48,878 | S/B 0.99 (median 1.00), C 0.966, D 0.884 |
| `tight.jsonl` exact minima 20–24 | 3,701 | 0.979; equal on 2,166 |
| small E variants, Lean-checked exact | 833 | 0.92–0.97 |
| planted shortcuts | 3,971 labelled | collapses to ~0.05 on ex falso and conclusion shortcuts |
| random sequents | 7,999 | ~0.41 |

- **Kill table:** brute-forced over 207,669 rule instances. Three deliberately wrong entries were all caught.
- **Real proofs:** on 72,697 proofs, every required set is hit by an actual line.
- **Lean:** all proofs used were Lean-accepted: 69,494 production proofs, 3,220 `e_raw` and 1,000 small E variants.

## e_raw results

| written | n | ≥ 20 | ≥ 30 | ≥ 40 | bound min / median / max | bound / written |
| --- | --- | --- | --- | --- | --- | --- |
| 35–39 | 350 | 350 | 350 | 0 | 32 / 36 / 39 | 0.975 |
| 40–49 | 700 | 700 | 700 | 631 | 36 / 43 / 49 | 0.971 |
| 50–59 | 700 | 700 | 700 | 700 | 44 / 52 / 59 | 0.962 |
| 60–69 | 700 | 700 | 700 | 700 | 54 / 61 / 69 | 0.952 |
| 70–80 | 770 | 770 | 770 | 770 | 62 / 71 / 79 | 0.942 |

- **Cumulative:** ≥ 40: 2,801, ≥ 50: 2,051, ≥ 60: 1,281, ≥ 70: 481.
- **Exact:** 469 records have bound equal to written length.
- **On B's output:** the bound alone gives ≥ 20 on 12,184 of 12,541 b2 records, 9,447 of them exact.

## Limits

- **Slack:** a required line is counted once, even where every proof needs it in several boxes. This mainly
  affects recycled hypothesis atoms. BOTE lines are not counted.
- **Soundness rests on:** the hand proof, the brute-force table, the real-proof test and one Codex review.
- **New rules:** a rule outside nd_verify's set would need a new table row.
- **nd_verify:** this folder only uses nd_verify's parser. Older code relies on nd_verify itself (`certify_fast`'s
  proof check, `certify.check`, `gen_long.py`).

## Files

| file | what |
| --- | --- |
| `sigbound.py` | the bound, with the argument in its docstring |
| `run_e.py` | runs the bound over a jsonl; output in `results/e_raw_sig.jsonl` |
| `test_sigbound.py` | kill-table brute force and the real-proof test |
| `validate.py`, `fuzz.py`, `adversarial.py` | the empirical checks |
| `lean_check.py` | the Lean wrapper |
| `results/` | all logs, including `codex_review.log` |
