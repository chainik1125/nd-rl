---
written_on: 2026-10-14
written_by: agent:claude
---

# Long proofs: certified-hard natural-deduction theorems

Robbie's goal (2026-10-13): 10,000 theorems whose minimal proof is certified to need at least 20 body lines
(lines after the premise block), plus exact-labelled theorems at 10–20 and a roughly uniform spread of lengths
to 50–80, with premise counts spread over 3–10. Data only; training on it is out of scope. Only code is in Git; the certified data is on Hugging Face
(see [DATASETS.md](../../../../docs/DATASETS.md)).

## Status (finished 2026-10-14; counts re-scored 2026-10-18)

Counts from `python certify.py --score production/v2/*_f20.jsonl` (50,289 distinct records):

- **21,046 theorems certified to need ≥ 20 body lines.** Written lengths: 20,263 at 20–29, 719 at 30–39, 60 at
  40–49, 4 at 50+. Premise counts: 3: 1,719, 4: 3,707, 5: 3,820, 6: 3,450, 7: 3,162, 8: 2,565, 9: 1,991, 10: 632.
- **21,882 with an exact minimal length below 20** (`min_body`): 2: 6, 3: 16, 4: 60, 5: 362, 6: 1,801, 7: 332,
  8: 620, 9: 896, then 1,400–2,400 per length for 10–19.
- 7,361 records timed out in the certifier and carry no label.
- **Exact lengths above 20** (stage 3, `production/tighten.py` → `production/v2/tight.jsonl`, certifying at floor =
  written length): 5,760 records at written 20–24, of which 3,531 are tight (minimum = written length).
- **Approach E** (`approach_e_long/`, written 35–80): 3,220 records in `production/v2/e_raw.jsonl`, 70 per written
  length. The exact certifier is too slow there; `certifier_scoped/` gives a signature lower bound
  (`certifier_scoped/results/e_raw_sig.jsonl`), which the long-RL pools use.
- Overlap audit (`production/audit_overlap.py`): 8 of our theorems (all ≤ 7 lines, from the short set) appear in the
  group's files (take-home targets/data, dan/dan data, charles/main jsonl); none at ≥ 10 lines. Sealed sets not checked.

History: the 10,000 goal was met at 03:05 on 2026-10-14 (10,091 certified); B and S kept running after that. At
02:17 oom-guard killed two certify workers (9 workers × 2 GB plus a subagent's 29 GB); B and S were resumed on the
unfinished records (`b_rest`, `s_rest` → `b2_f20`, `b3_f20` (reverse order), `s2_f20`) at `--mem 512`.

## How a theorem is certified

1. `certify.check`: verified with the take-home `nd_verify` (atoms P Q R S T U W X Y Z), no duplicate or uncited
   premises, conclusion not a premise, no F premise, premises jointly satisfiable and sequent valid (truth table).
2. `certifier_exact/certify_fast.py`: a C reimplementation of Dan's `minlen` (vendor/fork/minlen.py, branch
   dan/dan of the take-home fork). Same minimum over the same proof space, ~15,000x faster (median, 12-theorem
   benchmark). `cert_ge=20` means no proof with < 20 body lines exists in minlen's space; `min_body` is the exact
   minimum when one is found below the floor.

What "certified" means, precisely: minimal within minlen's search space (formulas from the theorem's
subformulas, their negations and double negations, plus minlen's rule shapes). Two known limits:
- Proofs using formulas outside that space are not searched (inherited from minlen).
- minlen never tries a NEGI box that is only its hypothesis F (`N1 | F : AS ; N2 ~F : NEGI N1 N1`), so for
  goals needing `~F` it over-counts by one line (found by Codex review, `certifier_exact/CODEX_REVIEW_2026-10-14.md`).

Validation of `certify_fast`: agrees with unmodified minlen on all 521 labelled records from approach D, on
15,500 fuzzed sequents against a literal port, on 300 production labels (0 disagreements) and 100 floor-16
certificates (0 contradicted). Codex reviewed the soundness arguments for its five speedups and found no
cost-increasing flaw. One API bug: `THM P SEQ P PRF` is falsely certified when called directly; `certify.check`
rejects such inputs first, so pipeline outputs are unaffected.

## What was tried (agents' reports are in each folder)

| folder | idea | outcome |
| --- | --- | --- |
| `approach_a_compose/` | chain hard pieces over disjoint atoms | additive for most pairs; certification was the wall |
| `approach_b_backward/` | deep backward generator (case splits over fresh atoms, IMPE chains, packed premises) | written proofs tight; with certify_fast 73/100 certified ≥ 20, premises 3–10 |
| `approach_c_premises/` | premise "gate" DAGs | tight proofs; 29/100 ≥ 20 plus exact 15–19 |
| `approach_d_theorem_first/` | sample theorems, let the search find the proof; wrote `minlen_sem` | exact labels 10–17; tree sampler used for the first overnight run |
| `certifier_exact/` | C rewrite of minlen | the unlock: ≥ 20 certificates in ms–s |
| `certifier_approx/` | restricted-lemma hunters + calibrated miss rates | 0 false certificates on 529 labels; superseded by the exact certifier |

Earlier finding (yield test): Robbie's take-home backward generator tops out at 16 body lines (27 with all knobs
raised); Dan's forward generator writes 80–120-line proofs but nearly all have jointly inconsistent premises or
shortcut to 5–11 lines.

## Production

- `production/out/`: first overnight run (approach D tree sampler, minlen_sem labels at floor 16), stopped at 23:50.
  Relabelled with certify_fast in `production/v2/d_f20.jsonl`.
- `production/v2/b_raw.jsonl` (24,000 from `approach_b_backward/gen_deep.py --seed 2001 --lo 20 --hi 80`) and
  `c_raw.jsonl` (9,000 from `approach_c_premises/gen_c.py --seed 2001 --min_body 15`), certified with
  `certify_fast.py --floor 20 --secs 15 --mem 1024 --proofs` into `b_f20.jsonl`, `c_f20.jsonl`.
- `s_raw.jsonl` (12,000 short theorems from `gen_deep.py --seed 3001 --lo 2 --hi 16 --body 1-19`) → `s_f20.jsonl`,
  `s2_f20.jsonl`: exact minimal lengths 2–19, premises 3–10; written proof was minimal in 5,740 of 5,750 checked.

## Caveats

- B's theorems come from one structural family (implication chains, case splits, conjunction-packed premises):
  distinct up to renaming, but similar in shape.
- Exact lengths above 20 exist only for the 5,760 stage-3 records (written 20–24). Every other certified record's
  minimum is in [20, written length].
- Records are deduplicated by `certify.theorem_key` (atom renaming, premise order; approximate for premises
  with equal skeletons). No audit against the group's evaluation sets yet.
