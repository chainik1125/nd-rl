---
written_on: 2026-10-02
written_by: agent:claude
---

# radical-scoping: radical departures for the capability-emergence question (rule knockout and FOL ranked first; CPU pilots all feasible)

Status: completed and reviewed 2026-10-02. No hard-constraint violation and nothing quarantined. **Every pilot number
reproduces**, and seven rewordings are applied in the nd-rl copy. This is desk scoping plus three CPU feasibility
pilots: **no model was trained or sampled**, so no checkpoint label applies. Every proof counted here is a generator or
hand-written proof, not a model proof. The deliverable is a **draft for Dan**:
[`docs/proposals/2026-10-03-radical-departures.md`](../../docs/proposals/2026-10-03-radical-departures.md) (the reviewed
copy of the executor's `radical_scoping/SCOPING.md`).

Question: which departures from the propositional natural-deduction path could best answer *to what extent RL against a
verifier creates genuinely new capability rather than eliciting rare-but-known behaviour, and what limits it*? The run
scopes 4–6 of them, ranks them by information per dollar and per week, and checks feasibility on CPU. Brief:
[`proposals/claude-heavy/BRIEF_radical-scoping.md`](../../docs/proposals/claude-heavy/BRIEF_radical-scoping.md)
(proposal 21).

## Configuration

- Fork `dan-pandori/nd-takehome`, branch `dan_radical-scoping`. Pre-registration
  `preregistration/radical-scoping.md` (`77e1a795`, committed before the pilot commit `57f7407a`). DONE `613c0432`. Review
  [`review_radical-scoping.md`](https://github.com/dan-pandori/nd-takehome/blob/5a71e1d2/review_radical-scoping.md):
  [`5e826fb4`](https://github.com/dan-pandori/nd-takehome/commit/5e826fb4) §Recount (blind),
  [`5a71e1d2`](https://github.com/dan-pandori/nd-takehome/commit/5a71e1d2) §Compare / §Verdict.
- Executor Fri 2026-10-02 16:52–17:07 UTC; reviewer 17:08–17:21. All work on the VPS (≤ 4 processes). **No pods, $0**
  (`podbudget radical-scoping`: 0.00 h of a $3 / 6 h budget).
- **P1, FOL.** The June sprint's generator and verifier (`radical_scoping/fol_pilot/fol/`, byte-identical to
  `code/experiments/archived/nd_sprints_20260609_11/fol/`) produced 1,000 theorems (`pool1k.jsonl`; seed 0, constants a–e,
  2–16 lines). A new translator, `fol2lean.py`, renders them as Lean 4 core terms. Controls: `neg_control.py` and
  `mut_agreement.py`.
- **P2, knockout.** `p2_move_counts.py` counts rule usage in the Stage-1 sets. `p2_necessity.py` is `minlen` (bound 12,
  20 s) with ORE or DN removed by subclassing, and it runs on `data/bs/textbook72.jsonl`.
- **P3, induction.** `ind_pilot/ind.lean`: a fresh inductive `N` in Lean 4 core (no Mathlib).
- Reviewer code: `review/radical-scoping-recount/` on the fork. It includes an independent FOL renderer `rv_fol.py` and
  exact sequent-calculus labellers `seq_provers.py` (G3c with and without L∨, G4ip).
- Bucket: `hf://buckets/dan-pandori/nd-rl/radical-scoping/radical_scoping`. `pool1k.jsonl` is the record: the generator
  depends on `PYTHONHASHSEED`, which was not recorded.

## Observations

Observed results below use the reviewer's wording where it differs from the executor's.

**P1. Lean 4 core checks the sprint's FOL proofs.**
- Lean accepts **1,000 / 1,000** of `pool1k.jsonl` under the executor's renderer. The reviewer's independent renderer
  also accepts 1,000 / 1,000, and another 1,000 / 1,000 on a regenerated pool. 992 use no axioms; the 8 DN proofs use
  the classical ones.
- **Controls:** 231 / 231 conclusion swaps and 41 / 41 eigenvariable clashes are rejected. On 578 one-line mutants, Lean
  and the sprint verifier agree on 577. In the one disagreement, the verifier's conservative freshness rule rejects a
  valid proof and Lean accepts it, correctly.
- **The reviewer's own mutants:** 943 / 994 rejected. All 51 accepted are valid (49 end in ORI1, 2 in BOTE).
- **The pool cannot test open-ended instantiation yet.**
  - The sampler *cannot* emit ∃-elim or ∨-elim: neither is in its rule list, and 0 / 1,000 proofs use them.
  - There are no function symbols.
  - The pool is shallow: line quartiles 3 / 4 / 5, max 16; inlined term size mean 4.3, max 12.
- Pre-registered P1 expectation (≥ 90 % Lean acceptance, generator revived in ≤ 1 h): **met**.

**P2. A rule knockout is a clean filter in `lean_seq` / `lean_staten`.**

| rule | `data/p2/train_depth3_f0_a1.jsonl` (155,000; the set the 3.2 M `lean_seq` and the 9.56 M best-cap6 `lean_staten` models were trained on; reviewer) | `data/train.jsonl` (154,990; the set the executor counted) |
|---|---|---|
| ORE (`Or.elim`) | 2,256 = 1.46 % | 2,277 = 1.47 % |
| DN (`Classical.byContradiction`) | 12,905 = 8.33 % | 12,950 = 8.36 % |
| classical reductio | 6.80 % | 6.81 % |
| NEGI | 9.93 % | 10.37 % |
| BOTE | 1.95 % | 1.90 % |

- Removing every ORE proof leaves 98.5 % of the set. The executor mislabelled which set the models used; the numbers
  move ≤ 0.05 pt, but pilot A must filter `train_depth3_f0_a1`.
- In the `lean_seq` grammar, `Or.elim` and `Classical.byContradiction` are the only ∨ eliminator and the only classical
  constant. In free-form Lean, `Or.casesOn`, `Or.rec`, `Or.resolve_*`, `Classical.em` and `Decidable.em` would also have
  to be blocked.
- **textbook72 needs the moves.** With bounded `minlen` labels (re-run identical row for row), the search proves 51 / 72:
  14 require ORE, 9 require DN, 1 both. For the other 21, no proof was found (6 of them timed out).
- **Exact, unbounded labels** (the reviewer's G3c / G4ip deciders): **21 / 72 need ORE, 19 / 72 need DN, 4 both**. They
  agree with the bounded labels on all 51 decided problems. So P2 is **stronger than written**.
- Renaming-class overlap with training: 0 / 72 (`train_depth3_f0_a1`) and 1 / 72 (`data/train.jsonl`).
- Pre-registered P2 ranges (ORE 1–10 %, contradiction 5–25 %, ≥ 70 % left, ≥ 1 requiring family): **all met**.

**P3. Induction over a custom `N` in Lean 4 core.**
- `z_add`, `s_add`, `add_comm` and `add_assoc` check with no axioms. `simp [add]`, `omega` and `decide` fail on `add_comm`.
- **Reviewer's extension:**
  - `induction n <;> simp_all [add]` proves `z_add` and `s_add` outright, so "the core library cannot trivialise the
    domain" is **overstated**. A tactic grammar for C must exclude `simp_all`.
  - `add_comm` has a one-theorem nested-induction proof with no lemma statement, so `ind.lean`'s comment "lemma
    invention is necessary" is false as written.

**The ranking (the executor's judgment, which the pilots make feasible but do not test).**
1. A. Rule knockout × exploration: `Or.elim`-free Stage-1, ε 0 vs 0.05 legal-action exploration, 3 seeds, ≈ $15.
2. B. FOL with function symbols: ≈ $15–20.
3. C. Induction arithmetic: ≈ $20–30, 5–8 Claude-days.
4. D. Library mining as a measure: $0.
5. E. EDL as a measure: ≈ $3.
6. F. Self-play conjecturing: dropped and folded into C.

The recommendation is A now, as proposal 20's target, and B's build in parallel. The ranking matches the
pre-registration.

**Rewordings the review required** (applied in the nd-rl copy, marked R1–R7):
- R1: the P2 counts named the wrong training set.
- R2: A's null is near-structural, not structural, so measure it. A softmax LM gives the never-targeted `Or.elim` token a
  non-zero probability. K0+ε tests *adoption* of a move supplied by a hand-built proposer, not unaided creation.
- R3: `minlen` is not complete, and the subformula property holds for normal proofs only.
- R4: §C's trivialiser and lemma-necessity claims (above).
- R5: the FOL sampler cannot emit ∃-elim or ∨-elim at all.
- R6: inherited numbers.
  - `round2-run5`'s reductio stays at 0 / 300 on 2 of 3 draws; s0 ignited to 0.170.
  - The 25 unsolved textbook72 problems belong to SN-cap12 (3.2 M `lean_staten`).
  - The sprint pass@8 and the `frontier-supply` numbers get model labels, and the token-format numbers from before
    2026-09-27 get their checker.
- R7: 6 of the 21 "no proof" problems timed out.

**Compute.** No GPU. The Lean checks took seconds on the VPS. No compute rows were recorded, and the reviewer says none
are needed.

## Limitations and next step

- **Not supported as worded:** nothing central. The claim closest to unsupported is A's "structural null", which is the
  premise of the recommended pilot (R2).
- The pre-registered ranges were wide, so "every expectation held" is weak evidence (reviewer).
- The bounded necessity labels depend in principle on `nd_verify`, through `minlen`'s self-check. That path never fired
  (0 errors in 216 searches), and the exact labels do not use it.
- Next measurement (the reviewer's, ≈ $1, before pilot A's ≈ $15):
  - Train one ORE-free best-cap6 Stage-1 on `train_depth3_f0_a1`, filtered (1,200 A40-s ≈ $0.16).
  - Report the frozen model's `Or.elim` probability at every legal position.
  - Report its pass@4,096 on the 21 exact-ORE textbook72 problems plus a `necessity.py` ORE-required pool.
  - If the pass rate is > 0, A needs a frozen-at-equal-attempts comparison, not "0 vs > 0".
- For B, before any GPU spend: probe on CPU whether adding EXE / ORE / function symbols to the sprint generator yields
  ≥ 1 % multi-instance proofs.
- The fork branch `dan_radical-scoping` is not merged into the fork's `dan` (documents and pilot scripts only).
