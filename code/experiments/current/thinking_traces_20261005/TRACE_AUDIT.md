---
written_on: 2026-10-05
written_by: agent:claude
---

# Audit of the programmatic thinking traces (`prog_traces.jsonl`)

**Scope.** `/scratch/cowifornia/nd-rl/think/traces/prog_traces.jsonl`: 154,977 records, `progtrace/1`, seed 0. The
generator is `code/thinking/progtrace.py` on branch `leon/thinking` (worktree `/scratch/cowifornia/nd-rl/think/nd-rl`,
HEAD `673e763`). That file is uncommitted; I audited the copy on disk, last modified 23:08:32 UTC. Sections A–E are
covered. Section F (the LLM traces) was dropped at the coordinator's request, so no LLM trace was audited. Everything
here was read-only, CPU-only and at most 16 processes. Scripts and raw outputs are in this folder (list at the end).

## Verdict: usable after listed fixes

- **Correctness: usable as-is.** Every hard check passed on every record. No trace prompt is in the renaming class
  of any eval set.
- **Informativeness: fix findings 1 and 2 before using this as the "thinking carries checked work" arm.**
  - About 63% of think tokens re-render the reference proof top-down: the plan lines plus the reference candidate.
  - The candidate `<no>` verdicts can be predicted by string-matching each candidate against the plan line, with no
    type checking.
  - 72% of all dead-end reductions are the same trivial ex-falso move.
  - Interpretation: a 9.5M model can fit most of the think tokens by plan-copying plus ritual. That is the r1_cot
    failure mode PROPOSAL.md §0 warns about.
  - The traces are still a correct, leak-free cold-start corpus for the format.

## Findings, ranked by severity

Percentages are of the stated denominator. "Decision blocks" means decision steps with a non-empty think: 491,506 of
546,592 decision steps.

### 1. HIGH: candidate checks are post-decision ritual, predictable from the plan by string match

**Observed:**
- The verdict rule, over the 584,720 candidates in decision blocks that come after at least one plan line:
  - The rule is "a candidate that literally matches the latest goal-reduction line is accepted; otherwise `<no>`".
  - A matching candidate is accepted in **272,236 of 272,236 (100%)**.
  - A non-matching candidate is rejected in 281,195 of 312,484 (90.0%).
  - That makes the rule **94.7% accurate** without reading a single formula.
  - This is a lower bound. Many of the misses are the DSL shorthand `Classical.byContradiction n3`, which stands for
    the action's `( fun hh => n3 hh )`.
- Order: 281,195 of the 322,871 rejected candidates (87%) are written after a plan line has already stated the
  reference reduction.
- The answer is visible before the action:
  - The action equals the term of the last plan line in 87.4% of decision blocks that have a plan (429,186 of
    490,821).
  - It equals the term of some plan line in 94.7%.
  - It appears verbatim as a candidate line in 61.9% of decision blocks.
- Token shares (all think lines):
  - plan lines plus the reference candidate: **62.6%**;
  - goal-level alternatives and dead goals: 15.2%;
  - wrong candidates, looks and `<no>`: 22.2%.

**Interpretation:**
- The decision is made in the plan, a deterministic top-down rendering of the K12 reference.
- The wrong candidates (swap, flip, wrong argument and so on, see finding 4) are decoys written after the decision.
- The looks "showing why" are not needed to predict `<no>`.

Verbatim (`k12_46270`, step 2): the plan names the answer, then ex falso is dismissed, then the swapped decoy is
rejected.
```
|- ( ~ ( F v P ) ) := n2 n1
|- ( ~ ( F v P ) ) := |- F .elim <no>
have n3 : ( ~ ( F v P ) ) := n1 n2 ;
n1 : ( R & ( R v R ) )
<no>
>>> have n3 : ( ~ ( F v P ) ) := n2 n1 ;
```

### 2. HIGH: ex falso dominates the dead ends, and the stats file's dead-end fractions overstate real search

**Observed:**
- Dead reductions:
  - **192,513 of 268,310 dead reductions (71.8%) are `⊢ G := ⊢ F .elim`**.
  - Of the 164,677 `⊢ X <no>` goal lines, 112,660 (68.4%) are `⊢ F <no>`.
  - The line `|- F <no>` occurs 101,120 times. It is the second most common line in the corpus, after `<no>`. It
    appears in 19.7% of decision blocks.
- Predictability: an ex-falso reduction is dead in 93.8% of cases (192,513 of 205,208). It is live only in
  inconsistent contexts. So "write `<no>` after ex falso" is 93.8% right with no reasoning at all.
- Per decision block:
  - **142,111 of the 205,377 decision blocks with any dead alternative (69.2%) contain only the ex-falso one.**
  - Only **63,266 of 491,506 decision blocks (12.9%)** contain a non-ex-falso goal-level dead end.
  - The stats file reports `decision_fractions.dead_end 0.418` and `backtrack 0.364`; these are mostly ex falso.
- Cause, in code: `alts_before` runs `if efq and (not dead or rng.random() < cfg.p_efq)`. Ex falso is therefore
  always inserted when no real dead alternative exists. `p_efq = 0.25` applies only when one does exist.
- Context:
  - 205,468 of 546,592 decision steps (37.6%) are in inconsistent contexts, where no goal-level `<no>` is possible.
  - Exploration that is neither ex falso nor plan, i.e. a non-ex-falso dead alternative or a rejected candidate,
    occurs in 266,580 decision blocks (54.2%).
  - 129,368 decision blocks (26.3%) are plan, reference candidate and looks only.

### 3. MEDIUM: working alternatives are abandoned with no reason, sometimes the better route

**Observed:**
- 36,143 decision blocks (7.4%) show a viable reduction and then silently restate the goal. This is by design ("no
  claim").
- In inconsistent contexts `p_viable` is doubled.

**Interpretation:** this teaches the model to drop a correct plan for no stated reason.

Verbatim (`k12_151622`, step 2; `n3 : ( ~ ( ~ ( R v S ) ) )` is in scope, so the third line closes the goal at once):
```
|- ( R v S ) := Or.inl |- R
|- R <no>
|- ( R v S ) := Classical.byContradiction |- ( ~ ( ~ ( R v S ) ) )
|- ( R v S ) := Or.inr |- S
|- S <no>
|- ( R v S ) := Or.elim n1 ( fun ( n5 : ( ~ ( R & Q ) ) ) => |- ( R v S ) ) ( fun ( n6 : S ) => |- ( R v S ) )
have n4 : ( R v S ) := Or.elim n3 ( fun ( n5 : ( ~ ( R & Q ) ) ) => by
n3 : ( ~ ( ~ ( R v S ) ) )
<no>
>>> have n4 : ( R v S ) := Or.elim n1 ( fun ( n5 : ( ~ ( R & Q ) ) ) => by
```

**Related (observed):** the traces rationalise roundabout K12 references.
- There are 12,207 decision steps where the goal is already an n-hypothesis in scope but the reference does
  something else. 1,747 of those thinks never mention that hypothesis.
- Example: `k12_5131` step 7 plans a second, redundant `Or.elim n1` while `n5 : P` (the goal) is in scope. The
  redundancy is in the reference (`N11 ORE N1 …; N10 R N7`), so this is not a generator bug.

### 4. MEDIUM: wrong candidates rarely come from the base model's real mistakes

**Observed:**
- Base-model distractors (waveA plus waveB, 31,326 in total) exist for only **12,063 proofs (7.8% of traces)**. All
  other rejected candidates are synthetic mutations of the reference.
- 2,360 of the 31,326 distractors (7.5%) cite names far outside the state, e.g. `<| n2 , n40 |>` (k12_120431 step
  10) or `n62 n1`. In a 40k-trace sample, 681 rejections are this "unbound n-name" kind.

**Interpretation:** the dead ends are not aimed at the model's real failure distribution, which was PROPOSAL §3.2's
intent. This is a design gap, not an error.

### 5. LOW: some rejections are environment conventions, not logic

**Observed:**
- 9,203 rejected premise candidates fail only `PR position`: premises restated out of order, which Lean would accept.
- In a 40k-trace sample, 228 rejected `exact n` candidates have `n : goal` and fail only the environment's
  "exact not last" rule.
- All rejections are true for the environment, as the DSL defines `<no>`.

**Interpretation:** they teach rules of the environment, not of logic.

### 6. LOW: think length is driven by formula restatement

**Observed:**
- 22.1% of all think tokens (10.3M of 46.7M) restate the state's own goal (`⊢ GOAL` line prefixes).
- 67.8% of think tokens are formula tokens.
- Per decision block, think length correlates with goal-formula length (Spearman ρ = 0.48; sample of 24,705 blocks)
  more than with the number of non-plan reductions explored (ρ = 0.33).
- The longest trace, `k12_120431` (1,904 think tokens), is mostly `( ~ ( ~ S ) )` repeated.

**Interpretation:** the "length tracks difficulty" signal is partly formula size.

### 7. LOW: smaller oddities

**Observed:**
- Premise-name citations: 7,061 reduction lines at premise steps cite `h`-names in skeletons, e.g.
  `Or.elim h3 ( … )` in `k12_133308`. Lean accepts this. The action language cannot do it: the environment treats
  `h` as unbound outside restatements.
- Off-target thinks at forced steps:
  - Premise-step thinks (19,579, 7%) plan the top goal. The forced restatement action does not use that plan.
  - Exact-step thinks (34,901, 8.5%) are always of the form "wrong `exact` + look + `<no>`".
- Plans that stop short (`plan_truncated` at the 240-token budget): the reference candidate then appears with no
  plan line leading to it. In `k12_120431` step 3, `Or.elim n4` appears after a plan that stops at
  `<| n2 , ⊢ X |>`.

**Interpretation:** the last of these blocks reads as unexplained.

### Not a problem (checked)

- Candidate placement is randomised:
  - the first candidate is the reference in 57.9% of blocks with candidates;
  - the last candidate line equals the action in 58.3%;
  - the reference index is spread over all positions.
- `<no>` position: the first `<no>` is on line 1 in 37.8% of decision blocks with a `<no>`, otherwise spread over
  lines 0–6.
- Shape diversity: 18,570 distinct line-kind shapes. The top shape (a single plan line) covers 10.4% of decision
  blocks.
- The reference is the last top-level reduction in 87.1% of blocks with at least 2 top-level reductions. The model
  cannot use that left to right without also knowing when to stop.
- Empty-block fractions match the docstring and the stats:
  - decision 10.1%;
  - premise 93.05%;
  - exact 91.5%.

## Checks that ran

**A. Hard correctness, all 154,977 records** (`a_hard.py`, 16 processes, 56 s): **0 failures.**
- Every id maps to its K12 row, and the prompt and `n_lines` match.
- Every action sequence equals `decompose(prompt, proof, canon=True)` exactly.
- `trace_dsl.validate(strict=True, lean=False)` passes on all, with 0 soft findings.
- Tokenization:
  - every think and action token is in the `lean_thinkn` vocabulary (114);
  - every name is within h1–h8 and n1–n64;
  - `data.trace_pairs` encodes and decodes each block exactly;
  - no empty think lines.
- Maximum pair length is **594** tokens, including `<eos>` (≤1,024). Histogram is in `a_hard.json`.
- There are no duplicate ids. 23 K12 rows have no trace; these are the generator's eval exclusions (22 heldout, 1
  textbook_dev).

**Lean** (`a_lean.py`):
- **300 of 300** random assembled proofs are accepted.
- **384 of 384** sampled candidates that the traces reject for a step_check reason are also rejected by Lean. All 13
  reason types are covered, including 120 `.elim` cases, which are the risky ones. So none of them teaches a `<no>`
  on a valid step.
- Harness control: 92 of 92 reference steps, built the same way, are accepted.

**Reproducibility:** 1,500 of 1,500 random records regenerate byte-identically from the current, uncommitted
`progtrace.py` with the current distractor files.

**B. Independent semantic check, all 545,986 non-empty think blocks** (`b_indep.py`):
- What is independent of `trace_dsl`:
  - my own tokenizer, formula parser and truth tables;
  - scope parsed from the state tokens;
  - the skeleton parser and bidirectional type checker;
  - anchoring and binder rules;
  - flagged-hole bookkeeping;
  - the commit rule;
  - my own Lean-style typing of atomic terms (`¬A ≡ A → False`).
- What is borrowed as ground truth: Dan's `Env` and `step_check`.
- Coverage: 1,084,115 reductions, 164,677 dead-goal lines, 670,810 candidates and 337,709 looks.
- **0 disagreements.** Specifically:
  - every goal `<no>` has a countermodel;
  - every `<no>` reduction is well-typed with a dead hole (0 ill-typed);
  - every unmarked reduction type-checks, and every dead hole is later flagged;
  - every rejected candidate fails, by env 38.3%, step_check 58.6% or a dead goal 3.1%;
  - every accepted candidate passes and equals the action;
  - the commit rule holds.
- Mutation cross-check (`b_mutate.py`, seed 7): 4,000 one-line mutants, checked by both the validator and my
  checker.
  - The two agree on 3,956: 3,507 both reject, 449 both accept.
  - The other 44 are all the validator's canonical-name rule for candidates, which my checker does not model.
  - Survivors that both accept (sample inspected) are true claims: renamed fresh binders, and one failing candidate
    turned into another.
  - An earlier run with the same design gave 3,948 agreements and 52 splits, all of the same kind.

**C. Statistics, all records:** `c_stats.py`, `c_more.py` and `e_more.py`, with numbers as above. Full tables are
in `c_stats.json`.

**D. Eval leakage, all 154,977 prompts** (`d_leak.py`): **0 hits** on each of four checks.
- The checks:
  - `data.EvalIndex(DEFAULT_EXCLUDE + textbook_train)`;
  - exact prompt match;
  - my own atom-renaming class key (minimum over the 24 permutations of P Q R S);
  - the same key with premise order ignored.
- The sets:
  - five OOD sets (840 items);
  - textbook_dev (58);
  - heldout (5,000);
  - textbook_train (14).

**E. Reading:** 12 whole traces, from 2 to 12 lines, including the longest (`k12_120431`); see
`e_show_{1,2,3}.txt`.

Reading reasoning quality, the plans are sensible and every claim is true. The genuine work is in the non-ex-falso
dead ends and box lookahead. Good examples, verbatim:

```
k12_111576 step 1 (real dead end, then backtrack)
|- ( ( R > S ) v ( ( P v R ) > ( ~ P ) ) ) := Or.inr |- ( ( P v R ) > ( ~ P ) )
|- ( ( P v R ) > ( ~ P ) ) <no>
|- ( ( R > S ) v ( ( P v R ) > ( ~ P ) ) ) := Or.inl n1
have n2 : ( ( R > S ) v ( ( P v R ) > ( ~ P ) ) ) := Or.inl n1 ;
```
```
k12_124343 step 2 (two levels of real search before the plan)
|- ( ( P v ( ~ P ) ) v S ) := Or.inr |- S
|- S <no>
|- ( ( P v ( ~ P ) ) v S ) := Or.inl |- ( P v ( ~ P ) )
|- ( P v ( ~ P ) ) := n2 |- Q
|- Q <no>
|- ( P v ( ~ P ) ) := Or.inl |- P
|- P <no>
|- ( P v ( ~ P ) ) := Or.elim n1 ( fun ( n4 : ( P v ( ~ P ) ) ) => n4 ) ( fun ( n5 : Q ) => |- ( P v ( ~ P ) ) )
```
```
k12_133308 step 7 (inconsistent context, n4 : F; candidate refuted by a look)
|- ( ( ( ~ S ) v ( P & Q ) ) v P ) := n4 .elim
have n7 : ( ( ( ~ S ) v ( P & Q ) ) v P ) := n1 .elim ;
n1 : P
<no>
have n7 : ( ( ( ~ S ) v ( P & Q ) ) v P ) := n4 .elim ;
```

Bad or weird examples, verbatim:
- `k12_46270` step 2 (finding 1) and `k12_151622` step 2 (finding 3), quoted above.
- `k12_120431` step 2: the plan stops at a hole, then an unrelated statement from the reference appears with no plan
  line leading to it:

```
|- ( ( ~ ( ~ S ) ) & ( ( ~ ( ~ S ) ) & ( ( ~ ( ~ S ) ) & ( ~ ( ~ S ) ) ) ) ) := |- F .elim
|- F <no>
|- ( ( ~ ( ~ S ) ) & ( ( ~ ( ~ S ) ) & ( ( ~ ( ~ S ) ) & ( ~ ( ~ S ) ) ) ) ) := <| n2 , |- ( ( ~ ( ~ S ) ) & ( ( ~ ( ~ S ) ) & ( ~ ( ~ S ) ) ) ) |>
have n4 : ( ( ( ~ F ) > P ) v ( ( ~ ( ~ S ) ) & ( ~ ( ~ S ) ) ) ) := Or.inr n3 ;
n3 : ( ( ~ ( ~ S ) ) & ( ~ ( ~ S ) ) )
>>> have n4 : ( ( ( ~ F ) > P ) v ( ( ~ ( ~ S ) ) & ( ~ ( ~ S ) ) ) ) := Or.inr n3 ;
```

A further ritual example (observed): in `k12_143519` step 4, `|- ( S > Q ) <no>` is established. After the plan has
already chosen `n2 n4`, the block re-tries `have n5 : ( S > Q ) := ( fun ( n6 : S ) => by` / `n6 : S |- Q <no>` /
`<no>`, re-deriving the same dead end.

## Not checked

- **F, the LLM traces:** out of scope per the coordinator. The 104 accepted records were snapshotted to
  `llm_snapshot.jsonl` before the scope change but were not audited.
- **Causal use of the think block:** needs training and a GPU (PROPOSAL §4, measure 5).
- **Whether the synthetic wrong candidates match the base model's error distribution:** needs model samples.
- **Lean coverage:** Lean was run on 300 traces and 384 rejected candidates, not all of them. Semantic dead ends are
  certified by truth tables (my own and the validator's, in agreement). Truth tables are complete for classical
  propositional logic over P Q R S.
- **Tier C ("hard in-distribution", transfer pool):** not yet mined, and not in `DEFAULT_EXCLUDE`. It must be added
  to the exclusion list when it exists.

## Files (this folder)

| What | Files |
|---|---|
| A: hard checks | `a_hard.py`, `a_hard.json`, `a_hard_fail.jsonl` (empty) |
| A: Lean | `a_lean.py`, `a_lean.json` |
| B: independent check | `b_indep.py`, `b_indep.json`, `b_disagree.jsonl` (empty), `b_feat.jsonl` (per-line features) |
| B: mutation test | `b_mutate.py`, `b_mutate.json`, `b_mutate_disagree.jsonl` (the 44 splits plus a sample of both-accept mutants) |
| B: convention-only rejections | `b_conv.py`, `b_conv.out` |
| C: statistics | `c_stats.py`, `c_stats.json`, `c_more.py`, `c_more.json` |
| D: leakage | `d_leak.py`, `d_leak.json` |
| E: reading | `e_show.py`, `e_show_{1,2,3}.txt`, `e_more.py`, `e_more.json` |

## Addendum: generator v2 (written by the coordinating agent, 2026-10-05)

`progtrace.py` now defaults to profile v2 (`Cfg()`). `Cfg.v1()` and `--profile v1` regenerate the
audited file byte for byte; this was checked on its first 5,000 records.

v2 changes, each tied to a finding above:
- **Leaf by candidates (finding 1).** When wrong candidates exist, with probability 0.7 the plan stops one level above the
  action, and the action is decided by candidates. When that happens:
  - with probability 0.35 there is no wrong candidate;
  - otherwise the wrong candidates come first and the reference comes last;
  - the passing candidate gets a check with probability 0.7.
- **No decoys after the plan (finding 1).** Once the plan has named the action, no wrong candidates follow.
- **Ex falso only at random (finding 2).** It is shown with probability 0.15. It no longer fills every block that has
  no other dead end.
- **No silently dropped reductions (finding 3).** `p_viable = 0`, so no working reduction is abandoned without a reason.
- **No convention-only rejections (finding 5).** Rejections that fail only `PR position` or `exact not last` are
  skipped.
- **No premise-step thinks (finding 7).**
- **More real dead ends.** `p_alt_top` and `p_alt_deep` go from 0.6 / 0.4 to 0.85 / 0.6.

Re-run of this audit's own `a_hard.py`, `b_indep.py` and `c_stats.py` on
`/scratch/cowifornia/nd-rl/think/traces/prog_traces_v2.jsonl` (scripts and outputs in `traces/audit_v2/`):

| | v1 | v2 |
|---|---|---|
| failing records (A) / disagreements (B) | 0 / 0 | 0 / 0 |
| "matches plan ⇒ accept, else `<no>`" accuracy | 94.7% | 51.7% |
| action = term of the last plan line (blocks with a plan) | 87.4% | 45.5% |
| verdict after a look line: reject / accept | — | 156,358 / 159,925 |
| first candidate is wrong | 42.1% | 55.5% |
| `\|- F <no>` lines | 101,120 | about 15,000 |
| think tokens per proof, mean | 313 | 238 |
| Spearman ρ, think tokens vs proof lines | 0.76 | 0.75 |
| longest pair, tokens | 594 | 598 |

Still open after v2:
- **Real dead ends are rare in K12 states.** Non-ex-falso goal-level dead ends occur in about 12% of decision blocks
  in both versions.
- **Base-model distractors cover few proofs (finding 4).** They cover 7.8% of proofs, so most wrong candidates are
  synthetic mutations of the reference.
- **Plans that are kept still copy the reference top-down (finding 6).**
