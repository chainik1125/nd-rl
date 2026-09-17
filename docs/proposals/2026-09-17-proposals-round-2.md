---
written_on: 2026-09-17
written_by: agent:claude
---

# Five experimental proposals after the novelty campaign

Written as the proposer role the orchestration proposal describes, until that role is
automated. Each is scoped to run from a brief on the existing infrastructure. Evidence base:
the [novelty campaign summary](../../experiment-summaries/2026-09-15-novelty-campaign/README.md)
and its reviews. The standing finding they build on: expert iteration crosses zero
pretraining coverage for a structural repetition of a learned move (a deeper box), amplifies
rare base-model generalisations for rule sequences, and ignites or not per Stage-1 draw
according to the pattern's pre-RL base rate — except for structural patterns, where three of
five zero-rate draws ignited anyway.

## 1. Cast natural deduction into Lean; measure novelty by model scale (Dmitry's idea)

**Question.** Does writing the same proofs as Lean 4 change what a pretrained code model can
do in context, and can "the smallest model that proves it in Lean" serve as a novelty
measure that is comparable across projects — unlike our base-model log-probabilities, which
are specific to one 3M-parameter model?

**Design.**
1. `nd2lean.py`: a deterministic translation of the ND proof format (spec.md) into Lean 4.
   Atoms become `Prop` variables; premises become hypotheses; each rule becomes a term or
   tactic step (`ANDI` → `And.intro`, `IMPE` → application, `IMPI` → `fun h => …`/`intro`,
   `ORE` → `Or.elim`, `NEGE` → application to `False`, `NEGI` → `fun h => …` at `¬`, `BOTE` →
   `False.elim`, `DN` → `Classical.byContradiction` or `not_not.mp`). Fitch boxes become nested
   lambdas. Check the translation on every proof in the take-home pools: `nd_verify` accepts
   ⇔ Lean accepts (`lake env lean`), with a report of any disagreement. This gives the project
   an independent second checker for free.
2. **v0 in-context test** (Dmitry's suggestion): Qwen3-Coder, 20 worked examples in context,
   one fresh theorem, greedy plus 8 samples; three surface forms — Lean 4, the take-home's
   token format, and a plain-English rule-name format as a control; validation-36 plus 200
   transfer theorems stratified by generating length; 5 example-set draws so the interval is
   real. Report accuracy by form and by length, and the paired delta Lean − tokens.
3. **Novelty by scale.** For every class of RL-found proof from the campaign (depth-3 from
   f = 0, reductio from f = 0, the 9-line transfer proofs), the smallest model in one family
   (Qwen3 0.6B → 32B, same prompt, same k) that proves the theorem in Lean. Compare the
   ordering with our log-probability ordering; where they disagree, look at the proofs.

**Prediction.** Lean helps a code model materially (≥ 10 pp) on theorems whose proofs are
nested-box heavy, because Lean's lambda nesting is a shape the model has seen millions of
times, and helps little on flat proofs. Scale-of-first-success will correlate with base
log-probability across proof classes but rank depth-3 proofs as *easier* than their
log-probability suggests, for the same reason.

**Cost.** One A100 pod with vLLM for Qwen3-Coder-30B-A3B (~$1.6/h, ~6 h), or hosted API
calls; Lean toolchain is CPU-only. ~$15 of pods. Needs: a decision on local vs hosted Qwen
(see questions), and whether Dmitry already has an ND → Lean casting to reuse.

## 2. Test the "structural repetition vs rule sequence" rule of thumb on new patterns

**Question.** The campaign's one-line conclusion is a hypothesis with two data points per
class. Does it survive four more patterns, chosen before the data is seen?

**Design.** Pre-register the class of each pattern, then run the f = 0 dial (two seeds) with
the campaign's machinery:
- structural: box depth 4 from data capped at depth 3; `IMPE` chains of length ≥ 5 from data
  with chains ≤ 3; three-way nested `ORE`;
- rule sequences: `IMPI` whose box contains an `ORE` (case split inside a conditional proof);
  `NEGI` whose box uses `ANDE` on the hypothesis; `ORI` immediately consumed by `ORE`.
Also add the measurement the ignition review asked for: the pattern's base rate on the
intermediate checkpoints (rounds 1–4) of zero-hit draws that later ignite, to see whether the
prior drifts before the first proof.

**Prediction.** Structural patterns acquired from f = 0 in ≥ 6 of 8 arms; rule sequences
≈ 0 except where a base draw generalised. A single clean counterexample kills the rule.

**Cost.** ~$40 on three pods; two days of wall clock.

## 3. How little outside data ignites an arm (quantifying the precursor idea)

**Question.** Sibling transfer ignited 11/11 dead arms with one training step. What is the
minimum, and does it have to be the pattern itself?

**Design.** From the round-4 state of each non-igniting arm (checkpoints exist): inject
(a) 1, 4, 16 sibling proofs; (b) 4 *generator-made* ≤ 6-line proofs containing the pattern
on unrelated easy theorems (allowed: generator data, cap 6); (c) 4 proofs of a *different*
pattern; (d) 4 proofs with the pattern's surface tokens but no valid structure (control).
One training step, then EI as usual; report ignition and plateau.

**Why it matters.** This is the sprint's "precursor curriculum" measured: if (b) works, the
recipe for a textbook schema is "a handful of generator-made short instances of its shape",
which the generator can produce on demand.

**Cost.** ~$20; one day.

## 4. Does GRPO ignite from zero? Algorithm and scale as variables

**Question.** Everything so far is expert iteration. The sprint used group-relative policy
gradient and saw nothing. Is "structural patterns cross zero coverage" a property of EI's
keep-any-success rule, or of RL on a verifier in general?

**Design.** On the depth-3 f = 0 sets (four exist): GRPO with the sprint's settings (groups of
8 and 32, binary reward, fixed loss divisor, no KL) vs EI at matched sample budgets, two seeds
each; record the fraction of groups with reward variance per round — the quantity the sprint
diagnosed — alongside acquisition. Then one arm on the sprint's own 85M model and
relative-unary codec, if its code can be run from the take-home pools.

**Prediction.** GRPO with groups of 8 does not ignite from a zero base rate (no variance to
learn from); with groups of 32 it ignites on the draws EI ignites, later; the 85M model
ignites earlier.

**Cost.** ~$30; needs the sprint's `nd_grpo.py` path or a reimplementation (~200 lines).

## 5. Target pools that require a pattern (fixing the campaign's open flaw)

**Question.** Both reductio and cap-8 derived-ORE dials were uninformative because no
target *required* the pattern. Can the bounded prover become a necessity oracle?

**Design.** Extend `minlen.py` with rule restrictions: a theorem requires `DN` if the search
without `DN` fails to bound 10; requires an `ORE` over a derived disjunction if the search
that forbids `ORE` on non-premise lines fails to bound 10 while the unrestricted search
succeeds. Build 300-target pools per pattern from the existing candidate files, verify by
hand on ten, rerun the two dials with two seeds at f = 0 and f_max.

**Prediction.** With necessity enforced, reductio f = 0 stays at 0 in every draw whose base
rate is 0 and the cap-8 derived-ORE dial rises with f. If the derived-ORE dial stays flat
even when the pattern is required, expert iteration cannot acquire it at this budget — a
real negative result.

**Cost.** ~$25; the prover extension is CPU work first.

## Order

5 → 2 → 3 are cheap and sharpen the existing claim; 1 opens the Lean track and needs a
decision on Qwen hosting; 4 is the one that speaks directly to the sprint's method.
