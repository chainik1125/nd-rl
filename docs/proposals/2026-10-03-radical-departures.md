---
written_on: 2026-10-02
written_by: agent:claude
status: draft for Dan (run `radical-scoping`, proposal 21); reviewed 2026-10-02; not queued
---

> **Librarian's note (2026-10-02).** This is the executor's `radical_scoping/SCOPING.md` from fork branch
> `dan_radical-scoping` (`613c0432`), copied after review. The review
> ([`review_radical-scoping.md`](https://github.com/dan-pandori/nd-takehome/blob/5a71e1d2/review_radical-scoping.md))
> found no hard-constraint violation, reproduced every pilot number, and asked for seven rewordings. They are applied
> below and marked *[reviewed: R1…R7]*, with R numbering following the review's §Verdict list. The recommendation
> (A, then B) is the executor's judgment, which the pilots make feasible but do not test. Every proof counted here is a
> generator or hand-written proof, not a model proof. Summary:
> [`experiment-summaries/2026-10-02-radical-scoping-departures-knockout-and-fol-feasible`](../../experiment-summaries/2026-10-02-radical-scoping-departures-knockout-and-fol-feasible/README.md).

# Radical departures for the capability-emergence question — scoping (DRAFT for Dan)

**Draft, 2026-10-02, run `radical-scoping` (proposal 21). Reviewed 2026-10-02 (see the note above).** CPU-only pilots on the VPS; $0 pod spend.
Fork branch `dan_radical-scoping`, directory `radical_scoping/`.

**Question.** To what extent does RL against a verifier create genuinely new capability rather than elicit
rare-but-known behaviour, and what limits it?

## The organising observation

In propositional natural deduction, the formulas a short proof needs are subformulas of the theorem, plus their
negations. That is the subformula property, and `minlen`'s search space is *restricted* to it (its own docstring: no
proof found is not a proof that none exists). The property holds for *normal* proofs, not for every proof: a Lean `have`
can introduce a non-subformula. So every *normal* proof our RL finds is a re-ordering of objects that are already in the
prompt. *[reviewed: R3; was "`minlen`'s complete search relies on it" and "every proof our RL finds"]* On that interface, the evidence so far says RL completes
proofs that pretraining brought within one improbable step:
- `trajectory` (best-cap12, 9.56M `lean_staten`, 3 seeds): the worst step of the eventual proof goes from −6.2 to −1.3 nats.
- `rl-from-ckpt`: no RL start reaches beyond the end arm.
- `textbook72` (SN-cap12 T1, 3.2 M `lean_staten`, a different model from the best-cap12 bullet above): the 25 problems
  that no checkpoint solves stay put. *[reviewed: R6; the model was unlabelled]*

Pure sampling cannot create anything outside this space; it can only make rare things common. There are two ways to
get a test where the answer is not "already in support":

1. **Remove something from the support.** Knock a move out of pretraining (§A).
2. **Change the domain** so that what a proof needs is not in the statement:
   - a witness term, in first-order logic (§B);
   - a lemma or induction hypothesis, in arithmetic (§C).

Measurement-only alternatives (§D, §E) and self-play (§F) rank lower.

## Ranked table

Columns:
- **Pilot $:** the smallest decisive pilot.
- **Info:** how much a clean result moves the project's answer.

Costs use the measured rates: A40 $0.49/h; one best-recipe Stage-1 = 1,200 A40-s ≈ $0.16; one 8-round T1 ladder ≈ $3–4.

| rank | departure | decisive pilot | pilot $ | Claude-days | weeks to answer | info | null result still useful? |
|---|---|---|---|---|---|---|---|
| **1** | **A. Rule knockout × exploration** (propositional, proof-state env) | `Or.elim`-free Stage-1; ε = 0 vs ε = 0.05 legal-action exploration; required pool; 3 seeds × 2 arms × 4 rounds | **≈ $15** | 2–3 | **1** | high | yes: it bounds what exploration can create |
| **2** | **B. FOL with function symbols** (whole-proof first) | pretrain on Herbrand complexity ≤ 1; RL on complexity 2 / witness depth 2; 3 seeds | **≈ $15–20** | 3–4 | 1.5–2 | high | yes: says whether the limit is the domain |
| 3 | C. Induction arithmetic, where lemmas are necessary | custom-`N` identities; RL must state an unseen lemma | ≈ $20–30 | 5–8 | 3+ | highest if it works | partly (risk is in the build) |
| 4 | D. Library mining of RL's own proofs (DreamCoder-style, as a measure) | mine reusable sub-derivations in existing EI solutions vs Stage-1 corpus | **$0** | 1 | < 1 | medium-low | yes |
| 5 | E. Description-length (EDL) measure | prequential code length of group-B vs group-A proofs on `trajectory` checkpoints | ≈ $3 | 2 | 1 | medium | partly (may restate log p) |
| 6 | F. Self-play conjecturing | — (fold into C as lemma proposal) | — | — | — | low here | — |

**Recommendation.**
- **Run A now, merged into proposal 20 (exploration).** It gives exploration a target where the null is *near*-structural: a
  never-targeted `Or.elim` token stays in the vocabulary with a small but non-zero probability, so the null must be
  measured (the frozen K0 model's `Or.elim` log p at legal positions and its pass@N on the required pool at equal
  attempts). K0+ε then tests whether RL *adopts* a move that a hand-built proposer supplies, which is a legitimate
  question but not unaided creation. A is also the cheapest decisive test. *[reviewed: R2; was "the null is
  structural … cannot be sampled … creation by construction"]*
- **Start B's build in parallel** as the domain departure Dan asked about. P1 shows that the Lean side of FOL already
  works. C is the natural follow-on if B shows that RL can push past the pretraining frontier in a domain with
  unbounded objects.

---

## A. Rule knockout × exploration

**Hypothesis.** RL against Lean can acquire an inference rule whose token never appeared in pretraining, but only with
exploration that proposes legal actions outside the policy's support. With plain sampling, acquisition is impossible.
At a small dose (≈ 0.1 % of proofs), plain RL elicits it.

**Why the current path cannot answer it.** Every move is in support at a non-zero rate, so "new" always means
"improbable". The prior knockouts were *patterns* built from known rules:
- `round2-run2`: 0 of 6 patterns acquired from a zero base rate by EI.
- `round2-run5`: reductio stays 0/300 at f = 0 on 2 of 3 draws (s1, s2); **s0 ignites to 0.170 by round 4**.
  *[reviewed: R6; was "stays 0/300"]*
- `run4-grpo-review`: GRPO "ignites" depth-3 from zero, but 67–71 % of those proofs open a vacuous box. The reviewer
  calls this a box-nesting habit, not a new rule.

None of these knocked out a *rule token*, and none used exploration. All three are token-format numbers from before
2026-09-27, counted by `nd_verify` (`run4-grpo-review`'s were also Lean-confirmed). *[reviewed: R6; checker added]*

**P2 feasibility (this run).** Rule usage in two cap-6 Stage-1 sets. The executor counted `data/train.jsonl`
(154,990 ND proofs, `radical_scoping/p2_move_counts_train.txt`). Pilot A's model, best-cap6 (9.56 M `lean_staten`), and
the `lean_seq` 3.2 M from-scratch models were trained on **`data/p2/train_depth3_f0_a1.jsonl`** (155,000); the
reviewer's counts for that file are in the second column. The two agree within 0.5 pt, so the conclusion stands, but
the knockout filter must be applied to `train_depth3_f0_a1`. *[reviewed: R1; the executor named `data/train.jsonl` as
the models' training set]*

| rule | `data/train.jsonl`: proofs | share | `train_depth3_f0_a1` (reviewer): share |
|---|---|---|---|
| ORE (`Or.elim`) | 2,277 | 1.47 % | 1.46 % (2,256) |
| BOTE | 2,951 | 1.90 % | 1.95 % |
| DN (`Classical.byContradiction`) | 12,950 | 8.36 % | 8.33 % (12,905) |
| NEGI | 16,068 | 10.37 % | 9.93 % |

In `lean_tok.py`, `Or.elim` and `Classical.byContradiction` are single tokens. Knocking out ORE therefore keeps
98.5 % of the set: filter on the `rules` field. This holds in the `lean_seq` / `lean_staten` grammar. Free-form Lean also
has `Or.casesOn`, `Or.rec`, `Or.resolve_*`, `Classical.em` and `Decidable.em` on `lean_check`'s allowlist, so a free-form
knockout would have to block all of them. *[reviewed: free-form caveat added]*

Requirement labels on textbook72 come from `minlen`'s bounded search (bound 12, 20 s) with ORE removed (a subclass;
`minlen.py` is not edited) or with DN removed (`radical_scoping/p2_necessity_textbook72.jsonl`):
- The search proves 51 of 72 within 12 lines. Of those 51, **14 require ORE** and **9 require DN** (1 requires
  both). For the other 21, no proof was found: 15 have none in the restricted space, and 6 timed out
  (`p2_necessity_summary.txt`). *[reviewed: R7; was "no proof within the bound"]*
- These are labels of minlen's restricted space, as in `necessity.py`.
- *[reviewed]* The reviewer's **exact, unbounded** labels (classical G3c with and without L∨ for ORE; intuitionistic
  G4ip for DN): **21 / 72 need ORE, 19 / 72 need DN, 4 need both**. They agree with the bounded labels on all 51
  decided problems, with 0 disagreements. So the required pool is stronger than written and needs no search bound.

So a required pool exists. A dedicated pool of ~300 `Or.elim`-required targets comes from `necessity.py` with this
subclass.

**Smallest decisive pilot.**
- **Model:** the 9.56M best recipe, `lean_staten`, Stage-1 1,200 s, on cap 6 with every ORE proof removed.
- **Seeds:** 3, the same in both arms.
- **Arms:**
  - (K0) T1 EI, ε = 0.
  - (K0+ε) the same T1 with proposal 20's E2 operator. With probability 0.05 per step, take a uniformly random
    rule-legal action; for `Or.elim`, a random in-scope disjunction with the current goal.
- **Budget:** 4 rounds, on the ORE-required pool split into RL and held-out halves.
- **Readouts:** required-pool solves, RL half and held-out half; `Or.elim` token rate; frozen pass@4,096; worst-step
  log p of the first accepted `Or.elim` proof (with `tj_score.py`).

**Expected.**
- K0: 0 required solves on every seed. This is near-structural, not structural, so measure it: the frozen K0 model's
  `Or.elim` log p at every legal position, and its pass@N on the required pool at equal attempts. If that is > 0, the
  design needs a frozen-at-equal-attempts comparison, not "0 vs > 0". *[reviewed: R2; was "(structural)"]*
- K0+ε: > 0 on ≥ 2/3 seeds, with held-out required solves > 0 by round 4. That is acquisition that transfers, not just
  memorised explored proofs.

**Falsifier.** K0+ε held-out required solves ≤ 1 on every seed. This would mean that exploration reaches the rule but RL
does not turn it into a policy.

**Follow-ups.**
- 5 seeds plus a dose arm at 0.1 % and 1 % (the Interplay paper's exposure threshold, arXiv 2512.07783).
- A full-data control.
- **Seeds.** With a (near-)structural-zero null, 3 vs 3 gives a one-sided exact permutation p of at least 0.05. 5 vs 5 reaches
  p = 0.004. Pre-register 5 for the claim.

**Reuses:** `state_env.py`, `state_ladder_ei.py`, the `best` recipe, `minlen` / `necessity.py`, `tj_score.py`, the
Lean judge, `podjob` orchestration. **New code:** the ε-legal-action operator (shared with proposal 20) and the knockout
filter.

**Cost:** pilot ≈ $15 (6 Stage-1 at $0.16 plus 6 half-ladders at ≈ $2); full version ≈ $60–80. 2–3 Claude-days.

**Risks.**
- The ε operator needs a finite legal-action set. In `lean_staten` the model writes formulas, so the operator must
  build them: the goal, or rule-determined formulas.
- An untrained token's input embedding may make the first explored `Or.elim` steps unreadable to the policy. Measure
  the per-step log p after the first update.
- Learning happens only on targets exploration happens to solve. The held-out half guards against reading memorisation
  as acquisition.

## B. First-order logic with open-ended instantiation

**Hypothesis.** RL can push the *instantiation frontier* beyond pretraining. If pretraining only uses one instance per
universal premise and witness terms of depth ≤ 1, can RL learn to use two instances, or terms like `f (f a)`?

**Why the current path cannot answer it.** Propositional proofs never need an object that is absent from the
statement. FOL proofs do, once function symbols exist: the needed term is open-ended, and validity is undecidable.

**P1 feasibility (this run, `radical_scoping/fol_pilot/`).**
- **The sprint's generator and verifier run unchanged** (stdlib only). They produce 1,000 theorems in 3 s on the VPS:
  - mean 4.25 lines;
  - 731 / 1,000 contain a quantifier.
- **Lean renderer** (`fol2lean.py`, new): Fitch → Lean 4 core term.
  - Mapping: `∀`-intro to `fun (p : U)`, `∀`-elim to application, `∃`-intro to `⟨t, _⟩`, `∃`-elim to `Exists.elim`.
  - **Lean accepts 1,000 / 1,000** in 13 s on one process.
  - **Controls:**
    - all 231 conclusion-swap mutants and all 41 eigenvariable-clash mutants are rejected;
    - on 578 one-line proof mutants, Lean and the sprint verifier agree on 577.
    - The one disagreement is a proof the verifier rejects for a conservative freshness rule: the generalised constant
      occurs in a premise only inside a conjunct that the cited line discards. Lean accepts it, correctly. *[reviewed:
      wording]*
- **Gaps found:**
  - **No function symbols, constants a–e only.** Instantiation is a choice among ≤ 9 names, so it is not yet
    open-ended.
  - **The sampler cannot emit ∃-elim (EXE) or ∨-elim (ORE)** at all: neither is in its rule list (`core.py:436`).
    This is structural, not under-sampling; 0 / 1,000 pool proofs use either. The pool is also shallow (line quartiles
    3 / 4 / 5, max 16). *[reviewed: R5; was "no ∃-elim, which matches the sprint's under-sampling"]*
  - **Formulas carry free variables.** These are rendered as parameters.
  - **`lean_check`'s allowlist** has no `Exists.intro` / `Exists.elim`, so it needs widening for a FOL judge.
  - **Lean's 100-error cap** is not lifted by an in-file `set_option`. Use `-DmaxErrors=`.

**Smallest decisive pilot.**
- **Generator:**
  - Add `f/1`, `g/2`, and EXE and ORE rules.
  - Label each theorem's Herbrand complexity: the instances needed per universal, from a bounded search, and the
    witness-term depth.
- **Training:**
  - Whole-proof `lean_seq`-style FOL Stage-1 (3.2M, fast recipe) on complexity ≤ 1 and depth ≤ 1.
  - Then T1 EI on complexity-2 / depth-2 targets.
  - Held-out: complexity 2, plus Pelletier problems 18–47 rendered as statements. Hand-written statements, not proofs.
- **Expected:** frozen pass@4,096 on complexity-2 targets ≤ 1 %; T1 ≥ 20 % on at least 2 of 3 seeds, which is
  frontier extension as in the propositional length ladder.
- **Falsifier:** T1 − frozen within the MDD (5 seeds), or T1 solves only targets with a complexity-1 proof that the
  label missed. Check by re-labelling every solve.

**Reuses:** the sprint generator and verifier, P1's Lean renderer, `expert_iter.py`, `lean_judge` (widened). The
proof-state env port is deferred.

**Cost:** pods ≈ $15–20 (3–5 seeds); 3–4 Claude-days.

**Risks.**
- **The sprint's FOL was easy** (pass@8 ≈ 0.94–0.96; June 2026 sprint FOL model, scored by the sprint verifier, not
  Lean). *[reviewed: R6; model label added]* Without function symbols and multi-instance targets, B
  reproduces the propositional result in new syntax.
- A forward sampler rarely makes multi-instance proofs (`generator-cannot-make-run2-shapes`). Probe yields on CPU
  first, as in P1.

## C. Induction arithmetic: lemmas are necessary

**Hypothesis.** RL can learn to state and prove auxiliary lemmas, formulas that occur nowhere in the goal, that it
never saw in pretraining. In arithmetic with induction, cut elimination fails: `add_comm` has no direct inductive proof
without `0 + n = n` and `s m + n = s (m + n)`, or a nested induction. (The reviewer checked a one-theorem nested-induction
proof with no `have`, so a lemma *statement* is not necessary; the comment "lemma invention is necessary" in
`ind_pilot/ind.lean` is false as written. *[reviewed: R4]*) So the needed object is a new formula, not a
subformula.

**P3 feasibility (this run, `radical_scoping/ind_pilot/ind.lean`).** In Lean 4 core:
- a fresh inductive `N` with `add` / `mul` by recursion, plus `z_add`, `s_add`, `add_comm` and `add_assoc` by
  `induction` + `rw`, checks in 0.7 s with **no axioms**;
- **`simp [add]`, `omega` and `decide` all fail** on `add_comm` over `N`. But `induction n <;> simp_all [add]` proves
  the helper lemmas `z_add` and `s_add` outright, so the core library *can* close the L0 / L1 lemma families. A tactic
  grammar for C must exclude `simp_all` (or `simp` with the definition). *[reviewed: R4; was "the core library cannot
  trivialise the domain"]*

Truth of polynomial identities over `N` is decidable by normal form, so the target supply is unlimited with known
labels.

**Smallest decisive pilot.**
- A mechanical prover produces training proofs: induction plus `rw` with a fixed lemma library L0. This respects the
  "no hand-written training proofs" rule.
- Hold out a lemma family L1. Test whether RL proves targets whose library proofs need L1, and whether it does so by
  stating an L1-like `have`.

**Expected:** frozen ≈ 0 on L1-needing targets; RL > 0 on ≥ 2/3 seeds, with ≥ 50 % of solves containing a new `have`
lemma.

**Falsifier:** solves come only via nested induction or inline expansion, with no lemma statement, or there are no
solves at all.

**Cost:** pods ≈ $20–30; **5–8 Claude-days.** The build is the prover and generator, plus a tactic-level grammar and a
judge allowlist (`N.rec`, `Eq.mpr`, `congrArg`).

**Risks.** A 3–10M from-scratch model may not learn `rw`-list proofs at all; the prover's style narrows what is
learnable; this is the biggest build.

## D. Library mining of RL's proofs (measurement, $0)

**Hypothesis.** If RL creates rather than elicits, its proofs contain reusable sub-derivations (derived rules such as
contraposition or De Morgan) that are rare in the Stage-1 corpus. If it elicits, the abstractions that compress RL's
proofs are already frequent in pretraining.

**Pilot.**
- Mine α-normalised `have`-blocks and sub-terms (Stitch-style top-k by compression) from the solved proofs in the
  `trajectory` and `best-state` buckets, and from the Stage-1 corpus.
- For each top abstraction, report its pretraining rate and its RL rate.

**Expected:** ≥ 90 % of the top-20 abstractions occur in ≥ 0.1 % of Stage-1 proofs.

**Falsifier:** ≥ 3 abstractions with a Stage-1 rate < 10⁻⁴ that appear in ≥ 1 % of RL solutions.

**Cost and risks.** $0, 1 Claude-day. It is a measure, not a departure. Turning it into "library-augmented
RL" (mined lemmas as new actions) is a later, DreamCoder-style step. LILO, Stitch and LEGO-Prover are not in
`references.bib`.

## E. Excess description length

**Hypothesis.** Elicitation costs few bits to teach and creation costs many. EDL (`donoway2026excessdescriptionlengthlearning`)
is the prequential code length of the RL-acquired behaviour relative to the final loss.

**Pilot.** On the `trajectory` checkpoints, fine-tune the frozen base prequentially on group-B proofs vs group-A
proofs. Compare EDL per theorem.

**Expected:** B's EDL is within 2× of A's.

**Falsifier:** B ≥ 5× A.

**Cost and risks.** ≈ $3 and 2 Claude-days. **Risks:**
- It may restate the existing per-step log p in bits (B's eventual proof −15 nats total at end of Stage-1).
- The astra notes warn that EDL's guarantees do not cover adaptive RL trajectories.
- Best used as a second read-out on A or B.

## F. Self-play conjecturing at our scale (dropped)

The literature review rejected it (STP needs a pretrained 7B model; Minimo stalled at about 11 steps). Our
`frontier-supply` was a null (+7.5 of 291, inside the MDD of 10.5; SN-cap12, 3.2 M `lean_staten`), and our generator already supplies unlimited targets. Conjecturing
matters only where conjectures are new objects: lemma proposal in C. Fold it there.

## What would change the ranking

- **If proposal 20's ε operator cannot be made finite** in `lean_staten`, A's cost doubles and B moves to first.
- **If B's generator probe yields under 1 % multi-instance proofs,** B needs a backward (goal-directed) generator,
  roughly +2 Claude-days.
- **If Dan prefers a domain change over a cleaner propositional test,** run B then C, and run A as proposal 20's target.
