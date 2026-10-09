# Pretrain dataset ideas (pretrain-data)

Brainstorm, 2026-10-03. Nothing here is committed to; the plan's § 3 holds the pools that are
proposed. Each idea names the gap it addresses, how the pool would be built, its cost, and the
risk. "Format fixed" means the idea fits the `lean_staten` state format and Dan's harness
unchanged; ideas that need a format change say so.

Design principles that shape all of these (plan § 2.1):
- Dan's pattern dial: a structure present at frequency 10⁻⁴ ignites RL as well as at 10⁻¹. So a
  pool can carry many rare structures as tiny slices, and the base pool (K12) can stay as the bulk.
- Steps, not data: at fixed Stage-1 wall-clock the model sees ≈ 1 epoch of 155k records; a pool
  that is 10× larger is not seen 10× more. Composition matters more than size.
- Noise floor: at one seed only the arm's own type cell is readable. Every idea states the type
  cell it should move.
- The sealed 72 are never seeds. The test split of the candidate sets is never a seed either.

## A. Where the problems come from

A1. **Sub-problems of textbook proofs** (plan P1). Every line of a verified reference proof is a
theorem; promote open assumptions to premises. 262 seeds → 1,359 classes, 323 with ≥ 4
premises. Cell: 4-premise, ∧-shuffle. Built. Risk: 25 twins of the 72 and 26 of other eval
problems must be removed by the audit.

A2. **Sub-problems of generator long-mode proofs.** Same extractor over K12's 7–12-line rows and
over the 3,000 RL targets' minimal proofs. Yields mid-length problems whose proofs are dependency
closures, so no padding, and whose premises include the enclosing assumptions, so more
premises. Cost: CPU minutes. Cell: premises 4–5, lengths 5–9. Risk: twins of RL targets (the
ladder trains on them anyway; audit against the eval sets only).

A3. **Truth-table-sampled theorems.** Sample goal formulas and premise sets from the formula
grammar, keep the entailments by truth table, find minimal proofs with `minlen.py`. This is
independent of the forward generator's bias: it emits classical tautologies (Peirce, excluded
middle, consequentia mirabilis, De Morgan) that forward construction never produces, and
RAA-heavy proofs. Cost: `minlen` at bound 40 is seconds per problem; 10k problems in hours on the
laptop. Cell: DN/classical, ∨I-goal, "neither group". Risk: proofs found by search, outside the
take-home's rule (allowed here).

A4. **Composed problems.** Chain two generator problems: the conclusion of one is a premise of the
next, then drop the intermediate as a premise and keep its proof as a lemma block. Produces
hypothetical-syllogism shapes, →E chains ≥ 4, and box-free proofs of 10–16 lines with 3–5
premises. Cost: CPU. Cell: long box-free, hypothetical syllogism. Risk: the joined proof may not be
minimal; re-run `minlen` and keep only when the composition is the minimal proof, or keep both.

A5. **Dualised problems.** From Γ ⊢ A make Γ, ¬A ⊢ ⊥ and Γ ⊢ ¬¬A; from A → B make ¬B → ¬A
(contraposition instance). Cost: trivial. Cell: NEGI/NEGE/⊥-goal, DN. Risk: proofs are
near-copies of the source proof plus a wrapper; a low-information slice. Use at 10⁻³.

A6. **Substitution instances.** Replace atoms in short proofs by compound formulas (P ↦ (Q ∧ R),
P ↦ ¬S, P ↦ (Q → R)). The proof skeleton is unchanged but formulas are larger, and ∧E/→E then
act on compounds. Cost: trivial. Cell: ∧-shuffles with compound conjuncts, formula length. Risk:
token length grows; the 256-token action cap in RL must still fit.

A7. **Premise enrichment.** Two variants. (a) Split a conjunctive premise into its conjuncts
(the proof gains ∧I lines or loses ∧E lines), giving 4–7 premises from existing problems. (b)
Add 1–3 irrelevant premises (distractors) to any problem, proof unchanged. Cost: trivial.
Cell: 4-premise (a), premise selection (b). Risk: (b) teaches ignoring premises; the generator's
"lazy premises" already does some of this. Keep (b) small.

A8. **Schemata families with instance depth** (plan P4, extended). The 26 textbook laws plus
exportation, distribution, absorption, material implication, each instantiated at depth 0–2
(atoms, then compounds for atoms). Cost: trivial. Cell: laws. Risk: proposal 10 called schema
instances "decorative"; test at 10⁻² only.

A9. **Other textbooks and corpora as seeds.** Open Logic Project, Carnap's exercise banks,
Lemmon-style sets, TPTP's propositional SYN problems, Lean's `propositional` examples. Transcribe
with last week's `transcribe.py`, prove with `prove.py`, extract sub-problems (A1). Cost: a day of
transcription per book. Cell: transfer to unseen books. Risk: must stay out of the test split's
books; licence check before release.

A10. **Contradiction-driven problems.** Random unsatisfiable clause sets (small DPLL/resolution
refutations) rendered as premises with goal ⊥, minimal proofs by `minlen`. K12 already has 12,389
contradictory-premise rows but all generator-shaped; the Calgary step-cap failure was a
contradictory-premise problem. Cost: CPU. Cell: ⊥ goals, premises ≥ 4. Risk: ⊥ goals are rare in
textbooks; keep small.

A11. **Join problems from the hard proofs** (from `hard_problems.md`, 2026-10-06). The hard
textbook proofs are made of pieces K12 already knows (85 % of their 2–6-line sub-theorems occur in
K12 in skeleton form) but joined in ways K12 almost never shows: ∧E feeding →E (↔-elimination,
≈ 90× rarer in K12 than in the hard proofs; one K12 proof has the skeleton `((P → Q) ∧ R), P ⊢ Q`),
¬I feeding ¬E (128×), nested ∧E (35×), ¬I feeding ∨I (De Morgan, 34×), two →I boxes feeding ∧I
(↔-introduction, 18×), nested ∨E (12×), →E chains (8×). Build 4–8-line problems whose minimal
proof contains each join, by taking the generator's problems and (a) wrapping a conditional
premise in a conjunction, (b) nesting a conjunction, (c) replacing a premise A by (A ↔ B) ∧ …,
(d) replacing a goal by its ↔ with a derivable formula, (e) requiring ¬I before ¬E by negating a
premise. Verified by `minlen`. Cost: CPU. Cell: ↔ problems (42 of 77 hard scraped), ∧-shuffles,
De Morgan. Risk: the three structural gaps (≥ 4 premises, depth ≥ 3, ↔ shape) are not fixed by any
small problem and need C2, A7 and A6 alongside.

## B. What the proofs look like

B1. **Minimal proofs everywhere** (plan P3). Replace every K12 proof by its `minlen` proof;
removes the manufactured splits (11.4 %) and R lines (17 %). Cell: padding, long proofs.

B2. **Multiple proofs per theorem.** For each theorem keep 2–3 distinct minimal or near-minimal
proofs (different rule orders, ORE vs RAA). A pretrain with proof diversity may give RL a wider
prior to explore from. Cost: `minlen` enumerates by changing tie-breaks. Cell: pass@k spread.
Risk: halves the number of distinct theorems at fixed records; test as a 10 % slice.

B3. **Recovery data.** Take a minimal proof, insert a dead line or a manufactured split in the
prefix, then continue with the minimal completion. The state format shows the padded prefix as
the state; the target is the clean continuation. Teaches "finish cleanly after a wasted step",
which is exactly what RL's accepted proofs fail to do (dead 1.06 + splits 3.9 lines excess).
Format fixed. Cost: trivial. Cell: padding, step-cap collisions. Risk: it also teaches that dead
lines occur; keep the padded fraction ≤ 20 % and never pad the target.

B4. **Valid-move supervision.** For every proof state in a reference proof, the target is not only
the reference's next line but any sound next line that lies on some minimal completion (from
`minlen`'s memo table). A broader prior over legal actions. Format fixed (one record per
state/action). Cost: `minlen` already computes `best(G, A, b)`; expose it. Cell: exploration,
pass@k. Risk: record count grows by × 3–5; at fixed steps the model sees fewer theorems.

B5. **Style canonicalisation.** Keep K12 but rewrite each proof into one normal form: no R,
no rederivation, premises cited in order, boxes closed as early as possible. Weaker than B1
(keeps K12's theorems and their ORE-heavy shapes) but tests whether style alone drives padding.
Cost: the padding parser plus a rewriter. Cell: padding.

B6. **De-padded RL proofs as data.** Shortest-of-k RL proofs on the RL targets (not the eval
sets), pruned by dependency closure. 72 % of problems' shortest-of-64 were ≤ the reference. A
self-distillation Stage-1 arm; the ladder already does expert iteration, so this tests
whether the same data helps earlier. Cost: existing rows. Cell: long proofs. Risk: RL seeds vary
(56 % vs 41 % splits); prune before use.

## C. How difficulty is distributed

C1. **Uniform length ladder 2–16.** K12 is flat 2–6 plus flat 7–12 long-mode. Extend to 16 with
`minlen` proofs. Cell: proofs ≥ 14 lines (54 of 67 never-solved are ≥ 14).

C2. **Depth ladder.** Generator depth ≤ 2 today. A 10⁻³ … 10⁻² slice at depth 3 and 4 (nested
∨E, ∨E inside →I inside ¬I). Cell: nested boxes. Risk: Dan's fork write-up says depth ≥ 4 is
unreachable by the generator; needs the goal-mode `reach()` with deeper targets, or A3.

C3. **Textbook-matched feature distribution.** Estimate the joint distribution of (goal
connective, premise count, minimal length, depth, rule set) on the seed split of the candidate
sets, then importance-resample the generator pool to match it. Uses no textbook problem as data,
only its shape. Cell: everything the test split measures. Risk: the seed split is 134 problems;
smooth the estimate.

C4. **Spice-rack pool.** K12 plus a 1 % total mixture of every rare structure (4 premises, ∧E
chains, →E chains ≥ 4, depth 3, RAA goals, ∨I goals, hypothetical syllogism), each at 10⁻³.
One arm tests whether "a little of everything" moves several cells at once; follow-ups ablate
one spice at a time. Cell: all; attribution needs ablations.

C5. **Rule-balanced resampling.** K12 is 44 % ORE and 4 % ∧E by rule use. Resample so each
rule's share is nearer uniform. Cell: ∧-shuffles, →E chains. Risk: changes the theorem
distribution as a side effect; report both.

C6. **RL-target-matched pretrain.** Match the pretrain length distribution to the RL targets'
(7–16, 300 per length) rather than to 2–6. Tests whether the pretrain should look like the RL
curriculum. Cell: the ladder's own solve rate. Risk: Dan's cap-12 result is already a step this
way.

## D. Multiplicity and invariance

D1. **Permutation multiplicity** (plan P1's × 24). Few theorems, many renamings, versus many
theorems, one renaming each. Also test the opposite on K12: collapse to one representative per
class and fill with new classes. Cell: renaming invariance; pass@k on renamed test problems.

D2. **More atoms.** The generator uses P Q R S; textbooks use up to 6 letters and some test
problems use letters the model has rarely seen in position. Add T U V W to the generator's atom
pool at a slice. Cost: a knob. Cell: letter-count ≥ 5 problems.

D3. **Formula-size invariance.** Same proof skeleton at formula sizes 1, 2, 3 (A6 systematically).
Cell: long-formula problems.

## E. Scale and repetition (cheap controls)

E1. **Pool size at fixed steps.** 15k unique records × 10 epochs vs 155k × 1 epoch. If equal,
every idea above can be tested with 10× smaller pools. Cost: one Stage-1 run.

E2. **Token budget per record.** Shorter prompts (fewer premises, smaller formulas) see more
records per step. Report records seen per Stage-1 run for every arm.

## F. Needs a format or harness change (out of scope unless Charles opens it)

F1. **Subgoal annotations.** Interleave "goal: X" lines in the proof text so the model plans
backward. The pinned verifier would need to ignore them and the state format would have to
carry them.

F2. **Negative examples.** Invalid lines marked as rejected. Not expressible as next-token
targets in the current format; would need a judge-conditioned format.

F3. **Normalisation pairs.** (padded proof → minimal proof) as an editing task. Different task
from proof writing.

## G. Analysis before any run (no GPU)

G1. **Coverage map.** For each never-solved textbook type (plan § 1 table), count the exact K12
rows that match it and compute what each idea above would add. Decides slice sizes.

G2. **Minimal-proof reachability.** For each candidate pool, the share of records whose proof is
minimal, by length. Separates "the data is padded" from "the data is short".

G3. **Twin audit by construction.** Each builder emits its records' curry-aware keys; the audit
against every eval set and the 72 runs before a pool is written, not after.

## Suggested order if everything above were on the table

Cheap, format-fixed, and each targeting a cell the textbook analysis measured:
B1 (minimal proofs), A1 (sub-problems), A11 (join problems, the direct answer to "which rules make
the hard problems hard"), A3 (truth-table-sampled classical theorems), C4 (spice rack) with G1 to
size the slices, then B3 (recovery data) as the direct test of the padding finding. A4 (composition) and C1 (length ladder) next. D1 and E1 as the two controls that make the
rest cheaper to run.

## H. Directions from the literature (2026-10-06)

What leading groups did when they faced the same shape of problem: a small model, a verifier, synthetic data,
and an RL stage that can only amplify what pretraining put within reach. Each item names the source, the analogue
here, whether it fits the fixed format, and the cell it should move. Project facts that decide which papers apply:
Robbie's 9.56M pretrain saw the 155,000-record pool **20.7 times** in its 1,200 s (`scale-d384-p1200_s0` summary:
3,180,545 records seen), the 38M model 7.9 times; both pretrains are single-task next-token on proofs; the RL stage is
expert iteration (STaR-style) with a verifier reward.

### H1. Synthesis by deduction closure and traceback (DeepMind, AlphaGeometry; HER)
AlphaGeometry (Trinh et al., Nature 2024) built 100M training proofs by sampling random premises, running a symbolic
deduction engine to closure, and **tracing back the minimal dependency subgraph of every derived fact** as its own
theorem. That is our forward generator plus dependency pruning, applied to *every* line rather than only the last one,
which is what the sub-problem extractor (A1/A2) does. Aygün et al. (ICML 2022) did the same for Lean with hindsight
experience replay: every intermediate state of a proof is a goal. **Do:** run the extractor over the whole forward
generator output before pruning (closure, not just the sampled conclusion), keep every distinct class with its minimal
proof, and prefer this to re-sampling theorems. Format fixed. Cell: ∧-unpacking, 4-premise, mid-length. The
AlphaGeometry "auxiliary construction" step has an ND analogue: proofs whose key move is an assumption not suggested
by any premise (a reductio hypothesis, an excluded-middle instance, a lemma assumed and discharged). Tag those records
and make sure a slice of them exists; the hard proofs' reductio boxes are exactly this.

### H2. Necessary-steps-only proofs and planning (Allen-Zhu & Li, Physics of LMs 2.1)
On synthetic grade-school math, models trained on solutions that contain **only the necessary steps** learn to plan
the dependency graph before writing ("level-2 reasoning"), while unnecessary steps in the data produce models that
compute everything reachable. K12's long half is 89 % unnecessary steps. Supports B1 (minimal K12) as the first arm,
and predicts the mechanism of the padding we measured. Format fixed. Cell: padding, long proofs.

### H3. Mistake-and-recovery data in pretraining (Physics of LMs 2.2; "Learning from mistakes", An et al. 2023)
Inserting wrong steps followed by a correction into *pretraining* data raised accuracy on reasoning problems, and
only when the errors were present at pretraining rather than added at fine-tuning. In Fitch a wrong line cannot be
written (the verifier rejects it) but a **useless** line can, so the analogue is B3: a padded prefix (dead line,
manufactured split) followed by the minimal completion. Format fixed. Cell: padding, step-cap collisions.

### H4. Search traces as training data (Stream of Search, Gandhi et al. 2024; Searchformer, Lehnert et al. 2024)
Training on the full search process (explored branches, dead ends, backtracks) instead of only the final solution
produced models that search at inference and could then self-improve to shorter traces. Fitch admits this without a
format change: a dead-end branch is an assumption box that is opened, explored and closed without being used, or a
derived line that goes unused. **Do:** render `minlen`'s own search (its iterative deepening tries) as proofs with the
abandoned branches kept, then bootstrap toward the pruned proof as Searchformer does. This is the aggressive version
of B3. Risk: it teaches the padding we want to remove, unless the RL reward prefers short proofs (it does not today).
Cell: never-solved problems where exploration, not knowledge, is missing.

### H5. Proof-artifact co-training (PACT, Han et al. 2021/22; GPT-f, Polu & Sutskever 2020)
PACT added auxiliary tasks built from the same proofs (predict the next lemma, the goal, the type, a skipped proof) and
got large gains in tactic prediction at fixed data. With 99 tokens we can add sequences that **reorder existing tokens**
and never appear at inference, because the harness only samples after `PRF`: given premises and proof, predict the
goal (`THM prem SEQ PRF proof QED SEQ goal`); given a proof, predict the rule of each line (mask the rule token);
given premises and goal, predict the premises actually used. Needs a trainer change (new sequence types in the
pretrain loader), not a format or harness change. Cell: premise selection, goal-directedness.

### H6. Statement curricula and a learned conjecturer (Polu et al. 2022; STP, Dong & Ma 2025; MetaGen, Wang & Deng 2020)
OpenAI's formal-math curriculum did expert iteration on a growing set of *statements* ordered by the model's pass
rate; STP trains a conjecturer to emit statements that are barely provable by the current prover; MetaGen trained a
neural theorem generator. We have a generator and a verifier but the generator's grammar is fixed. **Do:** (a) a
pass-rate ladder for RL targets (pick targets the current model solves 1–30 % of the time, re-estimated each round);
(b) train the pretrain itself to sample statements (`THM … SEQ … PRF` with nothing after) from the seed split's
statements only, prove the samples with `minlen`, keep the provable ones: a conjecturer for textbook-shaped theorems
without textbook proofs. (b) uses seed-split statements, never the 72. Cell: textbook transfer.

### H7. Problem variants at training and test time (AlphaProof, DeepMind 2024; Goedel-Prover-V2, Seed-Prover 2025)
AlphaProof's reported recipe fine-tunes on **variations of the target problem** (easier and related statements)
before attempting it; Goedel-Prover-V2 scaffolds data by synthesising problems of graded difficulty; Seed-Prover
proves lemma-first. For us this is the derived-nuance machinery (substitution, wrappers, distractors, conjoined) and
the sub-problem extractor turned into a generator of *graded variants* of each seed, from 2-line pieces up to the
seed. Allowed on the seed split; a test-time version on the test split or the 72 would be training on the test set
and is out. Cell: in-distribution seed-set transfer.

### H8. Data-constrained scaling: more unique data beats more epochs (Muennighoff et al. 2023)
Returns from repeating data fall off after about four epochs. The 9.56M pretrain runs **20 epochs** on 155,000
records in its budget. **Do (cheap, decisive):** a 1.5M-record pool from the same generator so the same wall-clock is
two epochs instead of twenty; and the reverse control (15,000 records × 200 epochs). Dan's "steps not data" was
measured at fixed pool size, so it does not answer this. Format fixed. Cell: everything; this is the first scaling
question the plan should settle because it changes how large every other pool must be.

### H9. Domain reweighting with a proxy model (DoReMi, Xie et al. 2023; Skill-it, Chen et al. 2023; Rho-1, Lin et al. 2024)
Instead of guessing slice fractions, train a small proxy on the uniform mix, measure per-family excess loss against a
reference, and up-weight families the model learns slowest (DoReMi), or order skills by which ones most reduce loss on
the others (Skill-it), or weight tokens by excess loss (Rho-1). Our families (K12 lengths, joins by family,
sub-problems, structural) are the domains. One 300 s proxy run gives the weights for the whole plan. Format fixed.

### H10. Importance resampling toward the target distribution (DSIR, Xie et al. 2023)
Resample the generator pool to match the feature distribution of the seed split (goal connective, premises, depth,
minimal length, rule set), estimated with smoothing from 134 problems. This is C3 with a method behind it; it uses
textbook *shape*, never textbook problems. Format fixed. Cell: everything the test split measures.

### H11. Rephrasing for diversity (WRAP, Maini et al. 2024; OpenMathInstruct, Toshniwal et al. 2024)
Multiple styles of the same content made small corpora go further; many solutions per problem helped math SFT.
Analogues: B2 (several minimal proofs per theorem, ORE-style vs RAA-style), A6 (formula-size variants of the same
skeleton), D1 (atom permutations). Cheap; test as 10 % slices.

### H12. Iterative over all-at-once proving (ProofWriter, Tafjord et al. 2021; RuleTaker, Clark et al. 2020)
Models trained to emit one inference at a time generalised to deeper proofs than models emitting whole proofs; mixed
depth curricula (D0–D5) transferred to unseen depths. Dan's state format is the iterative setting. For data: a depth
ladder with a controlled mix (C2) rather than the generator's accidental depth ≤ 2; ProofWriter's finding is that the
mix, not the maximum, carries the transfer.

### H13. Shortcut learning on skewed logic data (SimpleLogic, Zhang et al. 2022; "shortcuts to automata", Liu et al. 2023)
Transformers trained on logic data with a skewed label or rule distribution learn statistics of the distribution
rather than the rule; balancing the distribution removed the shortcut. K12 is 44 % ORE with 11 % manufactured splits:
a shortcut by construction. Supports C5 (rule-balanced resampling) and B1. Also a warning for every pool we build:
report the rule-use histogram next to the type cells.

### H14. Length generalisation depends on format details (Zhou et al. 2024; Lee et al. 2023)
Index hints, reversed order and scratchpads decide whether small transformers extrapolate in length. The line-number
shift augmentation is the project's index-hint trick already. Two cheap data-side tests: proofs with premises
re-ordered randomly at every presentation (so position carries no information), and a uniform length ladder to 16 (C1)
instead of the two-plateau K12.

### H15. Lemma libraries and refactoring (LEGO-Prover, Wang et al. 2023; REFACTOR, Zhou et al. 2024)
Growing a library of reusable lemmas raised miniF2F pass rates. Fitch has no lemma call, but "prove the lemma at the
top, then use it" is a proof shape (A4 composition, A12 lemma-fused) and the *library* can live in the data: the most
frequent join skeletons become a fixed slice present in every pool at the pattern-dial frequency.

### H16. The base model bounds RL (Yue et al. 2025; DeepSeek-R1, 2025)
RL with verifiable rewards sharpens the base model's distribution; pass@k at large k of the base bounds what RL can
reach. This is Dan's pattern dial measured from the other side, and it gives the cheapest readout for every pool:
**pass@256 of the frozen pretrain on the textbook type cells, before any ladder.** CPU sampling, no RL. Pools whose base
pass@256 does not move a cell will not move it after RL either; only pools that move it need the ladder. Format fixed.

### H17. Easy-to-hard and reverse curricula (Sun et al. 2024; R3, Xi et al. 2024)
Supervise on easy, RL on hard, with the reward trained on easy: the workstream's premise. R3 starts RL from states near
the end of a demonstrated proof and moves the start backward. Data side: the state trainer can take (state, action)
pairs from suffixes of textbook-shaped proofs, which is the sub-problem extractor again, applied to the state format.

### H18. Synthetic continued pretraining on a small corpus (EntiGraph, Yang et al. 2024)
Make a tiny corpus learnable by generating the graph of relations among its entities. For 134 seed problems: the graph
is sub-problems (A1), composition (A4), duals (A5), substitution instances (A6), premise enrichment (A7), permutations
(D1), generated to tens of thousands of records. One pool, "seed graph", instead of five slices.

### H19. Fill-in-the-middle and reverse training (Bavarian et al. 2022; Golovneva et al. 2024)
Predicting a missing proof segment, or writing the proof goal-first, as extra sequences. Both are reorderings of
existing tokens that never appear at inference, like H5, but a backward Fitch proof is not a valid object, so these
belong with the co-training tasks and need the trainer change.

### H20. Prune easy, keep hard (Sorscher et al. 2022; SemDeDup)
In the data-rich regime, pruning the easiest examples improves scaling. The pretrain's held-out greedy accuracy is
0.956–0.994: most of K12 is already solved. **Do:** score K12 with the frozen pretrain, drop records it gets right
greedily, refill with records from the new builders. Format fixed, one CPU pass. Cell: all; this is the cheapest way
to make room for the new material without growing the pool.

### Suggested additions to the pilot order
H8 (unique data vs epochs) and H16 (base pass@256 as the pool screen) first, since they change how every other arm is
sized and read. Then H2/B1, H1/A1–A2, H3/B3, H9 for slice weights. H5 (co-training) is the one trainer change worth
making if the data-only arms stall.

## I. Recent literature scan (2026-10-06): papers not previously in the repo

Four parallel searches (synthetic data for provers; pretraining data for reasoning; base model vs RL; length and
depth generalisation), each paper verified on its arXiv abstract page. Deduplicated; papers already cited in § H are
excluded. Grouped by what they change for us. Dates are arXiv submission dates.

### I-A. How much of a pattern pretraining needs, and where RL can act

I1. **Zhang, Neubig, Yue, Dec 2025, arXiv:2512.07783 (ICML 2026 spotlight)** "On the interplay of pre-training,
mid-training and RL." Fully controlled from-scratch study (100M model, DAG arithmetic): RL raises pass@128 only on
problems at the model's *edge of competence*, never in-distribution; a short mid-training phase narrowed onto that
edge beats the same compute spent on RL; ~1 % pretraining exposure to a new surface pattern lets RL transfer, 0–0.1 %
makes transfer fail. **For us:** this contradicts Dan's pattern dial (10⁻⁴ ignites at cap 6, 3.3M). The two results
differ in scale, task and the RL algorithm, so the fraction is the first thing to measure: one pattern (say
∧E-then-→E) at 10⁻⁴, 10⁻³, 10⁻², 10⁻¹ with one seed each, read as base pass@k and post-ladder pass@k. Also: an "edge
slice" of proofs just past K12's coverage (13–16 lines, depth 3) as the *last* 10 % of Stage-1 wall-clock rather than
mixed throughout, which the pretrain loader can do by ordering records. Cost: one Stage-1 ladder per fraction.

I2. **Gu, Lyu, Li, Zhang, May 2025, arXiv:2505.18091 (NeurIPS 2025 spotlight)** "Data mixing can induce phase
transitions in knowledge acquisition." A minority component is learned almost not at all below a critical mixing
ratio and quickly above it; the critical ratio grows as the model shrinks (power law), explained as capacity
allocation. **For us:** at 9.56M a rare proof type may need far more than 1 % to be learned at all by the *base*; the
slice sweep in I1 should look for a jump, not a slope. Measured on fact memorisation, so only the shape transfers.

I3. **Huang, Saphra, Lampinen, Potts, Lubana, May 2026, arXiv:2605.29548** "Why larger models learn more: capacity,
interference and rare-task retention." Inserts a novel task into pretraining at a controlled frequency (OLMo 4M–4B):
small models fail on rare tasks because frequent-task gradients overwrite slowly accumulating rare-task features.
**For us:** the one paper that is literally a pretraining data dial at our model size. Before crediting RL with
finding a 10⁻⁴ pattern, measure whether the *base* checkpoint already learned it (held-out loss on the pattern's
records per checkpoint); then repeat the dial at d=256 (3.2M) where the paper predicts non-retention.

I4. **Foster, Mhammedi, Rohatgi, Mar 2025, arXiv:2503.07453** "Is a good foundation necessary for efficient RL?"
Theory: base-model coverage of good responses lower-bounds the runtime of any efficient RL; the threshold is
"coverage above 1/(samples per round)", not frequency. **For us:** the principled reading of the pattern dial. The
ladder samples k=32 per target per round; a pattern survives if the base puts mass > 1/32 on some proof using it for
some target. Measurement: for each textbook problem, base-model probability mass on verified proofs (large-k
sampling), checked against which problems the ladder then solves.

I5. **Zhao, Kakade, Pehlevan, Jelassi, Malach, Apr 2025, arXiv:2504.07912** "Echo chamber: RL post-training amplifies
behaviours learned in pretraining." 150M–1B models pretrained from scratch on controlled mixtures; PPO, GRPO and
expert iteration all collapse onto one output format, and the winner is not the most frequent one (28 % beat 62 %);
doubling a component flips the winner; scale-dependent. **For us:** the closest cousin of the padding result (RL
amplified K12's manufactured-split style). Per-round measurement: which proof style the ladder converges to versus
its pretraining frequency and its verifier success per sample; a two-pattern dial (minimal style vs split style at
varied ratios) to find the flip.

I6. **Shen, Li, Rahman, Sun, Goldblum, Telgarsky, Izmailov, Jul 2026, arXiv:2607.16097** "Understanding reasoning
from pretraining to post-training." Chess testbed, models 5M–1B (brackets ours): post-RL performance at fixed RL
compute is predicted by pretraining loss; the RL reward slope grows ~linearly with pretraining tokens; on hard
problems RL *surfaces* moves nearly absent before RL, on easy ones it *sharpens* existing preferences. **For us:**
two things. (a) The cheapest screen for pools: held-out loss on a fixed textbook-shaped validation set (derived
problems from the seed split, never the 72), compared across pools before any ladder. (b) For every textbook problem
the ladder solves, log the base model's log-probability of the found proof and split "amplified" from "surfaced".

I7. **He, Fried, Welleck, Jun 2025, arXiv:2506.02355 (EMNLP 2025)** "Rewarding the unlikely." Formal theorem proving
with a perfect verifier: GRPO's rank bias reinforces already-probable correct proofs and neglects rare ones, so the
policy ends below "sample more from the base" at large N. **For us:** the ladder is expert iteration, not GRPO, but
the diagnostic is cheap: the base-policy rank of the proofs selected for each round, and pass@N versus the base at
large N, reported next to pass@1.

I8. **Song, Kempe, Munos, Sep 2025, arXiv:2509.06941** "Outcome-based exploration for LLM reasoning." Outcome-reward
RL reduces effective diversity even on the training set, and the loss transfers to unsolved problems through the
shared policy. **For us:** per-round count of distinct verified proofs (by canonical form) per problem on the targets
and on the textbook sets; if diversity on *unsolved* textbook problems falls as targets get solved, the only
pretraining-side counter is proof diversity per theorem (B2, H11). Related: Zhou 2026, arXiv:2607.20543 (rare correct
trajectories vanish before RL samples them enough); Ni et al. 2025, arXiv:2510.15990 (from-scratch GRPO: OOD gains
only when aligned with the pretraining bias); Abdulsalam, Patel, Saxe 2026, arXiv:2607.07646 (RL composes primitives
a rewrite-grammar transformer was pretrained on: the small-scale "teaches vs sharpens" analogue).

I9. **Akter, Prabhumoye, Nyberg, Patwary, Shoeybi, Choi, Catanzaro, Sep 2025, arXiv:2510.03264** "Front-loading
reasoning." Reasoning data in *pretraining* gives gains that later SFT cannot replicate; pretraining benefits from
*diversity* of reasoning patterns, post-training from quality; high-quality pretraining data has latent effects that
appear only after post-training. **For us:** build pools that differ in the number of distinct proof schemas and rule
combinations at fixed record count, and expect some pools to look identical at base and separate only after the
ladder. The "judge after RL, not at base" rule, with a citation.

### I-B. Recipes for the data itself

I10. **Morishita et al., Nov 2024, arXiv:2411.12498 (NeurIPS 2024)** "Principled synthetic logic corpus (ALT /
FLD×2)." Random forward chaining over ~50 FOL rules, 1–8 steps, 0–20 *distractor premises*, with four principles:
content-agnostic facts, unprovable samples, diverse rules and step counts, diverse surface forms; ablations put rule
diversity and distractors above step-count diversity. The closest published recipe to `gen.py`. **For us:** add
distractor premises (sweep 0–4), enforce rule-pair coverage quotas (the joins table), and vary step counts; A7(b)
with evidence. Risk: 7B–70B models; distractors at 9.56M may hurt above a small ratio.

I11. **Xia et al., Apr 2025, arXiv:2504.20213** "Can LLMs learn formal logic? A data-driven framework." Backward
random generation of Boolean-logic proofs plus *template transformation*: substitute each propositional variable by a
random formula at rate ~0.7, keeping the justification sequence; 9k examples sufficed; helps width generalisation and
small models most. **For us:** A6 (substitution instances) with a measured optimum: apply variable→formula
substitution at 0.7 to K12 or to the join pool, re-verify (the skeleton is unchanged so the proof re-verifies after
renaming). Risk: Hilbert-style proofs, so box nesting and ∨E untested; formula length against the 256-token action cap.

I12. **Gu et al., Oct 2025, arXiv:2510.15700** "ProofOptimizer." A Lean proof simplifier trained by expert iteration
+ RL shortens proofs by 49–87 %, and the simplified proofs reused as training data improve the downstream prover;
trivial theorems filtered by an automation tactic (518k → 307k). **For us:** direct support for B1 (minimal K12,
built) and for H20 (prune the trivially solved). Keep a controlled fraction of padded proofs rather than none.

I13. **Yin et al., Feb 2025, arXiv:2503.04772** "LeanNavigator: millions of Lean theorems by exploring state
transition graphs." Every proof state with a path to QED is a theorem; alternative paths give alternative proofs;
4.7M theorems from Mathlib. Related: Lai et al. 2025, arXiv:2505.12031 (proof-state exploration for diverse tactics).
**For us:** A1/A2 (sub-problems) plus B2 (alternative proofs), from the same walk: every inner box and every suffix is
a problem; branch at each state to collect distinct proofs. Dedup by renaming class and audit, as now.

I14. **Tian et al., Oct 2025, arXiv:2510.00732** "EvolProver: evolving formalised problems via symmetry and
difficulty." Statement augmentation along two axes: AST-level equivalent rewrites (symmetry) and evolved
harder/easier variants (difficulty). **For us:** purely symbolic rewrites for ND (premise reordering, ∧/∨ commutation,
contraposition, curried/uncurried, double-negation wrappers) and a difficulty ladder by adding one premise or one
nesting level and re-proving with `minlen`. Each variant needs its own verified proof; commuting ∧ changes which ∧E
is used.

I15. **Sileo, Aug 2026, arXiv:2608.05148** "Reasoning Core." 50 procedural generators with difficulty controls; "semantic
validity alone does not ensure training utility": compact targets and calibrated difficulty decided the gains, and
audits found subtle generator bugs in generation, rendering, targets and scoring. **For us:** an audit checklist for
`gen.py` and every builder (rendering, padding, difficulty histogram against the textbook shape); the 89 %-padding
finding is exactly the kind of bug this paper means.

I16. **Vompa & Tammet, May 2026, arXiv:2605.04330 (TMLR)** "Scaling properties of implicit deductive reasoning."
Horn-clause deduction with 8-layer d=256 models: data balanced by rejection sampling per (depth, label) bucket plus
*counterfactual minimal pairs* (remove a critical premise, add transitive distractors so surface statistics stay the
same but the answer flips) removes shortcut learning; explicit chains are still needed to extrapolate past trained
depth. **For us:** balance K12 by depth and rule (C2, C5), and add minimal pairs: same premises with one critical
premise swapped so a different, longer proof is required, plus distractors. The unprovable half has no Fitch proof,
so only the provable side carries over. Closest domain match in the scan.

I17. **Saparov, He, Kim et al., Dec 2024, arXiv:2412.04703 (ICLR 2025)** "Transformers struggle to learn to search."
DAG reachability (equivalent to proof search in simplified logic): small transformers learn a real search algorithm
only under a balanced distribution of lookahead and distractor structure; naive random graphs give shortcuts, and
scale or CoT do not fix larger graphs. **For us:** instrument the generator with dependency-graph statistics
(premise-to-goal distance, unused premises, multiply-used premises, branching) and balance them; the 4-premise
problems should include unused and reused premises. No extra compute.

I18. **Cai, Lee, Schwarzschild, Oymak, Papailiopoulos, Jun 2025, arXiv:2506.09251** "Extrapolation by association:
length generalisation transfer." Co-training on a longer, related, *easy* auxiliary task transfers length
generalisation to a target task whose own data is short; transfer tracks re-use of attention heads. **For us:** an
auxiliary stream in the same format: 20–40-line proofs made only of ∧I/∧E/reiteration chains and long premise
lists, so line citation and box bookkeeping are trained at lengths K12 never shows. One extra generator; cheap.

I19. **Wang, Nichani, Bietti, Damian, Hsu, Lee, Wu, May 2025, arXiv:2505.23683 (COLT 2025)** "Learning compositional
functions from easy-to-hard data." Gradient descent on k-fold compositions succeeds only if the data has both shallow
and deep compositions (mixed or as a curriculum); hard-only needs exponentially many samples, easy-only never reaches
depth k. **For us:** K12 at depth ≤ 2 is the easy-only regime. Keep K12 and add a 5–10 % tail of 13–30-line,
depth 3–4, 4–6-premise proofs. The depth-3 side pool from the size ladder (486k records) is that tail, already built.
Related: Kordi et al., EACL 2026, arXiv:2511.21692 (easy-only and hard-only both fail); He 2025, arXiv:2512.02677
(nesting depth is a separate axis from length).

I20. **Tong, Cakar, Pehlevan, Feb 2026, arXiv:2602.14404** "Boule or baguette? Task topology, length generalisation
and reasoning traces (PITA)." 23M propositional statements with Lean ND proofs: models trained up to the median length
generalise to longer proofs on *broad, shallow* splits but fail on *narrow, deep* ones; breadth = unique statements per
atom count, depth = proof states including backtracking. **For us:** audit K12 for breadth per size (unique theorems
per atom and premise count; the 2-line saturation at 105k is a breadth ceiling) and raise breadth at each size; PITA
itself, if released, is a source of propositional theorems to re-render in Fitch style after a leak audit.

I21. **Lee, Cai, Schwarzschild, Lee, Papailiopoulos, Feb 2025, arXiv:2502.01612** "Self-improving transformers
overcome easy-to-hard and length generalisation." A plain transformer extends from 10- to 100-digit addition by
sampling its own solutions on slightly harder problems, filtering by correctness, retraining, and stepping the
difficulty up. (Huang et al., NeurIPS 2025, arXiv:2511.07378 prove the mechanism.) **For us:** a *data* loop with the
verifier as filter: generate 13–14-line theorems, let the frozen pretrain attempt them, keep verified proofs, add to
Stage-1, retrain, step up. Several Stage-1 rounds; audit each round. This is self-distillation at the pretrain
stage, distinct from the ladder.

I22. **Qin, Alvarez-Melis, Jelassi, Malach, Apr 2025, arXiv:2504.07052 (COLM 2025)** "To backtrack or not to
backtrack." 3M–144M models on DFS traces vs solution-only: direct solutions win on CountDown, backtracking wins on
Sudoku; fixed traces can lock in a suboptimal strategy; after GRPO the backtracking model gains far more. **Cai,
Wang, Chen, Dhingra, May 2025, arXiv:2505.24273** "How much backtracking is enough?": optimal synthetic backtrack
depth scales with task difficulty (0 to 5+). **For us:** the two sides of H4/B3. Test search-trace data (opened and
discarded boxes, then the clean proof) against clean-only, reading base pass@k *and* post-ladder gains; sweep the
number of discarded branches per proof (0, 1, 3). The verifier accepts unused lines, so no scratchpad is needed, but
the RL reward does not prefer short proofs, so padding can be re-amplified.

I23. **Hua et al., Jul 2025, arXiv:2507.13332** "TAIL: Turing-machine imitation is a length-generalisable reasoner."
Atomic-state dumps and explicit memory fetches in the trace, not thinking style, gave length generalisation. Related:
Phua et al. 2026, arXiv:2605.22221 (traces must expose the current search state locally or models condition on
history). **For us:** Dan's state format already exposes the proof state per action; the data-side analogue is an
auxiliary sequence with an explicit "fetch line k" before each citation. Format-adjacent; park under F unless the
state trainer can take it.

I24. **Li et al., Feb 2025, arXiv:2502.12143 (ACL 2025 findings)** "Small models struggle to learn from strong
reasoners." Models ≤ 3B do worse on long, intricate chains and better on short ones matched to capacity; mixing
long and short beats either. **For us:** supports necessary-steps-only as the default at 9.56M, and suggests a mixed
pool (minimal K12 plus a minority of longer, structured proofs) over all-minimal.

I25. **Kopiczko, Vaze, Blankevoort, Asano, Feb 2026, arXiv:2602.11149** "Data repetition beats data scaling in long-CoT
SFT." At a fixed update budget, 128 epochs on 400 samples beat one epoch on 51,200 (7B, long CoT); training-token
accuracy signals saturation. **For us:** the opposite prior to H8; the size ladder (E7) now tests both directions,
and the pretrain's training-token accuracy per epoch is the saturation signal to log. Chen et al. 2026,
arXiv:2606.16246 (token-replacement augmentation delays overfitting in multi-epoch pretraining) is the augmentation
version: atom permutations per presentation (D1) as an epoch-stretching device.

### I-C. Seen and set aside

LeanConjecturer (Onda 2025, arXiv:2506.22005; "aesop cannot prove it" as a triviality filter: our analogue is "no
proof ≤ 2 lines"); GAR (Wang 2025, arXiv:2510.11769); Chen 2026 self-play theory, arXiv:2606.01861 (random-walk
conjecturing over a theorem graph with a diversity measure); Pythagoras-Prover (Leang 2026, arXiv:2606.12594);
CPL (Kasaura 2025, arXiv:2509.14274); OctoThinker (Wang 2025, arXiv:2506.20512; long CoT destabilises RL);
Thinking-augmented pretraining (Wang 2025, arXiv:2509.20186); Kang et al. 2025, arXiv:2510.01631 (~30 % synthetic
fraction optimum in web pretraining); Zhang et al. 2025, arXiv:2506.11300 (curricula as warm-up cut steps 18–45 %);
Javanmard 2026, arXiv:2603.01293; Sun 2026, arXiv:2601.06911; He 2026, arXiv:2606.04466; Loop-Think-Generalize
(Kohli, COLM 2026, arXiv:2604.07822) and MTP-induced planning (Huang, COLM 2026, arXiv:2604.11912): architecture
or objective changes, out of scope.

### What this scan changes in the plan

1. **The slice fraction is contested.** Dan's 10⁻⁴ (cap 6, 3.3M) versus Zhang's "0.1 % fails, 1 % works" (100M,
   arithmetic) and Gu's "small models need a larger fraction". Add **E8: fraction dial** for one join pattern at
   10⁻⁴ … 10⁻¹, one seed each, read at base pass@k and after the ladder. Until it runs, the spice-rack and join
   slices should be built at 1 % and 10 %, which they are.
2. **Two cheap screens before any ladder:** held-out loss on a textbook-shaped validation set (I6) and base pass@k
   on the type cells (H16, I4). Both run on CPU on the frozen Stage-1 checkpoint.
3. **Keep K12, add the tail.** Easy-only never reaches depth (I19); the depth-3 side pool exists. Add the pool
   **k12_tail10** (K12 with a 10 % slice of depth-3, 13+-line records) to `mix.py`.
4. **Three new measurements for every ladder run:** base log-probability of each RL-found textbook proof
   (amplified vs surfaced, I6), pass@N versus base at large N (I7), distinct verified proofs per problem per round
   on unsolved textbook problems (I8).
5. **Generator audit items** (I15, I17, I20): dependency-graph statistics, breadth per size, padding share, all in
   `pools.md` next to the type cells.

## J. Screening pools for signal before spending a ladder (2026-10-07)

Charles: "Are there ways to quickly verify whether or not a pool would have good signal?" The problem: 27 pools, about
11 ladders (plan § 8). A ladder costs a Stage-1 (≈ 20 A40-minutes) plus Dan's T1 ladder (≈ 3 A40-hours). Screens
below are ordered by cost. "Validation set" throughout = textbook-shaped problems from the *seed split only* (v0,
lean_v2 and the derived problems built from them, 286 problems with reference proofs from the release), never the
sealed 72 and never the test split; it is fixed once and reused for every pool.

### J-A. Zero training (data only, minutes on the laptop)

J1. **Need-weighted cell coverage.** `pools.md` already reports hard-piece coverage and join rates. Weight each type
cell (rule bigram × depth × premise count) by how many *unsolved* textbook problems need it, and score a pool by the
weighted share of cells it fills above a floor (say 10 records). A pool that only fills cells the K12 model already
solves scores zero. Cost: one script over existing stats.

J2. **Nearest-neighbour distance in skeleton space.** For each validation reference proof, the distance (edit distance
on the rule sequence, or 0/1 on skeleton equality) to its nearest pool proof. Pools with many validation proofs at
distance 0–1 give the model something to copy; distance ≥ 4 everywhere means no local support. Cost: minutes.

J3. **Shallow next-step predictor.** Train a count model (next rule given the goal's main connective, the last two rules
and the box depth) on each pool in seconds, then measure its accuracy and perplexity on the validation proofs' steps.
It is Dan's state-action decomposition with a lookup table instead of a transformer; Shen (I6) says pretraining loss
predicts post-RL, and this is the cheapest loss proxy there is. Cost: seconds per pool.

J4. **Compression distance.** Normalised compression distance (gzip) between a pool sample and the validation proofs.
Crude, free, and a useful sanity check that J3 is not measuring something odd.

J5. **Diversity at fixed size.** Distinct skeletons per 1,000 records, rule-bigram entropy, breadth per theorem size
(I20). Akter (I9) says pretraining benefits from pattern diversity; record it now, test the correlation when ladders run.

### J-B. Frozen checkpoints, no training (needs the K12 checkpoint, CPU minutes to an hour)

J6. **Novelty under the K12 model.** Mean per-token loss of each pool's *new slice* under Robbie's K12 Stage-1
checkpoint (`scale-d384-p1200`). Three regimes: near K12's own loss (the slice is redundant, e.g. substitution may be),
moderate (the "edge of competence", where Zhang I1 says RL acts), very high (unlearnable at this size, or a format the
model has never seen). The regime predicts whether a pool can matter at all. Cost: one forward pass over 15,500
records per pool, minutes on CPU.

J7. **Base pass@32 on the slice's own problems.** Sample the K12 checkpoint on the new slice's prompts. If the base
already solves them at a high rate, the slice teaches nothing new; at 0 % it may be too far; in between is the edge.
Pairs with J6 as the two sides of "edge". Cost: minutes on GPU, under an hour on CPU for a 10 % slice.

J8. **Retention risk (Huang, I3).** Which K12 cells does the slice compete with? Measure, for the K12 checkpoint, the
loss on K12 records of the cells the slice displaces (the replaced rows). A slice that removes the only records of
a rare cell will hurt that cell; the replaced-row list is in every pool (`slice` tag). Cost: free, from the data.

### J-C. Small training (minutes of GPU each, or hours on the laptop)

J9. **Continued pretraining from the K12 checkpoint (the strongest cheap screen).** Start from Robbie's K12 Stage-1
checkpoint, continue for a fixed 2,000 steps on the pool, and compare with 2,000 more steps on K12 itself (the control
for extra steps). Read (a) loss on the validation set, (b) base pass@32 on the type cells, (c) J8's retention cells.
This is exactly the mid-training Zhang (I1) found beats RL compute, so a pool that fails to move anything here is
unlikely to move the ladder. About 5 GPU-minutes per pool, 27 pools ≈ 2.5 A40-hours ≈ $1.50. The confound: it
measures *marginal* information on top of K12, not the pool as a from-scratch Stage-1; a pool whose value is in
*removing* something (minimal, wrong) needs J10.

J10. **Micro-scale from-scratch Stage-1.** Dan's pattern dial ran at d=256 (3.3M) on cap 6 and its result is the
reason the slice sizes are what they are, so there is a precedent for a small-scale screen in the repo. Train d=256
for 300 s per pool (one third of the pinned budget), read validation loss and base pass@32 on cells. Gu (I2) warns
that the critical mixing ratio grows as the model shrinks, so a slice that fails at 3.3M may still work at 9.56M:
a positive screen is informative, a negative one is not. About 27 × 5 GPU-minutes.

J11. **Learning-curve slope, not endpoint.** In J9 or J10, log the validation loss every 15 s (the pinned trainer
already writes a curve) and compare slopes over the last third. Kopiczko (I25) uses training-token accuracy as the
saturation signal; a pool that is still improving at the budget's end is under-trained, one that saturated early
is giving epochs away.

### J-D. One RL round instead of eight (the ladder's own first step)

J12. **Round-1 yield (Foster's coverage test, I4).** After a full Stage-1 on a pool (20 A40-minutes), sample k = 32
per target on the ladder's targets and the validation set, verify, and count distinct cells with at least one hit.
Dan's ladder is expert iteration, so round 1's hit set is the only thing round 2 can train on: a pool whose round-1
yield on a cell is zero cannot reach that cell by round 8 except through transfer. 27 Stage-1s ≈ 9 A40-hours, about
a quarter of the budget, and every later ladder reuses its Stage-1.

J13. **Two-round mini-ladder.** Rounds 1–2 only (≈ 45 GPU-minutes instead of 3 hours), read the per-cell slope. Dan's
ladder curves in `preregistration/best-state.md` are front-loaded, so round 2 predicts round 8 for most cells. Use on
the top five pools from J9 and J12.

### J-E. Using what already ran

J14. **Found-proof support.** The RL runs that exist found verified proofs for 183 scraped problems (per_problem.csv).
For each pool, the share of those found proofs whose 2–6-line pieces all occur in the pool, and the same for the
reference proofs of the 86 never-solved problems. A pool that raises the second without lowering the first is the
shape we want. Free.

J15. **Solved-shape saturation.** Which cells did the K12 pretrain (44 solved) and the K12 ladder (183) already cover?
A pool whose new records fall mostly in covered cells is spending its slice on what RL already does. Free.

J16. **Calibrate the screens on the ladders that run.** Whatever pools go through the full ladder, run every screen
above on them first and record the rank correlation afterwards. Shen (I6) predicts loss correlates; Akter (I9)
predicts some pools separate only after RL. Until the correlation is known, a screen may *rank* pools but must not
*drop* a cheap, theory-motivated one.

J17. **Negative control for every screen.** The `k12_wrong100` pool should score at the bottom of every training
screen (J9–J12) and near K12 on every data screen that ignores validity (J2, J4). If a screen does not separate it
from K12, the screen is measuring format, not signal.

J18. **Pool-pair ablations inside one Stage-1.** Interleave two pools' slices with distinct tags and read per-tag loss
from one run (the records carry `slice`), halving Stage-1 count for pairs that are only being compared to each other
(10 % vs 30 % doses).

### Ranked: what to actually do

1. **J3 + J2 + J1 now** (an afternoon, no compute): a `screen_data.py` that writes one row per pool: shallow-predictor
   perplexity on the validation set, nearest-skeleton distance distribution, need-weighted coverage. Negative control J17.
2. **J6 + J7 + J8 next** (K12 checkpoint, CPU): novelty regime and base pass@32 of each new slice; the replaced-cell
   retention list. Classifies every pool as redundant / edge / unreachable before any GPU.
3. **J9 as the GPU screen** (≈ 2.5 A40-hours for all 27): continued pretraining from the K12 checkpoint, validation
   loss and cell pass@32 against the K12-continued control. Add J10 only for the pools whose value is in removal
   (minimal, wrong) since J9 cannot see those.
4. **J12 for the survivors** (Stage-1 + round-1 yield), then full ladders for the top pools plus the E8 dial, which is
   not a screening question but a measurement in its own right.
5. **J16 throughout**: record every screen's prediction before the ladder result is known.

Expected cost of screening everything: data screens free; checkpoint screens free; J9 ≈ $1.50; J12 on ten survivors ≈
$2. That leaves the ladder budget intact for the pools the screens choose.

## K. Diversifying the pool from what was hard (2026-10-09)

Charles asked for ways to diversify the pool, taking inspiration from the pools that were harder to solve. The checkpoint
screens (Robbie's base and Dan's Stage-1 / round-8) say which those were: joins (Dan base 0.36 / 0.30 minimal), the ↔
dial (0.35), counterfactual pairs (Robbie 0.33), distractors (0.63 and the one slice RL did not lift). Each put a
structure the generator never produces on its own into the data at a difficulty the base can almost reach. The far pool
(`k12_aux10`, 0.00 everywhere) marks the other edge: too far and nothing transfers. Target band for anything below:
Dan-base pass@32 0.2–0.5 with need-weighted coverage above K12's 0.51.

Compound the axes that were hard alone:
- K1 joins × distractors: join problems with 1–2 unused premises (shortcut-checked as `scan_pools.py` does). A join plus
  two distractors is a four-premise problem, the textbook cell at 0 of 7.
- K2 joins × bigger formulas: the depth-1 substitution applied to the join slice only (`hard_problems.md` § 5: 25 pieces
  K12 knows by skeleton, not by formula width). Substitution alone was redundant.
- K3 join chains: trigrams and 4-grams from the hard proofs' rule sequences (the ∧E→→E→∧E→→E modus ponens chain no model
  solved), problems whose minimal proof contains the whole path, not the bigram.

Open the generator knobs the hard cells need:
- K4 premise-count dial: 4–5 premises (generator `max_prem`), fractions 10⁻³ … 10⁻¹; the surfacing dial
  `screens_analysis.md` § 6 recommended.
- K5 the long relevant pool, 13–16 lines, depth ≤ 3, textbook-shaped skeletons only (otherwise it is another aux10).
- K6 goal-connective rebalancing in goal mode: quotas for → goals (8 % of K12; hypothetical syllogism never solved),
  ↔ as (A→B)∧(B→A), negated compounds.
- K7 nested ∨E where the two cases need different sub-proofs (0.00 at base in the per-family follow-up; 1,129
  instances in the join pool to seed from).
- K8 long reductio: goal reachable only by assuming its negation, contradiction ≥ 4 lines deep (Dan's cap-8 arm was the
  only one writing accepted 8+-line reductio proofs).

Let difficulty pick the data:
- K9 model-hard mining: Dan's Stage-1 base at pass@32 over the 2.2M generator rows, keep problems solved < 0.5, build a
  pool from them. Selects on the ladder's own base, so audit and read on the textbook sets only.
- K10 minimal-length floor: records with minimal proof ≥ 8 lines only, topped up from the generator (H20, prune easy).

Extend the pairs:
- K11 case-flip pairs: one premise swap changes which ∨E case or which conjunct carries the proof (the current pairs
  change proof length).
- K12 reuse pairs: a derived line cited twice, in two branches (the generator's forward mode rarely reuses a line; the
  ¬E←→E join is usually a reused implication).

First to build: K1, K4, K9 (cheap, existing builders, each aimed at a measured zero), then K5 (largest known gap, most
build work). Screen all with the three screens; keep what lands in the target band.
