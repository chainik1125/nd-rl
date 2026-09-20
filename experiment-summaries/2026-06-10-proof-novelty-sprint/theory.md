# Theory: verifier-RL on ND proving as budgeted search amortization

*W0 deliverable, sprint 2 (2026-06-10). All numbers cited from
`results/numbers.md`. Claims are tagged: **[D]** derivation (follows from
stated assumptions), **[C]** calibration (model fit to sprint-1 numbers, a
consistency check rather than a fit-free prediction), **[P]** falsifiable
prediction (pre-registerable), **[X]** conjecture.*

The facts a theory must explain: (i) a length cliff that tracks the training
cap and ignores 2× data; (ii) a token wall (P(N9)=8.3e-7 vs 3.9e-2) that no
RL budget crossed; (iii) knowledge compounding with a depth-graded retraining
multiplier (×1.2/2.6/8 at 9/10/11 lines) but R_k decay in every protocol;
(iv) a 12-line ceiling for 4-layer models across 3 protocols / 2 seeds /
3 instances, moving to 13 at depth 8 via a 4-deep nested ORE, saturating at
16 layers; (v) HER ≈ token-matched oracle on transfer while avoiding the
oracle's 14pp in-distribution collapse.

## 1. RL as search amortization

**Emission model.** Assume **[A1]** the probability that the policy emits one
*specific* valid L-line proof τ factorizes per line, costing (1−ε) per line
inside the consolidated training envelope (iid greedy 0.876 over ~5.5-line
proofs gives ε ≈ 0.024 at greedy; larger at T=1); **[A2]** each line beyond
the envelope costs an additional factor β ≪ 1, combining the learned length
prior (mass on `QED` after ~7-8 lines) and unconsolidated structure; **[A3]**
any never-trained token multiplies by its probability floor g. Then

    p(τ) ≈ A · (1−ε)^L · β^(L−C)_+ · g^(#gated tokens),      [D given A1-A3]

with C the cap. Per-target success at budget k is 1−(1−p)^k ≈ kp in the rare
regime, so the *found-proof frontier* at total budget K sits where
N_targets · K · p(L) ~ O(1): **exponential rarity past the cap, linear-in-L
log-cost** — the cliff is the generic prediction of any per-step error model,
no reasoning-failure assumption needed.

**Calibration [C].** From chain-2's round-1 beyond-cap histogram
(152×9, 29×10, 2×11) and the static control (188/38/3/0): β ≈ 0.07–0.20 per
extra line, shrinking with depth (the shrink is the capacity wall bending β
downward, §3). Taking β ≈ 0.15 and requiring ≥5 finds (the study's "robust"
criterion): frontier ≈ 8 + 1 + ln(5/152)/ln(0.15) ≈ **10.8 lines** —
observed robust frontier 10 (round 1) to 11 (round 3). The same model makes
the token wall quantitative: a 9-line fixed-format proof contains `N9` once,
so across the entire fixed study (~1,200 × 32 × 5 ≈ 1.9e5 rollouts) the
expected number of 9-line emissions is ≤ 1.9e5 × 8.3e-7 ≈ **0.16** — zero
observed, exactly as predicted; the offset model's g ≈ 4e-2 puts 9-liners
comfortably inside budget. The format change moved one factor by ~5×10^4;
five rounds of RL move frontier-class probabilities by ≤ ~10×/round. That is
the whole explanation of "a tokenization detail was worth more than five
rounds of RL."

**What EI/GRPO are, formally.** With binary verifier reward, both expert
iteration (sample k, keep accepted, distill) and GRPO (group-relative clipped
PG with KL anchor) have fixed points in the family
π* ∝ π_ref(τ) · 1[verifier accepts τ] restricted to what sampling actually
surfaces — Monte-Carlo EM / rejection-sampling-then-distill (the known
RAFT ≈ EI ≈ binary-reward-PG correspondence) **[D for EI, X for the exact
GRPO fixed point]**. Consequence: define the **effective support at budget
K**, S_K(π) = {τ : π(τ) ≳ 1/K}. One round of verifier-RL can only promote
proofs *from* S_K(π_t); what it buys is that S_K(π_{t+1}) ⊋ S_K(π_t) — the
ratchet. RL therefore converts inference-time search into weights at a rate
bounded by what the current round's search can find: **amortization, not
generation**. Gated proofs (g ≈ 8e-7) sit 4-5 orders of magnitude below any
feasible 1/K and are never promoted **[D]**. This is "sharpening vs
expansion" made precise: expansion of *effective* support is real and
iterable; expansion of support is impossible, and the boundary is set by
sampling budget, not algorithm — consistent with GRPO matching hindsight-EI's
pass@64 endpoint (.508) and ceiling (12) while using 3.5× the samples.

## 2. Criticality: when does the ratchet self-sustain?

**Definition.** R_k = N_k / N_{k−1}, where N_k = number of *new* verified
beyond-cap theorems banked at round k. (Caveat first: on a *fixed* finite
target pool R_k → 0 trivially by depletion — static controls decay too
(310/17/14/14/9). R_k is only meaningful against a non-depleting target
stream; sprint-1's measured decay conflates policy subcriticality with pool
exhaustion. Fix in sprint 2: fresh per-round target bands.)

**Consolidation model.** Training on n verified examples of a frontier class
(e.g. "9-line proofs") boosts that class's emission probability by a factor
m(n) that saturates at full consolidation (1/β per line) once n exceeds a
sample-complexity threshold n*; below n*, partially. Then the frontier
recursion is, in the two limiting cases **[D given the model]**:

- **Full consolidation each round** (N_k ≥ n* at the frontier): the new
  frontier line stands in the same relation to the consolidated mass as the
  old one did, so next round's frontier yield equals this round's:
  **R ≈ 1, frontier advances ~1 line/round** until capacity. Criticality —
  not supercriticality: with a depth-uniform pool, R > 1 requires target
  density *growing* with depth (true in unrestricted theorem space, which is
  exponentially branching; false in a fixed band) or a mechanism that
  shortens conditional proofs (lemma citation, §4).
- **Partial consolidation** (N_k < n*): R_{k+1} ≈ min(1, N_k/n*), and since
  each level's raw yield is β × the previous level's, the input shrinks
  geometrically while n* stays constant → R decays after ~log_β(n*/N_1)
  rounds. **Subcriticality is the generic fate of a fixed per-round budget.**

**Calibration [C].** N_1 = 152 nine-liners visibly consolidated length 9
(round-2 10:9 find-ratio jumped 0.19 → 0.83); observed R_2 ≈ 0.40 (73/183
beyond-cap) brackets n* at roughly **10^2–10^2.5 examples** (order of
magnitude only; depletion inflates the estimate). HER's failure to raise R
despite raising data value is then expected, not anomalous: its harvest was
short (39/1,418 round-1 prefixes beyond 8 lines) — it adds mass *below* the
frontier, which the model predicts should raise bulk competence/transfer
(observed: +9-11pp) without feeding R (observed: R declined identically).
Transfer level ∝ consolidated bulk; R ∝ frontier yield. Different quantities,
different knobs.

**The cleanest test [P].** Criticality condition: per-round frontier yield
≥ n*. Yield is ~linear in per-target k in the rare regime, so raising k from
32 to ~256 should multiply round-1 frontier yield ~8× (152 → ~10^3 nine-liners,
29 → ~230 ten-liners ≥ n*), and the 4-layer chain should then hold R ≈ 1 for
~3 rounds, hit the 12-line capacity ceiling early, and *then* crash. If R
still decays from round 1 at k=256, the consolidation model is wrong. This
cleanly separates budget-limited from capacity-limited subcriticality for
~8× one round's GPU cost.

## 3. Capacity ↔ depth: the ceiling is a nesting-depth ceiling in disguise

**Claim.** Generating a valid discharge line (NEGI/IMPI/ORE citing a
subproof) requires computing which assumption is innermost-open at the
current position — exactly bounded-depth balanced-bracket (Dyck-(k,D))
matching over the AS/discharge structure. Known transformer results (Yao et
al. 2021: depth-D bounded Dyck with ~D+1 soft-attention layers; constant-depth
constructions need precision/width growing with length) make the natural
small-width regime **one nesting level resolved per O(1) layers**, with 1-2
layers of overhead for lexing/copying (induction-head copying alone needs
≥2 layers — which is why the 1-layer model cannot emit the format at all,
iid .000). So tracked nesting depth d* ≈ layers − c, c ≈ 1. **[X as a bound;
the Dyck reduction itself is D]**

**The line-length map [C].** A depth-d nested case analysis costs ~3 lines
per level (AS + body + discharge) plus 1: L_min(d) ≈ 3d + 1. Depth 4 → 13
lines: the depth-8 model's ceiling-breaking 13-liner is *exactly* the
minimal-length depth-4 object. The 4-layer ceiling of 12 = depth-3 minimum
(10) + ~2 affordable padding lines (padding is taxed by β per line, so ~2 is
what the length prior affords). The "universal 12" was never about the
number 12: **it is nesting-3 expressed in lines, and the 8-layer 13 is
nesting-4.** All chain-1 discoveries had nesting ≤ 3, mirroring training; no
4-layer model across any protocol ever emitted nesting 4.

**Why 16 ≈ 8 [X→P].** At depth ≥ 8 the binding constraint flips from
capacity to *data*: nesting-4 is 0.08% of training proofs and nesting-5
essentially absent, so there is nothing for extra layers to amortize. Two
predictions: **(P1)** enrich training/seed data with nesting-4/5 proofs and
the 16-layer model separates from the 8-layer (frontier 16 ≈ 3·5+1); the
4-layer model does not move (capacity-gated). **(P2)** On a *zero-nesting*
family (§7, P-iii) the 4-layer frontier should run well past 12 lines — if
it still ceilings at ~12 on shallow proofs, the nesting account is wrong and
something positional/length-prior-like is binding.

## 4. Where verifier-RL provably wins: ladder distributions

From §1-2, RL's edge over SFT-on-more-of-the-same is exactly one thing: it
*moves the sampling distribution between rounds*. So the regimes where RL
wins biggest are those with **short conditional description length at every
level, long absolute description length** **[D given §1-2]**: a family
T_0, T_1, …, T_J where the minimal proof of a T_j theorem given consolidated
T_{j−1} patterns costs ≤ c lines (constant), but from scratch costs ~c·j.
Then base-policy probability is β^{cj} unconditionally vs β^c after
consolidating level j−1: SFT/sampling needs ~β^{−cj} samples to reach level
j; iterated verifier-RL reaches it in j rounds of constant cost S ≥ n*/yield.
**Exponential-to-linear separation** — the clean "RL creates knowledge"
regime. Two structural requirements on the distribution: *thick rungs*
(≥ n* instances per level, else consolidation starves) and *within-capacity
rungs* (each level's new structure inside d*).

One calculus-level catch: our Fitch format has **no lemma-citation rule** —
every proof is self-contained, so "lemma reuse" can only act through weight
amortization of *patterns*, not through proof compression. Adding a `LEM`
rule (cite any banked theorem as a line; verifier checks the bank) makes the
conditional-length mechanism available in the calculus itself and is the
single highest-leverage change for observing sustained R ≥ 1 **[X]**: with
citation, consolidating level j−1 shortens level-j proofs *for the verifier*,
not just for the policy.

## 5. A theory of proof value as RL training data

**Definition.** The training value of a proof π for policy θ is the expected
gain in discovery probability it induces over future targets:
V(π; θ) ≈ Σ_t [p_{θ′}(t) − p_θ(t)]. Decomposing via §2's consolidation
model, V factors approximately into **(rarity under the current policy) ×
(downstream usage) × (learnability gate)** **[D-ish, from the model]**:

    V(π; θ) ∝ bits(π) := −log π_θ(π)   [the amortization gain if learned]
            × U(π)    [how many targets share its patterns/structure]
            × 1[nesting(π) ≤ d*] · 1[class count ≥ ~n*]  [learnable at all]

Three immediate consequences. (1) **Value is peaked at the zone of proximal
development**: in-distribution proofs have bits ≈ 0 (the flat primitives
control, observed); beyond-capacity or singleton-class proofs have
learnability 0 (the oracle arm bought transfer only via its
*within-capacity* content, and the fixed chain's N9-gated long proofs would
be worth nothing to it). (2) **Style is a multiplier on the gate, not on the
value**: the oracle's 14pp iid collapse vs HER's none, at matched tokens and
matched transfer, says off-style data costs integration without buying
transfer — predicting W1's style-matched oracle recovers iid retention at
equal transfer **[P]**. (3) For sprint-2's measures:

- **bits / description length**: should be the best single correlate of ICL
  *ladder height* (both measure rarity under a competent prior), with
  discontinuous ladder jumps at nesting-depth increments rather than smooth
  length scaling **[P]**.
- **lemma-value** (# targets unlocked): the direct estimator of U; predicted
  best single predictor of seeding payoff on R_k and transfer **[P]**.
- **multiplicity**: high multiplicity ⇒ high reachability ⇒ resampling finds
  it anyway ⇒ *low marginal* seed value at matched length; predict negative
  partial correlation with seeding payoff controlling for length **[P]**.
- **genus/cyclomatic complexity**: inverted-U — structural reuse packs more
  pattern per token (value ↑) until it outruns capacity/class-thickness
  (value → 0) **[X]**.

**Seeding predictions for W5 [P].** Seed HER-EI with (a) high-novelty
(high-bits, thick-class, within-capacity) vs (b) low-novelty proofs at
matched token budget: (a) raises round-1 frontier yield and the retraining
multiplier ⇒ higher R_1, R_2, a longer-lived ratchet, and a pass@64 transfer
gain; (b) tracks the primitives control (flat). The sharpest version is an
**interaction prediction**: nesting-4 seeds are among the *most* valuable
for an 8-layer model and near-worthless for a 4-layer one — value rank
*inverts at the capacity boundary*. A 2×2 (seed novelty × model depth)
design tests the whole framework at once; the framework is falsified if
high-bits seeds help the 4-layer model beyond its nesting ceiling, or if
low-bits seeds match high-bits seeds within capacity.

## 6. Predicate-logic outlook (brief)

Quantifiers add eigenvariable side conditions (∀I/∃E freshness): validity
acquires *global* constraints, and a locally-valid prefix can be doomed (a
bad early instantiation makes the goal unreachable). Predictions **[P/X]**:
(i) per-step error model gains a term growing with the number of live
variables, so β is smaller and amortization is worth *more* (bigger RL-vs-SFT
gap) but criticality is harder (lower yields); (ii) **HER degrades**: valid
prefixes stop implying extendability, so harvested prefixes include dead
ends — expect the HER-vs-vanilla gap to shrink or invert unless harvesting
filters for goal-consistency; (iii) a new token wall: a bounded variable
inventory x1..xn in training makes proofs needing n+1 fresh variables
literally unwritable — audit before any capability claim (the sprint-1
lesson, predicted to recur); (iv) capacity now tracks variable-dependency
bookkeeping (quantifier alternation rank), not just subproof nesting —
depth requirements should scale with alternation depth.

## 7. Concrete ND distributions for sprint 2+

- **P-i. Lemma-ladder with a citation rule.** Add `LEM` (cite banked
  theorems; verifier checks the bank). Generate J-level DAGs: level-j
  theorems provable in ≤4 lines citing one level-(j−1) lemma, ≥ ~4j lines
  from scratch; ≥200 theorems/level (thick rungs); atoms/connectives as
  sprint 1. Prediction: with citation, R ≥ 1 for many rounds (supercritical
  until pool edge); without, subcritical at the usual rate. *The* criticality
  testbed.
- **P-ii. De Morgan / negation towers (pure nesting).** Targets like
  ~(φ1 ∨ (φ2 ∨ (… ∨ φd))) ⊢ ~φ1 ∧ (~φ2 ∧ …), recursing to depth d;
  independently pad length with conjunct width. Sweeps d at fixed L and L at
  fixed d → de-confounds the §3 claim. Prediction: 4-layer solves to d ≈ 3
  regardless of L; 8-layer to d ≈ 4-5; data enrichment moves 16-layer only.
- **P-iii. Implication/resolution chains (zero nesting, pure length).**
  Premises P1, P1→P2, …, P_{L−1}→P_L plus distractor implications (random
  DAG; distractors defeat copy heuristics and stress long-range index
  citation), target P_L; nesting 0, length exactly controllable. Prediction:
  the 4-layer ratchet runs well past 12 lines here (falsifier for §3); the
  cliff still tracks the cap (the cliff is per-step, not nesting).
- **P-iv. Keystone distributions (support-expansion test).** 99% routine
  targets + 1% requiring one reusable trick the SFT generator never writes
  (style hole, like sprint-1's explosion at k=256), with a graded bridge of
  near-miss targets in the RL pool only. Prediction: SFT on more generator
  data never finds the keystone; HER-EI climbs the bridge and amortizes it —
  the cleanest demonstration that RL's win is moving the sampler, with the
  bridge density as the dial that turns the win on and off.
- **P-v. Multiplicity-matched twins.** Pairs of families matched in length
  and nesting, differing ≥10× in enumerated proof multiplicity. Seed RL with
  proofs of one twin; predict seeding helps the low-multiplicity twin most
  (high-multiplicity is found by resampling anyway) — the causal validation
  of multiplicity as a (negative) value signal.

Cross-cutting protocol fix: measure R_k on **fresh per-round target bands**,
never a fixed pool, or criticality is masked by depletion.

## What the experiments should test

The framework reduces to four falsifiable load-bearing claims, each with a
cheap test: (1) *budget-limited vs capacity-limited subcriticality* — raise
per-target k 32→256 for one 4-layer round and check R holds ≈1 until the
12-line ceiling (§2, ~8× one round's cost); (2) *the ceiling is nesting, not
length* — run the zero-nesting chain family (P-iii) on a 4-layer model and
the nesting towers (P-ii) across depths, expecting frontier >12 on the
former and a depth-quantized d* ≈ layers−1 on the latter, with the 3d+1 line
map; (3) *proof value = bits × usage × learnability* — W4 should find ladder
height tracking −log π_θ with jumps at nesting increments, and W5 should
find the novelty×depth interaction (high-nesting seeds help 8-layer, not
4-layer) plus low-multiplicity > high-multiplicity seeds at matched length,
with W1's style-matched oracle recovering the 14pp iid loss; (4) *RL's win
condition is ladder structure* — P-i (lemma citation) should produce the
study's first sustained R ≥ 1, and P-iv the first clean policy-support
expansion that SFT data scaling provably cannot match. If (1) and (4) both
fail, verifier-RL in this testbed is pure amortization with no critical
regime, and the right conclusion is that scale (depth) plus format audits do
all the work; if they succeed, we have the recipe — thick-rung,
within-capacity ladders plus frontier-yield-≥n* budgets — for making
self-training self-sustaining.
