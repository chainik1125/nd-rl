# Literature landscape: verifier-driven RL for natural-deduction proving

Scope: actionable references for our sprint, organized by the five questions. Each entry has a 1–2 line "how we'd use it." Verification notes and unconfirmed flags are marked **[FLAG]**.

---

## 1. Computable proxies for proof complexity / interest / novelty

The deepest theory comes from structural proof theory, where the **cut rule = lemma use**: a cut introduces an auxiliary formula (a lemma) that is then consumed. Cut-elimination removes lemmas at the cost of (worst-case hyper-exponential) blow-up, so the *gap between a proof and its cut-free normal form* is a principled measure of how much "lemma leverage" a proof exploits.

- **Carbone, "Logical structures and genus of proofs," Ann. Pure Appl. Logic 161(2):139–149, 2009.** Defines the **genus** of a proof as the topological genus of its logical flow graph (the graph tracing atom/formula occurrences through inference rules). Cut-free proofs have acyclic flow graphs (genus 0); cuts + contraction create cycles, and the number of independent cycles ≈ genus is a non-trivial structural invariant. *How we'd use it:* compute the logical-flow-graph genus (or simply the **cycle count / first Betti number** of the flow graph) of each ND proof as a cheap "structural richness" proxy — high genus ⇒ heavy reuse of intermediate results. Verified: paper exists, journal/volume confirmed via ScienceDirect/dblp.
- **Buss, "The undecidability of k-provability" / Carbone, "Interpolants, cut elimination and flow graphs for the propositional calculus," APAL 1997.** Origin of the **logical flow graph**; establishes acyclicity of cut-free and contraction-free proofs. *How we'd use it:* gives us the exact construction to build the flow graph from an ND derivation; acyclicity is the baseline against which we score cycles.
- **Carbone, "Duplication of directed graphs and exponential blow up of proofs," APAL 100, 1999.** Formal link between flow-graph cycles and proof-size blow-up under cut-elimination. *How we'd use it:* justifies **cut-elimination size ratio** (|cut-free normal form| / |original|) as a complementary, very-cheap-to-approximate "lemma value of the whole proof" measure.
- **Lemma usefulness / lemma mining — Kaliszyk & Urban, "Learning-assisted theorem proving with millions of lemmas," J. Symbolic Computation 2015 (arXiv:1402.3578); Kaliszyk, Urban, Vyskočil, "Lemma Mining over HOL Light," 2013.** Operational definition: **a lemma's usefulness = how many later theorems it (re)proves or shortens**, mined from the inference DAG of a large library. Adding the top-ranked mined lemmas improved ATP coverage by ~5% (conservative) on HOL Light/Flyspeck. *How we'd use it:* directly portable — score a subproof/lemma by its **reuse frequency across our generated proof corpus** (a PageRank-style centrality on the lemma-dependency DAG). This is our most concrete "usefulness" metric.
- **Kolmogorov/MDL view & proof mining.** Kolmogorov complexity / MDL formalize "interest = compressibility," but K is **uncomputable** **[FLAG: do not implement K directly]**; use compressed proof length (gzip of the serialized derivation, or DAG-size after common-subexpression elimination) as an MDL surrogate. **Proof mining (Kohlenbach, *Applied Proof Theory*, 2008)** extracts quantitative bounds from proofs — relevant conceptually (proofs carry extractable content) but heavyweight and analysis-specific; **not** a near-term tool for us.

**Implication for our sprint:** We have three *computable* structural proxies that need no human labels: (i) flow-graph cycle count / genus, (ii) cut-elimination size ratio, (iii) lemma-reuse centrality on the proof DAG. These can rank our generated proofs by "interest/novelty" for the curriculum experiment in §4.

---

## 2. In-context learning for formal / symbolic proofs

The consistent finding: **plain few-shot ICL is weak at *constructing* proofs**, though strong at single-step / answer-only deduction. Backtracking-search agents and self-generated context close much of the gap.

- **Saparov & He, "Language Models Are Greedy Reasoners" (PrOntoQA), ICLR 2023.** Synthetic ND-style multi-hop deduction; CoT GPT-4 reaches very high *answer* accuracy on single-rule (modus ponens) chains but degrades sharply with proof depth and distractors. *How we'd use it:* PrOntoQA's generator is a near-drop-in template for our **100-shot ND-completion ladder**; gives a published baseline curve to compare against.
- **Han et al., FOLIO (2022) and downstream evals.** First-order, harder; few-shot GPT-4 proof/label accuracy drops well below modus-ponens-only tasks; **proof accuracy <45% once ≥3 steps** are required in tutoring-system studies (arXiv:2505.04736, 2025). *How we'd use it:* sets expectations — our ladder should expect a steep depth cliff; measuring *where* the cliff is per-model is the point.
- **Thakur et al., "An In-Context Learning Agent for Formal Theorem-Proving" (COPRA), 2023/2024 (arXiv:2310.04353).** Few-shot prompting alone is *not* effective for proof generation; wrapping GPT-4 in a backtracking search with environment feedback + retrieval is. *How we'd use it:* motivates treating raw 100-shot ND as the **lower bound** of capability; our RL/expert-iteration loop is the "internalize the search" upgrade.
- **"Discovering New Theorems via LLMs with In-Context Proof Learning in Lean" (arXiv:2509.14274, 2025); TheoremLlama (2024).** Show proof ability improves when **the model's own verified proofs are fed back as context**. *How we'd use it:* validates seeding the few-shot context with *self-generated, verifier-passed* ND proofs — bridges §2 to §3/§5.

**Implication for our sprint:** Budget the 100-shot ND eval as a *capability ladder by proof depth*, expecting a sharp drop around depth 3–5. Plain ICL is the floor; the interesting deltas come from (a) self-generated verified exemplars in context and (b) amortizing search via RL. **[FLAG]** I did not find a paper reporting clean k-shot ND *completion* accuracy at k=100 specifically — our eval appears to fill a genuine gap.

---

## 3. RL as amortized search / expert iteration / STaR / ReST

The unifying frame: **RL (and rejection-sampling SFT) compiles expensive inference-time search into the policy weights** — you search once, keep the wins, train on them, and the next round needs less search.

- **Polu & Sutskever, "Generative Language Modeling for Automated Theorem Proving" (GPT-f), 2020; Polu et al., "Formal Mathematics Statement Curriculum Learning," 2022.** Introduced **expert iteration** for formal math: alternate proof search ↔ SFT on verified proofs; EI beats search-only at fixed compute and *automatically discovers a curriculum* (it learns to solve progressively harder statements). *How we'd use it:* this is our **core bootstrapping loop**; cite as the canonical "amortized search" result and the origin of emergent curricula (ties to §4).
- **Zelikman et al., "STaR: Bootstrapping Reasoning with Reasoning," NeurIPS 2022.** Generate CoT, keep only those reaching the correct answer (with rationalization for failures), fine-tune, repeat. *How we'd use it:* the minimal self-training recipe for ND when we have a verifier as the correctness oracle.
- **Singh et al., "Beyond Human Data: Scaling Self-Training" (ReST/ReST-EM), 2023; Yuan et al., "Scaling Relationship... RFT," 2023 (arXiv:2308.01825).** ReST-EM = generate (E-step) / filter+train (M-step); RFT shows **diverse correct self-generated paths multiply gains** (LLaMA-7B GSM8K 35.9%→49.3% by pooling diverse rejection samples). *How we'd use it:* justifies sampling *many diverse* ND proofs per goal and prioritizing **distinct proof structures** (links to our §1 structural proxies — dedupe by flow-graph signature, not surface form).
- **DeepSeek-Prover (V1 2024 / V2 2025); InternLM2.5-StepProver (2024).** Large-scale expert iteration in Lean: generate → Lean-verify → retrain, repeating until no gains. *How we'd use it:* the modern, verifier-in-the-loop instantiation we are emulating at ND scale; evidence the loop keeps yielding harder solved problems.

**Implication for our sprint:** Frame our pipeline explicitly as "search now, amortize into the policy." Use a verifier as the filter, prioritize **structurally diverse** wins (not just count), and expect an emergent difficulty curriculum à la Polu et al.

---

## 4. Curriculum / data-selection for RL and self-training

Evidence converges on **easy→hard scheduling**, with a twist directly relevant to our novelty-seeding plan: hardness helps *only when the model can occasionally succeed*, and **novel/diverse** examples often matter more than raw difficulty.

- **Parashar et al. (and related), "Curriculum RL from Easy to Hard Tasks Improves LLM Reasoning," 2025 (arXiv:2506.06632).** Easy→hard schedule for RLVR; crucially, **fading out easy tasks prevents overfitting**, and curricula reach a target with fewer samples than flat training. *How we'd use it:* schedule our RL by an *automatically computed* difficulty (model pass-rate or proof depth), not a fixed set; fade easy ND goals out over training.
- **"How Difficulty-Aware Staged RL Enhances Reasoning," 2025 (arXiv:2504.00829).** Staged exposure to harder data + difficulty-bucketed data selection markedly improves RL optimization. *How we'd use it:* bucket generated ND proofs by difficulty and stage them.
- **RFT diversity result (Yuan et al., 2023, above).** Gains are *multiplicative in diversity × correctness* of self-generated data — i.e. **which** self-data you keep matters as much as how much. *How we'd use it:* keep proofs that are **novel by structure** (high genus / new lemma-reuse pattern), not duplicates of already-solved templates — exactly the high- vs low-novelty seeding contrast we plan to test.

**Implication for our sprint:** Our planned "seed RL with high- vs low-novelty proofs" is well-motivated: theory says easy→hard + diverse retained data wins, but there is genuine open disagreement about whether *novelty* specifically (vs plain difficulty) is the active ingredient. **[FLAG]** I found no paper that isolates structural-novelty-selected proofs as the RL seed — our experiment is a real contribution if it cleanly separates novelty from difficulty (control for pass-rate while varying genus/lemma-novelty).

---

## 5. Model collapse vs verifier-filtered self-training

Naive recursive training on self-data collapses; **accumulation** and especially **verification** break the curse — with at least one line of evidence that *verified synthetic can beat real data*.

- **Shumailov et al., "AI models collapse when trained on recursively generated data," Nature 2024 (orig. "The Curse of Recursion," arXiv:2305.17493).** Iteratively *replacing* data with model outputs erases distribution tails → irreversible degradation. *How we'd use it:* the failure mode we must avoid; motivates never training purely on unfiltered self-output.
- **Gerstgrasser et al., "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data," 2024 (arXiv:2404.01413).** Theory + experiments: **accumulating** (not replacing) bounds test error independent of iteration count → no collapse. *How we'd use it:* always **accumulate** verified ND proofs alongside seed data rather than swapping; gives the safe default mixing policy.
- **"Self-Verification Provably Prevents Model Collapse in Recursive Synthetic Training," NeurIPS 2025 (OpenReview X5Hk8aMs6w).** Confidence-based self-filtering alone yields finite-sample bounds that prevent collapse, even fully-synthetic. *How we'd use it:* our **verifier is a stronger-than-confidence filter** (it's sound), so we should be strictly safer than this regime.
- **Yi, Liu, Cheng, Xu, "Escaping Model Collapse via Synthetic Data Verification," 2025 (arXiv:2510.16657).** Injecting an external verifier prevents collapse and gives near-term gains; **verifier-based synthetic retraining can outperform the real-data baseline when verifier quality is high enough**, but estimates drift toward the verifier's "knowledge center" and plateau/reverse if the verifier is imperfect. *How we'd use it:* this is the closest published echo of our finding (self-generated verified > external data). **[FLAG — important caveat]** Their theory predicts the advantage is bounded by *verifier soundness*; a Lean/ND checker is essentially perfect, so the plateau-and-reverse failure should **not** bite us — but we should state this explicitly, because their result does *not* guarantee unbounded improvement for noisy verifiers.

**Implication for our sprint:** Our "self-generated verified beats external" result is consistent with Yi et al. (2025) and the self-verification line, and is *strengthened* by ND having a sound checker (verifier quality ≈ perfect, so no center-of-mass drift). This is a defensible, well-supported claim — frame it as "with a *sound* verifier we sidestep the verifier-quality ceiling that limits noisy-verifier results."

---

## Top 3 tools/measures to implement (ranked)

1. **Lemma-reuse centrality on the proof DAG (§1, §4).** Build the dependency DAG over all generated ND (sub)proofs; score each lemma by reuse frequency / PageRank. Directly operationalizes Kaliszyk–Urban "lemma usefulness," is cheap, and doubles as the **novelty/usefulness signal for curriculum seeding**. Highest value-for-effort.
2. **Logical-flow-graph genus / cycle count (§1).** Per-proof structural-richness invariant (Carbone). Compute Betti number of the flow graph (or genus if feasible); use to **dedupe by structure** and to define "novelty" independent of difficulty for the §4 high- vs low-novelty experiment. Moderate effort (graph construction from ND rules), high theoretical payoff.
3. **Cut-elimination size ratio (§1).** |cut-free normal form| / |original proof| as a one-number "lemma leverage" proxy, plus gzip-length as an MDL/compressibility surrogate. Cheapest of the three; good sanity cross-check for (1)/(2). Implement normalization on small ND proofs first (blow-up can be large).

### Verification notes
- Confirmed: Carbone genus paper (APAL 161(2):139–149, 2009); Kaliszyk–Urban lemma mining + ~5% figure; Shumailov (Nature 2024); Gerstgrasser (arXiv:2404.01413); Polu–Sutskever GPT-f; STaR; RFT 35.9%→49.3%; Yi et al. (2510.16657) "verifier can beat real-data baseline, drifts to verifier center."
- **[FLAG] Unconfirmed / gaps:** no published clean **100-shot ND-completion** accuracy curve (our eval fills this); no paper isolating **structural-novelty-selected** proofs as an RL seed (our §4 experiment is novel); the "<45% at ≥3 steps" figure is from one 2025 tutoring-systems study (arXiv:2505.04736), treat as illustrative not definitive. Kolmogorov complexity is uncomputable — use compressed-length surrogates only.
