---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - venhoff2025base
---

# Constructive model diffing: does the base model, steered at the right moments, reproduce the RL model? (gap recovered)

Paper: [@venhoff2025base] (Venhoff, Arcuschin, Torr, Conmy, Nanda; "Base Models Know How to Reason, Thinking Models
Learn When")
Source: arXiv 2510.07364v4 (dated 7 Jul 2026; HTML rendering read: Abstract, Sec. 1-6, App. E.1-E.2; App. A-D, G by
headings); v1 abstract page also read. All figures below are v4's. **Version warning:** v1 (Oct 2025) claimed, over
"three base and four thinking models", that the hybrid "recovers up to 91% of the performance gap ... while steering
only 12% of tokens" and that "pre-training is when models acquire most of their reasoning mechanisms" (v1 Abstract).
v4 splits RL-trained from SFT-distilled pairs and finds the general claim holds only for RL (below); cite v4.

## Learnings

- **Question.** "What do thinking language models learn during training that their base models lack?" (Abstract).
  The diff is decomposed into "reasoning mechanisms (category vectors that can induce a reasoning behavior in the
  base model) and reasoning heuristics (a classifier determining when a mechanism should fire)" (Abstract).
- **Headline.** Nine base/thinking pairs (four RL-trained Open-Reasoner-Zero, four SFT-distilled R1-Distill, one
  mixed QwQ-32B): "hybrid models recover roughly 76% of the RL base-to-thinking gap but only 11% of the SFT gap",
  read as "RL primarily teaches heuristics for orchestrating pre-existing base mechanisms, whereas SFT-distillation
  installs new ones" (Abstract). QwQ-32B (SFT then RL) "recovers ∼20%" (Sec. 3.7).
- **Pipeline.**
  1. Taxonomy: small Top-K SAEs (5-50 latents) on sentence-averaged activations of thinking-model traces; LLM-labelled
     categories (Sec. 2).
  2. Category vectors, trained *in the base model*: thinking-model rollouts are teacher-forced "through the base model,
     identifying token positions where the two models disagree"; each such position is labelled with the SAE's top
     category from the thinking model's hidden state; then "Jointly optimize a per-category vector and a small MLP that
     predicts a steering coefficient, minimizing cross-entropy on the thinking model's next token" (Sec. 3.3). V ∈
     R^{K×d} at "approximately 37% of model depth"; coefficient MLP "Linear (d,512)" → GELU → K Softplus heads reading
     the base residual (Sec. 3.3; App. E.2).
  3. Hybrid: the base model generates; "First, we check whether the base- and the thinking model disagree in their next
     token prediction"; at disagreements the SAE's top category is steered with its vector (Sec. 3.5-3.6).
- **The measure.** "Rec.% is the fraction of the base-to-thinking gap that the hybrid recovers", i.e. (hybrid − base) /
  (thinking − base), on GSM8K, MATH500 and a disjoint Hendrycks-MATH subset; "Base and hybrid models decode greedily;
  thinking models sample 3 rollouts at temperature 0.6" (Table 1 caption). Examples: ORZ-0.5B 83.0 / 95.3 / 107.4;
  R1-Distill-1.5B −3.0 / 5.7 / −8.5 (Table 1). A second, independent signal: RL pairs' category vectors "converge to
  substantially lower cross-entropy than the SFT-distilled R1 models" (Sec. 3.4, Fig. 3).
- **Sparsity and controls.** RL pairs "steer only ∼5-12% of tokens per problem" (Sec. 3.7, Table 2), and "with only
  5-15 distinct category vectors applied sparsely, the hybrid cannot succeed by memorizing outputs" (Sec. 3.6).
  Ablations on ORZ-1.5B / 32B: full pipeline "∼77%"; random category "drops recovery to 20-28%"; norm-matched random
  vectors give −28 % and −13 %; random positions −8 % / 28 % (Sec. 3.8, Fig. 4).
- **Proposed reason.** "RL training optimizes the model's own outputs against a reward signal, encouraging it to
  leverage what it already knows" (Sec. 3.7).

## Evidence and limitations

- The heuristic is not extracted into a standalone component: the thinking model runs alongside the base model at
  every token to detect disagreements and to pick the category (Sec. 3.5). The measure therefore asks whether the
  base model's own mechanisms suffice *given an oracle for when and which*; the oracle supplies up to log2 K bits per
  steered position plus the coefficient MLP's input-dependent magnitude.
- The category vectors and the MLP are trained by gradient descent on the thinking model's tokens, so the hybrid
  contains learned parameters (K × d + ~d × 512); the random-vector ablation shows the directions carry the signal,
  but the fitted capacity is not compared with the size of the base-to-thinking weight update.
- The authors' own alternative: "the lower recovery for SFT-distilled models could reflect either genuine mechanism
  modification or limitations in our category vector optimization" (Sec. 5). RL vs SFT is confounded with model
  family (ORZ is Qwen2.5-based; distilled models include Llama) and with the teacher.
- Greedy accuracy at one budget (2048 tokens); no pass@k, no likelihood of reference solutions.

## Connections and questions

- **Definition offered:** the base model 'has' the thinking model's mechanisms to the extent that a small set of
  base-model steering directions (one per behaviour category), applied only where the two models disagree, recovers
  the thinking model's accuracy: capability-in-base = gap recovered (Rec.%), with category-vector held-out CE as a
  second readout.
- **New vs better access:** explicit and graded. High Rec.% (they see ~76 % for RL) → the fine-tune mainly learned
  *when* to use existing mechanisms (elicitation / orchestration); low Rec.% (~11 % for SFT) → it installed or
  changed mechanisms (creation). No threshold is stated; the four ablations act as the null distribution.
- **Null / floor:** random category, norm-matched random vectors, magnitude-only, random positions (Sec. 3.8). The
  base model itself is the floor of the gap. 'Any k' is not addressed (greedy decoding).
- **Transfer to our setting:** possible, with two adjustments. Protocol: teacher-force r16's proofs (holdout250, the
  excluded-middle family) through pend; at positions where pend's greedy next token differs from r16's, label the
  step by tactic category (our steps are explicit, so no SAE is needed: e.g. intro / apply / cases / exact /
  classical-step); train K vectors (K × 384) at layer 2 of 6 (≈ 37 % depth) on pend to maximise r16's token at those
  positions; build the hybrid and report gap recovered on held-out theorems with Lean (pass@1 greedy and pass@256).
  For the excluded-middle schema specifically: does a 'classical step' vector exist in pend whose application, at
  r16's disagreement points, makes pend produce A ∨ ¬A proofs? Cost: one teacher-forced pass over r16's proofs, a
  few minutes of vector training, Lean checks of hybrid samples. Adjustments / failure modes: (1) **information
  leak** — our proofs are 10-40 tokens and our categories nearly name the next tactic, so the oracle's log2 K bits per
  disagreement can carry most of the proof; report the oracle's bits next to the proof's teacher-forced bits under
  pend (cf. expert bits in `phuong2024dangerous.md`) and keep K small; (2) **capacity** — a d×512 MLP is ~2 % of our
  9.6 M parameters, not negligible as in 32 B models, so use a scalar coefficient per category or count the fitted
  parameters as part of the elicitation cost; (3) r16's own proofs are selected for high π_r16, so also run the
  construction on reference proofs.
- Related: `ward2025repurposes.md` (base-derived vector works only in the tuned model), `prakash2024finetuning.md`
  (cross-model patching), `jain2023mechanistically.md`, `hofstatter2025elicitation.md` (steering as elicitation),
  `greenblatt2024passwordlocked.md`, `aghajanyan2020intrinsic.md` and `mukherjee2025subnetworks.md` (size of the
  update as the alternative 'how much was added' measure).
