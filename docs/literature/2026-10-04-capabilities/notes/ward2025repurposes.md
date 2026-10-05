---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - ward2025reasoning
---

# Reasoning fine-tuning repurposes a base-model direction: the 2 × 2 steering test (vector source × steered model)

Paper: [@ward2025reasoning] (Ward, Lin, Venhoff, Nanda; short paper in ICML template)
Source: arXiv 2507.12638v1 (HTML rendering read in full: Abstract, Sec. 1-5, App. A-C)

## Learnings

- **Claim.** Backtracking in DeepSeek-R1-Distill-Llama-8B "is in part driven by a repurposed direction already
  present in base model activations": a direction in base Llama-3.1-8B's residual stream "which systematically
  induces backtracking when used to steer the distilled reasoning model", while "this direction does not induce
  backtracking in the base model, suggesting that the reasoning finetuning process repurposes pre-existing
  representations to form new behavioral circuits" (Abstract). Framing: reasoning models repurpose base
  representations "rather than learn new capabilities from scratch" (Abstract); "base models may possess latent
  reasoning capabilities which are unexpressed until they are extracted by the finetuning process" (Sec. 1).
- **Method.** Difference-of-means steering vectors, v = MeanAct(D+) − MeanAct(D), where D+ are positions labelled as
  backtracking by GPT-4o and D all positions (App. A, Eq. 3). Two twists: (1) a negative token offset (best "∼−13
  to −8" tokens before the event at layer 10, Sec. 3.1), so the vector captures what precedes the decision, not the
  "Wait" token itself; (2) the vector is computed "separately on residual stream activations from both the base and
  reasoning models on the same reasoning traces" (Sec. 3): the base model reads the reasoning model's traces.
- **The 2 × 2 result** (Sec. 3.2, Fig. 3). Base-derived vectors "reliably induce backtracking when used to steer the
  reasoning model, and have comparable performance to their reasoning-derived counterparts"; "neither base-derived
  nor reasoning-derived steering vectors invoke backtracking behavior in the base model" — "the base model never
  exhibits backtracking behavior, even when steered with the reasoning model-derived backtracking-inducing vector"
  (Fig. 3 caption). The two vectors have "high cosine similarity of ∼0.74" (Sec. 3.2).
- **Controls.** "the backtracking steering vector significantly outperforms all tested baselines": mean activation,
  Gaussian noise, self-amplification, and difference-of-means vectors for other sentence types (deduction,
  initializing) (Sec. 3.3, Fig. 4). Caveat they report: "adding Gaussian noise to activations has a nontrivial effect
  on the fraction of output words" that are "Wait" (Sec. 3.3).
- **Not a token shortcut.** Logit lens: "the base-derived steering vectors do not decode to backtracking keywords, yet
  are successful in eliciting backtracking in the fine-tuned model" (Sec. 4.1, Fig. 5; Eq. 2 score).
- **Not the whole mechanism.** The direction "is densely present in model activations" and does not cleanly track
  backtracking as a probe (Sec. 4.2); there are "instances both where backtracking occurs while the direction is not
  present, and instances where the direction is present but backtracking does not occur" (Sec. 5).

## Evidence and limitations

- One model pair: "we examined a single reasoning model" (Sec. 5); results are "an existence proof for latent
  reasoning-related representations in base models, rather than a comprehensive explanation of reasoning behavior"
  (Sec. 5).
- Behaviour metric: fraction of words in {wait, hmm} (Sec. 2.2, Eq. 1); "∼83%" of keyword-flagged sentences are real
  backtracking by the authors' own judgement and keyword-vs-GPT-4o F1 is 54-65 % (App. C, Table C.1). No task
  accuracy is reported for the steered models.
- The fine-tuned model is a *distilled* reasoning model (Abstract wording). That the distillation was supervised
  fine-tuning on R1 outputs, not RL, comes from the DeepSeek-R1 report and was not checked here; the paper itself
  does not separate SFT from RL effects.
- Short paper (four pages plus appendix): figures carry most results; no statistics beyond one-standard-deviation error bars (Fig. 3).

## Connections and questions

- **Definition offered:** implicit. A capability's *representation* is present in the base model if a direction
  computed from base activations (difference of means at positions preceding the behaviour, on the tuned model's own
  traces) induces the behaviour when added to the tuned model. Its *mechanism* (the downstream circuit that turns
  the direction into behaviour) is present only if the same addition induces the behaviour in the base model.
- **New vs better access:** gives a three-way split that maps directly onto our question:
  1. base-derived vector induces the behaviour in **both** base and tuned model → representation and mechanism
     both exist in the base: *elicited* (steering alone unlocks it);
  2. base-derived vector works in the tuned model, **nothing** works in the base (their finding) → *repurposed*:
     representation old, downstream use new;
  3. only tuned-derived vectors work, and only in the tuned model (low cosine to any base-derived vector) → *new
     representation*.
  The paper sits in case 2 and calls it "rather than learn new capabilities from scratch"; for us case 2 is
  partial creation (RL built the reader), and should be reported as such rather than as elicitation.
- **Null / floor:** baselines = mean-activation, noise, self-amplification and other-category vectors (Fig. 4). No
  random-init model and no answer to 'any k': the behaviour metric is a frequency in sampled text.
- **Transfer to our setting:** cheap and direct. On r16's excluded-middle proofs, take residual activations at the
  positions just before the classical step (by_contra / Classical.em / double-negation elimination) and at matched
  non-classical steps, in **both** pend and r16 run on the same r16 traces; difference of means per layer (6 layers ×
  384). Steer: (a) r16 with the pend-derived vector on goals where r16 does *not* use reductio (does it start to?);
  (b) pend with the pend-derived and the r16-derived vector on held-out excluded-middle goals, scoring the
  teacher-forced probability of the classical step and pass@256 with Lean. Baselines: Gaussian noise of the same
  norm, mean activation, a vector for an intuitionistic step type, and the same pipeline on init activations. Our
  behaviour metric is crisp (the classical tactic token), unlike their keyword proxy. Cost: one forward pass per
  state plus a few hundred steered samples per setting: minutes. Failure modes: (i) the noise effect they saw means
  steering 'success' must beat a norm-matched noise vector; (ii) at 384 dimensions a difference of means may mix the
  goal's surface form (¬¬ present) with the decision, so use held-out goal shapes; (iii) case 2 vs case 1 depends on
  the steering scale searched, so the scale sweep must be identical for pend and r16.
- Related: `venhoff2025base.md` (same group; steering the base with thinking-model vectors at chosen times),
  `prakash2024finetuning.md` (cross-model activation patching), `jain2023mechanistically.md` (wrappers),
  `hewitt2019control.md` (decodable is not used), `hofstatter2025elicitation.md` (steering as an elicitation method).
