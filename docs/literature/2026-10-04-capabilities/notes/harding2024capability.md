---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - harding2024capability
---

# What is it for an ML model to have a capability? (conditional analysis of model abilities, CAMA)

Paper: [@harding2024capability] (Harding, Sharadin; British Journal for the Philosophy of Science, 2024,
DOI 10.1086/732153)
Source: arXiv 2405.08989v1 (HTML rendering, read in full: Sec. 1-6). The journal version was not read; the
arXiv text includes referee acknowledgements, so it is probably close to it.

## Learnings

- **Core proposal.** "crudely, a machine learning model has a capability to X just when it would reliably
  succeed at doing X if it 'tried'" (Abstract).
- **Three desiderata (Sec. 3).**
  - *Reliability*: "the more reliably the model φs, the stronger the evidence that it has an ability to φ" (Def. 5,
    Sec. 3.1). The UNIFORM model "always produces the next token by sampling uniformly from the token
    vocabulary"; it could produce any LLM output, yet "very few ability claims are actually true of UNIFORM"
    (Sec. 3.1). This is Dan's random-weights objection, answered by reliability.
  - *Competence vs performance* (Chomsky 1965): "a performance failure does not entail the absence of an
    underlying competence" (Sec. 3.2); the account "should explain how a model's ability to φ can be invariant
    across different methods for eliciting and measuring that capability" (Def. 6).
  - *Non-coincidence*: successes by accident (memorisation, heuristics that happen to coincide, luck) are not
    evidence of the ability (Def. 8, Sec. 3.3).
- **Orthodox analysis, which they reject as insufficient.** "M is able to φ_c iff there exists some set of
  background conditions B in which M reliably φ_c s across queries" (Def. 7, Sec. 3.3); it fails
  non-coincidence.
- **CAMA (Def. 10, Sec. 4.1).** "M is able to φ_c iff there exists some set of background conditions B in which
  the following conditional reliably holds across queries q_φc: if the output M produces is best explained by its
  being directed at φ_c ing, then M successfully φ_c s."
- **Behavioural test for "trying" (Def. 11, Sec. 4.2).** The output is directed at φ_c iff M "is sensitive to
  φ_c-relevant perturbations to the input" and insensitive to φ_c-irrelevant ones. Limits: a behavioural test
  "cannot be used to disambiguate between capabilities φ_c and ψ_c which are behaviourally indistinguishable"
  (Sec. 4.2).
- **Protocol (Def. 12, Sec. 5.1).** Generate many queries; for several background conditions, sample outputs;
  ("Rejection Sampling") "Throw away those outputs … which are not explained by being directed at φing"; the
  model is able iff for some B it reliably succeeds on the remaining (tried) queries. Caveat: "we should be wary
  of making claims about the model's ability to φ_c in domains in which it only 'tries' to φ_c on some very narrow
  range of queries" (Sec. 5.1).
- **Individuation.** The model is p_θ; "we treat tokenizers, inference procedures, and other scaffolding for p_θ
  as part of the set of background conditions" (Sec. 2.3). Fixed-point perspective: "a model individuation is
  principled iff it identifies a fixed-point in the conditions under which the model is evaluated" (Def. 3);
  complex scaffolding counts as a new model (Sec. 2.3).
- **Fine-tuning (Sec. 5.2.2).** "fine-tuning changes the model being evaluated"; "there is no in-kind difference
  between tuning an existing model and training a new model using the old model's weights as a parameter
  initialisation; the difference between tuning and training proper is a matter of degree" (Sec. 5.2.2, at fn. 38). Still, "if a
  model can be fine-tuned with very little compute (relative to e.g. its original training compute) to φ
  successfully, this does provide indirect evidence that it already had the ability to φ"; "the fine-tuning
  compute budget required to elicit a capability might be a natural proxy for the difficulty of eliciting the
  capability; future work could explore this" (Sec. 5.2.2, at fn. 38; the HTML inlines footnotes, so footnote vs main text is not separable).
- **RLHF changes propensity, not capability.** The conditional "is unaffected by RLHF (for most capabilities φ),
  the degree to which the model tries to φ changes" (Sec. 5.2.2).
- **Graded "practical availability" (fn. 42).** Under the Mandelkern et al. amendment, an action's practical
  availability for an LLM "is given by the probability mass of the output strings which count (according to the
  operationalisation construct) as implementing the action"; "an action is practically available for an LLM in an
  evaluation situation to the extent that it's something the LLM is likely to do".
- **Fair inter-model comparison (Sec. 5.2.3).** Same background conditions for all models conflate performance
  with competence; "what's relevant is finding – for each model – the set of background conditions on which the
  model is most successful at φing", with the trying condition ruling out gerrymandered conditions.

## Evidence and limitations

- Conceptual paper, no experiments. The behavioural test needs perturbations that separate φ from rival
  explanations ψ; the authors note that for some capabilities every perturbation is relevant, and that
  behavioural evidence underdetermines "best explanation" (Sec. 4.2).
- "Reliably" is never given a threshold; it "will depend on the capability at issue" (fn. 21). The fine-tuning
  budget is suggested, not defined.

## Connections and questions

- **Definition offered:** M can φ iff, under some admissible background conditions (prompting, inference
  procedure, sampling), M succeeds reliably on those queries where its output is directed at φ (sensitive to
  φ-relevant, insensitive to φ-irrelevant input changes). Graded variant: availability = probability mass of
  strings implementing the action.
- **New vs better access:** yes, conceptually. Changing background conditions (prompting, sampling, search) is
  better access to the same capability; changing the conditional "if it tries, it succeeds" is a new capability;
  changing how often it tries is propensity (their RLHF analysis). Fine-tuning formally makes a new model, and
  only a small fine-tuning budget relative to training compute is evidence of prior possession; no threshold.
- **Null / floor:** the UNIFORM example is precisely the random-weights objection. CAMA excludes it twice:
  UNIFORM's successes are unreliable, and they are coincidental — its output does not change when the theorem
  changes, so none of its outputs count as "trying", whatever k is.
- **Transfer to our setting:** (1) A *goal-sensitivity* test for "trying": for each theorem, compare the model's
  proof distribution under φ-irrelevant perturbations (renaming atoms or hypotheses, reordering premises; the
  project already varies name bases) with φ-relevant ones (a changed goal or a non-theorem). Teacher-forced
  log p makes this cheap: e.g. the mutual information between goal and first proof steps, or the log p drop of a
  proof when the goal is swapped. The random-init model scores ≈ 0 by construction, which gives a principled floor
  independent of k. Then measure reliability (p̂ per theorem) only on goal-sensitive outputs. (2) The fn. 42 /
  Mandelkern reading suggests a two-factor decomposition: availability of the crux action (the project's
  worst-step log p) × reliability once it is taken (log p of the rest of the proof given the crux). RL that raises
  only availability is better access to a route the model could already complete reliably; RL that raises the
  conditional reliability changes the competence. (3) Fair comparison: give pend its best background conditions
  (temperature, best-first search, guided sampling) before comparing it with r8/r16 — but by the fixed-point
  perspective, environment scaffolding that writes proof tokens (the claim audit found "The environment writes the
  box-closing and Or.elim-branch tokens") is part of a different model, so state-env vs whole-proof gains are not
  pure access gains. Cost: perturbation scoring is a teacher-forced pass (minutes); conditional-reliability
  sampling needs forced-prefix sampling (pod time). Main failure mode: every output of our models is a proof
  attempt, so "trying" degenerates unless it is defined by goal sensitivity; and the theory gives no reliability
  threshold or fine-tuning budget, so the numbers still have to come from calibration (positive and negative
  controls as in greenblatt2024passwordlocked.md and deeb2024unlearning.md).
- Related notes: maier2025abilities.md (the philosophy behind CAMA: conditional analysis, general vs specific
  abilities), firestone2020performance.md (performance constraints), vanderweij2024sandbagging.md (capability vs
  propensity), greenblatt2024passwordlocked.md ("reasonable prompt" ↔ non-gerrymandered background conditions).
