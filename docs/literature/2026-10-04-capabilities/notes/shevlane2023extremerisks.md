---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - shevlane2023extremerisks
---

# Model evaluation for extreme risks: capability evaluations vs alignment (propensity) evaluations

Paper: [@shevlane2023extremerisks] (Shevlane, Farquhar, Garfinkel, Phuong, Whittlestone, Leung, Kokotajlo,
Marchal, Anderljung, Kolt, Ho, Siddarth, Avin, Hawkins, Kim, Gabriel, Bolina, Clark, Bengio, Christiano, Dafoe
2023)
Source: arXiv 2305.15324v2 (HTML rendering read: abstract, Sec. 1-6, Tables 1-2; appendix skimmed)

## Learnings

- **Two kinds of evaluation.** "Developers must be able to identify dangerous capabilities (through "dangerous
  capability evaluations") and the propensity of models to apply their capabilities for harm (through
  "alignment evaluations")" (Abstract). "These evaluations can be organised into two categories: (a) whether a
  model has certain dangerous capabilities, and (b) whether it has the propensity to harmfully apply its
  capabilities (alignment)" (Sec. 1).
- **Capability evaluation assumes worst-case use.** "a model should be treated as highly dangerous if it has a
  capability profile that would be sufficient for extreme harm, assuming misuse and/or misalignment" (Sec. 2).
  "Partly, agency is a question of the model's capabilities" (Sec. 4).
- **Latent capabilities must be surfaced.** Table 2 desideratum "Surfacing latent capabilities": "Researchers will
  need to bring latent capabilities to the surface (for example, by prompt engineering or fine-tuning)" (Sec. 4,
  Table 2). Limitation "Capability overhang: Models sometimes have capabilities that the AI research community
  does not realise", e.g. chain-of-thought months after GPT-3 (Sec. 5.1, item 3a).
- **Model lifecycle and system level.** "the results from the end of a long development process will likely fail to
  convey relevant information about the base model" (Table 2); "Evaluations should study models both with and
  without these augmentations" (Table 2, "Model-level and system-level").
- **How large a change makes a new model.** For post-deployment updates: "The magnitude of the change to the model
  could be assessed in terms of the amount of additional training that it has gone through (as a percentage of the
  original training length), or the model's improvement on key performance benchmarks" (Sec. 3.2, fn. 4).
- **Behavioural evaluations are not enough.** Evaluations "should eventually also involve looking mechanistically
  at how the model produced that behaviour" (Table 2); deceptive alignment "is one reason not to rely solely on
  behavioural evaluations" (Sec. 5.1).
- **Wide difficulty range.** Capability evaluations should span wide difficulty so progress can be tracked
  toward thresholds; "evaluations would ideally provide a quantitative score, although this will not always be
  practical" (Table 2).

## Evidence and limitations

- A policy/agenda paper; no definitions beyond the capability/propensity split, no elicitation budget, no
  threshold. It is the origin of the vocabulary later papers formalise (van der Weij et al. cite it for
  capability vs alignment/propensity evaluations).

## Connections and questions

- **Definition offered:** capability = what the model could do under worst-case use (misuse or misalignment),
  after surfacing latent capabilities by prompting or fine-tuning; propensity = whether it would apply them.
- **New vs better access:** only implicitly. Capability evaluation is supposed to be invariant to propensity, so
  anything a model does once "surfaced" counts as present. The paper's fn. 4 supplies the only quantitative hook:
  training added, as a percentage of the original training length, as the measure of how far a modified model is
  from the original — the seed of the later "< 1 % of training cost" convention.
- **Null / floor:** not discussed.
- **Transfer to our setting:** the capability/propensity split maps onto our two candidate readings of RL:
  (a) RL raises a *propensity* — how often pend's existing route gets taken (the sharpening reading); (b) RL
  raises the *capability* — what the model can do at all under the best admissible elicitation. Operationally:
  capability(pend, T) = success under pend's best admissible elicitation (best temperature, search, few-demo
  SFT within budget); propensity = default-sampling p̂. RL that moves pend's propensity up to pend's own
  elicited capability is elicitation; RL that exceeds pend's elicited capability is creation relative to that
  elicitation set. fn. 4 suggests reporting each elicitation's added training as a % of Stage-1 tokens. Cost:
  bookkeeping plus the elicitation runs themselves. Failure mode: "best admissible elicitation" is open-ended
  (capability overhang), so the verdict is only ever relative to the elicitation set tried.
- Related notes: vanderweij2024sandbagging.md (formalises actual vs exhibited capability, 1 % budget),
  harding2024capability.md (propensity to try vs conditional success), greenblatt2024passwordlocked.md.
