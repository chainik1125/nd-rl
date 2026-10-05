---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - firestone2020performance
---

# Performance vs. competence in human–machine comparisons

Paper: [@firestone2020performance] (Firestone, PNAS 117(43):26562–26571, 2020; doi 10.1073/pnas.1905334117)
Source: PMC7604508 (PMC HTML of the published article, read in full; figures are images and were not checked)

## Learnings

- **The distinction.** "Cognitive science traditionally distinguishes what a system knows (competence) from what
  it does (performance). Competence is a system's underlying knowledge: the internal rules and states that
  ultimately explain a given capacity, often in idealized terms" (section "Internal Knowledge vs. External
  Expression"). "intelligent creatures often know more than their behavior may indicate, because of "performance
  constraints"" (same section). A failure may arise "not because the system lacks the relevant knowledge or
  internal capacities ("competence"), but instead because of superficial constraints on demonstrating that
  knowledge ("performance")" (Abstract).
- **Both directions.** Behavioural similarity can hide different processes ("performance without competence")
  and "different behaviors can arise from similar underlying processes ("competence without performance")"
  (section "What Species-Fair Comparisons Show").
- **Method: accommodate the constraint, then re-test.** Infants seem not to know gravity, but with a measure that
  bypasses motor control they "stare measurably longer at unsupported floating objects"; "infants knew about
  gravity all along; they just failed to display that knowledge in their natural behavior" (section "When
  Superficial Differences Hide Deep Similarities"). Conversely Nim Chimpsky, taught sign language to bypass the
  vocal tract, still failed: "Since accommodating performance constraints still failed to produce humanlike
  behavior, we might feel safer concluding that chimpanzees lack the competence for language" (section "When
  Superficial Differences Are Deep Ones Too"). Principle: "Allowing other minds to demonstrate their knowledge
  requires accommodating their performance constraints, so that their success or failure won't depend on those
  constraints" (section "Fair Comparisons").
- **Three factors** for "species-fair" comparison: limit machines like humans, limit humans like machines,
  species-specific task alignment (sections 1-3, Fig. 4).
- **A confirmed competence gap.** CNNs on same/different (SVRT): "the best performing CNN model for this problem
  could not get significantly above chance from 1 million training examples" (quoting Kim et al.; section "A
  Case Study: Same vs. Different."). Even then: "it is conceivable that some other test could reveal efficient
  same/different abstraction", but "the space of superficial alternative explanations has narrowed considerably"
  (same section).
- **Elastic boundary.** On the CNN shape bias, re-training on a style-transferred dataset produced shape-based
  classification: "it wasn't that CNNs couldn't give shape-based classifications of images; they just didn't
  employ that humanlike strategy until their environment invited it" (section "3. Species-Specific Task
  Alignment"). Here a new training set is treated as revealing, not creating, a competence.

## Evidence and limitations

- A perspective piece built from case studies in machine vision; no formal definition of competence beyond
  "underlying knowledge … internal rules and states". The verdict "deep difference" is relative to the
  accommodations tried (explicitly defeasible, same/different section).
- The shape-bias example shows that the essay draws no line between accommodation and re-training; taken
  literally, any capability reachable by training would count as competence.

## Connections and questions

- **Definition offered:** competence = the internal knowledge that explains a capacity; performance = its
  expression under particular constraints (input, output format, memory, speed). A competence claim is
  supported when performance appears once the constraints are accommodated; a competence gap is supported when
  it does not appear even after accommodation (a training budget of 1 million examples in the CNN case).
- **New vs better access:** yes, as a method: an accommodation (changing the measurement or response mode while
  leaving the system unchanged) reveals existing competence; failure under all reasonable accommodations is
  evidence of absence. No budget or threshold; the shape-bias case even counts re-training as accommodation.
- **Null / floor:** not discussed. The Nim and same/different cases mark "absence" only relative to the
  accommodations and training budget tried.
- **Transfer to our setting:** the "looking-time" move has a direct analogue: measure pend without making it
  *generate*. (1) Discrimination/likelihood probes: does pend assign the RL model's crux step a high rank among
  candidate next steps, or prefer a valid proof over minimally corrupted invalid ones (teacher-forced log p
  differences)? Competence without performance = high relative preference for the correct step but low absolute
  generation probability; Nim-like absence = no preference even under forced choice. Cost: teacher-forced passes
  only (minutes), given candidate steps from the environment. (2) Response-mode accommodation: guided sampling
  (each step checked, failing steps redrawn) removes the "one slip ends the proof" constraint, and state-env
  interfaces remove the naming/bookkeeping constraint; gains that vanish once these are accommodated were
  performance gains. Failure mode: as the shape-bias example shows, "accommodation" has no natural stopping
  point — an interface that writes proof tokens (the state env writes box-closing and `Or.elim` branch tokens,
  per the claim audit) supplies part of the competence. Guard: allow only accommodations that carry no
  task-specific information (cf. "reasonable prompt" in greenblatt2024passwordlocked.md, the information
  criterion in deeb2024unlearning.md and hofstatter2025elicitation.md).
- Related notes: harding2024capability.md (competence vs performance as desideratum; Firestone cited there),
  maier2025abilities.md (general vs specific ability, masking), burden2023triangulation.md (competence as latent
  variable inferred from performance + task demands).
