---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - arora2023skills
---

# A theory for emergence of complex skills: competence = success rate on items requiring a skill (or a k′-tuple), driven by excess cross-entropy

Paper: [@arora2023skills] (Arora, Goyal, 2023; v2 title "A Theory for Emergence of Complex Skills in
Language Models")
Source: arXiv 2307.15936v2 (HTML rendering read: abstract, Sec. 1-8; appendix proofs not read).

## Learnings

- **Claim.** (a) "A statistical framework that relates cross-entropy loss of LLMs to competence on the
  basic skills that underlie language tasks"; (c) "competence at “complex skills,” which involve t-tuples
  of basic skills, emerges essentially at very similar scaling as competence on the elementary skills
  themselves" ("slingshot generalization") (Abstract).
- **Skill graph and competence.** A bipartite graph between skills and text-pieces; each text-piece needs a
  random k-tuple of skills (Def. 4-5). Competence on a skill "is a fraction between 0 and 1 corresponding to
  the model's success rate when presented with all cloze questions from text-piece selected randomly among
  all text-pieces adjacent to the particular skill node"; competence on a k′-tuple is the same over pieces
  that use all skills in the tuple (Sec. 1.1; Def. 7). "competence on anaphora resolution" is not 0/1
  (Sec. 4.1). Accounting is conservative — a wrong answer counts against every skill in the piece — so
  Def. 7 "can be thought of as a lower bound on the model's true “competence” individual skills", and it
  "does not capture out-of-distribution generalization" (Sec. 4.1).
- **Link to loss.** Loss = entropy + excess entropy (KL to the true distribution) (Sec. 2.1, Eq. 6);
  "there is no way to compute excess entropy using just the corpus". The Cloze Sufficiency Assumption: "The
  pre-trained model's average (multiclass) prediction loss on Cloze questions" tracks excess cross-entropy
  (Assumption 2); Thm. 3 (Pinsker): excess entropy ε at a position gives a binary cloze question answered
  wrongly with probability at most √(2ε).
- **Emergence of tuples.** Random-graph bounds (Thm. 8, 14) give performance curves; tensorisation gives
  Cor. 13: when the loss falls from δ to δ/k′, "the performance curve inferred by our method for" k′-tuples "of skills using" M_2 "is identical" to the single-skill curve of M_1 (Sec. 5.1.1). Takeaways: halving the
  error fraction (≈ 10× scale) raises competence on 2k′-tuples to the previous k′-tuple level;
  k′-tuples "that include more frequent skills will tend to emerge faster" (Sec. 7).
- **Paucity of stimulus.** Tuples outnumber the corpus, so "if the model displays competency on even" 10% of the k′-tuples of skills "then it must have somehow acquired competence in" k′-tuples "that were not seen during training" (Sec. 1.1).

## Evidence and limitations

- Purely theoretical, conditional on scaling laws and the cloze sufficiency assumption; skills are never
  identified, only posited. "Competence is guaranteed only on text-pieces drawn from the data
  distribution" (Sec. 8): the theory is about in-distribution competence of a pretrained model.
- Nothing about fine-tuning or RL. The bounds are lower bounds; the actual rate could be faster.

## Connections and questions

- **Definition offered:** competence on a skill (or k′-tuple of skills) = success rate on the
  sub-distribution of items whose solution requires that skill (tuple), with partners drawn at random;
  a lower bound under conservative accounting. Its driver is excess cross-entropy on the data
  distribution.
- **New vs better access:** not addressed. Two usable consequences (our inference): (1) in a synthetic
  setting excess entropy is *computable* (the generator's distribution is known), so the theory's premise
  can be checked: competence that rises while held-out excess entropy on the pretraining distribution does
  not fall (or rises, as under RL forgetting) is not "slingshot" competence from the data distribution — it
  is a shift of the policy toward a sub-distribution, i.e. a change of access/propensity rather than of
  modelled knowledge. (2) Competence per skill and per k′-tuple gives a *profile*, and the theory predicts
  how the tuple profile follows the single-skill profile; RL gains on tuples that exceed what the
  single-skill competences predict are compositional gains the base's statistics do not explain.
- **Null / floor:** none for sampling; competence is a success rate (a fixed-budget quantity). A random
  model has competence at chance on cloze tasks; for generation tasks the budget must be fixed.
- **Transfer to our setting:** skills = inference schemas in the generator; text-pieces = theorems (each
  needs a tuple of schemas). Competence(s) = solve rate (at fixed k, e.g. pass@1 or pass@32) over held-out
  theorems whose minimal proofs use s, competence(s1, s2) over theorems using both, for pend, r8, r16.
  Excess entropy: teacher-forced loss of each checkpoint on held-out generator proofs minus the
  generator's entropy (if the generator's step probabilities are recorded; otherwise compare losses
  across checkpoints, entropy cancels). Cost: J1-type scoring plus J2 sampling over schema-labelled sets.
  Failure modes: the generator's notion of "required skill" (minimal proof) is not what the model uses;
  competence on rare tuples is noisy (few theorems per tuple); the theory's random-tuple assumption fails
  for our generator, whose schema co-occurrences are structured.
- Related: `yu2023skillmix.md` (the empirical test built on this theory), `okawa2023multiplicative.md`
  (multiplicative emergence of compositions), `keysers2020cfq.md`, `schaeffer2023mirage.md` (emergence
  as a metric artefact).
