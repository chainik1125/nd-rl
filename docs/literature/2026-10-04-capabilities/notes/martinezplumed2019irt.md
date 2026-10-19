---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - martinezplumed2019irt
  - martinezplumed2016making
---

# Item response theory for ML classifiers at the instance level (Martínez-Plumed, Prudêncio, Martínez-Usó, Hernández-Orallo)

Papers: [@martinezplumed2019irt] ("Item response theory in AI: Analysing machine learning classifiers at the instance
level", Artificial Intelligence 271:18-42, 2019, doi 10.1016/j.artint.2018.09.004) and its conference precursor
[@martinezplumed2016making] ("Making Sense of Item Response Theory in Machine Learning", ECAI 2016, FAIA 285:1140-1148,
doi 10.3233/978-1-61499-672-9-1140, CC BY-NC).
Source: **the ECAI 2016 paper was read in full** (open-access PDF from IOS Press, converted with pdftotext;
`~/cd_sources/martinezplumed2016irt-ecai.txt`; page numbers below are proceedings pages). The AIJ 2019 version could
not be accessed: the RiuNet repository copy sits behind an anti-bot proof-of-work wall and ScienceDirect returned 403.
Every claim below is from the ECAI paper; anything about the journal version's additional content is **UNVERIFIED**.

## Learnings

- **Mapping (Sec. 1-2).** Instances are items and classifiers are respondents; "IRT shows a dual behaviour in the way
  that classifier ability and instance difficulty are estimated at the same time, both depending on the other
  classifiers and instances" (Sec. 1, p. 1140).
- **Model and estimation (Sec. 2.2, 3).** 3PL ICC P(U_i = 1 | θ_j) = c_i + (1 − c_i)/(1 + exp(−a_i(θ_j − b_i)))
  with difficulty b_i, discrimination a_i, guessing c_i (Eq. 1); joint MLE with the R package ltm, "which implements
  the previously mentioned Birnbaum's method" (alternating item and ability steps; Sec. 2.2, 3). "For estimating good
  IRT models, a reasonably high number of individuals is needed" (Sec. 3, p. 1142).
- **Pool (Sec. 3).** "we used 128 classifiers arising from 15 different families", plus artificial respondents: three
  random classifiers, majority/minority, and "an optimal/pessimal classifier (Opt, Pess)" that use the test labels
  (Sec. 3, p. 1142).
- **Difficulty is the easy part (Sec. 4).** Difficulty "is not equal but well correlated to the percentage of
  classifiers that predict it correctly" (p. 1143).
- **Negative discrimination flags noise (Sec. 4.1).** On the Cassini toy set, "From the 200 instances, 180 had
  positive slopes"; "negative discrimination values were observed for 20 instances", and "noisy instances put on
  purpose are exactly those that have negative slope" (p. 1144).
- **Guessing parameter is not chance level (Sec. 4.2).** "the guessing parameter has to be interpreted as an extra
  degree of freedom to fit the logistic models, but not linked to the class distribution" (p. 1144).
- **Abilities are distorted by anomalous items (Sec. 5.1).** "the instances with a negative value of the discrimination
  parameter greatly affect the estimation of the ability parameter of the classifiers"; after removing them "Now the
  Optimal and Pessimal classifiers are actually the best and worst classifiers respectively" (p. 1145), because "IRT
  penalises those classifiers that respond correctly to the instances with negative discriminations".
- **Classifier characteristic curves (Sec. 5.2).** "A CCC is a plot for the response probability (accuracy) of a
  particular classifier as a function of the instance difficulty" (binned b_i), also drawn against discrimination
  (p. 1146).
- **Calibration and relativity (Sec. 6).** "we could scale the difficulties and abilities values such that they are
  zero for random classifiers"; "the optimal classifier does not get the highest ability" when anomalous items are
  kept; "IRT evaluates classifiers in terms of the other classifiers that we include in the pool" (p. 1147).
- **Outlook (Sec. 7).** "a good estimation of ability using adaptive testing could be done with about a dozen
  instances"; extension suggested "for weakly supervised machine learning (e.g., reinforcement learning)" (p. 1147).

## Evidence and limitations

- Small illustrative datasets (Cassini 200 instances, Heart-Statlog 270); one binary response per (classifier,
  instance) from cross-validation folds.
- Results are descriptive; no held-out validation of the IRT fit itself.
- The AIJ journal version (more datasets, extended analyses) was not read — UNVERIFIED.

## Connections and questions

- **Definition offered:** classifier ability = latent θ jointly estimated with instance difficulty / discrimination
  from the pool's instance-level responses; its profile over difficulty is the classifier characteristic curve.
- **New vs better access:** not addressed. *Our interpretation:* two of their tools transfer directly. (1) Negative /
  low discrimination identifies items on which weaker respondents do better than stronger ones — in our pool, theorems
  that early or base checkpoints solve but later RL checkpoints lose (the grpo-best "trades theorems" pattern) would
  surface this way, and should be analysed separately rather than allowed to distort θ. (2) The CCC of each checkpoint
  over pretraining-calibrated difficulty bins shows *where* RL gains sit: a uniform lift (all bins) vs a lift
  concentrated in the hardest bins vs gains in bins pend never touched.
- **Null / floor:** artificial respondents (random, pessimal, optimal) are put in the pool to anchor and interpret the
  scale, with the suggestion to set random = 0. In our pool the random-init checkpoint p0 is the natural anchor, but it
  solves nothing (0 of 287,680), so under joint MLE its θ is −∞ unless items it can solve exist (see
  `truong2026irsl.md`, `hofmann2025fluid.md` for floor tactics).
- **Transfer to our setting:** use 2PL, not 3PL — proof generation has no guessing floor, and the paper itself shows c
  absorbs misfit rather than chance. Calibrate on pretraining checkpoints (p50…pend × seeds) with binomial counts
  (k = 256), then score RL checkpoints with item parameters fixed, to avoid the pool-relativity they describe (adding
  RL checkpoints to the calibration pool changes every item parameter). Screen items with negative/low
  discrimination before interpreting abilities. Cost: CPU minutes. Failure mode: their "optimal classifier demoted"
  effect — a strong checkpoint that solves theorems weaker ones fail for idiosyncratic reasons gets a distorted θ; and
  the pool is not a population (three seeds' trajectories).
- Related notes: `hernandezorallo2017evaluation.md` (agent characteristic curves; same group), `burden2023triangulation.md`,
  `zhou2025adele.md`, `polo2024tinybenchmarks.md`.
