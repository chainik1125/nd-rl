---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - burnell2023revealing
---

# Revealing the structure of language model capabilities: factor analysis over a population of LLMs

Paper: [@burnell2023revealing] (Burnell, Hao, Conway, Hernández-Orallo, 2023)
Source: arXiv 2306.10062v1 (HTML rendering read: full text incl. App. A-B; figures are images and were not read)

## Learnings

- **Approach (Abstract).** They investigate "the structure of LLM capabilities by extracting latent capabilities from
  patterns of individual differences across a varied population of LLMs", with data from "29 different LLMs across
  27 cognitive tasks" (HELM) (Abstract; Sec. 2.1).
- **Positive manifold (Sec. 3.1).** "mean correlation of r = 0.56" between tasks across models.
- **Number of factors (Sec. 3.2; App. B).** Scree plot: "The patterns suggest a 3 or 4-factor solution is most
  appropriate" (App. B); the Hull method: "This method suggests that a 3-factor solution is most appropriate"
  (Sec. 3.2); a Bayesian factor analysis also puts most posterior mass on 3 (App. B, Fig. 7).
- **Fit and variance (Sec. 3.2, Table 2).** Maximum-likelihood EFA with oblimin rotation; fit statistics "somewhat
  poor (CFI = 0.70, TLI = 0.61, RMSEA = 0.26)"; cumulative variance explained 0.33 / 0.64 / 0.82.
- **Factors (Abstract; Sec. 3.2).** "three well-delineated factors that represent reasoning, comprehension and core
  language modeling"; the language-modelling factor collects next-token-prediction tasks "(as measured by
  bits-per-byte)" (Sec. 3.2).
- **Small-sample caution (Sec. 3.2-3.3).** "Studies using frequentist factor analytic typically employ larger samples
  than it would be possible to obtain from modern LLMs", hence a Bayesian FA because "Bayesian methods are much more
  robust to small sample sizes".
- **Post-training moves models differently from scale (Sec. 3.5, Table 3).** Log model size correlates positively
  with all three factors; "Instruction tuning was negatively correlated with language modeling, but positively
  correlated with reasoning" (r = −0.50 [−0.72, −0.17] and 0.44 [0.11, 0.69]). "There were no models that underwent
  RLHF without instruction tuning, so we did not analyse the relationships with this property" (Sec. 3.5). Hence
  "changes to a model that improve one ability might simultaneously impair others" (Abstract; Sec. 4).
- **Caveats (Sec. 4).** "The results of any factor analysis depend on the tasks included in the analysis"; more data
  could come "by performing ablations of different models and testing each version on the same benchmark".

## Evidence and limitations

- N = 29 models for 27 variables is far below usual factor-analytic practice; the poor fit indices are acknowledged.
- Factor interpretation is post hoc, guided by an expert's task annotation (Sec. 2.3).
- Correlations with model properties are across heterogeneous, confounded models (size, data, tuning vary jointly).
- Hernández-Orallo's later ADeLe paper criticises such populational factors as volatile when new models enter the
  pool (see `zhou2025adele.md`).

## Connections and questions

- **Definition offered:** a capability = a latent factor that accounts for the covariance of task scores across a
  population of models; a model's level = its factor score.
- **New vs better access:** not addressed directly, but the instruction-tuning result is the closest precedent in this
  thread for *post-training moving models along a different direction than pretraining scale*: scale loads on all
  factors, instruction tuning raises one and lowers another. *Our interpretation:* a decision rule in this spirit —
  fit the factor model on pretraining checkpoints, then add RL checkpoints; RL is "more of the same" if their factor
  scores move along the direction traced by pretraining (all factors rising in the pretraining proportions) and "new"
  if (a) an extra factor is needed (Hull / parallel analysis / Bayesian posterior on the number of factors rises when
  RL checkpoints are added) whose scores are flat across pretraining checkpoints, or (b) RL moves a factor that
  pretraining barely moves (the instruction-tuning pattern). Negative factor correlations with RL (an ability lost,
  e.g. pass@256 on theorems "traded" by GRPO) are also informative.
- **Null / floor:** not discussed. Tasks at floor have no variance and cannot load; random-init and early Stage-1
  checkpoints would be outliers on every factor.
- **Transfer to our setting:** variables = theorem classes (≈ 10–30, e.g. by reference length, number of ∨E / ¬I / →I
  boxes, classical-only vs intuitionistic, atoms), each scored per checkpoint as a logit of pooled per-attempt success
  (from k = 256 counts, continuity-corrected); persons = ≈ 150 checkpoints; or item factor analysis directly on the
  322 binomial items (multidimensional 2PL, see `polo2024tinybenchmarks.md`). Cost: CPU minutes. Main failure modes:
  (1) dependence — checkpoints are three seeds' trajectories, so the effective N is small (the paper's own concern at
  N = 29 applies with more force); (2) a "training-stage" factor (format learned early, long proofs late) can appear
  within pretraining alone and be mistaken for an RL dimension — compare factor solutions on early-vs-late pretraining
  and on replay-only ladder checkpoints as negative controls; (3) class construction decides which factors can appear
  ("depend on the tasks included").
- Related notes: `ruan2024observational.md` (PCA version, scaling), `hofmann2025fluid.md` (one pooled axis hid a
  decrease), `burden2023triangulation.md` and `hernandezorallo2017evaluation.md` (same group's non-populational
  alternatives), `zhou2025adele.md`.
