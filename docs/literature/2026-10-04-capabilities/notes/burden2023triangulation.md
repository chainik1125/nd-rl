---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - burden2023triangulation
---

# Inferring capabilities from task performance with Bayesian triangulation (measurement layouts)

Paper: [@burden2023triangulation] (Burden, Voudouris, Burnell, Rutar, Cheke, Hernández-Orallo)
Source: arXiv 2309.11975v2 (HTML rendering read: abstract, Sec. 1-8, App. A; App. C-F skimmed)

## Learnings

- **Capabilities as latent variables tied to task demands.** "we introduce measurement layouts which model how
  task-instance features interact with system capabilities to explain performance" (Abstract). The cognitive
  profile is a tuple ⟨C, B, R⟩: "Capability levels represent what the system can do, bias values represent some
  other preferences or limitations that may impact performance in a less monotonic way, and robustness levels
  account for reliability issues, unexplained or random effects (noise)" (Sec. 2).
- **Demand units and the 50 % convention (Sec. 6).** Each demand (an observable meta-feature of an instance) is
  linked to a capability, typically by σ(capability − demand); this "causes the capabilities to have the same unit
  as the demand, and allows us to interpret a capability with value x as consistently succeeding on the demand
  with meta-feature x in 50% of instances, higher if capability exceeds demand, and lower otherwise" (Sec. 6).
  Biases enter as σ(capability − demand + bias) (Sec. 6). Multiple demands combine non-compensatorily: "our
  capabilities were non-compensatory, so we often opted for taking the product of probabilities" (Sec. 6).
- **Floor built in.** "We use half-normal priors on these latent capabilities, reflecting the fact that there is a
  meaningful 0 value … representing no capability" (Sec. 3). A "random action agent that moves stochastically in
  the environment" is among the fitted subjects (Sec. 5).
- **Single-subject inference.** Unlike IRT/FA/SEM, "we infer the cognitive profile of a single subject from their
  performance data alone" (App. A), with capabilities specified from domain knowledge first, which "avoids
  post-hoc interpretation of statistical factors as "capabilities."" (Sec. 7).
- **Validation.** Synthetic agents with known profiles (designed blind by a separate team): "a correct description
  of the agent in around 75% of cases" (Sec. 4); held-out prediction better than aggregates ("consistently a better
  predictor of success (lower Brier score …)", Sec. 4; Abstract: "significantly more predictive"). On real data
  (PPO, Dreamer-v3, children 4-7), "All agents except children have low object permanence capability" (Sec. 5).
- **Known failure.** The layout was "unable to spot "fraudsters" making use of strategies (such as "go to the last
  seen location of the reward") to mimic OP capability" (Sec. 4).
- **Cost.** "If many parameters must be inferred simultaneously then many evaluation instances are needed" (Sec. 8).

## Evidence and limitations

- Fitted with NUTS in PyMC, 4 chains, 1000 warm-up, 2000 draws (Sec. 3-5). Validation is strongest on synthetic
  agents; real-agent profiles have no ground truth.
- Layout structure (which demands, which linking functions) is hand-built; a wrong layout yields confident but
  wrong profiles, and strategy-mimics are not detected (Sec. 4).
- Not checked: figures (images), appendix tables of linking functions.

## Connections and questions

- **Definition offered:** a capability is a latent level on the scale of an instance demand, inferred jointly with
  biases and a robustness/noise term from instance-level successes; capability = x means 50 % success on instances
  with demand x.
- **New vs better access:** not discussed, but the profile separates candidate mechanisms. *Our interpretation:*
  a training intervention that raises success uniformly across demand levels (robustness/noise terms, or a bias
  term toward an irrelevant feature removed) looks like better access/reliability; one that moves a capability
  level on a specific demand dimension (e.g. the classical-reasoning or `Or.elim`-nesting dimension) past demand
  levels where pend was at floor looks like a new capability. This gives a structured version of the project's
  group B / group C split.
- **Null / floor:** handled by construction: capabilities have a meaningful zero (half-normal priors), a random
  agent is fitted like any subject, and possession is graded on the demand scale (50 % convention), so a random
  model's astronomically rare successes give a capability far below any demand instead of "eventually solves".
  The 50 % point is a convention, but unlike k it is tied to the item scale, not to the sampling budget.
- **Transfer to our setting:** demands = theorem meta-features the project already computes (`L_true`/reference
  length, depth, inference-node term size — the organism-analysis's best single predictor —, number of ¬I /
  `Or.elim` boxes, classical-only vs intuitionistic). Outcomes = per-theorem success counts (k = 256 attempts as
  binomial data, not a solved/unsolved bit). Fit one layout per checkpoint (random-init, Stage-1 steps, pend, r8,
  r16; seeds as replicates) and compare posterior capability levels and robustness terms; held-out theorems test
  predictive validity. Cost: CPU-only fitting on existing count tables (minutes to an hour per checkpoint with
  NUTS on ~300-5,000 theorems). Main failure mode: proofs admit alternative routes, so a model can succeed by a
  "fraudster" strategy that does not use the demanded skill (the paper's own failure; the project's own finding
  that RL policies route around the reference proof); and the 50 % convention is a choice — the same arbitrariness
  as a threshold on p̂, though at least on a scale shared across models.
- Related notes: hernandezorallo2017evaluation.md (ability-oriented evaluation, difficulty scales and agent
  characteristic curves; same group), firestone2020performance.md (competence vs performance),
  vanderweij2024sandbagging.md (capability profiles vs difficulty to detect underperformance).
