---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - kwa2025measuring
---

# METR time horizon: capability as the task difficulty (human time) at which an agent succeeds 50 % of the time

Paper: [@kwa2025measuring] (Kwa, West, Becker, …, Barnes, Chan; METR), "Measuring AI Ability to Complete Long Software
Tasks"
Source: arXiv 2503.14499v4 (v4 of 10 Jul 2026, HTML rendering; read: abstract, Sec. 1-3, App. A.3, B.1.3, C.3-C.4,
E.1-E.4, H.1-H.2; Sec. 4-5 skimmed)

## Learnings

- **Metric (Abstract).** "50%-task-completion time horizon, the time humans typically take to complete tasks that AI
  models can complete with 50% success rate". Task suite "totaling 170 tasks with a wide range of difficulty";
  "12 frontier models from 2019 to 2025" (Sec. 1); "We perform 8 runs" per agent/task pair (Sec. 2.3).
- **Fit (Sec. 3.1).** Per agent, a logistic regression p_success(agent, task) = σ((log h_agent − log t_task)·β_agent),
  "where t_task is the geometric mean time of successful human baselines, and h_agent and β_agent are learned
  parameters, with h_agent representing the 50% time horizon". Continuous scores are binarized: "Continuously scored
  tasks are binarized via a task-specific threshold chosen to represent human performance". Task weights: "a task from
  a family of size n gets weight 1/√n" (App. C.4).
- **Relation to IRT (Sec. 3.1 fn. 5; App. C.4).** "unlike IRT, we use difficulty ratings directly based on human
  baseline time rather than ratings learned from agent performance". Joint IRT calibration means "the ability score of
  one agent may depend on other agents' performance; in our method each agent's time horizon is computed
  individually"; and "the 2PL model uses one slope parameter per task, whereas we use one slope parameter per agent"
  (App. C.4).
- **Floor handling (App. B.1.3, C.3).** Very short SWAA tasks were added: "we focused on tasks in the 2-second to
  15-second range with SWAA in order to measure GPT-2 and GPT-3" (Fig. 7 caption); and "we imputed a score of zero for
  GPT-2 on all tasks in RE-Bench and HCAST" (App. C.3).
- **Results.** "GPT-2 has a 50% time horizon of only 2 seconds, while o3 has a 110-minute time horizon"; horizon
  "doubled every 207 days" (Sec. 3.2). "models' 80% time horizons are 4-6x shorter" (Sec. 3.2.1), while "The doubling
  time in 80% time horizon (204 days) is similar to the doubling time of 50% time horizon (207 days)" (Sec. 3.2.1).
  Between-task success correlation across models "approximately 0.73" (Sec. 2.3). Per-model horizons have correlated
  errors because "tasks at the same human time rating vary widely in difficulty for models" (Sec. 3.2).
- **Interpretation caveats (App. E.3-E.4).** "AI agent success rate is imperfectly predicted by human
  time-to-complete, meaning that other factors also substantially influence the difficulty of tasks"; "time horizon
  is always measured relative to a task distribution and baseliners' levels of context and skill". Measuring an X %
  horizon of t "requires many tasks of human length t that the AI agent completes with a success rate of about X%".
  Elicitation: "while our results are a reasonable lower bound, some models may have somewhat greater capabilities
  than we demonstrate" (App. E.4). Continuous scoring "overstates the time horizon of recent models, since it is
  easier to achieve an average score of 0.5 on most tasks than to match human performance 50% of the time" (App. H.1).

## Evidence and limitations

- Difficulty is external (human time), so the scale does not move when agents are added — but human time is a noisy
  proxy (R² ≈ 0.8 of mean success vs log time, Sec. 2.3) and depends on the baseliner population.
- Few agents (12 frontier) and few tasks per length bin; the 50 % point is a convention (they check 80 %).
- Not checked: Sec. 4 (SWE-bench Verified, messiness), Sec. 5 extrapolations, appendix tables.

## Connections and questions

- **Definition offered:** capability = the location h on an externally calibrated difficulty axis where per-attempt
  success is 50 % (with a per-agent slope β as a reliability/sharpness parameter), fitted for each agent separately.
- **New vs better access:** not addressed (App. E.2 only speculates that agentic RL post-training may speed horizon
  growth). *Our interpretation:* the two per-agent parameters give a natural split for RL checkpoints:
  (a) **sharpening** — β rises and the 80 % horizon grows while the 50 % horizon stays near pend's (success on items
  already within reach becomes reliable; items above the horizon may even lose probability); (b) **extension** — the
  50 % horizon moves beyond pend's; (c) **re-ordering** — success no longer follows the difficulty axis (poor logistic
  fit for RL checkpoints, theorems becoming easy out of order), the signature of a new dimension. A 1 % horizon (the
  project's "solves bar" p̂ ≥ 0.01) measures reach; 50 % measures reliable capability.
- **Null / floor:** handled by design: the horizon is a per-attempt quantity, so k never enters; weak models are
  placed on the scale by adding easy items (SWAA) rather than by extrapolation, and tasks a model cannot run are
  imputed as failures. Random weights would get a horizon at (or below) the easiest items, not "eventually
  everything".
- **Transfer to our setting:** the analog of human time is a difficulty for each theorem that does not depend on the
  RL models. Candidates: (1) intrinsic — reference length L_true or inference-node term size (the project's best
  single predictor); (2) **pretraining time** — the Stage-1 step s_i at which the pretraining trajectory first reaches
  50 % (or 1 %) per-attempt success on theorem i, interpolated from the 14 Stage-1 checkpoints per seed (p0 … p20000, pend) (my proposal;
  it puts difficulty in pretraining-compute units, so an RL checkpoint's horizon reads directly as "pretraining steps
  equivalent", and theorems with s_i undefined are exactly those beyond the pretraining axis); (3) an IRT difficulty
  b_i calibrated on pretraining checkpoints (`polo2024tinybenchmarks.md`). Fit per checkpoint
  n_ok ~ Binomial(256, σ(β_m(log h_m − log d_i))) on the 322 theorems; report h_50, h_80 and h_1 for p0…pend and
  r1…r16, per seed. Cost: seconds per checkpoint on existing reads. Failure modes: (i) a single difficulty axis
  ignores route dependence (the policy proves a theorem by a route unlike the reference); (ii) Stage-1 checkpoints
  reach 50 % on few hard theorems, so s_i is censored for most of group B/C — use a lower success level or IRT
  extrapolation; (iii) RL sharpens the sampler (entropy falls r0 → r1, organism-analysis), which changes β and can
  move h_50 without any change in "competence"; report both parameters.
- Related notes: `burden2023triangulation.md` (50 % convention on demand scales), `zhou2025adele.md` (18 demand
  scales, same convention), `hernandezorallo2017evaluation.md` (area under agent characteristic curve),
  `phuong2024dangerous.md`, `vanderweij2024sandbagging.md` (elicitation lower bounds).
