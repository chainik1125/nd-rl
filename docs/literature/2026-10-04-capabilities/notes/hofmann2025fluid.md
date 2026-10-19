---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - hofmann2025fluid
---

# Fluid Benchmarking: 2PL abilities for pretraining checkpoints, adaptive items, and the cost of forcing one dimension

Paper: [@hofmann2025fluid] (Hofmann, Heineman, …, Hajishirzi, Smith; AI2 / UW / CMU), "Fluid Language Model
Benchmarking" (2025)
Source: arXiv 2509.11106v1 (HTML rendering; read: abstract, Sec. 1-4, 6-8, App. A-E; result tables in Sec. 5 / App. F
skimmed)

## Learnings

- **Premise (Abstract).** "the relative value of benchmark items depends on an LM's capability level, suggesting that
  evaluation should adapt to each LM"; IRT "increases validity, while dynamic item selection reduces variance".
- **Model (Sec. 3.1).** 2PL p(u_ij = 1) = logistic(a_j(θ_i − b_j)), fitted "using Markov chain Monte Carlo … with
  hierarchical priors on all parameters"; a new checkpoint's θ is obtained by "maximum a posteriori estimation" with
  item parameters fixed (Eqs. 2-3). Low-discrimination items are often bad items: "many of them are mislabeled"
  (Sec. 3.1; Sec. 6: mislabeled items per session 0.01 vs 0.75 for random subsets).
- **Information moves with training (Sec. 3.2, Fig. 2).** Fisher information a²σ(1 − σ) is "maximized when θ_i=b_j
  (where I=a_j²/4)"; over a simulated training run the most informative items move "from very easy items at the
  beginning of training, to very difficult items at the end of training". Items are selected adaptively by maximum
  information (Eq. 5); with a standard-error stopping rule the number of items needed varies "from around 20 at the
  beginning to over 80 midway" (Sec. 6, Fig. 5).
- **Setting (Sec. 4.1; App. C-D).** Six pretraining runs (Amber-6.7B, OLMo1/2-7B, Pythia-2.8B/6.9B, K2-65B); "we evenly
  select between 61 and 94 checkpoints"; "over 13 million item-level evaluations". Item parameters calibrated on "a
  final set of 102 LMs" from the Open LLM Leaderboard, excluding derived models: "Finetuned, merged, fused, distilled,
  or continually pretrained LMs were excluded, as they can lead to clusters of highly similar models, potentially
  skewing the IRT model" (App. D).
- **Dimensionality (Sec. 4.1; App. E).** Multidimensional models and a single cross-benchmark unidimensional model
  "yielded worse results"; per-benchmark MIRT with 2-5 traits "did not yield consistent improvements in model fit".
  Crucially, the single pooled dimension "substantially reduced construct validity": it "emphasized TruthfulQA items
  aligned with general trends, obscuring the fact that Amber-6.7B actually becomes less truthful during pretraining"
  (App. E).
- **Ceiling analogue of our floor (Sec. 6).** Items "not answered correctly by any train LM … are effectively assigned
  the same maximum difficulty"; "a fixed IRT model cannot distinguish finer levels of difficulty among them".
- **Quality metrics along training (Sec. 4.2, 6).** Step-to-step total variation (Eq. 6) and monotonicity (Spearman of
  score vs checkpoint index); HellaSwag monotonicity "0.91 for Random, compared to 0.99 for Fluid Benchmarking". They
  confirm that static IRT subsets (tinyBenchmarks, metabench) "increase step-to-step variance" (Sec. 6). Possible
  extension: "holds potential value for posttraining as well" (Sec. 6).

## Evidence and limitations

- Only pretraining checkpoints are evaluated; no post-trained or RL checkpoints.
- Item banks are calibrated on an external population, then applied to checkpoints of other runs; whether a checkpoint
  population would calibrate differently is not tested.
- Validity is operationalised as rank agreement with a sister benchmark, which favours a general factor.

## Connections and questions

- **Definition offered:** a checkpoint's ability = its 2PL θ (MAP) on a benchmark-specific item bank calibrated on a
  population of pretrained models.
- **New vs better access:** not addressed. *Our interpretation:* App. E is the key cautionary result for us — a single
  pooled axis can report *rising* ability while one construct falls. So an RL-specific change can be hidden by a
  unidimensional pretraining axis. Decision rule: estimate θ separately per theorem class (classical-only, ∨E-heavy,
  long, short) and pooled; "more of the same" if class abilities move in lockstep with the pooled θ as they did along
  pretraining; "new" if one class's θ rises far more than the pooled θ predicts (class-level DIF), replicated across
  seeds.
- **Null / floor:** early checkpoints are measured with very easy items (adaptive selection) and MAP keeps all-wrong
  patterns finite; items no calibration model solves collapse to one maximum difficulty — the mirror image of our ≈ 30
  never-solved theorems and of group C at pend, whose difficulty a pretraining-only calibration cannot resolve.
- **Transfer to our setting:** calibrate on Stage-1 checkpoints (p50…pend, 3 seeds × 2 caps; RL and replay-only ladder
  checkpoints excluded, following their App. D rationale), using binomial counts; score r1…r16, GRPO arms and ladders by
  MAP with items fixed. Adaptive selection is unnecessary (full matrices exist), but item information I(θ) tells which
  of the 322 theorems are informative at each checkpoint — for p0…p400 essentially none are, which argues for adding
  easy items (single-step or 1-3-line theorems, or step-level teacher-forced items) to place early checkpoints and the
  random init on the scale. Their TV and monotonicity metrics are directly usable to compare candidate capability
  measures along our trajectories (θ vs solved@256 vs reference-proof log p). Cost: CPU minutes. Failure mode:
  near-duplicate checkpoints within a run (the clustering concern they cite); calibrating on pretraining leaves
  RL-only theorems unidentified (see the censored-item bound in `truong2026irsl.md`).
- Related notes: `polo2024tinybenchmarks.md`, `truong2026irsl.md`, `burnell2023revealing.md`, `ruan2024observational.md`.
