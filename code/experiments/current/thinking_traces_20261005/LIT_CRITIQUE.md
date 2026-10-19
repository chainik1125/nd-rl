---
written_on: 2026-10-06
written_by: agent:claude
papers:
  - guoDeepSeekR1IncentivizesReasoning2025
  - ren2025deepseekproverv2advancingformalmathematical
  - yueDoesReinforcementLearning2025
---

# Adversarial literature review of the thinking-traces experiment

Scope: [PROPOSAL.md](PROPOSAL.md), [TRACE_AUDIT.md](TRACE_AUDIT.md), the
[Stage-1 summary](../../../../experiment-summaries/2026-10-05-thinking-traces/README.md),
`code/thinking/{ei,grpo,effort,rootplan}.py` (branch `leon/thinking`, HEAD `b3c78cd`; `grpo.py` and
`rootplan.py` are uncommitted), and the stored outputs under `/scratch/cowifornia/nd-rl/think/runs/`
(Stage-1 evals, EI round 1, the GRPO smoke run). This is a deliberately hostile reading, as Leon asked. No GPU
was used; nothing existing was edited.

Labels used throughout:
- **Observed:** a number read or recomputed from stored outputs in this review (method at the end).
- **Reported:** what a cited paper states. Papers are cited by arXiv id.
- **Interpretation:** my reading, which may be wrong.

## 1. Verdict

No. As designed, these experiments cannot support a test-time-scaling claim, and the current evidence
points the other way on the metrics Leon has chosen.

- **Stage 1, textbook72.** Thinking does not beat no-thinking at pass@1 or pass@5: every 95% interval over
  theorems and seeds includes zero. It wins only at pass@256.
- **Stage 1, OOD.** Thinking is below no-thinking at pass@1 in every one of the nine thinking cells
  (three models × three budgets), while using 1.1–4.6× the tokens.
- **The pattern.** Thinking helps only at large k. Lean-STaR (arXiv 2407.10040) attributes this same pattern
  to output diversity. RL is documented to trade large-k diversity for pass@1 (arXiv 2504.13837).

**The single biggest threat is the budget knob itself.** It is a hard cut that closes the think block mid-line
at B tokens per step:
- Intermediate budgets are already *worse than not thinking at all*. On OOD pass@1, B64 is below B0 in all
  three thinking models.
- In the GRPO smoke run, at initialisation, pass@1 was 4% at `<e16>` and 7% at `<e64>`, against 22% at
  `<e0>` and 27% at `<e256>`.

So any "more effort, more success" curve that the effort runs produce will be manufactured by damaging the
low budgets, not by thinking paying at the high end. Unless every curve is anchored at B0 and at
matched-token parallel sampling plus guided redraws from the same checkpoint, and the think content is shown
to matter (swapped-trace and filler controls), a positive result will not be interpretable as test-time
scaling.

## 2. Ranked critiques (most damaging first)

### C1. The baselines a TTS claim must beat are missing from the RL comparison, and Stage 1 already trips the proposal's own falsifiers

**Issue.** The RL arms (EI from T0, T1, T1v1 and T1v1+effort; GRPO next) are compared with one another. A
test-time-scaling claim, however, has to beat *parallel sampling with a perfect verifier* and *guided
redraws* from the same checkpoint at matched cost.
- OOD_CHECK.md says exactly this. The PROPOSAL §6 falsifiers encode it:
  - "thinking is unused if B = 0 equals B = 256 in pass@1 on Tier A";
  - "not worth its tokens if at or below T0-guided at matched tokens on Tier A long".
- Neither falsifier is reported in the Stage-1 summary.

**Observed (Stage 1, stored evals).**
- **textbook72, pass@1 / pass@5, 3-seed means.** 95% intervals resample theorems and seeds.

  | arm | pass@1 | pass@5 |
  |---|---|---|
  | T0 | 17.2% | 26.0% |
  | T1 B256 | 15.6% (−4.8 to +1.6 vs T0) | 27.9% (−2.8 to +7.4) |
  | T1v1 B256 | 16.8% (−3.9 to +3.3) | 29.3% (−1.8 to +9.5) |

  The README headlines pass@256, the only metric where thinking wins. At pass@256, only T1v1 excludes zero
  (+3.7 to +18.5 pp). T1 does not (−0.9 to +13.0).
- **OOD sets without textbook_dev** (840 theorems, k 16, pass@1):
  - T0: 1.74% and 1.93% (seeds 0, 1);
  - every thinking cell: 0.55–1.58%, at 1.1–4.6× T0's tokens per theorem (2.2–2.3k for T0, 2.5–10.1k
    for the thinking cells).
- **Tier A (Charles long, n = 80), k 16:**
  - T0 with guided redraws: pass@16 12.5% / 20.0% at 6.6k / 6.9k tokens per theorem;
  - T1 B256, plain: 6.2% / 1.2% at 6.4k / 6.1k tokens.

  At matched tokens, the second falsifier fires at Stage 1.
- **Tier A, pass@1, B0 against B256:** T1 s1 0.3% vs 0.3%; T1 s0 0.2% vs 0.8%.

  The first falsifier also fires or nearly fires, although at this base rate it has no power (see C4).
- **"T1 guided + B256: 18.0%" against "T0 guided: 13.7%"** (README OOD table): this compares 20.8k tokens per
  theorem with 6.1–7.1k. The summary does not say so.

**Reported.**
- Snell et al. (arXiv 2408.03314): the best test-time strategy "critically varies depending on the
  difficulty of the prompt", and a compute-optimal allocation beats best-of-N by more than 4×. Sequential and
  parallel compute must be compared at matched FLOPs, per difficulty band.
- s1 (arXiv 2501.19393) claims that sequential beats parallel. That comparison is s1-32B with budget forcing
  against majority voting *on the base model* (Fig. 4), which is not the same model and has no verifier. With
  Lean as the verifier, our parallel baseline is pass@k, which is much stronger than majority voting.
- DeepSeek-Prover-V2 (arXiv 2504.21801, Tables 1 and 3), 7B:
  - CoT pass@1 is 58.6% against non-CoT 55.5%, at 4,488 against 443 output tokens.
  - At roughly matched total tokens CoT does win: pass@1024 at about 4.6M tokens gives 79.9%, against
    non-CoT pass@8192 at about 3.6M tokens, 75.0%. That is my arithmetic from their tables.

  So the CoT advantage there is real but appears at large budgets. That matches our Stage-1 crossover.

**Interpretation.** Stage 1 says that a think token is worth less than a sample token below about 5k tokens
per theorem, and is worth nothing at pass@1 OOD. RL that does not include the parallel and guided baselines
can only show that one thinking policy beats another.

**Fix.**
- Every RL arm's final checkpoint gets the same read: a pass@k curve against *sampled tokens per theorem*,
  for plain sampling, guided redraws, and thinking at B ∈ {0, …, 256}.
- Also run a **T0-guided EI arm**, i.e. EI with the logical arm in rollouts. It is the strongest no-think
  competitor.
- State the falsifiers in the summary and evaluate them after RL.

### C2. The budget/effort knob is hard mid-line truncation, its labels are a difficulty proxy, and the effort run changes three things at once

**Issue (code).**
- `sample.py` forces `</think>` after B think tokens per step, wherever it falls in a line.
- In EI, the forced close is **trained as a natural close**. The pipeline README says "trained like the
  others"; `ei.entry_pairs` wraps the truncated content in `<think> … </think>`.
- `effort.relabel_records` labels each retained Stage-1 record with the smallest level that holds *its own
  longest block*.
- In GRPO the effort tokens are fresh, untrained embeddings, from `expand_to_effort` with no SFT pass.
- The effort EI run (`ei_T1v1e_s0`) differs from `ei_T1v1_s0` in three ways:
  - budgets {0, 16, 64, 256} against {16, 64, 256};
  - visible `<eB>`;
  - `--select shortest`.

  It is one seed, from T1v1 only.

**Observed.**
- **Truncation hurts.** On OOD, B64 is the worst budget in every thinking model:
  - pass@1: T1 s0 0.80% vs 1.08% at B0; T1 s1 0.82% vs 1.12%; T1v1 0.55% vs 0.89%;
  - same ordering at pass@16.
- **GRPO smoke** (`runs/grpo/smoke`, effort init from T1v1, lr still in warm-up, so essentially the initial
  policy). Pass@1 by level over iterations 1–3:

  | level | it 1 | it 2 | it 3 |
  |---|---|---|---|
  | e0 | 21.7% | 27.3% | 23.4% |
  | e16 | 4.0% | 0.4% | 3.8% |
  | e64 | 7.3% | 12.9% | 7.4% |
  | e256 | 26.7% | 36.8% | 41.3% |
- **The labels are not a budget.** Effort labels of the retained records (`logs/ei_T1v1e_s0.log`):

  | label | records | share |
  |---|---|---|
  | `<e0>` | 32,715 | 21%, the Dualformer-emptied copies |
  | `<e16>` | 1,065 | 0.7% |
  | `<e64>` | 14,241 | 9% |
  | `<e256>` | 106,956 | 69% |

  In retained data, `<e256>` means "this proof contained a 65–240-token block", which is a property of the
  proof. In RL data, `<eB>` means "sampled under cap B". The two semantics are mixed, and `<e16>` has almost
  no supervised examples.
- **EI round 1 trains on truncated blocks.**
  - T1: 16% of kept B16 steps and 6% of kept B64 steps end at the cap.
  - T1v1: 28% and 16%.
  - Together these are 4–7% of all kept steps, each taught as a deliberate close.
  - In the non-effort arms the model never sees which budget produced them.

**Reported.**
- **s1 (arXiv 2501.19393), Table 3.**
  - Budget forcing has 100% control and a positive slope.
  - Token-conditional control, i.e. the budget in the prompt without forcing, has 40% control and *negative*
    scaling (−24): "current models cannot count tokens".
  - Rejection sampling to a length shows inverse scaling (−35). Their explanation: shorter samples are the
    ones "on the right track from the start".
- **L1 (arXiv 2503.04697).**
  - Says s1's hard forcing "severely degrades performance compared to the underlying model".
  - LCPO samples the target length *uniformly and independently of the problem* (n_min 100, n_max 4,000),
    uses a soft penalty α·|n_gold − n_y| with α = 0.0003, and starts from an RL'd reasoning model.
- **Qwen3 (arXiv 2505.09388).** On reaching the budget it inserts a natural-language transition before
  `</think>`. The budget ability "emerges naturally" from thinking-mode fusion; it was not trained by
  truncation.
- **Elastic Reasoning (arXiv 2505.05315).** Trains GRPO with budget-constrained, truncated rollouts so that
  the model learns to act from a cut-off think.
- **Kimi k1.5 (arXiv 2501.12599 §2.3.3).**
  - The length penalty exists to *curb* overthinking for efficiency.
  - It "may slow down training during the initial phases", so it is warmed up.
- **OLMo 3 (arXiv 2512.13961 §4.4.1, footnote 36).** A length-control (L1-style) verifier was tried and "did
  not find it useful for downstream performance".

**Interpretation.**
- `--select shortest` is a length penalty by selection, in a project whose goal is to make longer thinking pay.
  It also selects for "lucky" short trajectories (s1's rejection-sampling result).
- Effort-conditioned GRPO starts with a template mismatch: the prompt ends in a token that has never been
  trained. Dr. GRPO (arXiv 2503.20783) finds that a template mismatch "can destroy reasoning capabilities
  before RL reconstructs it".
- The effort token as built cannot be read as a budget. It reads as "empty copy / not empty", plus a hint
  about proof length.

**Fix**, in this order:
1. **Graceful truncation.** Close at the last line boundary before B, or let the current line finish up to a
   hard cap. In EI, do not train forced closes (drop those steps or mask the `</think>`).
2. **Randomised hindsight labels for retained data.** Draw the label uniformly from the levels ≥ the record's
   need, so that the label stops encoding difficulty. Give `<e16>` real support, e.g. by Dualformer-style
   dropping down to 16 tokens.
3. **Factor the effort run into a 2 × 2:** effort token yes/no × selection random/shortest, all on budgets
   {0, 16, 64, 256}. Use at least 2 seeds, or drop `shortest` until effort works.
4. **For GRPO,** do a short SFT warm start on relabelled data before RL, as EI already does implicitly. Then
   consider L1's soft target in place of, or alongside, the hard cap.

### C3. "The model uses its thinking" is not established: the B0 ablation is confounded, and the content controls are missing

**Issue.** The README's first observed claim is that "forcing the block empty costs 12.5 pp (T1) and 19.9 pp
(T1v1)". Its second is that "trace training costs the no-think policy 6–9 pp". Neither isolates the
*content* of the thinking.

**Observed.**
- **B0 mainly breaks the action syntax.** On textbook72, all seeds pooled, the share of attempts that fail on
  syntax-like environment errors (`term`, `no semicolon`, `malformed`):

  | arm | syntax-like failures |
  |---|---|
  | T0 | 1.4% |
  | T1 at B0 | 23.8% |
  | T1v1 at B0 | 27.4% |
  | T1 at B256 | 10.4% |
  | T1v1 at B256 | 11.0% |

  - The B0 drop is largely the model losing its action format when the context it expects is missing.
  - The cause was not checked. One candidate: think-vocabulary tokens are not masked inside actions
    (`mask_action_think` defaults to False).
- **T0e explains the "6–9 pp cost".** T0e (empty tags, no traces) is 6.0 pp below T0 at pass@256, with
  interval −14.4 to +1.4.
  - T1 at B0 against T0e is −0.5 pp (−7.9 to +7.9); at pass@1 it is −0.4 pp (−4.3 to +4.0).
  - So the "trace training cost" is not separable from the format, or from seed noise.
  - The choice of no-think baseline (T0 or T0e) changes the pass@256 gain of thinking from +6 to +12 pp.
- **Not run yet:**
  - the causal-corruption test (PROPOSAL §4, measure 5);
  - a length-matched filler arm;
  - a swapped-trace arm.

**Reported.**
- **Gandhi et al. (arXiv 2503.01307).** Use two controls: an empty CoT and a length-matched placeholder CoT
  (`<think>. . . .</think>`). Both stay at the base model's ≈30–35%. Models primed with *incorrect* solutions
  that show the right behaviours do as well as those primed with correct ones.
- **Stechly et al., "Beyond Semantics" (arXiv 2505.13775).** Transformers trained from scratch on A* maze
  traces. With traces *swapped between problems*, accuracy is equal or better:
  - Wilson mazes: 83.3% against 79.9% with correct traces;
  - OOD Drunkard mazes: 11.7% against 0.0%.

  GRPO raised accuracy "without any improvements in trace validity". Trace length correlated little with
  problem complexity.
- **Pfau et al. (arXiv 2404.15758).** Meaningless filler tokens can supply computation, but learning to use
  them "requires specific, dense supervision".
- **Goyal et al. (arXiv 2310.02226).** Pause tokens help when present in both pretraining and fine-tuning.

**Interpretation.**
- Every Stage-1 result is consistent with "the think block is extra serial compute plus a source of sampling
  diversity, whose content barely matters".
- The fact that v1 (ritual traces) ≥ v2 (audited traces) is what Stechly et al. would predict. It is not
  evidence that "variety and length" are what matters.
- The v1 − v2 difference is not significant either: −2.3 to +13.0 pp at pass@256. v1 also spends 31% more
  tokens per attempt.

**Fix.** These are cheap and decisive; each Stage-1 run takes about 45–60 min on a 3090.
1. **T1-swap:** each think block taken from a different state with the same step type and length bin.
2. **T1-filler:** the same per-step think lengths, filled with one dummy token.
3. **Inference-time corruption of T1 and T1v1 at B256:** swap blocks between states, shuffle lines, flip
   `<no>`. Measure the accuracy drop.
4. **3–5 more seeds of T0 and T0e,** to settle which of them is the baseline.

Until (1)–(3) are run, do not write "the model uses its thinking". Write "the model depends on the block's
presence".

### C4. Statistical power: textbook72 and the OOD sets cannot resolve the effects being claimed

**Issue.** The summary gives point estimates over 3 seeds, says "no CI" in its captions, and interprets
differences of 2–5 pp.

**Observed.**
- **textbook72's informative core is about 34 problems.** 14 problems are solved by every arm and seed, and
  24 by none. One problem is 1.39 pp.
- **Reading noise.** The same T0e model, read twice, gave 37.5% and 40.3%.
- **Seed SD at pass@256:** 2.1 pp (T0) to 4.2 pp (T1v1 B256).
- **Paired differences.** At pass@1 (3-seed means), the per-theorem difference against T0 has SD 8.7–9.1 pp.
  - Theorem sampling alone would detect 3 pp with 72 theorems.
  - Seed noise is the binding term. Seed SDs at pass@1 are 1.6 pp (T0) and 2.1 pp (T1v1 B256). Detecting a
    3 pp pass@1 difference needs about 6 seeds per arm.
- **The RL arms have one seed each.** T1v1 has two. They cannot resolve differences under about 5 pp on
  textbook72 whatever is done to the read.
- **OOD.** Only 110 of the 840 OOD theorems are solved by *any* plain cell at k 16. At pass@1 ≈ 1.5%, the
  whole OOD read rests on the equivalent of about 13 theorems.
- **pass@256 with n = 256** is the "any success" estimator, the highest-variance point of the unbiased
  estimator.

**Reported.**
- Hochlehnert et al., "A Sober Look" (arXiv 2504.07086): small benchmarks (AIME'24, 30 problems) are "highly
  sensitive" to seeds and decoding. Under standardised evaluation, RL gains shrink to marginal.
- Miller, "Adding Error Bars to Evals" (arXiv 2411.00640): clustered standard errors, paired differences,
  power analysis.
- Practice in the cited reports:
  - OLMo 3 measures eval variance over 3 runs of 14 models and buckets benchmarks by it.
  - DAPO reports avg@32.
  - L1 uses 16 seeds.
  - Prover-V1.5 and V2 report ±SD.

**Fix.**
- Report hierarchical (theorem × seed) bootstrap intervals for every paired arm difference. Cluster the
  generated families by (family, knob).
- Use n ≥ 2k samples for any pass@k point that is plotted.
- Pre-register the one comparison each RL run is meant to decide.
- Run at least 2 seeds for any RL arm whose result will be claimed.
- Move pass@1 comparisons to a set whose pass@1 sits in 10–60% (C5).

### C5. The OOD testbed is dominated by floor items and a length cliff, mixes in the dev set, and is not sealed

**Observed.**
- **Floors.** `ood_case_product` is 0/40 for every model, budget and method. In `or_reorder`, `and_reorder`
  and `commute`, knobs n ≥ 5 are at about 0% pass@16 everywhere: 480 of the 720 generated items. For
  example, T0 s0 is at 0/40 on and_5 to and_8, and on or_5 to or_8.
- **The cliff is proof length.** Mean reference length per knob:

  | family | n = 3 | n = 5 | n = 8 |
  |---|---|---|---|
  | or_reorder | 12 lines | 25 | 46 |
  | and_reorder | 7 lines | 13 | 22 |
  | commute | 9 lines | 17 | 28 |

  K12 trains on 2–12 lines.
- **Against OOD_CHECK's own advice.** The sets include n = 7–8, which OOD_CHECK said to drop: states reach
  396 tokens, above K12's 386 maximum.
- **textbook_dev is not OOD.** It is part of the "OOD" read and of textbook72, and it has been the project's
  dev set throughout.
- **Selection.** Tier A was filtered with the SOTA model's failure analysis ("within reach").
- **Contamination at the class level is clean.**
  - The trace audit found 0 leaks.
  - This review checked `rl_targets.jsonl` (4,495) against all five OOD sets, textbook_dev, heldout and
    textbook_train with `data.EvalIndex`: **0 hits**.

**Reported.**
- Zhang, Neubig and Yue (arXiv 2512.07783): RL yields real capability gains (pass@128) only when RL data
  "target the model's edge of competence". Contextual generalisation needs "minimal yet sufficient
  pre-training exposure".
- OLMo 3 (§6.2) runs a spurious-reward negative control to check that RL-Zero evals are not contaminated.

**Interpretation.**
- With per-step accuracy s, a proof of L lines succeeds with probability of roughly s^L. Even 97% per step is
  under 50% at 25 lines.
- The n ≥ 5 cliff is a long-horizon compounding and length-extrapolation problem. Per-step Markov thinking
  attacks it only by raising s. Guided redraws attack the same quantity for free.
- The OOD sets as built measure "has the model ever seen 13+-line proofs", not "can thinking buy accuracy".

**Fix.**
1. Split the read into separately reported tiers:
   - in-distribution held-out;
   - near-OOD (13–20-line problems from the K12 generator, and n = 3–4 of the families, with fresh seeds);
   - far-OOD (n ≥ 5), reported as a floor check only.
2. Make the near-OOD band large: ≥ 300 items, with pass@1 in 5–50%.
3. Seal half of it now, untouched until the final read.
4. Report textbook_dev separately as "dev".
5. Optionally, borrow OLMo 3's spurious-reward control, i.e. EI or GRPO with random reward, as a check on the
   protocol.

### C6. RL trains on a saturated, in-distribution pool; EI arms are not compute-matched, and EI's "distinct" inflates the thinking arms' data

**Observed.**
- **The pool.** RL targets are the ladder's `rl_targets.jsonl`. The SOTA model solves 97.2% of them at
  pass@256 (OOD_CHECK).
  - Round 1 at k 32 from these Stage-1 inits:

    | init | targets solved |
    |---|---|
    | T0 | 3,681 / 4,495 (81.9%) |
    | T1 | 3,688 (82.0%) |
    | T1v1 | 3,496 (77.8%) |
    | T1v1+effort | 3,528 (78.5%) |

    That is no coverage gain from thinking at 2–3× the tokens.
  - EI runs have `--transfer` and `--heldout` unset, so no held-out signal is logged during EI.
- **"Distinct" counts think variants.**
  - T1 round 1: 79% of solved targets have ≥ 4 kept entries, against 45% for T0.
  - The share with ≥ 4 *distinct proofs* is the same, 46% against 45%.
  - Mean distinct (think, action) sequences per target: T1 9.6 against T0 4.0. Mean distinct proofs: 3.7
    against 4.0.
  - So the thinking arms' fine-tune mix has many more RL-weighted records (up to 4 × rl_weight 4 per target)
    that repeat the same proof with different thinks.
  - That is a different RL-to-retain ratio than T0's, and a push toward action-level collapse.
- **Compute.** EI fixes attempts (8 rounds × k 32), not tokens. A thinking attempt costs 2.8–3.6× a T0 attempt
  at B256 on textbook72, so T1's EI sees roughly 2× T0's sampled tokens. This is an estimate: T1's round-1
  token counts are not written yet.

**Reported.**
- OLMo 3 (§4.4.2): offline difficulty filtering removes prompts with pass rate > 62.5% (8 rollouts, T 1.0).
  Active sampling keeps batches full of non-zero-advantage groups and "greatly reduced loss variance"
  (Fig. 26).
- DeepSeek-Prover-V1.5 (arXiv 2408.08152 §2.3): RL prompts are theorems where the SFT model "has a moderate
  success rate" (about 4.5k kept).
- Havrilla et al. (arXiv 2403.04642): EI performs best in most cases, with sample complexity similar to PPO;
  "models fail to explore significantly beyond solutions already produced by SFT models".
- Xiong et al. (arXiv 2504.11343): rejection-sampling fine-tuning (RAFT) is competitive with GRPO. GRPO's
  edge comes mainly from discarding all-wrong prompts.

**Interpretation.** EI and GRPO on this pool will raise in-distribution pass@1, which is what EI does
(pass@k distilled into pass@1, as OOD_CHECK notes). They will not create the OOD headroom Leon wants. EI-T1
against EI-T0 at the end of 8 rounds is confounded by token budget and mix size.

**Fix.**
- Dedupe "distinct" by action sequence for the cap, and keep at most one think variant per proof per round.
- Match EI arms on sampled tokens: T0 at k ≈ 64–90, or plot every arm against cumulative sampled tokens.
- Turn on `--heldout`.
- Build an edge-of-competence RL pool: 13–20-line generator problems, decontaminated, filtered offline to
  pass rate ≤ 62.5% for the init being trained.

### C7. GRPO choices against R1, OLMo 3, DAPO and Dr. GRPO

| choice | `grpo.py` | literature (reported) | risk here |
|---|---|---|---|
| group | 16 per (target, budget) | R1 16; DAPO 16; OLMo 3 8; Prover-V1.5/V2 32 | fine |
| KL | none | OLMo 3, DAPO, Dr. GRPO: none; R1 0.001 (reference reset every 400 steps); Prover-V1.5 0.02 | fine only if entropy is watched (below) |
| clip | symmetric 0.2 | DAPO: ε_low 0.2, ε_high 0.28, because symmetric clipping causes "entropy collapse"; OLMo 3 0.2/0.272; R1 (Nature version) ε = 10, saying low values truncate gradients for many tokens | collapse of exactly the diversity that is Stage 1's only win |
| lr | 5e-5 default, lr-check at 5e-5 and 1.5e-4 | DAPO 1e-6; OLMo 3 1e-6 (7B Think), 2e-6 (32B); R1 3e-6; Prover-V1.5 5e-6 | not comparable across 9.5M and 7B; the right check is approximate KL and clip fraction per update, not the lr value |
| updates per batch | 4 minibatches, PPO ratio | DAPO 16; R1-Zero 16 minibatches, 1 epoch; OLMo 3 Think 1 | fine with clipping |
| zero-gradient groups | skipped, no refill | DAPO dynamic sampling; OLMo 3 active sampling | smoke: 48–62% of groups skipped, so the effective batch halves and swings |
| curriculum | p(1 − p) + 0.03 over (target, budget) | LILO (arXiv 2502.12272): learnability sampling, 3× fewer steps | good, but it starves budgets with p ≈ 0 (e16, e64 above) and spends the 0.03 floor on a 97%-solvable pool |
| loss normaliser | tokens in the minibatch | DAPO and OLMo 3 the same; Dr. GRPO: a constant (MAX_TOKENS), since per-batch normalisation still varies | mixing budgets in a batch changes the per-token step size with the budget mix; minor |
| advantage | r − mean, no std | Dr. GRPO; OLMo 3 | fine |
| truncated thinks | forced `</think>` masked; truncated content keeps the attempt's advantage | DAPO: penalising truncated samples is "reward noise"; overlong *filtering* (masking) "significantly stabilizes training" | failures caused by the cap push down sound partial reasoning |
| logging | no entropy, no think validity, no probe pass@k | DAPO §4.3: monitor entropy and length; Stechly: validity decays under GRPO | collapse would not be seen until the final eval |
| credit | one advantage on every step's tokens | VinePPO (arXiv 2410.01679): Monte Carlo values from intermediate states, up to 3.0× less wall-clock than PPO | 40–56% of tb72 attempts finish in the environment and are then rejected by Lean, so failure credit lands on whole proofs |

On R1's failed process reward model (PRM): R1 (arXiv 2501.12948 §4.2) lists three problems. Fine-grained
steps are hard to define, step correctness is hard to judge, and a *model-based* PRM "inevitably leads to
reward hacking". The first two do not apply to a proof-state environment, whose steps are defined and checked
by rules (`step_check`, sound with respect to Lean). The third applies to learned PRMs only. R1 is therefore
not an argument against using the rule-based step check inside RL rollouts, which is what Dan's +9 pp guided
redraws already do at test time.

**Observed failure mix** (textbook72, share of all attempts):

| arm | finished in the environment, then rejected by Lean |
|---|---|
| T0 | 51.8% |
| T1 B256 | 44.7% |
| T1v1 B256 | 40.2% |

This is the largest failure class in every arm.

**Fix.**
- Add clip-higher (0.28).
- Log, per iteration:
  - token entropy, separately for think and action;
  - the think-validity rate (`trace_dsl.check_step` on a sample);
  - pass@k (k up to 64) on a fixed probe set.
- Refill skipped groups, or report the effective batch size.
- Mask the think tokens of truncated blocks in failed attempts.
- Run one arm with the logical arm (step_check redraws or early termination) in rollouts.

### C8. Per-step Markov thinking against whole-proof thinking; R and M as specified are re-renderings of the reference proof

**Reported.**
- **Lean-STaR (arXiv 2407.10040).** Per-step thoughts written *retrospectively* by GPT-4 from the ground-truth
  tactic, which is our programmatic set-up with a stronger author.
  - Gains are modest and reported at pass@32/64 only: InternLM2-plus 41.3% (SFT), 43.4% (Lean-CoT),
    45.5% (Lean-STaR).
  - EI without thoughts reaches 43.0%.
  - No matched-token comparison; a single run.
  - Their Table 10 explanation: thought gains grow with K because "thoughts increase the diversity of outputs".
- **DeepSeek-Prover-V1.5 (arXiv 2408.08152).** Whole-proof generation with truncate-and-resume. RL CoT is
  51.6% against non-CoT 50.5% at pass@128 (±0.5 SD).
- **DeepSeek-Prover-V2 (arXiv 2504.21801).**
  - A whole-proof *sketch with holes* (`have … := by sorry`): DeepSeek-V3 decomposes, and a 7B prover
    searches each subgoal.
  - Under RL, generated proofs "frequently diverge" from the CoT decomposition. They added a consistency reward
    to enforce it.

**Observed (`rootplan.py` docstring).**
- **R** writes at step 0 "the reference term of a frame's goal … top-down, depth first". The root plan *is*
  the reference proof, and the following actions are copies of it.
- **M**'s memo is "the remaining plan of the current frame … from the current statement on". Its target is
  its input minus the head, and the head is the action.

Both bring back the property TRACE_AUDIT finding 1 rated HIGH, "the plan names the action" (94.7% predictable
by string match in v1). That is the r1_cot design that lost at matched tokens (PROPOSAL §0).

**Interpretation, on whether R and M are well posed:**
- **R has no re-planning.** Environment states arrive in context with no loss, but the plan is never
  re-emitted. A plan that the environment contradicts mid-proof cannot be revised, so the model learns to
  ignore observations.
- **R has one knob per attempt** (exploration at the top two levels). There is no per-step lever where the
  errors happen.
- **R's context grows with proof length.**
  - Stage-1 pairs never exceed 598 tokens.
  - A 9.5M ALiBi/NoPE model must extrapolate to thousands of tokens on exactly the long OOD proofs.
  - Kazemnejad et al. (arXiv 2305.19466): ALiBi does poorly at length generalisation, NoPE best, and a
    scratchpad "is not always helpful".
- **M teaches copy-and-shrink under teacher forcing.** At test time it carries its own memo, so a wrong memo
  persists: exposure bias.
- **M's cost.** It re-samples up to 256 memo tokens at every step. On a 25-line proof that is thousands of
  sampled tokens per attempt, and the inputs carry the previous memo as prefill.
- **Cost metric.** "Sampled tokens" undercounts R, whose observations are prefill, and M, whose memo is
  re-read. Define one cost (sampled tokens plus total forward tokens, or FLOPs) before comparing formats.

**Fix.**
- Before training R or M, run the TRACE_AUDIT "action = term of a plan line" check on their data. If the plan
  determines more than about 90% of actions verbatim, the format is a re-rendering.
- Restructure R as Prover-V2 does: a sketch whose hard sub-goals are holes `⊢ F`, filled at later steps, with
  a re-plan line allowed when a state contradicts it.
- Restructure M so that the memo holds only the *open subgoals*, not the remaining proof term.

### C9. Programmatic traces rationalise a known answer; they are not a record of search

**Observed (TRACE_AUDIT).**
- Reference actions are fixed. Dead ends are chosen by a generator that knows the answer.
- Base-model distractors cover 7.8% of proofs; the rest are synthetic mutations.
- Real (non-ex-falso) goal-level dead ends occur in about 12% of decision blocks.
- Think length tracks proof length (ρ 0.75–0.76) more than exploration (ρ 0.33 in v1).
- **All EI runs have `--validate_think off`**, so the RL data keep think blocks whose claims may be false.

**Reported.**
- **Stream of Search (arXiv 2404.03683).** GPT-Neo 250M, trained from scratch on traces from real heuristic
  BFS/DFS solvers, including failures (only about 57% of the trajectories reach a solution). SoS reaches
  51.27% against 25.73% for optimal paths only, and STaR/APA then solve 36% of the previously unsolved
  problems.
- **Searchformer (arXiv 2402.14083).** Learns from A* execution traces. It outperforms models that predict the optimal
  plan directly while using a 5–10× smaller model and a 10× smaller training set.
- **Dualformer (arXiv 2410.09918).** Randomly dropped trace parts give fast and slow modes:
  - slow mode 97.6% optimal on 30 × 30 mazes, against 93.3% for Searchformer;
  - fast mode 80% against 30% for solution-only.
- **Gandhi et al.** Behaviours matter more than correctness. RL "amplifies empirically useful behaviors while
  suppressing others": backtracking and verification are kept, backward chaining and subgoal setting fade.
- **STaR (arXiv 2203.14465).** Rationalisation (writing a rationale given the answer) is a bootstrap device.
  Its value comes from the subsequent self-generated, filtered rationales.
- **R1 (arXiv 2501.12948).**
  - Only "thousands" of cold-start examples, for readability.
  - The behaviours come from RL on a strong base.
  - For a 32B student, distillation (72.6% AIME) beats RL on its own (R1-Zero-Qwen-32B, 47.0%).

**Interpretation.** The literature's from-scratch successes (SoS, Searchformer, Dualformer) all train on
traces of a *real search whose length is its actual cost*. Ours are decorated rationalisations whose length is
a sampled hyperparameter. Expect RL to keep whichever line types correlate with success, likely the
candidate-and-look check, and to drop plan lines (Gandhi's finding).

**Fix.**
- Generate traces from a search that does not see the reference, e.g. the truth-table-pruned backward search
  run blind, with the model's own proposals as candidates, so that dead ends are the model's real errors.
- Turn on `--validate_think drop` in at least one EI arm, so that RL data keep only true thinks
  (STaR-style).
- Report the think-validity rate per round.

### C10. Can a 9.5M model trained from scratch show test-time scaling at all, and what would show it

**Reported, in favour.**
- SoS (250M), Searchformer and Dualformer, Stechly et al. (about 380M), and Physics of LMs 2.2
  (arXiv 2408.16293: error-and-correction data in pretraining beats error-free data) all show that small
  transformers trained from scratch learn to *use* trace tokens.
- Merrill and Sabharwal (arXiv 2310.07923): CoT length strictly increases a transformer decoder's
  expressivity (logarithmic, then linear, then polynomial steps).

**Reported, against.**
- R1 §4.1: small models "relying on large-scale RL require enormous computational power", and distillation
  wins.
- Havrilla et al. and Yue et al. (arXiv 2504.13837): RL mostly re-weights what is already sampleable. Base
  models have higher pass@k at large k.
- Dr. GRPO: "aha" self-reflection already exists in base models before RL.

**Interpretation.**
- The capacity question is secondary. The binding constraint is that our think language asks for *verdicts
  without derivations*. `⊢ G <no>` asserts unprovability with no countermodel written out (PROPOSAL §2.4
  lists countermodel display as "optional later"), so a 6-layer model must guess a truth-table result in one
  forward pass.
- The CoT expressivity results apply only when the intermediate computation is serialised.

**What evidence would show test-time scaling, all on held-out near-OOD, with seed CIs:**
1. **Within-model, interventional.** pass@1 rises monotonically in B ∈ {0, 16, 32, 64, 128, 256} with graceful
   truncation, and keeps rising under minimum-think forcing beyond the natural length (s1's "Control" and
   "Scaling" metrics).
2. **Dominance.** The thinking pass@k-against-tokens curve lies above plain sampling *and* guided redraws from
   the same checkpoint over the 1k–30k tokens-per-theorem range.
3. **Content-dependence.** Swapped and filler controls do not reproduce the gain, and inference-time
   corruption removes it.
4. **Adaptivity.** Gains grow with difficulty, and at a free budget the model allocates more thinking to the
   harder held-out problems (Snell et al.).

### C11. Smaller design issues

- **Train14.** It is still reported. The 16+-line bin has n = 13: "8% vs 0%" is one problem per seed.
- **Token accounting.** Cost is k × the set-mean tokens per attempt (`fig_tokens` caption). That assumes
  uniform per-theorem cost. Report per-theorem allocation if adaptive effort is claimed.
- **Extrapolation is untested.** Stage-1 blocks are at most 240 tokens, and B256 is effectively free.
  Minimum-think forcing (the s1 "Wait" analogue, PROPOSAL §2.4) has not been run.

## 3. What the literature predicts for our runs (falsifiable)

Each prediction gives the measurement and what would falsify it.

**P1. EI sharpens and narrows** (Havrilla et al.; Yue et al.).
- After 8 rounds, tb72 pass@1 rises for every arm.
- The Stage-1 pass@256 advantage of thinking over T0 (+6.0 pp T1, +10.6 pp T1v1) shrinks.
- *Falsified if:* the EI-T1 − EI-T0 pass@256 gap on tb72 is larger than at Stage 1.

**P2. Think length falls under EI without an effort token** (forced closes trained as natural; 46% of kept
T1 entries come from B16 or B64).
- Mean think tokens per attempt at free budget (B256) on tb72 drop by at least 20% by round 8 in EI-T1 and
  EI-T1v1.
- *Falsified if:* they rise.

**P3. `<e16>` collapses to `<e0>`** (s1's token-conditional result; 0.7% supervised support; the truncation
damage in the smoke run).
- After the effort EI or GRPO run, pass@1 at e16 lies within noise of e0.
- The mean e16 think length is under 8 tokens per decision step.
- Any rise of the e0 → e256 curve comes from e64 → e256.
- *Falsified if:* e16 beats e0 by 3 pp or more with a seed interval excluding 0.

**P4. Entropy collapse without clip-higher** (DAPO).
- In GRPO with symmetric 0.2 clipping and no KL, action-token entropy falls monotonically.
- tb72 pass@256 at the end of GRPO falls below the same init's Stage-1 pass@256, even as pass@1 rises.
- *Falsified if:* final pass@256 ≥ the Stage-1 value.

**P5. No R1-style length growth** (hard per-step caps; budgets never compared within a group; no
per-response normalisation, the Dr. GRPO length artefact).
- Mean e256 think tokens per decision step change by less than ±25% over GRPO.
- *Falsified if:* they grow by more than 25% with flat or rising accuracy.

**P6. Trace validity decays under RL** (Stechly et al.).
- The share of sampled think blocks that pass strict `check_step` falls between the init and the end of RL
  in every arm without `--validate_think`.
- *Falsified if:* it rises.

**P7. OOD gains stay at the short knobs** (Zhang et al.; the RL pool is 2–12-line problems).
- After RL, n ≥ 5 family items stay at 2% pass@16 or below in every arm.
- Any OOD gain is concentrated at n = 3–4 and in Charles's 11–15-line problems.

**P8. The frontier empties without active sampling** (DAPO Fig. 3b; OLMo 3 Fig. 26).
- The GRPO `groups_skipped` fraction rises above 60% within about 50 iterations on `rl_targets`.
- `frontier` (the share with 0.1 < p < 0.9) declines monotonically.

**P9. Swapped traces match** (Stechly et al.).
- A T1-swap Stage-1 arm lands within the seed CI of T1 at tb72 pass@256.
- *Falsified if:* T1 beats T1-swap by more than the 95% interval.

**P10. The R plan diverges under RL** (Prover-V2's consistency-reward finding).
- After RL on format R, the share of actions that match the root plan verbatim falls, unless a consistency
  reward is added.

**P11. The matched-token crossover persists** (Lean-STaR's diversity reading; Prover-V2's token ratio).
- After RL, at ≤ 2k tokens per theorem on tb72, T0-RL ≥ T1-RL. The crossover stays at 3k tokens per theorem
  or more.

## 4. Recommended changes, prioritised

**Cheap: CPU, existing data, or a minute of GPU.**
1. **Correct the Stage-1 summary.**
   - Add hierarchical CIs.
   - Report pass@1 and pass@5 next to pass@256.
   - Report OOD without textbook_dev, and per-knob tables.
   - Report the PROPOSAL §6 falsifiers.
   - Retract "trace training costs the no-think policy 6–9 pp" (T0e explains it) and the claim that v1's lead
     "supports … variety and length" (the difference is not significant).
2. **Fix the knob before spending GPU on effort RL.**
   - Truncate at line boundaries.
   - Do not train forced closes in EI.
   - Use randomised hindsight effort labels.
   - Warm-start the effort tokens with SFT before GRPO.
3. **Factor the effort run.** Effort × selection, or drop `--select shortest`.
4. **Instrument GRPO.**
   - Log entropy for think and action, think validity, probe pass@k, and effective batch size.
   - Add clip-higher 0.28 and a constant loss normaliser.
   - Mask truncated think tokens in failed attempts.
5. **Fix the EI bookkeeping.**
   - Make "distinct" mean distinct action sequences.
   - Turn on `--heldout`.
   - Plot EI against cumulative sampled tokens.
6. **Audit R and M** with the TRACE_AUDIT plan-names-the-action script before any training.

**Medium: about 1–10 GPU-h each.**
7. **Inference-time corruption tests** on the existing T1 and T1v1 checkpoints (PROPOSAL §4.5).
8. **Missing Stage-1 controls:** T1-swap, T1-filler, and 3–5 more seeds of T0 and T0e.
9. **The interventional TTS read** on every Stage-1 and RL checkpoint: B grid × minimum-think forcing, with
   plain and guided parallel baselines, on a new sealed near-OOD band. Build the band (CPU) first.
10. **One RL arm with step_check in the rollout** (logical arm) and a T0-guided EI arm, as the baselines thinking
    must beat.

**Expensive: tens of GPU-h.**
11. **An edge-of-competence RL pool** (13–20-line generator problems, offline-filtered to pass rate ≤ 62.5%)
    in place of the saturated `rl_targets`.
12. **Search-generated traces** in place of rationalised ones: a blind search with the model's own candidates,
    SoS-style. Then STaR-style self-traces with `--validate_think drop`.
13. **L1-style effort RL:** uniform random targets, a soft length reward, started from a checkpoint that has
    already passed the step 7–9 checks.
14. **Sketch-with-holes R**, Prover-V2-style, with a re-plan line. Only after the step 6 audit.

## How the new numbers in this note were computed

All of this was CPU work on stored files, using short ad-hoc scripts that were not committed.

- **Stage-1 textbook72.**
  - Source: `runs/eval/tb72_*_s{0,1,2}/rows_plain_B{0,256}.jsonl`.
  - Unbiased pass@k from `n_ok` of `k` = 256 per theorem.
  - Intervals: a 4,000-draw bootstrap that resamples the 72 theorems and, independently, the 3 seeds of each
    arm, for the difference of arm means.
- **OOD.** `runs/eval/ood_*/rows_plain_B*.jsonl` and `summary.json`, with the `textbook_dev` rows removed for
  the "OOD-only" numbers. Knob tables join each row to `ood_check/sets/*.jsonl` by `name`.
- **Failure mix.** The per-attempt `status` and `ok` fields of the same rows. "Finished in the environment,
  then rejected by Lean" means `status == "done"` and `ok == 0`.
- **EI.** `runs/ei/ei_*_s0/found_1.jsonl`: entries by `budget`; steps whose think length ≥ budget; distinct
  action sequences per target name.
- **Effort-label counts.** `runs/logs/ei_T1v1e_s0.log`.
- **GRPO smoke.** `runs/grpo/smoke/metrics.jsonl`.
- **Contamination check.** `code.thinking.data.EvalIndex` over `DEFAULT_EXCLUDE` plus `textbook_train`,
  queried with every prompt of `rl_targets.jsonl`.
