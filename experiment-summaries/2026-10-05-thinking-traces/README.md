---
written_on: 2026-10-05
written_by: agent:claude
---

# Thinking traces for the proof-state model (test-time scaling, LLM tradition)

Status (2026-10-08):
- Stage 1 is done: 4 arms × 3 seeds, with reads on textbook72 and the OOD sets (below).
- The first RL phase is done (EI and fixed-loss GRPO on the rl_hard12 pool, to iteration 78). See
  [RL phase](#rl-phase-2026-10-06-to-08) and the full write-up,
  [RL_RESULTS.md](../../code/experiments/current/thinking_traces_20261005/RL_RESULTS.md).
- **The LLM-written-trace arm is dropped.** Claude subagents asked to write these traces were blocked by a
  `[reasoning_extraction]` safeguard, so no Claude-written trace is used for training.

Question: if the small proof-state prover (state_env_october, 9.56M parameters) is pretrained to write a short
think block before each step, can RL later make it use more thinking for more success (test-time scaling)?
Leon's assignment from the [2026-10-02 group meeting](../../docs/research_calls/group_meeting/group_meeting-02-10-2026.md).

Working folder: [`code/experiments/current/thinking_traces_20261005/`](../../code/experiments/current/thinking_traces_20261005/)
(branch `leon/thinking`, built on leon/main + origin/dan 816e553).

## What is settled so far

- **Headroom exists outside the generator's pools** ([OOD_CHECK.md](../../code/experiments/current/thinking_traces_20261005/OOD_CHECK.md);
  existing data only). In-distribution pools are saturated (95.6-100% at pass@256). Charles's long problems are not:
  pass@1 7.7-13.0% and pass@256 34.4-46.7% (n=90), with guided redraws adding 9.1 pp at matched tokens. Most
  "wildly OOD" axes (≥4 premises, formula depth ≥7, >4 atoms, name exhaustion) fail on format, which thinking
  cannot fix. Structural depth with small formulas (nested case splits, reordering) is the usable axis.
- **Prior attempt** (r1_cot, 2026-09-09, old stack): a think block that re-rendered the proof lost to direct
  proofs at matched tokens (10/58 vs 11/58 at k 16). The Stage-1 results below change that picture: both
  trace versions here are used by the model, including v1, whose plan lines largely re-render the reference.
- **Design proposed** ([PROPOSAL.md](../../code/experiments/current/thinking_traces_20261005/PROPOSAL.md)):
  - a per-step Markov think block, `state <think> … </think> action <eos>`;
  - the 111 existing tokens plus `<think> </think> <no>`;
  - every think line is a claim the validator checks: goal reductions, looks, candidates, `<no>` dead ends.
  - Traces were to be written by Claude, with programmatic search traces as filler and control. Only the
    programmatic traces were used; the LLM route was blocked (see Status).
  - Pilot: 8/8 hand-written traces valid, all Lean-checked. This was written by an agent, so it is not a
    production acceptance rate.

## Pipeline (built, tested, used for Stage 1)

- `code/thinking/`: the model side.
  - **Vendoring.** Dan's fork is vendored unmodified at 4dcd0e4f and loaded under private module names.
  - **Tokenizer.** The `lean_thinkn` tokenizer has 114 ids, ref0 50.
  - **Data builder.** Takes traces or plain K12, with no-think modes `empty` / `none` and eval-set exclusion by
    renaming class. The full K12 build is 1,239,820 pairs, max 394 tokens.
  - **Training.** Stage-1 training uses Dan's best recipe, forced offline. A cold start from a `lean_staten`
    checkpoint leaves old-token logits exactly unchanged.
  - **Sampler.** A thinking-aware state-env sampler with per-step budget forcing (max B, min m), plus guided
    redraws of the whole think + action. Every sampled token is counted.
  - **Eval and EI.** An eval over a think-budget grid, and expert iteration that keeps the sampled think tokens.
  - Commands and 3090 estimates: [`pipeline/README.md`](../../code/experiments/current/thinking_traces_20261005/pipeline/README.md).
- `code/thinking/gen/`: the LLM side.
  - **Pipeline.** Stratified selection with eval exclusion, then base-model distractor sampling, then Batch API
    requests: structured output, prompt caching, effort. A collector validates replies with up to 2 feedback
    retries and a cost ledger.
  - **Safety.** Dry run is the default. A real call needs `--submit --max-usd`, and the key is never written.
  - Calibration commands and cost: [`gen/README.md`](../../code/experiments/current/thinking_traces_20261005/gen/README.md).
- **Checks run:** 93 thinking tests pass (CPU, including slow), 27 validator tests, and `make -C code check`
  (ruff, format, pyright) passes.

### Estimates

On RTX 3090s; not measured:
- no-think control Stage 1: about 20-30 min;
- traced Stage 1: 24-40 min;
- cold start: 3-4 min;
- 4 arms × 3 seeds: 4.4-6.4 GPU-h;
- first eval: about 25 min per checkpoint.

API:
- calibration of 50 proofs × 3 models: about $4 (worst case $17);
- 2,000 proofs on Opus: about $96 (range $57-174).

## Decisions taken (Leon, 2026-10-05)

The PROPOSAL.md §8 items, as decided:
1. **Format:** per-step Markov think blocks with `<think> </think> <no>`.
2. **Reference actions stay fixed.** Traces never change the proof; backtracking inside proofs is emphasised.
3. **LLM traces:** planned via Sonnet subagents instead of the Batch API. That route is now blocked by a
   safeguard (see Status) and is not pursued.
4. **Programmatic traces:** built (`code/thinking/progtrace.py`).
5. **First run:** from scratch, compute approved.

Data mix:
- matched compute, 24k steps per arm;
- all 155k K12 proofs;
- no sweep;
- 20% of traced copies with emptied think blocks, Dualformer-style (`data.py --empty-frac 0.2`).

## Programmatic traces (built and audited 2026-10-05)

An Opus audit of the first version found it correct but ritual-heavy
([TRACE_AUDIT.md](../../code/experiments/current/thinking_traces_20261005/TRACE_AUDIT.md)):
- every hard and semantic check passed on all 154,977 traces;
- 0 eval-class leaks;
- Lean accepted 300/300 sampled proofs and rejected 384/384 candidates marked `<no>`.

Its findings:
- In v1 the plan named the action before any candidate was checked, so a string match predicted 94.7% of the
  `<no>` verdicts.
- 72% of dead ends were the trivial ex-falso move.

Generator v2 fixes this: the action is decided by candidates checked before the answer is named, and ex falso is
no longer padding. The same audit scripts on v2 report:
- 0 failures and 0 disagreements;
- the string-match rule falls to 51.7%;
- after a look line the verdict is 50/50.

The v2 traces are `/scratch/cowifornia/nd-rl/think/traces/prog_traces_v2.jsonl`, with `.stats.json` beside it:
- 154,977 traces;
- think tokens per proof: mean 238, median 201, p90 483, rising with proof length (Spearman ρ 0.75);
- longest pair 598 tokens.

Arms: T1 trains on v2. T1v1 trains on the audited v1 traces as a "ritual traces" control.

Still open:
- Real dead ends are rare in K12 states, about 12% of decision blocks.
- Base-model mistakes are used as wrong candidates for only 7.8% of proofs.

## Stage-1 results (2026-10-06, 3 seeds per arm)

**Setup:**
- Stage 1 from scratch with Dan's best recipe: 24,000 steps on one RTX 3090, 41–60 min a run.
- Runs in `/scratch/cowifornia/nd-rl/think/runs/stage1/`.
- textbook72 reads: 58 dev + 14 train, k 256, plain sampling, T 0.8. B is the per-step think budget: 0 forces an
  empty block, 256 leaves it effectively free (training blocks are at most 240 tokens).
- Evals in `runs/eval/tb72_*` and `runs/eval/ood_*`.

**textbook72, pass@256:**

| arm | B | seed 0 | seed 1 | seed 2 | mean | tokens / attempt |
|---|---|---|---|---|---|---|
| T0, no think | – | 43.1% | 47.2% | 45.8% | 45.4% | 116 |
| T0e, empty think | 0 | 37.5% | 37.5% | 43.1% | 39.4% | 113 |
| T1, v2 traces | 0 | 41.7% | 36.1% | 38.9% | 38.9% | 129 |
| T1, v2 traces | 256 | 54.2% | 47.2% | 52.8% | **51.4%** | 320 |
| T1v1, v1 traces | 0 | 34.7% | 34.7% | 38.9% | 36.1% | 125 |
| T1v1, v1 traces | 256 | 59.7% | 51.4% | 56.9% | **56.0%** | 420 |

Noise: one T0e model, which never thinks, read twice with different budgets gave 37.5% and 40.3%.

**Matched sampled tokens per theorem** (3-seed means; [fig_tokens_tb72.png](figures/fig_tokens_tb72.png)):

| tokens / theorem | 1k | 3k | 10k | 29k |
|---|---|---|---|---|
| T0, no think | **28.8%** | **34.4%** | 40.1% | 45.2% |
| T1 (v2), B 256 | 24.2% | 32.8% | 41.7% | 47.4% |
| T1v1 (v1), B 256 | 23.5% | 32.3% | **42.2%** | **49.7%** |

**By reference length** ([fig_lengthbin_stage1.png](figures/fig_lengthbin_stage1.png), pass@256, 3-seed means):

| arm | 1–5 | 6–10 | 11–15 | 16+ | train14 |
|---|---|---|---|---|---|
| T0 | 100% | 72% | 27% | 0% | 43% |
| T1 (v2), thinking | 100% | 79% | 29% | 8% | 52% |
| T1v1 (v1), thinking | 100% | 88% | 38% | 3% | 57% |

**OOD sets** (k 16, pass@16, all 898 theorems, plain unless noted):

| arm | result |
|---|---|
| T0 s0 / s1 | 7.0% / 6.7% |
| T1 s0 / s1, B 0 | 6.2% / 5.0% |
| T1 s0 / s1, B 256 | 8.4% / 5.9% |
| T1v1 s0, B 256 | 8.7% |
| T0 s0 / s1, guided redraws | 13.7% / 13.7% |
| T1 s0, guided + B 256 | 18.0% |

Guided redraws plus thinking reach 23.8% on Charles's long problems and 25.0% on commute, at about 21k tokens per
theorem. Dan's Stage-1 checkpoint with guided redraws at k 64 (about 24k tokens) gets 20.0% and 23.8%.

**Observed:**
- **The trace-trained models use their thinking.** Forcing the block empty costs 12.5 pp (T1) and 19.9 pp (T1v1)
  on every seed.
- **Trace training costs the no-think policy 6–9 pp.** With thinking forced empty, T1 and T1v1 read 38.9% / 36.1%,
  against 45.4% for T0.
- **At matched tokens, thinking loses at small budgets and wins at large ones.** It is 4.6–5.3 pp behind T0 at 1k
  tokens per theorem, level around 5k, and 2.2–4.5 pp ahead at 29k. The gain is concentrated in the 6–15-line
  problems and the train14 problems.
- **The audited v1 traces do at least as well as v2.** At B 256 v1 leads by 4.6 pp in mean pass@256, though seed
  ranges overlap (51.4–59.7 vs 47.2–54.2). It also leads at matched tokens above 10k, thinks longer, and depends
  more on its thinking.
- **OOD, at Stage 1, thinking is worth about as much as more samples.** Plain-sampling differences are within seed
  noise.

**Interpretation, open:**
- This is the expected Stage-1 picture. The model has the format and uses it, but a think token is not yet worth
  more than a sample token except at large budgets.
- Making thinking pay at small budgets, and OOD, is the job of the RL phase.
- v1's lead supports Leon's view: what the traces must supply is variety and length, not audited rigor.
- Which init is better for RL is not settled by these reads.

## RL phase (2026-10-06 to 08)

**Setup.**
- Runs live in `/scratch/cowifornia/nd-rl/think/runs/`. Arms:
  - EI (Dan's ladder, rounds of 5,000 × k 32, stopped after round 4–5 and compared at rounds 1–2);
  - GRPO (Dr. GRPO, clip-higher, 256 × 16 per iteration, stopped at iteration 78 on 2026-10-07).
- Both arms start from T0 (no thinking) and T1v1 with effort tokens `<e0>`–`<e256>`, and are matched on
  sampled attempts.
- Pool: `rl_hard12`, 5,000 targets of 7–12 lines with T0 seed 0 at 2–16/32. Sealed test split:
  `hard12_test_1k`.

**Retracted (2026-10-07).** Two earlier readings came from a buggy GRPO loss: "thinking shrinks under RL" and
"no-thinking GRPO improves faster than thinking".
- The sampler masks `<eos>` and other tokens inside think blocks, but the loss scored the full vocabulary.
- Mass drifted onto the never-sampled `<eos>`, and the thinking GRPO run on hard12 collapsed twice.
- The fix is `7b082b7`, with a sampler-equality test, dual clip, alarms and rolling checkpoints.
- Details: [GRPO_COLLAPSE_AUDIT.md](../../code/experiments/current/thinking_traces_20261005/GRPO_COLLAPSE_AUDIT.md).
- EI, no-thinking GRPO, Stage 1 and all evals are unaffected.

**Observed:**
- **Overlong masking teaches looping.**
  - In no-thinking GRPO, attempts ending at the 48-step cap rose from 0% (iteration 40) to 21% (iteration 182),
    and batch pass@1 fell from 93% to 75%.
  - Under masking, a loop is neutral while a failure is penalised.
  - The relaunch scores capped attempts as failures (`--overlong fail`).
- **The hard12 test set is partly a selection effect**
  ([seed control](../../code/experiments/current/thinking_traces_20261005/stage1_stats/hard12_seed_control.md)):
  - T0 seed 0 reads 27.8% pass@1, against 42.4–42.5% for seeds 1–2.
  - T1v1 at B 256 is level with unbiased T0 at pass@1 (42.6% vs 42.4%) and ahead at pass@5 (80.8% vs 75.2%,
    +5.6 pp [+2.9, +8.5]).
  - At matched tokens T1v1 at B 256 is behind: −15 pp at 1k tokens per theorem, −4.5 pp at 3k.
  - Within T1v1, thinking adds +12.5 pp pass@1 and +19.1 pp pass@5 across 3 seeds.
- **Thinking allocation is adaptive, and RL strengthens it** ([ADAPTIVITY.md](../../code/experiments/current/thinking_traces_20261005/ADAPTIVITY.md)):
  - At e256, think tokens per attempt correlate with the checkpoint's own difficulty. ρ is −0.23 at the init
    and −0.43 after one EI round.
  - The e256 gain over e0 is largest where the model rarely succeeds without thinking: +15 to +31 pp.
  - The gain is slightly negative where it nearly always succeeds.
  - Short budgets are worse than none: e16 is −11 pp at the init and −26 pp after EI round 1.
- **Filler thinking hurts.** T1f, trained with filler think blocks (1 seed), loses 3.3 pp pass@1 at B 256
  against its own B 0 on textbook72 ([−4.9, −2.0], paired over theorems). The same seed of T1 with real traces
  gains 1.7 pp. The content of the traces matters.
- **The goal-stack state (all open goals) changes nothing resolvable on textbook72**
  ([table](../../code/experiments/current/thinking_traces_20261005/stage1_stats/tb72_goalstack.md)): all pairwise differences are within ±4 pp, and every interval
  includes 0.
- **The sealed 21–30-line test set (2,775 theorems) is at the floor for Stage-1 models.** T1v1 at B 256 reaches
  0.22% pass@8. Built by composing measured short pieces:
  [test_21_30/README.md](../../code/experiments/current/thinking_traces_20261005/test_21_30/README.md).

**Results of the fixed runs** (2026-10-07; details, tables and caveats in
[RL_RESULTS.md](../../code/experiments/current/thinking_traces_20261005/RL_RESULTS.md)).
- Matching: one EI round = 160k attempts = 39 GRPO iterations, so r1 ↔ it39 and r2 ↔ it78.
- 13–20 refers to the sealed 13–20-line set (986 items).

- **GRPO vs EI.**
  - GRPO is ahead in distribution at pass@1 only. hard12 no-think: 90.3% vs 78.0% at it39/r1 (+12.3 [+10.6, +13.8])
    and 92.8% vs 82.7% at it78/r2. pass@5 ties (−1.1 to +1.4), so GRPO sharpens rather than widens.
  - Out of distribution it is mixed. On 13–20, EI with thinking at round 2 (11.4% pass@1, 28.6% pass@5) is level
    with or ahead of thinking GRPO at it78 (seed 1: 10.1% / 23.3%).
- **Thinking vs no-think GRPO.**
  - In distribution: +2–3 pp pass@1 in both seeds (seeds 0+1 pooled: +2.9 [+1.4, +4.3] at it39).
  - On 13–20 the gain is mostly at pass@5: +14.8 (seed 0) and +9.7 (seed 1) at it39, and +9.4 for seed 1 at it78.
  - At matched sampled tokens the 13–20 gain is about 0 at 1,000 tokens per problem and +3 to +5 at 2,000.
  - The seed-0 run lost its 13–20 gain after it39 (−4.3 pass@1 at it78), at the same time as a clip-low spike at
    it48–52. Seed 1 kept its gain to it78.
  - The earlier reading that the 13–20 gain decays with training came from this one run. Seed 1 does not show it.
- **Variants** (one seed each, so not ranked: no-think seeds alone differ by 3.2 pp on 13–20).
  - Token cost cut tokens per attempt by about 40% at the same accuracy.
  - Graded-length Stage-1 traces gave the best 13–20 read at it78: 11.9% / 27.9% at B 256.
  - LCPO steered think length per effort level but did not raise accuracy.
- **Budget forcing lowers accuracy** at every forced minimum. On textbook72, the RL'd one-budget model goes from
  32.9% to 5.3% pass@1 at a 128-token minimum.
- **Graded-length Stage 1** (T1v1g4) removes the no-think penalty on hard12 (e0 41.5% vs T0 42.4%) and is +6.2 pp
  over T0 at e256. It brings no gain on textbook72.

Figures: `figures/fig_rl_matched`, `fig_rl_variants`, `fig_rl_t1320_curves`, `fig_rl_t1320_tokens`, `fig_forcing`
and `fig_graded_stage1` (png, pdf and svg).

## Baseline figure

Dmitry's progress figure for Dan's checkpoints is [figures/fig_lengthbin_baseline.png](figures/fig_lengthbin_baseline.png):
- textbook72 at pass@256, plain sampling, 3 seeds each;
- caption in `figures/fig_lengthbin_baseline.caption.md`.

| | 1–5 | 6–10 | 11–15 | 16+ | train14 |
|---|---|---|---|---|---|
| Pretraining only | 100% | 58% | 29% | 0% | 38% |
| After RL (round 8) | 100% | 92% | 69% | 28% | 67% |

The trace arms will be added to the same figure.
