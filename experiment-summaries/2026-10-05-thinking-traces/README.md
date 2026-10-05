---
written_on: 2026-10-05
written_by: agent:claude
---

# Thinking traces for the proof-state model (test-time scaling, LLM tradition)

Status (2026-10-05, evening): programmatic traces built for all of K12 and under audit; the no-think controls
(T0, T0e) are training; the programmatic-trace arm (T1) follows the audit. **The LLM-written-trace arm is
dropped:** Claude subagents asked to write these traces were blocked by a `[reasoning_extraction]` safeguard,
so no Claude-written trace is used for training.

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
  proofs at matched tokens (10/58 vs 11/58 at k 16). The traces here must carry checked work that the action does not.
- **Design proposed** ([PROPOSAL.md](../../code/experiments/current/thinking_traces_20261005/PROPOSAL.md)):
  - a per-step Markov think block, `state <think> … </think> action <eos>`;
  - the 111 existing tokens plus `<think> </think> <no>`;
  - every think line is a claim the validator checks: goal reductions, looks, candidates, `<no>` dead ends.
  - Traces are written by Claude Opus 5.5 through the Batch API, with structured output, strict validation and
    retries. Programmatic search traces serve as filler and control.
  - Pilot: 8/8 hand-written traces valid, all Lean-checked. This was written by an agent, so it is not a
    production acceptance rate.

## Pipeline (built, CPU-tested, not yet run)

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

## Baseline figure

Dmitry's progress figure for Dan's checkpoints is [figures/fig_lengthbin_baseline.png](figures/fig_lengthbin_baseline.png):
- textbook72 at pass@256, plain sampling, 3 seeds each;
- caption in `figures/fig_lengthbin_baseline.caption.md`.

| | 1–5 | 6–10 | 11–15 | 16+ | train14 |
|---|---|---|---|---|---|
| Pretraining only | 100% | 58% | 29% | 0% | 38% |
| After RL (round 8) | 100% | 92% | 69% | 28% | 67% |

The trace arms will be added to the same figure.
