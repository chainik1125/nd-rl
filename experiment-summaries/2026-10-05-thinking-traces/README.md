---
written_on: 2026-10-05
written_by: agent:claude
---

# Thinking traces for the proof-state model (test-time scaling, LLM tradition)

Status: proposal written and pipeline built and tested on CPU (2026-10-05). No training, GPU evaluation or paid
API call has run.

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

## Decisions pending (Leon)

See PROPOSAL.md §8:
1. Approve the format.
2. Allow traces to shorten roundabout reference steps.
3. Staged spend: calibration about $7, then 2,000 proofs ($37-154), then 20,000.
4. Build programmatic traces.
5. Choose the first training run (cold start vs from scratch) and approve GPU time.

Also pending: an OOD calibration eval of about 2 GPU-h.
