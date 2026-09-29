---
written_on: 2026-09-29
written_by: agent:claude
---

Ok I actually like your current ideas a lot. I want you to go implement the dumb ideas here first before we get too fancy. Specific things I want you to try, all on our 'best' model to date (using best pretrain, new RL methods, lean tokenizer)
report per method:
* pass@k on heldout set of final model. Do pass at up to 64
* total time spent on inference, so we can tell wehter our gains come from a massive compute scale up
* num proofs of length >10 pass @k on held out set

sampling methods to try
0) no resampling, what we do rn
1) resample a line whenever it's invalid, try up to 10 times per proof. Report rate of resampling, and num proofs we 'salvaged' when we output a failed line and then got it right (how to define 'line' in the lean regime?
2) beam-search / sequential montecarlo (I want you to define this more rigorously)
3) idk what line-level search is but I think I want this? again define

no expert iteration, no learned value head for now

---

# Test-time scaling: initial plan

Everything below is the agent's plan. Nothing here has been run yet.

## Setup shared by every method

**Model.** The best legal pipeline to date is combined-pipeline run **c005**:

- 186-claude pretraining.
- Leon's GEAR expert-iteration recipe, with verifier-guided RL sampling.
- `lean_seq` tokenizer, 9.56M parameters.
- 3-seed dev metric 904.3.

Its three checkpoints are `runs/combo-c005_s{0,1,2}/final.pt` in the HF repo. See
[the combined-pipeline summary](../../../../experiment-summaries/2026-09-28-combined-pipeline/README.md).
c004 ties it (905.0) but overran the 2,400 s budget. The two differ only in their RL deadline, so
c005 it is.

**Pool.** The holdout half of `transfer.jsonl`, which the dev metric has never scored:

- 1,177 theorems, all with reference length 7-14.
- **114 have reference length > 10**: 50 at 11, 48 at 12, 8 at 13, 8 at 14.

"Length > 10" means the reference length `n_lines` (shortest known proof), the same length the
dev metric bins on. I use the full holdout rather than `holdout250` because `holdout250` has only
about 25 theorems longer than 10 lines, too few to compare methods.

**Tuning.** Any knob a method has (expansion width, length normalisation) is tuned on the dev half,
seed 0 only. Holdout is scored once per final configuration, on all three seeds.

**Sampling constants** unless stated: T = 0.8, max_new = 512 kept tokens, and the harness's Lean ∧
nd_verify gate on every finished proof. These are the frozen harness eval's settings. Note that c005
sampled at T 1.0 during RL.

### What a "line" is in the Lean regime

A `lean_seq` proof is a sequence of Lean tactics separated by `;`:

```
have n1 : F1 := h1 ; have n7 : ( P → R ) := ( fun ( n3 : P ) => by have n4 : Q := n1 n3 ; exact n4 ) ; exact n7 <eos>
```

**A line is one segment: the tokens after the previous accepted `;`, up to and including the next
`;` or `<eos>`.** The segment is parsed with the fork's own `lean_tok.inverse`, which yields the ND
lines it denotes. Each new ND line is checked by Leon's `LineChecker`, which runs nd_verify's rule
checks one line at a time. A segment is **valid** if it parses in the Lean grammar and every ND line
it adds passes.

Most segments are exactly one ND line. The exceptions are boxes:

- **Opening a box.** `have n7 : ( P → R ) := ( fun ( n3 : P ) => by have n4 : Q := n1 n3 ;` is one
  segment carrying two ND lines: the `AS n3` and `n4`.
- **Closing a box.** `exact n4 ) ;` yields the `IMPI` line `n7`.
- **The final segment.** `exact n7 <eos>` is valid only if the proof is complete.

This is the segmentation `guided.py` already uses for c004/c005's RL sampling. The c004 CPU tests
showed 0 false rejects on 23,936 line checks and 0 false accepts on 9,000 mutants. Line validity is
necessary for Lean to accept, but not sufficient, so the Lean gate still runs on every finished
proof.

### Compute accounting (every method)

Each (method, seed) run records:

- **Forward tokens per theorem**, the hardware-independent x-axis. This is decode tokens including
  every discarded segment, plus prefill tokens (the prompt, plus any prefix that search re-encodes).
  FLOPs ≈ 2 × 9.56M × forward tokens; attention is negligible at ≤ 800 tokens.
- **Wall-clock** on one A40, split into:
  - GPU decode;
  - CPU line checks;
  - Lean gate;
  - total.

The first search implementations will be naive, so wall-clock will overstate their cost. Tokens is
the fair comparison and wall-clock is the honest one. Both get reported.

Headline chart: solve rate on the y-axis, forward tokens per theorem on a log x-axis, one line per
method, with a panel for reference length > 10. For i.i.d. methods (0 and 1) the pass@k curve maps
onto the token axis by k × mean tokens per sample.

For search methods, "pass@k" becomes **pass@k-equivalent**: the solve rate at a token budget equal
to k plain samples' mean cost, for k = 1, 2, 4, ... 64.

## Method 0: plain sampling (current eval)

- **Procedure.** `harness.sample_and_judge`, 64 i.i.d. samples per theorem, sampling seed
  `seed * 1000 + 777`. This is exactly `score_holdout.py --pool holdout`.
- **pass@k.** Unbiased estimator (Chen et al., 2021) for k = 1..64 from n = 64 samples.
- **Correctness check.** `--pool dev` must reproduce c005's recorded dev metric (915 / 893 / 905).

## Method 1: resample invalid lines (≤ 10 rejections per proof)

**Procedure.** `guided.generate` from the combined pipeline:

1. When a row finishes a segment, check it.
2. If the segment is invalid, mask its tokens out of the KV cache.
3. Resample the segment from the saved distribution at its start.
4. Treat a premature `<eos>` (proof incomplete or unparseable) the same way.

Settings: `guide_retries = 10` and `guide_rollbacks = 10`. The binding cap is therefore 10 rejections
per sample in total, however they are spread across lines. That is my reading of "up to 10 times per
proof". The 11th rejection abandons the sample and counts it as a failure.

Samples are still i.i.d. across attempts, so pass@k uses the same estimator as method 0.

**Extra reports:**

- **Resampling rate:**
  - rejections per sample;
  - rejected segments ÷ all checked segments;
  - share of samples with ≥ 1 rejection;
  - share abandoned at the cap.

  Each is split by cause: invalid ND line, grammar parse failure, premature `<eos>`.
- **Salvaged proofs.** Samples that pass the final Lean gate *and* had ≥ 1 rejection.
  - A rejected segment is one the verifier would reject, so without the rollback every salvaged
    sample would have failed. The count is therefore an exact per-sample counterfactual.
  - At the theorem level, salvage is fuzzier: another plain sample might have solved it anyway.
    For that level I also report theorems solved at k = 64 by method 1 but not by method 0, taken
    as a 3-seed aggregate.

**Caveat.** c005 was RL-trained with this very sampler (4 retries per segment, 24 per sample), while
its eval used plain sampling. So method 1 matches the distribution the policy was trained under.

## Method 2: line-level beam search and sequential Monte Carlo

Both use one primitive:

- **`extend(prefix, m)`** samples m next segments from the policy given the valid prefix (T = 0.8,
  stopping at `;` or `<eos>`). It checks each segment and returns the valid, deduplicated children
  with their log-probs.
- A child whose segment is a valid final `exact` goes to the Lean gate. If Lean accepts, the theorem
  is solved and the compute spent so far is recorded.

With no value head, only two signals exist: the policy's log-prob, and validity.
**Beam search uses log-prob to optimise. SMC uses validity to sample.**

### 2a: Beam search (width B, expansion E)

1. Start with beam = {empty prefix}.
2. At each depth, `extend` every beam entry with m = E (default 4).
3. Pool the valid children across the beam and drop duplicates.
4. Keep the top B by cumulative log-prob of the whole prefix (summed over tokens, at T = 1).
5. Stop when solved, when the beam empties, or at 512 tokens.

- **Compute knob.** B ∈ {1, 2, 4, ..., 64} at fixed E, one run per B.
- **Expected failure mode.** Ranking by log-prob favours short, common lines and collapses the beam
  onto near-duplicates. Snell et al. (2024) found beam search falls below best-of-N at large budgets
  for this reason. Beam search is deterministic given its samples, so it has no coverage mechanism
  beyond B × E.

### 2b: Sequential Monte Carlo (N particles; a bootstrap particle filter)

1. Start with N particles, all at the empty prefix.
2. Each step, every live particle proposes one segment from the policy.
3. Each particle's incremental weight is 1 if the segment is valid, else 0.
4. Finished particles (valid final segment) go to the Lean gate and leave the pool.
5. Resample the live pool back to N particles, drawn uniformly (systematic resampling) from the
   survivors.
   - The draw is uniform because 0/1 weights make every valid prefix equally weighted.
   - If no particle survives, the theorem fails.

- **Target distribution.** The policy conditioned on every line being valid.
  - Method 1 approximates the same target but is biased toward prefixes where valid continuations
    are rare (Lew et al., 2023; Loula et al., 2025), because retry-until-valid is a *locally*
    normalised sampler.
  - SMC corrects that bias, and it moves dead particles' budget onto live prefixes, rather than
    retrying the dead line where it stands.
- **Compute knob.** N ∈ {1, 2, 4, ..., 64}, one run per N.
- **Expected failure mode.** Degeneracy: after a few resamples, every particle descends from one
  ancestor. Report the number of distinct surviving ancestors at the end.
- **Sanity check.** N = 1 equals method 1 with 0 retries.

## Method 3: line-level best-first search

This is the GPT-f (2020) / BFS-Prover (2025) algorithm, with a segment as the move.

- **Node.** A valid prefix.
- **Frontier.** A priority queue holding every discovered, unexpanded node for the theorem.
- **Score.** log p(prefix) / (lines in prefix)^α.
  - α = 0 is GPT-f's cumulative log-prob.
  - α > 0 is BFS-Prover's length normalisation, which stops the search from preferring shallow nodes
    forever. Tune α ∈ {0, 0.5, 1} on dev.
- **Loop.**
  1. Pop the best node and `extend` it with E = 8.
  2. Push the valid children not seen before.
  3. A child that is a valid final segment goes to the Lean gate.
  4. Stop at the token budget, or when the frontier is empty.
- **Deduplication.** v1 dedups on the exact ND-line sequence. v2, later, uses a canonical state:
  the set of derived formulas with their box context, which merges prefixes that derived the same
  things in a different order.
- **How it differs from beam search.** Beam search advances all B hypotheses in lockstep and discards
  the rest for good. Best-first keeps every discovered prefix and can jump back to any of them.
  That jump is the backtracking.
- **Anytime.** One run per theorem at the maximum budget (64 plain-sample equivalents) records the
  tokens spent at first solve. That gives the whole solve-rate-vs-budget curve from a single run.
- **Sanity check.** E = 1 with the frontier capped at size 1 is plain sampling.

## Implementation order

1. **Script.** `ttscale.py` in this folder. It loads a c005 checkpoint the way `score_holdout.py`
   does (the run's own `pretrain.py` plus `final.pt`), runs one method on one pool, and writes
   per-theorem rows (successes, per-sample tokens, rejections by cause, salvage flags) plus compute
   counters.
2. **Methods 0 and 1.** Correctness check on dev seed 0 first, then holdout on 3 seeds. These reuse
   existing samplers, so they are the quick part.
3. **The `extend` primitive.** Start naive: re-encode the prefix for each expansion, batched across
   theorems, with no KV reuse between tree nodes. CPU unit tests with a tiny random model cover the
   search bookkeeping.
4. **Methods 2a, 2b and 3.** Tune on dev seed 0, then holdout on 3 seeds.
5. **Writeup and summary.**
   - One table per method: pass@k (or pass@k-equivalent) at k = 1..64; > 10-line counts; tokens and
     wall-clock per theorem; method 1's resampling and salvage stats.
   - The headline chart.
   - An experiment-summaries entry.

**Cost guess, unmeasured.** Method 0 is 75k samples per seed, roughly 5-10 A40-minutes including the
Lean gate. Method 1 costs about 1.2-1.5× that, going by c004's RL slowdown. The search methods are
unknown until `extend` exists. Everything together should be a few pod-hours, a few dollars.

## Out of scope for this plan

- Expert iteration and learned value heads (Robbie).
- Temperature sweeps and sampling without replacement are candidates for the next round.
- Test-time training on the evaluation theorems is ruled out (Robbie, 2026-09-29). Holdout is only
  ever scored at the end; see [value_head_plan.md](value_head_plan.md).
- The pretraining loop, its frozen harness, and its data are untouched. Everything here only reads
  c005's checkpoints.
