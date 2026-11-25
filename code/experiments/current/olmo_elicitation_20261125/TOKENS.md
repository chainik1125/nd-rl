---
written_on: 2026-11-25
written_by: agent:claude (EDL build agent)
papers: []
---

# Token-level profile of the distance to s2300 (method note and first readout)

## Summary

This note covers item R3 of `docs/literature/2026-11-25-elicitation-metrics/REVIEW.md`, in the scope the coordinator set
on 2026-11-25.

- **Method.** The step-0 pass of each EDL run stored per-position statistics on `s2300`'s 134 pool responses. `analyze_tokens.py` compares each checkpoint with `s2300` position by position, on CPU only.
- **Readout.** On the five mini-run checkpoints (observed):
  - The L0 gap shrinks with RL step.
  - The gap grows more concentrated in a few positions.
  - The gap sits on positions where `s2300` itself is uncertain (high entropy).
- **Main caveat.** This is the forward direction only: `s2300`'s samples, teacher-forced, which is off-policy for the checkpoint. The reverse direction would need another forward pass.

## Code and outputs

- **Code:** `analyze_tokens.py`, `fig_tokens.py`. The tests are in `test_olmo_edl.py` (`-k tokens`):
  - truncated JS against the full-vocabulary JS;
  - the lower-bound property;
  - top-share arithmetic;
  - an end-to-end run on the tiny model: excess equals the L0 gap from `step0.json`, `s2300` against itself is 0, the decile counts are right and the figure renders.
- **Inputs:** `/scratch/cowifornia/nd-rl/olmo_elicit/runs/<ckpt>/edl/step0_tokens.npz`, written by `olmo_edl.py` on the pods.
- **Outputs:**
  - `/scratch/cowifornia/nd-rl/olmo_elicit/tokens_summary.{json,csv}`;
  - `/scratch/cowifornia/nd-rl/olmo_elicit/figures/tokens_profile.{pdf,svg,png}`.
- **Branch:** `leon/olmo-elicitation` (uncommitted at the time of writing).

## Definitions

All readouts are per position of `s2300`'s responses, by segment (total, thinking, answer) and by pool set. The set "all" pools train, test and reach (563,967 positions; 13,940 of them in answers).

### Divergence

- **JS (approximate).** The Jensen-Shannon divergence in nats between t's and `s2300`'s next-token distributions.
  - It is taken on the union of their two top-20 supports. Each distribution is truncated to its own top 20 and renormalised.
  - This is an approximation, not a bound.
  - Top-20 mass is 0.997–0.998 on average for t and 0.999 for `s2300`. 0.2–0.4 % of positions have under 0.9 of t's mass in the top 20.
  - In a synthetic test, the truncation error stayed below the discarded mass. That is an empirical check, not a proof.
- **JS lower bound.** The exact JS of the distributions coarse-grained to {each id in both top-20 lists} + {rest}.
  - Every probability in that partition is known.
  - Coarse-graining cannot increase JS, so this is a lower bound, up to the fp16 storage of the log-probabilities.
  - It stays within 0.001 of the approximation at every checkpoint.

### Ranks

- **Rank shares.** The share of positions where `s2300`'s sampled token is top-1, in the top 5, or outside the top 20 under t.
  - `s2300`'s own rates are the ceiling: 83.4 % top-1 and 98.4 % top-5 overall, because its samples are not greedy.
- **Argmax agreement.** The share of positions where t's argmax equals `s2300`'s argmax.
- **`s2300`'s argmax in t's top 5.** The share of positions where `s2300`'s argmax is among t's top five tokens.

### Excess

- **Excess.** e = NLL_t − NLL_s2300 at each position. Its token mean is the L0 gap.
- **Concentration.** The share of the net excess carried by the top 1 / 5 / 10 % of positions, ranked by e.
  - This share exceeds 1 when the other positions sum to a negative excess.
  - The share of the positive part, Σ max(e, 0), and the share at the top positions ranked by JS are reported beside it.
- **Entropy deciles.** Ten equal-count bins of `s2300`'s entropy at the position. For each bin: the mean excess and its share of the net and of the positive excess.

### Intervals, trends and reading rule

- **Intervals.** 95 % bootstrap over responses, B = 1000, for the ratio readouts on "all".
- **Trends.** Spearman with RL step over the four non-reference checkpoints.
- **Check.** The per-response NLL sums in each npz equal `step0.json`, with a maximum difference of 1.2e-11 nats.

The reading rule of review R3 is descriptive, not pre-registered:

- **Rule.** "Reranking" if at least 99 % of `s2300`'s tokens are in t's top 5 and the top 5 % of positions carry at least 50 % of the gap.
- **Problem with the 99 % threshold for the total segment.** It is unattainable there because `s2300`'s own sampled tokens reach only 98.4 %. The argmax readouts avoid this sampling floor.

## First readout (observed; mini run, all 134 responses, reference `s2300`)

| Checkpoint | Segment | Excess, nats/token [95 % CI] | JS > 0.1 | Argmax agree | `s2300` argmax in t's top 5 | Sampled token top-1 (s2300 self) | Top 5 % of positions carry (net / positive) | Top entropy decile carries |
|---|---|---|---|---|---|---|---|---|
| dpo | total | 0.117 [0.112, 0.122] | 7.1 % | 88.1 % | 99.56 % | 80.1 % (83.4 %) | 69 % / 45 % | 29 % |
| s0100 | total | 0.109 [0.105, 0.113] | 6.3 % | 88.5 % | 99.62 % | 80.3 % | 69 % / 44 % | 28 % |
| s0500 | total | 0.081 [0.078, 0.084] | 3.9 % | 89.9 % | 99.78 % | 81.0 % | 78 % / 46 % | 30 % |
| s1100 | total | 0.040 [0.038, 0.041] | 0.7 % | 92.7 % | 99.95 % | 82.1 % | 110 % / 51 % | 31 % |
| dpo | answer | 0.157 [0.125, 0.198] | 6.7 % | 95.0 % | 99.34 % | 93.5 % (96.9 %) | 89 % / 82 % | 44 % |
| s0100 | answer | 0.126 [0.098, 0.159] | 5.7 % | 95.6 % | 99.58 % | 94.0 % | 91 % / 82 % | 47 % |
| s0500 | answer | 0.059 [0.046, 0.076] | 2.8 % | 97.0 % | 99.84 % | 95.4 % | 104 % / 86 % | 60 % |
| s1100 | answer | 0.013 [0.009, 0.017] | 0.5 % | 98.6 % | 99.99 % | 96.5 % | 160 % / 92 % | 79 % |

### Observed patterns

- **Monotone trends.** Across dpo → s1100, every distance readout falls monotonically with RL step (Spearman ±1 over four points) in all three segments: mean JS, the share with JS > 0.1, the excess, and the rank shares.
- **The thinking segment carries the total.** Thinking is 97.5 % of the positions, so the total and thinking rows are nearly identical.
- **Concentration rises with RL step.** For example, the top 5 % of positions carry 69 % of the net excess at dpo and 110 % at s1100.
- **The gap sits on high-entropy positions.** The bottom half of `s2300`'s entropy positions carries 4–9 % of the net excess in total. In the answer, the top decile carries 44–79 %.
- **Tokens outside the top 20 are rare.** `s2300`'s sampled token is outside t's top 20 at 0.14–0.44 % of positions, against 0.12–0.13 % for `s2300` itself.

### Interpretation (not a result)

- Even at dpo, `s2300`'s argmax is in the checkpoint's top 5 at 99.3–99.6 % of positions. The excess concentrates on positions where `s2300` itself is uncertain.
- This is the pattern review R3 calls token-level reranking.
- The concentration shares above 1 at later checkpoints mean those checkpoints are more confident than `s2300` on many low-excess positions, while the remaining gap sits in a few.
- None of this says the full responses are reachable by sampling. Per-token support is not sequence-level reachability (review R3, failure modes).
