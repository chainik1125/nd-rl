# Independent review of the ignition study (2026-09-17)

Method as in the earlier reviews: recompute from raw files with independent code (own
written-form depth counter and reductio predicate; per-proof hit counts from the coverage
files; per-round pattern-theorem counts from `found_1..8.jsonl` with the round taken as the
minimum over raw records of each normalised proof; `nd_verify` re-run on samples).

## What reproduces (everything checked)

- **Pre-RL base hits** for all 21 draws match `artifacts/ign/summary.json` exactly once the
  per-proof `count` field is used (e.g. depth-3 s9: 1,112 hits in 600k samples; reductio s0:
  92 in 3×10⁶). The zero / non-zero split is identical to the summary for every draw.
- **Per-round pattern-theorem counts** for all 10 depth-3 and 11 reductio arms match the
  summary's `per_round` lists element for element; ignition rounds (threshold 20 / 12
  theorems) and final acquisitions match. 0 verifier failures in 100-proof samples per arm.
- **The reductio claim holds as stated**: the four draws with a non-zero pre-RL rate (s0, s3,
  s7, s9) are exactly the four that ignite, and the seven zero-rate draws end at 0/606.
- **The depth-3 claim holds**: draws with rate ≥ 2.5×10⁻⁴ ignite by round 3; of the five
  zero-hit draws, s2, s3 and s8 ignite (rounds 3, 5, 5) and s5, s7 never do.
- **Sibling transfer**: in every one of the 11 intervention arms the round after the injected
  training step shows the pattern on 143–161 (depth-3) or 24–32 (reductio) targets, i.e.
  above threshold immediately, and the arms reach the plateau. Confirmed.

## Refinements

1. **k = 128 and T = 1.0: "no detectable effect" is the more accurate summary.** For the
   two reductio arms where the study says these interventions "ignited" the arm (s3, s9), the
   intervention arms' per-round trajectories are close to the uninterrupted arm's: reductio s3
   with T = 1.0 reaches 1 / 2 / 14 theorems at rounds 6–8 against 1 / 4 / 16 without it; s9
   similarly. Only reductio s3 with k = 128 shows an acceleration (14 theorems at round 6
   vs 1). Depth-3 s3 (a zero-hit draw that self-ignites at round 5) shows 23 / 19 theorems at
   round 5 under k = 128 / T = 1.0 against 28 without — same thing. So the finding is not
   "these interventions ignite arms that would have ignited anyway" but "they change almost
   nothing, in either direction", which supports the study's conclusion more cleanly.
2. **Ignition threshold is a choice.** 20 theorems (2% of 1,000) for depth-3 and 12 (2% of
   606) for reductio. The conclusions do not move for thresholds between 10 and 40, because
   igniting arms jump from single digits to > 100 within one or two rounds.
3. **The 300-target pre-RL sample** (the study's own caveat) under-covers reachable targets
   for some draws: reductio s3 and s9's first RL hits were on targets outside the sampled 300.
   The "exactly those four ignite" statement therefore rests partly on hits in the first RL
   round, not only on the pre-RL sample. A pre-RL sample over all targets at k = 2,000 would
   close this; it costs about 4× the sampling.
4. **What the study did not measure and would settle the depth-3 puzzle**: the pattern's base
   rate on the *intermediate* checkpoints (rounds 1–4) of the zero-hit draws that later ignite
   (s2, s3, s8). If the rate rises from 0 before the first pattern proof appears, ignition is
   the length/box prior drifting under training on depth-≤2 successes; if it stays at 0 until
   a single lucky sample, it is not. This is a two-hour experiment on saved checkpoints.

## Verdict

The study's numbers and its three conclusions reproduce. Wording refinement (1) should be
carried into any summary; (3) and (4) are the next measurements.
