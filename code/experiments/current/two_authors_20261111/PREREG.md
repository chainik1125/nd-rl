---
written_on: 2026-11-11
written_by: agent:claude (orchestrator), for Leon
---

# Pre-registration: two authors

Frozen 2026-11-11, 1:30 am PT, before any base of this experiment exists. Design and build are in
`SPEC.md`, `CORPUS.md` and `RUNNER.md`. Leon asked for the experiment; he has not reviewed this
file, and the thresholds and predictions are the orchestrator's.

## What the corpus build changed in the plan (known before freezing)

- **Step order is the primary marker.** It is realised at 100 % of A's sites with room and 0 % of
  B's, costs no lines, and is carried by the state.
- **Reiteration is a secondary marker**, read only on proofs whose fully reiterated form fits in
  12 lines (`reit_fits`), because the cap leaves A in its form at about 55 % of sites overall.
- **Conjunct order is reported and carries no weight** (free sites in 2.4 % of records).
- Length signals the author (A's proofs average 7.65 lines, B's 6.81). Every comparison below is
  against the control base, which has the same habit rates and lengths without the bundle.

## Stage 1: the gate (bases only)

Bases: `authors_half`, `control_half`, `authors_p10`, `control_p10`, seed 0, Dan's recipe and step
count as in the tag experiment. Each is measured with `authors_measure.py run` (k 32, tags A, B and
none, sets `marker_plain`, `route_unseen`, `marker_heldout_class`) and read with `authors_gate.py`.
Intervals are 95 % bootstrap over theorems.

| | quantity (authors base) | passes if |
|---|---|---|
| G1 | step-order A-form share on `marker_plain`, tag A minus tag B | ≥ 50 points |
| G2 | untagged, proofs with room for two or more habits: share all-A or all-B, minus what independent habits at the same rates would give; and that excess minus the control base's | excess ≥ 10 points, and the difference from the control has an interval above 0 |
| G3 | untagged, theorems with both route and non-route accepted proofs (pooled over the three sets): step-order A-form share in route proofs minus in non-route proofs, authors base minus control base | ≥ 10 points with the interval above 0; **no reading** below 20 such theorems |

- **Which version goes to RL:** the weaker tie, `half`, if it passes G1, G2 and G3; otherwise `p10`
  if it passes all three.
- **Neither passes G2:** stop. The finding is that these models do not hold an author without a
  tag (they see the proof state, not the history).
- **G2 passes but G3 fails or has no reading on both:** the habits bundle but the route is not
  tied to the bundle. No RL is launched on this rule; the result goes to Leon with the numbers.
- The chosen version gets seeds 1 and 2 (authors and control), measured the same way. The RL
  stage needs G1 and G2 to hold on at least 2 of the 3 authors seeds.

## Stage 2: RL (after the gate, on Runpod, with Leon's OK on the spend)

Six lines: the chosen authors bases and their control bases, seeds 0 to 2. The knockout's `dd1`
GRPO recipe on **untagged** prompts on Dan's RL pool, to x 1.5 or 7 GPU-h, with the
reached-and-held stop off so every line runs to the same compute. At the base and at every
snapshot: Dan's measure on the strict 106, and `authors_measure` (k 8, tags A, B and none) on
`marker_plain` and `route_unseen`. `marker_heldout_class` is read at the base and at the end.

**Precondition.** A line "has the route" from the first snapshot where Dan's measure crosses his
bar of 14. A seed is read only if its authors line has the route by x 1.5. If fewer than 2 of 3
authors lines do, there is no reading: RL did not move the rewarded habit.

**Primary measure.** D = change from base to the end of the line in the untagged step-order
A-form share on `marker_plain` (theorems that need no reductio), in points. Per seed,
ΔD = D(authors line) − D(control line), with a bootstrap interval over theorems. The same is
reported at the snapshot where the authors line first has the route.

| reading | rule, on at least 2 of 3 seeds |
|---|---|
| RL shifted the default author | ΔD ≥ +10 points with the interval above 0 |
| RL learned a router | not the above, and inside route proofs on `route_unseen` the step-order A-form share at the end is ≥ 10 points higher in the authors line than in the control line, interval above 0 |
| RL edited the one habit | neither of the above, with ΔD's interval inside ±10 points |
| unresolved | anything else, or the seeds split |

**Secondary measures** (reported, no reading rule):
- the same for reiteration on `reit_fits` proofs, where the control line shows RL's own drift on a
  habit that costs lines;
- the same under tag A and under tag B: does untagged RL move what the tags produce;
- `marker_heldout_class` (theorems mentioning ⊥, a class absent from the RL pool): ΔD there;
- observed against implied: with the base's route rate and marker share under tag A and under
  tag B as the two authors, the mixture weight that reproduces the line's untagged route rate
  implies a marker share; report observed over implied at every snapshot;
- the share of accepted proofs in a combination no author writes (mixed bundle class) over x;
- route use counted with the detector that follows reiterated lines (`route`), next to Dan's
  `knock`, since a model trained on author A can write a reiterated double negation that `knock`
  misreads as the route.

## What would make the result uninterpretable

- The control lines' markers move with the route as much as the authors lines' (then the effect
  is not about authors).
- The authors and control bases differ in ordinary proving by more than 5 points of untagged
  accepted share on `marker_plain` at the base.
- A line that stops on the GPU-hour cap before x 1.0 is reported and not read.

## Predictions (the orchestrator's priors)

| | prediction | probability |
|---|---|---|
| P1 | G1 passes on both versions | 95 % |
| P2 | G2 passes at `half` | 75 % |
| P3 | G3 passes at `half` | 35 % |
| P4 | G3 passes at `p10` | 65 % |
| P5 | given RL is read: default shift | 25 % |
| P6 | given RL is read: router | 35 % |
| P7 | given RL is read: one habit alone | 30 % |

## Compute

Runpod A40 secure pods at $0.59/h, one job per pod, at most 4 at once, $40 in total, no new
launches past $36; each pod deleted when its results are back on serrano with matching checksums.
Nothing sealed goes to a pod; of `transfer_track832.jsonl` only the 106 strict rows.
