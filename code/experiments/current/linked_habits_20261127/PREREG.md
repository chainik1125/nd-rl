---
written_on: 2026-11-27
written_by: agent:claude (orchestrator), for Leon
---

# Pre-registration: linked habits (19 against 4)

Written 2026-11-27 12:12 am PT. At this time no corpus, base or run of this experiment exists; the two build agents
started at 12:10 am and work on CPU. Design and terms: `SPEC.md`. Amendments are appended with a time stamp, each
before the data it changes. One seed (0) throughout: this is an exploratory test and says nothing about seed spread.

## What is being tested

Leon's theory for the Nov 18 result (rewarding one of 4 linked habits moved its linked habit by about 9 points where a
whole-author shift predicted about 47): holding an author is not worth it for 4 habits. Prediction of the theory:
with 19 linked habits the base holds an author without a tag, and RL on one habit moves the others.

Three bases from three corpora that differ only in what is linked: `l19` (all 19 habits follow the author), `l4` (the
Nov 18 corpus plus notes drawn at random), `l0` (nothing linked).

## Stage 1: gate on the bases (before any RL)

Measured with k = 8 samples per theorem, Lean-accepted proofs only, on the measurement sets, points with 95 % bootstrap
intervals over theorems. A-share of notes = A-form notes / (A-form + B-form notes).

- **G0, notes are written.** Untagged, at least half of the premises of accepted proofs carry a note, in each base.
  If not, the note measures have too little to read and the experiment stops for repair.
- **G1, the tag is followed.** Note A-share under tag A minus under tag B, pooled over the 15 classes: at least 50
  points on `l19`. Expected within ±5 on `l4` and `l0`.
- **G2, an author without a tag.** Untagged, proofs with at least two notes: A-share of the later notes when the
  first note is A-form minus when it is B-form. At least 30 points on `l19` with the interval above 0. Expected within
  ±10 on `l4` and `l0`.
- **G3, notes and the old habits (reported, no bar).** Untagged: A-form reiteration share per site, and A-form step
  order share, in proofs whose first note is A-form minus B-form; `l19` − `l0` and `l4` − `l0`.

Decision:
- G2 on `l19` at least 30 with the interval above 0: the RL stage runs.
- G2 on `l19` between 10 and 30 with the interval above 0: the RL stage runs and every report calls the link weak.
- Otherwise: no RL stage. The finding is that 19 linked habits do not make this model hold an author without a tag.
  Pods are deleted and the result is reported.
- G1 failing on `l19` is reported and does not by itself stop the RL stage (the RL prompts carry no tag).

## Stage 2: RL (8 lines, 60 iterations each)

Lines and recipe: `SPEC.md` (the Nov 18 Amendment 3 recipe, guard off, unlikeliness term off, untagged prompts).
Read at iteration 60, or at the last snapshot before a collapse (collapse rule of `../two_authors_20261111/PREREG.md`).
All readings are on untagged samples; the same quantities under tag A and tag B are reported beside them.

For a habit h and corpus c: `D_h(c, bonus)` = A-share of h in the bonus line minus in the `ub0` line of c.

**Preconditions.** No reading of a bonus line unless the rewarded habit rose: `D` of the rewarded habit at least +20
points (reiteration for `styA`, the rewarded note class for `noteA`).

**P-A, the direct 4 against 19 comparison (reward reiteration, read step order; the Nov 18 marker).**
`ΔD19 = D_order(l19, styA) − D_order(l0, styA)` and `ΔD4 = D_order(l4, styA) − D_order(l0, styA)`, on `marker_plain`.
- The theory is supported on this marker if `ΔD19 ≥ +10` with the interval above 0 and `ΔD19 − ΔD4` has its interval
  above 0.
- `ΔD4` is the repeat of Nov 18 (it was +8.7 [4.8, 12.9] at iteration 60).

**P-B, notes under the reiteration reward.** The mean over the 15 note classes of `D_h(l19, styA) − D_h(l0, styA)`:
at least +10 with the interval above 0 means the notes came along. The same for `l4` is the control (expected inside
±10, since its notes are tied to nothing).

**P-C, reward one note habit, read the others.** The mean over the unrewarded note classes of
`D_h(l19, noteA) − D_h(l0, noteA)`: at least +10 with the interval above 0 means the linked notes came along. Also
reported: the number of unrewarded classes that moved toward A by more in `l19` than in `l0` (sign test), the same
mean restricted to the first note of a proof (no earlier note in view, so a move there is a shift of the default
author), and `D` for reiteration and step order.

Readings:

| reading | rule |
|---|---|
| the author moved as a unit | P-C at least +10 (interval above 0) and P-A's `ΔD19` at least +10 (interval above 0) |
| the notes moved together, the old habits did not | P-C at least +10 (interval above 0), `ΔD19` inside ±10 |
| habits edited one by one, as with 4 | P-C's interval inside ±10 and `ΔD19` inside ±10 |
| unresolved | anything else |

**Size against a whole-author shift (reported, no bar).** As on Nov 18: fit the weight of a tag-A-like author to the
rewarded habit's untagged share (the base's tag-A and tag-B profiles as the two components) and give, for each other
habit, the observed move as a share of the move that weight implies.

**What would make a reading unusable.**
- Notes disappear under RL: a class with fewer than 50 notes at a snapshot is not read at that snapshot; if the note
  rate per premise in a line falls under 10 %, its note measures are reported as too thin.
- A line that collapses is read at its last snapshot before the trigger and marked.
- The sets differ between lines only by sampling; contrasts are paired over theorems.

**Orchestrator's priors.** G2 passes on `l19` (at least 30): 65 %. P-C at least +10 with the interval above 0: 45 %.
`ΔD19` at least +10 with the interval above 0: 25 %. `ΔD19 − ΔD4` interval above 0: 20 %. "Habits edited one by one":
30 %.

## Compute and stopping

- Runpod, three RTX 4090 pods (one per corpus; secure cloud, $0.89/h each by the catalog read of 12:13 am PT,
  stock "medium"), about 5 hours each: about $14. Recorded in `LH/pods.tsv`. Estimate's top: $25. No lab GPU.
- Each pod is deleted when its jobs have finished and its results are fetched and md5-checked (Leon's rule of
  2026-11-05: at job end, not at a clock deadline). If the spend reaches $32 (the top plus 30 %), I ask Leon and
  keep the pods until he answers. A pod whose setup fails is deleted at once.
- Only the orchestrator creates and deletes pods. No sealed set goes to a pod.

## Amendment 1 (2026-11-27 1:00 am PT; no base, pair or tuned model exists): version 2

Leon changed the design after the pitch (his words are quoted in `SPEC2.md`): more natural habits, the 15 notes
kept on top, a performance control, a way to keep the habits from vanishing during training, and preference tuning
("this sort of rlhf") instead of a reward bonus. `SPEC2.md` is the contract. Everything above this amendment about
`l19` / `l4` / `l0` and the lines `styA` / `noteA` is replaced by what follows; G0, G1 and the stopping rules keep
their meaning. What I had seen when writing this: version 1's corpus statistics only (no model was trained).

**Bases.** Four corpora, seed 0: `c24` (24 habits linked), `c9` (the nine natural ones linked, notes unlinked),
`c4` (the Nov four linked), `c0` (none). The pretraining step count is scaled so that every base makes as many passes
over its data as the Nov 14 `half` bases; the line cap is raised to fit. Every sampler and judge runs with the step
limit 51 (48 before), for every base.

**Performance control (Leon).** The Nov 14 bases without notes (`authors_half_s0`, `control_half_s0`) are measured
with the same code, sets and limits. Reported for every base, untagged: accepted % of attempts, pass@1, pass@8, mean
lines, share of attempts ending at the step limit. Pre-set flag: a `c` base whose pass@8 on `marker_plain` is more
than 5 points below `control_half_s0` (interval excluding 0) is reported as degraded by the added lines. The flag does
not stop the experiment, because the comparisons are among `c` bases that all carry the same lines.

**Gate.** G2 is read per habit group. Decision rule as before, applied to `c24`'s link between notes: at least 30
points with the interval above 0 runs stage 2; 10 to 30 runs it and calls the link weak; otherwise no stage 2. The
link among the natural habits in `c9` and `c24`, and the links between notes and natural habits, are reported with
no bar.

**Stage 2: a rater who prefers one habit.** Rated habits: reiteration (all four bases) and the `atom` premise note
(`c24`, `c0`). Pairs are the base's own accepted proofs and their twins. Arms: DPO against the frozen base (main);
reward model plus GRPO with the KL anchor to the base (second); narrow fine-tune (third); and the no-preference runs.

Read at **matched movement**: the snapshot at which the rated habit's untagged A-share is closest to 30 points above
the base (and, beside it, the last snapshot). Precondition: the rated habit is at least 20 points above the base at
the snapshot read.

For a habit h, base c and arm: `D_h(c)` = A-share of h in the rated arm minus in that arm's no-preference run, at the
matched snapshot. Sets: `marker_plain` for the natural habits, `note_plain` (or `marker_plain` if no `note_plain` is
built) for the notes. A habit with fewer than 50 sites at a snapshot is not read there.

- **P1, the common markers (4 against 9 against 24), reiteration rater, DPO arm.** `M(c)` = mean of `D_h(c)` over
  step order and conjunct order (linked to reiteration in `c4`, `c9` and `c24`) minus the same mean in `c0`.
  The theory is supported on these markers if `M(c24) ≥ +10` with the interval above 0 and `M(c24) − M(c4)` has its
  interval above 0. `M(c4)` is the repeat of Nov 18 by another method.
- **P2, everything linked, reiteration rater, DPO arm.** `S(c)` = mean of `D_h(c) − D_h(c0)` over the unrated habits
  that are linked in c (2 in `c4`, 7 in `c9`, 22 in `c24`; the route is left out, it is not a per-site choice).
  At least +10 with the interval above 0 in `c24` means the linked habits came along. Also given for the first site
  of a proof only (no earlier evidence in view).
- **P3, note rater, DPO arm.** The same `S` in `c24` over the 14 unrated notes, and over the natural habits.

| reading | rule |
|---|---|
| the author moves as a unit with 24 links and not with 4 | P1 holds, and P2's `S(c24)` at least +10 (interval above 0) |
| linked habits move, but no more with 24 than with 4 | `M(c24)` and `M(c4)` both at least +10 (intervals above 0), their difference's interval includes 0 |
| habits edited one by one, as on Nov 18 | the intervals of `M(c24)` and `S(c24)` inside ±10 |
| unresolved | anything else |

The second and third arms are read with the same quantities and reported beside the DPO arm; where the arms
disagree the report says so and draws no single conclusion. `c9` against `c4` and `c24` against `c9` are reported
as the natural-only and the notes-added steps.

**Vanishing (Leon).** The anchor to the base is part of every arm (DPO's reference, the trainer's KL anchor in the
GRPO arm). Reported for every no-preference run: each habit's A-share and number of sites per proof relative to the
base. If a no-preference run loses more than half of a habit's A-share by the matched snapshot, that habit's `D` is
marked as read under drift.

**Size against a whole-author shift** as in the original text, with the base's tag-A and tag-B profiles.

**Orchestrator's priors.** `c24`'s note link passes (at least 30): 65 %. P2 `S(c24)` at least +10 with the interval
above 0: 40 %. P1 holds: 20 %. "Habits edited one by one": 35 %. A `c` base flagged as degraded: 25 %.

**Compute.** Four RTX 4090 pods (one per corpus; secure cloud, $0.89/h each by the catalog read of 12:13 am PT),
about 5 hours each: about $18. Estimate's top: $30; at $39 (the top plus 30 %) I ask Leon and keep the pods until he
answers. Deletion rule unchanged (at job end, results fetched and md5-checked; a pod whose setup fails is deleted at
once). No lab GPU.

## Amendment 2 (2026-11-27 1:40 am PT; the four corpora exist, no base, pair or tuned model exists): 22 linked habits

What I had seen when writing this: the corpus builder's report, the manifests and `habit_table.json` (corpus statistics
only), and the corpus tests passing when I ran them (101 passed). No model has been trained on these corpora.

**The linked set is 22, not 24.** The builder left two natural habits out of the linked set (`CORPUS2.md`):
- redundant case split: the form without the split leaves no trace in a proof, so its sites cannot be read on a
  model-written proof;
- ex falso style: the reductio form is the signature of the contradiction route (habit 1), so writing it would turn
  ordinary records into route records.
Both stay as in the source in every corpus. The corpus names are kept as labels: `c24` = all 22 linked (7 natural +
15 notes), `c9` = the 7 natural habits linked (route, step order, reiteration, conjunct order, disjunct, both halves,
restate before use), `c4` = the Nov four, `c0` = none. Every report says "22" and "7", not "24" and "9".

**Thin habits.** Disjunct has room in 0.61 % of records and conjunct order in 2.1 %. The 50-site rule stands: where
conjunct order cannot be read at a snapshot, `M` is step order alone and the report says so.

**Counts in P2.** The unrated linked habits under the reiteration rater are 2 in `c4` (step order, conjunct order),
5 in `c9` (those two, disjunct, both halves, restate) and 20 in `c24` (those five and the 15 notes).

**Lengths.** The two authors' proofs differ by 0.06 lines in `c24` and `c9`, by 0.93 lines in `c4` (A longer, because
only A's reiteration is linked there) and by 0.01 in `c0`. Mean lines are reported for every arm and snapshot.

**What counts as evidence of a held author (added before any model exists).** Notes are written first and stay in
view for the rest of the proof. So a habit decided *after* a rated habit's sites can follow it by reading what is on
the page, with no author held. The held-author reading therefore rests on habits decided *before* the rated habit's
first site: under the reiteration rater these are the notes and step order (its site is the first non-note line).
P2's `S` is reported separately for "decided before the first reiteration site" and "decided after it", and the first
row of the readings table ("the author moves as a unit") requires the "before" part to be at least +10 with its
interval above 0. P3 (note rater) is reported, but movement of the later habits there is called "following the visible
note", not a held author; of P3 only the other notes' *first-site* shares bear on a held author.

**Audit.** Before any pod is created, an independent auditing agent reads this design and the build; its findings and
my replies are kept in `discourse/design_audit.md` (Leon, 2026-11-27: "lets get some discourse here on experiment
design").

## Amendment 3 (2026-11-27 2:05 am PT; no base, pair or tuned model exists): changes from the design audit

An independent audit of the design and the corpus side is in `discourse/design_audit.md` (findings F1 to F16, my
replies below them). All findings are accepted. What I had seen when writing this: that audit (corpus statistics and
CPU smoke runs on toy pairs only). The binding changes:

1. **DPO is the decisive arm.** Only in the DPO arm do the rows shared by the two pair members (the notes and step
   order, written before any reiteration site) get no direct gradient, so only there does their movement test a held
   author. The readings table applies to the DPO arm alone. The sentence "where the arms disagree the report ...
   draws no single conclusion" is withdrawn: the other two arms are reported with their own predictions (points 5
   and 6) and do not change the table's reading.
2. **DPO pairs** are subsampled, seeded, to equal numbers of A-sampled and B-sampled pairs (train and held-out, per
   corpus and rater). Counts before and after are reported.
3. **Snapshots and matched movement.** DPO and fine-tune snapshots at steps 25, 50, 100, 150, 200, 300, 400, 600, 800,
   1000, 1500, 2000. Every snapshot gets a cheap probe (untagged, `marker_plain`, k 8: the rated habit's A-share).
   The full measurement runs on the base, the two snapshots that bracket +30 points over the base, and the last one.
   `D` is read at +30 by linear interpolation between the bracketing pair; the no-preference run is read at the same
   steps with the same weights. Beside it: `D` per point of rated movement, and the KL to the base at the read
   snapshots. If +30 is not reached, the last snapshot is read and marked (the +20 precondition stands).
4. **Seeds.** DPO rated and swapped runs with two seeds (0, 1); `D` is the mean, both seeds are shown. Intervals
   resample theorems and exclude base-to-base variance; every report says so. If `M(c24)`, `M(c4)` or `S_before(c24)`
   lies between 0 and +20, a second base (seed 1) is trained for `c24` and `c4` and the DPO arm repeated on it
   before any conclusion is drawn.
5. **Fine-tune arm.** Data: the preferred members of all train pairs, unbalanced (not the balanced file, and not the
   "narrow" file, which the smoke run showed to be nearly empty). Control: the sampled members of the same pairs,
   same steps and seeds. Each file's A-share per habit is reported beside the base's.
6. **GRPO arm.** (a) Before it runs, the reward model is probed on held-out twins of every other flippable habit
   (15 notes, restate, both halves, conjunct order, disjunct). A habit whose A member it prefers in more than 55 %
   or fewer than 45 % of twins is marked reward-driven, and its movement in this arm is not read as spillover.
   (b) Chain prediction for a habit h decided before the rated site: the base's P(h is A | the proof's reiteration
   form) applied to the line's shares of reiteration forms. The reading is observed minus predicted, against the
   anchor-only line treated the same way.
7. **Readout.** `S_before(c)`: mean of `D_h(c) − D_h(c0)` over the 15 notes and step order. `S_after(c)`: the same
   over conjunct order, disjunct, both halves and restate. The first row of the readings table uses `S_before(c24)`
   in place of `S(c24)`. P3 is also given on first sites. First-site notes: pooled over sites (primary); per class
   only for `atom`, `vcc`, `vac`, `>ca`, `~c`, `>cc`, `vca`, `&aa`, `&ac`, `vaa`, `>ac`. Per-class `D` is also grouped
   by whether A's form is the copy, the doubled disjunction, or the longer form.
8. **Gate.** The primary G2 is the link from a proof's first site to later sites of *other* habits (other classes,
   for notes). Case split and ex falso leave the site sequence and the pooled G1; ex falso is treated as part of the
   route habit. `c24 − c0` is printed beside `c24`'s value. The bar (30) is unchanged.
9. **Performance control.** The flag is on pass@1 (accepted share of attempts): more than 5 points lower, interval
   excluding 0. Comparisons: `c0` with `control_half` and `c24` with `authors_half` (described as "equal passes over
   the data, 1.74 times the steps, other pods"), and the clean one: a fifth base `c0n`, which is `c0` with the notes
   stripped, same recipe, equal passes over its data. `c0` against `c0n` is what the notes cost.
10. **Detector.** Per snapshot the readout logs near-misses (unused and-elimination lines away from their sibling,
    unused restatements in boxes, cited header copies, premises with two notes, late notes). A near-miss of the
    both-halves habit counts as "other", not as "no site".
11. **Two different steps.** In `c24` the notes put the author on the page before any reiteration site; in `c9` and
    `c4` nothing but step order does. `c9 − c4` is therefore the step "more linked habits, author not on the page",
    and `c24 − c9` the step "plus notes that show the author". An effect in `c9` with none in `c24` would be read as
    page evidence standing in for a held author, not as a refutation. `c0` is built from another source corpus than
    the other three, so `c24 − c4` and `c9 − c4` are the comparisons free of that difference.

**Compute.** One more base (`c0n`), second seeds and more snapshots add about $3. Estimate about $21, top $30 and the
$39 rule unchanged. A second pair of bases under point 4 would add about $6.

## Amendment 4 (2026-11-27 2:41 am PT; no base, pair or tuned model exists): trims to stay inside the stated budget

The runner's estimate for the plan of Amendment 3 is about 51 pod-hours (about $46), above the top of $30 that Leon
was told. Leon is asleep and left the night's decisions to me; I do not raise a cap in his absence, so the plan is
trimmed. What I had seen: the runner's dry-run time estimates only (no model). Most of the cost is measurement, and the
trims remove measurements that do not bear on the decisive readings (the DPO arm with the reiteration rater):

1. **Probes** run only on rated runs (DPO and fine-tune with the preferred form), in step order, and stop at the first
   snapshot more than 30 points over the base; the last snapshot is always probed. Control runs (swapped DPO,
   fine-tune on sampled members) are not probed: they are measured at the rated run's steps.
2. **Tuned snapshots are measured untagged only.** The tag-A and tag-B conditions are measured on the bases (G1 and
   the whole-author comparison need them) and on the last snapshot of each corpus's seed-0 reiteration DPO run.
3. **The note rater (P3)** keeps its DPO arm with one seed (rated and swapped) on `c24` and `c0`. Its fine-tune arm and
   its GRPO line are dropped. P3 was already secondary, and after Amendment 2 only its first-site reading bears on a
   held author.
4. **GRPO arm:** the reiteration-rated line and the anchor-only line on each corpus, snapshots measured at iterations
   10, 20, 40 and 60, untagged.
5. Unchanged: four bases and `c0n`; the gate; the reiteration rater's DPO arm with two seeds on all four corpora; its
   fine-tune arm with its control; the reward-model probe and chain prediction; the readout of Amendment 3.

**Compute after the trims** (my estimate from the runner's per-task times): about 27 pod-hours, about $24; top $30 and
the $39 rule unchanged. If the gate fails, stage 2 does not run and the cost is about $7.

## Amendment 5 (2026-11-27 3:05 am PT; no base, pair or tuned model exists): earlier snapshot steps

The trainers' CPU smoke (on a Nov 14 base, toy pairs) passed 90 % held-out preference within 18 steps, so the rated
habit may move 30 points before the first snapshot of Amendment 3 (step 25). Snapshots are added at steps 2, 5, 10 and
15; the rest of the grid, the learning rate (1.5e-4, as SPEC2 fixed) and the read rule are unchanged. If the trainer
audit recommends a lower learning rate, that change is recorded here before any tuned model exists.

## Amendment 6 (2026-11-27 3:37 am PT; no base, pair or tuned model exists): the GRPO arm is shortened

The runner's estimate after Amendment 4 is about 34 pod-hours of work (about $33 with waiting), still above the top
of $30; the GRPO arm is the largest item (11.6 pod-hours) and, after the audit, the least decisive one. What I had
seen: time estimates only. Changes:

1. GRPO lines run 40 iterations (not 60), with snapshots measured at iterations 10, 20, 30 and 40, untagged.
2. No GRPO lines on `c9`. The arm runs on `c24`, `c4` and `c0` (the reiteration-rated line and the anchor-only
   line on each), which keeps the 22-against-4 comparison and its unlinked reference.
3. Pods are created in the order of their stage-1 length (`c0`, `c4`, then `c24` and `c9`) so that fewer pod-hours
   are spent waiting at the gate.

Everything else stands. **Compute after this change** (from the runner's per-task times): about 28 pod-hours of work,
about $27 with waiting; top $30 and the $39 rule unchanged.

## Amendment 7 (2026-11-27 3:56 am PT; two bases are pretraining, none is finished; no pair or tuned model exists): changes from the trainer audit

Part 2a of the audit (`discourse/design_audit.md`, findings T1 to T11) found no blocking fault in the trainers and
five things to fix. What I had seen: that audit (CPU smokes on a Nov 14 base with toy pairs) and nothing from the new
bases. Binding changes:

1. **Learning rate and grid (T3, T5).** DPO and fine-tune runs use learning rate 3e-5 (not 1.5e-4: at 1.5e-4 the
   30-point crossing fell inside the 10-step warmup). Snapshots at steps 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150,
   200, 300 and 400; 400 is the last step (not 2,000). Read rule unchanged. This replaces the grids of Amendments 3
   and 5.
2. **Tags on a tuned snapshot (T5).** The tag-A and tag-B measurement of Amendment 4 point 2 is taken on the upper
   bracketing snapshot of each corpus's seed-0 reiteration DPO run, not on its last snapshot.
3. **No-preference DPO control (T4).** Every batch of a swapped run is exactly half swapped within each sampled side,
   with a fixed swapped set; the two control seeds use different orientations (swap seed = run seed).
4. **Fine-tune arm (T8).** The loss is divided by the same fixed constant in the rated run and its control.
5. **Pairs per corpus (T7).** At most 4,000 balanced train pairs per corpus and rater (2,000 per sampled side), drawn
   with a fixed seed, so that every corpus trains on the same number of pairs; a corpus with fewer uses what it has
   and is flagged in the readout.
6. **Reward model in the GRPO arm (T1, T2).** The bonus is paid only on accepted proofs in which the detector finds
   at least one reiteration site; a proof the detector cannot read gets none, and such proofs are counted. The
   reward-model report gains two probe rows from the base's own samples (a proof without a site against one with a
   site, same theorem; two same-form proofs of different length, same theorem). A habit is marked reward-driven when
   the bootstrap interval of its mean bonus difference, 0.3 x [sigmoid(score of A member) - sigmoid(score of B
   member)], lies outside plus or minus 5 % of the rated habit's own mean bonus difference; a habit with fewer than
   100 probe twins is reported as not probed. (This replaces the 45 / 55 % rule of Amendment 3 point 6a.) The reward
   model stays an ordinary whole-proof scorer; the audit's option of scoring only the rows that differ is declined,
   because no real rater works that way. Its measured preferences for length and for proofs that have a site are
   reported as properties of this arm.
7. **A check on the pod (T11).** The first logged training row of every DPO run must show loss 0.6931 within 0.001
   (policy equals reference); a run that does not is stopped and reported.

## Amendment 8 (2026-11-27 4:03 am PT; two bases are pretraining or being measured, no gate has been read, no pair or tuned model exists): changes from the launcher audit

Part 2b of the audit (`discourse/design_audit.md`, findings R1 to R13) found stage 1 sound as it runs, two faults in
the readout that would print a verdict from an invalid read, and that the second-base trigger of Amendment 3 point 4
would fire almost always. What I had seen: that audit (synthetic rows and corpus records) and the operator's note
that pretraining runs at 29 to 40 steps per second; no measurement of any new base. Binding changes:

1. **G0 (R1).** G0 is the share of premises that carry a note, (sites - none) / sites, pooled over classes. The
   decision check applies G0 at least 50 on `c24`, `c9`, `c4` and `c0` before the G2 rule; a base under 50 means
   "stop for repair", as the original text says.
2. **A second base, run in any case (R5).** The trigger of Amendment 3 point 4 is withdrawn. A second base (base seed
   1) is pretrained and measured for `c24` and for `c4` on their pods after their stage 2, and the reiteration
   rater's DPO arm (rated and swapped, trainer seed 0, probes and full measurements as for the first base) is run on
   it. No fine-tune, GRPO or note-rater arm on the second bases. Their gate quantities are reported and gate nothing.
3. **How the two bases enter the readings.** `M(c24)`, `M(c4)` and `S_before(c24)` are computed per base seed
   against the one `c0` base and averaged over the two base seeds (and, on the first base, over its two trainer
   seeds). The readings table is applied to the averages. If the two base seeds give opposite signs for
   `M(c24) - M(c4)` or for `S_before(c24)`, with either value outside plus or minus 5, the affected rows read
   "unresolved (bases disagree)".
4. **Valid reads only (R3, R4).** No row of the readings table is printed as a reading unless, for `c24`, `c4` and
   `c0` and for every run that enters it, the 30-point crossing was reached and bracketed by two fully measured
   snapshots and the precondition holds. A missing bracket measurement gives "not read" with the reason, never the
   last snapshot. Two trainer seeds of opposite sign with either outside plus or minus 5 give "not read: seeds
   disagree".
5. **GRPO read (R7).** The GRPO arm is read at matched movement like the other arms (the snapshot whose
   rated-habit share is closest to 30 points over the base), with iteration 40 beside it; the chain prediction is
   computed at both, with the mixing weight taken from the same site counts as the conditionals.
6. **Sets for tuned snapshots (R13).** Tuned snapshots are measured on `marker_plain` only (every `D` is read
   there); `route_unseen` is measured on the bases.

**Compute.** Pretraining runs about twice as fast as estimated and Amendment 7 cut the tuning runs from 2,000 to 400
steps; the second bases add about 4 pod-hours. My estimate is about 26 pod-hours, about $24 with waiting; top $30 and
the $39 rule unchanged.

## Gate record 1 (2026-11-27 4:56 am PT; gate read for `c24` and `c4` only; `c0`, `c0n` and `c9` not yet measured)

Read with `lh_gate.py` at commit e7b3388 (`LH/gate/gate_v2.txt`, 12:55 UTC), pooled sets, 95 % bootstrap over
theorems:

| base | G0 noted premises | G2 notes given notes (other classes) | G2 natural given natural | pass@1 untagged, marker_plain |
|---|---:|---:|---:|---:|
| `c24_s0` | 99.2 | +97.0 [+94.3, +99.0] | +41.0 [+36.0, +45.8] | 45.7 |
| `c4_s0` | 99.5 | -3.8 [-8.1, +1.0] | +10.7 [+6.4, +14.9] | 44.7 |
| `authors_half_s0` (Nov 14, no notes) | | | | 61.9 |
| `control_half_s0` (Nov 14, no notes) | | | | 63.2 |

**Decision.** `c24`'s primary G2 is at least 30 with its interval above 0: stage 2 runs. **Performance flag fired:**
`c24_s0` is 16.2 points of pass@1 below `authors_half_s0` on `marker_plain` [-18.7, -13.7], and `c4_s0` is about as
low, so the cost does not come from the number of linked habits; `c0` against `c0n` will say how much of it is the
note lines. By the rule written before the run the flag is reported and does not stop stage 2.

**How `go` is given (a clarification written before any `c0` or `c9` gate number was seen).** Amendment 8 point 1
checks G0 on four bases before the G2 rule. G0 is a property of one base, so each pod gets its `go` when its own
base has G0 at least 50 and `c24`'s G2 rule is met; a base under 50 gets no `go` and is stopped for repair. `c24`
and `c4` get `go` now, on the code of commit e7b3388; `c0` and `c9` when they reach the gate. If `c0` failed G0,
no reading that uses `c0` could be made until it was repaired.

## Gate record 2 (2026-11-27 5:16 am PT; all four bases and `c0n` measured; no tuned model measured)

`lh_gate.py` at e7b3388 with every base in prints "Decision check: stage 2 runs". Pooled sets, 95 % bootstrap over
theorems; pass@1 untagged on `marker_plain` (where the flag is read):

| base | G0 noted premises | G2 notes given notes | G2 natural given natural | pass@1 | pass@8 |
|---|---:|---:|---:|---:|---:|
| `c24_s0` | 99.2 | +97.0 [+94.3, +99.0] | +41.0 [+36.0, +45.8] | 45.7 | 81.2 |
| `c9_s0` | 99.5 | +0.2 [-4.0, +4.2] | +16.4 [+11.8, +20.9] | 53.1 | 87.6 |
| `c4_s0` | 99.5 | -3.8 [-8.1, +1.0] | +10.7 [+6.4, +14.9] | 44.7 | 82.4 |
| `c0_s0` | 99.3 | +2.0 [-2.1, +5.9] | +7.0 [+1.7, +12.1] | 53.0 | |
| `c0n_s0` (no note lines) | 1.3 | | -3.4 [-7.2, +0.3] | 58.1 | |
| `authors_half_s0` (Nov 14) | | | | 61.9 | 90.0 |
| `control_half_s0` (Nov 14) | | | | 63.2 | 91.4 |

**Performance flags (pass@1, `marker_plain`), all three fired:** `c0` - `c0n` = -5.1 [-7.9, -2.3] (the cost of the note
lines alone); `c0` - `control_half` = -10.3 [-13.0, -7.5]; `c24` - `authors_half` = -16.2 [-18.7, -13.7]. The new
bases solve fewer theorems than the Nov 14 bases; about 5 points of that is the note lines. `c24` and `c4` are a
further 8 points below `c0` and `c9`, and I have no explanation for that yet (`c9` links the route too, so linking
the route is not it). Reported, stops nothing; every stage-2 reading is on proofs Lean accepts.

**`go` times (UTC), all on e7b3388:** `c4` 12:57, `c24` 12:58, `c0` 13:06, `c9` 13:15.

**Balanced train pairs for the reiteration rater (cap 4,000):** `c4` 2,354, `c24` 2,966; `c0` and `c9` are added to
the read-out when their pair builds end. `c24`'s note rater: 1,900. Both under the cap: the pre-set flag.

## Amendment 9 (2026-11-27 5:31 am PT; all four pods in stage 2; `c9`'s first DPO runs had started one minute earlier, at 13:30:47 UTC; no training row, probe or measurement of any tuned model had been seen)

From part 2d of the audit (`discourse/design_audit.md`, W1 to W7). What I had seen: the gate records above and the
pair counts (balanced train pairs for the reiteration rater: `c0` 2,292, `c4` 2,354, `c24` 2,966; `c9`'s count,
2,968, reached me while I was writing this).

1. **Passes over the pairs at the crossing (W6).** No corpus reached the cap of 4,000, so the corpora do not train on
   equal numbers of pairs. For every rated run the read-out reports the number of passes over its train pairs at the
   30-point crossing (64 x step / pairs), per corpus and base. The pair counts stay as they are: the swapped control
   of each corpus sees the same pairs and batches as its rated run.
2. **A pre-set control, run only if the passes differ widely.** If `c4` or `c0` crosses after more than 2 passes
   while `c24` crosses within 1 (or the reverse), then before any reading of `M` is stated, `c24`'s reiteration DPO
   run and its swapped control are repeated on a subsample of `c24`'s pairs equal in size to the smaller corpus's
   set (about 1 pod-hour), and `M(c24)` is reported for both. Otherwise no such run is made.
3. **Second-base pass.** It runs from commit ad001ea, which the audit found to leave every first-base command and
   read-out number unchanged.
4. **What is kept from a pod.** Results, logs, pair files, probe outputs, bases, each run's final checkpoint, the
   fully measured snapshots, every snapshot of a GRPO line and the reward models are fetched and md5-checked; the
   other snapshots are left on the pod and are lost when it is deleted (they can be regenerated from the seeds). A
   pod is deleted only after that fetch has been checked a second time, independently of the fetch script.

## Amendment 10 (2026-11-27 8:35 am PT; first-base stage 2 finished and read; the second-base passes for `c24` and `c4` are running; the run below does not exist)

**What I had seen.** The first-base read-out (`LH/read/read_firstbase.txt`, code 12fc233) and the first part of the
audit of it (`discourse/design_audit.md`, Part 3). In the reward-model + GRPO arm on `c24` (one RL seed, one base)
the notes moved as a block: at iteration 40 the share of A-form notes before the first reiteration site is 59 % in
the base, 89.5 % in the reward line and 24.3 % in the anchor line. The anchor line (acceptance only, no bonus) moved
by itself (step order A 42 to 13 %, note_atom 67 to 30 %), so the printed difference between the lines (about +62 on
note_atom) is the sum of two opposite drifts; against the base the reward line moved notes by +20 to +40. The
reward model's bonus difference on held-out note twins is between -0.008 and +0.001, against about 0.26 for the
reiteration flip. None of this was predicted in advance in this form, so it is an observation from one seed.

**The run (a replication, nothing else changes).** On the `c24` pod after its second-base pass: the two GRPO lines
again from the first base `c24_s0`, RL seed 1 (`rl_c24_rlhfreit_s1`, `rl_c24_anc_s1`), with the same reward model
(`rm_c24_reit_s0`, not refitted), the same recipe, 40 iterations, snapshots at 10 / 20 / 30 / 40 measured untagged on
`marker_plain`. About 2 GPU-hours, about $1 to $2; the top of $30 is unchanged.

**Quantities.** For each line, the change against the base of the pooled A-share of notes before the first
reiteration site, and of step order, with 95 % bootstrap intervals over theorems; and D(notes) = reward line minus
anchor line; at iteration 40 and at the iteration where the reward line's rated-habit movement is closest to 30.

**Predictions from seed 0, fixed now.** Anchor line: notes at least 20 points toward B. Reward line: notes at least
15 points toward A. D(notes) at least +30. Step order toward B in both lines.

**Reading rule.** The block movement of the notes under on-policy RL on `c24` **replicates** only if both lines move
the notes in those directions with intervals excluding 0, and D(notes) is above +20 with its interval above 0. If the
anchor line does not drift in seed 1, the report says that seed 0's D was enlarged by the anchor line's drift and
leads with the reward line against the base. If the reward line's notes do not move toward A (interval includes 0),
the seed-0 observation is reported as not replicated. Whatever the outcome, both seeds are reported, and no claim is
made about corpora other than `c24` or about a second base: GRPO lines on `c9` (natural links without linked
notes) and on the second `c24` base are not run here and are left to the PI.

**What this does not change.** The decisive arm is DPO (Amendments 3 and 8), and its reading comes from the two bases'
average when the second bases are in.

### Erratum to Amendment 10's "What I had seen" (2026-11-27 9:12 am PT; the seed-1 lines have not started)

Two figures in that paragraph were taken from the audit's first message and do not match the read-out's own
estimator; the predictions and the reading rule above are unchanged. (1) The pooled A-share of notes before the
first reiteration site is 59.5 % in the base, 88.5 % in the reward line and 26.2 % in the anchor line at iteration
40 (+29.0 and -33.3 against the base), not 59 / 89.5 / 24.3. (2) Against the base the reward line moved the note
classes by +10 to +41 per class (the lowest value is a class with fewer than 50 sites), not "+20 to +40". Both were
checked independently by the writer of the summary and by the auditor (`discourse/design_audit.md`, Part 4, Y12).

## Result record: Amendment 10, second RL seed on `c24` (2026-11-27 11:20 am PT)

Both seed-1 lines ran on rp-lhc24 from 9:24 am to 11:15 am PT with code 507c9fd, from the first base
(`stage1_c24_s0.pt` as `--init` and `--kl_anchor_ref`), RL seed 1, measurement seed 0, and the seed-0 reward model
(`rm_c24_reit_s0`, same sha256 as the seed-0 line recorded; no refit, no new probe). The operator checked the two
commands token by token against the seed-0 line's recorded command before the start (only name, output folder and
seed differ). Read-out: `LH/read/read_final.{txt,json,stdout}` (code 507c9fd, B 1,000).

**Observed, iteration 40** (also the iteration closest to +30; rated-habit move +29.8). Pooled A-share of the notes
before the first reiteration site, against the base (59.5 %): reward line +26.5 [+23.6, +29.7]; anchor line
-20.9 [-24.4, -17.5]; D(notes) +47.4 [+44.2, +50.8]. Step order against the base (41.6 %): reward line
-17.6 [-22.6, -12.9]; anchor line -21.7 [-26.5, -16.7].

**Against the predictions.** Anchor line notes at least 20 points toward B: held at the point value, narrowly
(-20.9; the interval reaches -17.5; seed 0 gave -33.3, and over iterations 10/20/30/40 the seed-1 anchor line reads
-27.8 / -13.6 / -23.4 / -20.9 in the operator's digest, so its size is not stable). Reward line notes at least 15
points toward A: held (+26.5). D(notes) at least +30: held (+47.4). Step order toward B in both lines: held.

**Reading by the rule: replicates.** Both lines move the notes in the predicted directions with intervals excluding
0, and D(notes) is above +20 with its interval above 0.

**Scope.** Two RL seeds, one base (`c24_s0`), one reward model (the seed-1 reward line reuses the seed-0 reward
model by design). Not run: GRPO lines on `c9`, which would separate the number of links from the notes being
written on the page before the rated line, and on the second `c24` base. Those stay Leon's decision (Amendment 10).
The interpretation of the block movement (selection among the base's two authors) stays an interpretation
(`discourse/design_audit.md`, Part 3 and Part 4, Y3).
