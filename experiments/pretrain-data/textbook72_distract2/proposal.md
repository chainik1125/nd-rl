# textbook72_distract2: the sealed 72 with two unused premises each, on Robbie's pretrains and Dan's RL models

Pre-registered 2026-10-09 (UTC-7 evening), before any run. Charles: "Copy the original 72 problem textbook set but add
in 2 irrelevant premises for each problem, just like the distractor set. Re-evaluate the latest pretrain model
(probably from Robbie) and then the RL models from Dan on this and see the performance."

## Hypothesis

The matched distractor check on generator problems (`pools.md`, 2026-10-09) found that Robbie's base solves no prompt
with four or more premises, needed or not (0 of 71), while problems that stay at three or fewer premises keep most of
their solve rate (0.74 vs 0.96 at three). If that is a property of the models rather than of generator problems, the
same wall appears on the textbook problems: adding two unused premises should wipe out the solved problems that cross
to four or more premises and only dent the rest.

The 72 have 0 / 1 / 2 / 3 / 4 premises for 16 / 30 / 22 / 3 / 1 problems (aggregate count). With two distractors,
46 stay at two or three premises and 26 go to four or more.

## Sealed-set handling

The 72 are the sealed test set. This experiment uses them for scoring only: the distracted copies, the sample dumps and
any per-problem output live in the gitignored `docs/textbook-problem/textbook72/`; the run folder holds aggregates
only; nothing is reported per problem; no model is trained or selected on anything here. The distracted copy is a
derived problem set and is never used as a seed, template or training record. Recorded in
`docs/pretrain-data/questions.md` 9 with the default followed.

## Construction (`code/pretrain_data/textbook72_distract.py`)

For each of the 72, exactly two distractor premises, built as `scan_pools.py`'s distractors are: random formulas of
depth ≤ 2 and ≤ 9 tokens over the four atoms; not a subformula of any premise or the goal; the premise set with both
distractors stays satisfiable (no explosion shortcut); inserted at random positions among the original premises. The
shortcut check differs because the 72 carry no reference proof locally: `minlen` is run on the distracted problem with
the bound `reference_lines + 1` and a 10 s limit; if it finds a proof shorter than the reference plus the two premise
lines, the distractors are redrawn (up to 6 draws); a timeout counts as "unknown" and the problem is kept. Seed 0.
The number of problems per shortcut status is reported in aggregate.

## Models and protocol

- Robbie's pretrains: `scale-d384-p1200_s0` (9.56M, lean_seq, the screens' base) and `scale-d768-p1200_s0` (38M),
  through `textbook_problem/eval_ckpt.py` as the 2026-10-02 pretrain evaluation ran it: k = 32, temperature 1.0,
  max 400 new tokens, seed 0, pinned verifier.
- Dan's RL models: `la_T1_best12_s{0,1,2}_r8` (best-state cap-12 T1 ladder, round 8) and, if its checkpoint is in his
  bucket, the Stage-1 base `stage1_best12_s0_b1200`, through his environment sampler (`state_sample.env_generate`,
  temperature 1.0, 48 steps, 256 tokens per action) with every proof judged by the pinned verifier, k = 32, seed 0.
  This is the screens' protocol, not the 2026-10-02 evaluation's (which gated through Lean); the original 72 are
  re-run under the same protocol so the comparison is paired.
- Both the original 72 and the distracted 72 are run for every model in the same job.

## Metrics (aggregate only)

Per model: average solve rate (mean over problems of solved samples / 32) and problems solved at least once, on the
original and the distracted set; the paired change; and the same split by the original premise count (0, 1, 2, 3+),
reported as counts per bin. Across Dan's three seeds: mean and range.

## Expected result (written before the run)

| model | original any-of-32 | distracted any-of-32 | of the problems crossing to ≥ 4 premises (26) |
| --- | --- | --- | --- |
| Robbie d384 | 10–14 of 72 | 3–8 | 0–1 solved |
| Robbie d768 | 12–18 | 4–10 | 0–1 |
| Dan r8 (each seed) | 40–50 | 18–30 | 0–3 |
| Dan Stage-1 (if run) | 20–35 | 8–18 | 0–2 |

The decisive cell is the last column. **If it is 0–3 for every model, the premise-count wall holds on the textbook
problems and the premise pool is the first Stage-1 to run.** If the RL models solve 8 or more of the 26, the wall is a
property of the generator problems or of Robbie's base, not of premise count, and the distractor reading moves back
to premise selection. The problems staying at ≤ 3 premises are expected to keep 60–85 % of their solved count.

## Budget and stop rule

CPU only (laptop), about 25 minutes per model for both sets; no paid compute. Stop and record if the distractor builder
cannot place two satisfiable, non-overlapping distractors on a problem (report the count), or if any model's original-72
number differs from its earlier published figure by more than 8 problems under the same protocol (then the protocol,
not the distractors, is what changed; the earlier Dan figures used Lean gating and are not the same protocol).
