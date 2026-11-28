---
written_on: 2026-11-28
written_by: agent:claude (orchestrator of this measurement, Fable 5.2)
---

# Pre-registration: weight interpolation between the pre-RL Olmo model and the final RL model

Written 2026-11-28 2:56 am PT (`date`), before any pod was requested, any weight was read and any sample existed.
Amendments are appended at the end with their own time; nothing above an amendment is edited after data exist.

Folder: `code/experiments/current/olmo_weight_interp_20261128/` (called W below), branch `leon/olmo-elicitation`.
Data root: `/scratch/cowifornia/nd-rl/olmo_elicit/interp/`. Run steps: `RUNBOOK.md`. Audit: `discourse/audit.md`.

## Why

Leon, 2026-11-28 about 2:55 am PT: "can you also do like the model diffing metric we did historically ... weight
diffing i mean". The historical measurement is the "Weight interpolation" line of the 2026-09-30 benchmark suite
(`experiment-summaries/2026-09-30-benchmarking/README.md`): on the small provers, the model
theta(alpha) = base + alpha x (RL - base) went from 0 % pass@1 at alpha 0 to the RL model's rate at alpha 1, mostly
between alpha 0.5 and 0.75 (sealed set: 0.31 % at 0.5, 12.2 % at 0.75, 19.0 % at 1). The same measurement has not been
made on the Olmo RL run. Metrics 2 (`olmo_metrics2_20261127/`) only extrapolated the early update
(dpo + 3.5 x (s0500 - dpo), called ExPO there) and measured spectra of the weight difference.

## Models, prompts, settings (all as in metrics 2)

- `dpo` = `allenai/Olmo-3-32B-Think-DPO` at commit `97604023f2602a7dfe6d31b1fc229985e0002022`: the model before RL.
- `s2300` = `allenai/Olmo-3.1-32B-Think`, revision `step_2300`, commit `29884259e8c87406883e817e0628fa5ce9b529c3`:
  the last step of the RL run. Both are downloaded from the public hub on the pod and every shard's sha256 is
  compared with the hub's record before use.
- Prompts: the 48 "reach" prompts (`olmo_ifcurve/pids_reach.txt` in Dan's pinned clone, commit `8e744a89`). They are
  the IFBench / IFEval prompts that `dpo` never passed in 768 stored plain draws.
- Sampler: Dan's unedited `olmo_ifladder/sample.py` through metrics 2's `sample_wrap.py`: vLLM 0.11.0, his
  `olmo_thinker` chat template, bf16 weights, temperature 1.0, top_p 1.0, at most 32,768 new tokens per draw, prefix
  caching on, the `dpo` tokenizer files (`/workspace/models/tok`).
- Checker: Dan's strict IFBench / IFEval checker (`olmo_ifomega/grade.py`) through metrics 2's `tools/grade_job.sh`.
  "Pass" below always means "the strict checker passes". Nobody has read these answers yet; a separate blinded
  reader step follows this run and is not part of it.

## Cells

A cell is one engine load that samples 48 reach prompts x 4 draws = 192 draws. "alpha" is the position on the
straight line in weight space: theta(alpha) = dpo + alpha x (s2300 - dpo).

| order | cell | weights served | why |
|---|---|---|---|
| 1 | `a100` (alpha 1.0) | the stored `s2300` checkpoint itself, as downloaded (not a merged copy) | the same-engine yardstick; it must exist even if the run is cut |
| 2 | `a050` (alpha 0.5) | merged checkpoint written on the pod | the cell the reading rule uses |
| 3 | `a075` (alpha 0.75) | merged checkpoint | where the small provers switched |
| 4 | `a025` (alpha 0.25) | merged checkpoint | the low end |

Every cell uses the same pod, the same engine build, the same sampler arguments and the same request layout
(`--pids @inputs/spec_reach4.txt --n 1 --chunk 1`: one request per draw, so a row is written when its draw ends; this
is the layout of metrics 2's ExPO reach cell). The only things that differ between cells are the model directory and
its labels (`--model`, `--name`, `--rev`, `--job`).

Seed: `--seed 710012` in every cell (the engine seed of metrics 2's ExPO reach cell). Dan's sampler has no per-draw
seed; the number seeds the vLLM engine once per load. Equal seeds therefore do not pair draws across cells (the
weights differ, so the token streams diverge at once), and a cell is not reproducible draw for draw. The seed is held
equal only so that no setting other than the weights differs.

No alpha 0 cell is run. The alpha 0 end is `dpo` on prompts chosen because `dpo` never passed them: 0 % by
construction (0 of 768 stored draws), and 0.26 % (1 of 384) in metrics 2's fresh same-stack run of `dpo` with a zero
steering vector. Both are shown as stored points, never as a cell of this run.

## Construction of the merged checkpoints

- Writer: metrics 2's `rank1/expo_write.py` (imported unedited; alpha plays the role of its gamma and `s2300` the role
  of its checkpoint). For every floating-point tensor it computes dpo + alpha x (s2300 - dpo) in fp32 and rounds once to
  bf16 (round to nearest, ties to even). That is all 707 tensors of the checkpoint: the 448 attention and MLP matrices,
  the 257 norm weights, the input embedding and the output head. This is the route of metrics 2's ExPO checkpoints, not
  the adapter route (which left embeddings, output head and norms at `dpo`).
- Checks on every write, by `W/interp_write.py`, before the cell may sample (a failure stops the chain):
  1. Census of the two inputs: how many tensors differ between `dpo` and `s2300`, by class (matrix, norm, embedding,
     output head, other). A tensor that differs and is not floating point, or a tensor present in only one checkpoint,
     fails the write, because the writer would copy `dpo`'s version.
  2. Exactness: every tensor of the written checkpoint (all 707, not a sample) is compared with the exact rational
     result ((q x dpo + p x (s2300 - dpo)) / q with alpha = p / q, in fp64) rounded to bf16, using metrics 2's fixed
     verifier (`expo_write.verify_tensors`); every element must be within 1 bf16 step.
  3. `config.json` of `dpo` and `s2300` may differ only in fields that do not change the computation (listed in
     `interp_write.py`); the merged folder carries `dpo`'s files, the alpha 1.0 cell `s2300`'s own.
  4. sha256 of every written file in a manifest; the manifest and the check report are kept after the checkpoint is
     deleted.
- Rounding (a known limit, measured, not corrected). Dan's weight analysis found that about a third of the weights
  differ between `dpo` and `s2300` by exactly one bf16 step. A served bf16 model cannot hold a quarter of a step: at
  alpha 0.25 such a weight stays at `dpo`, at alpha 0.75 it lands on `s2300`, at alpha 0.5 it goes to whichever
  neighbour has an even last bit. So the merged model is the nearest bf16 model to the point on the line, not the point
  itself. For each merged checkpoint the check report gives the "realised fraction" (the projection of
  theta(alpha) - dpo on s2300 - dpo, divided by the squared length of s2300 - dpo; 0.5 would be exact at alpha 0.5),
  the length ratio, the share of the written change that lies off the line, and the shares of differing weights that
  sit at `dpo`'s value, at `s2300`'s value and strictly between. The reading rule uses the nominal alpha; the realised
  fraction is reported beside every alpha, and if it is more than 0.05 away from the nominal alpha the write-up says so
  in the sentence that states the verdict.

## Quantities

For each cell, from its per-draw rows (192 planned):

1. Strict-checker pass@1: for each prompt the share of its 4 draws that pass, averaged over the 48 prompts, with a
   95 % interval from a bootstrap over prompts (10,000 resamples, seed 20261128, percentiles 2.5 and 97.5). All four
   cells are resampled with the same prompt draws, so every comparison below is paired over prompts.
2. r(alpha) = pass@1 at alpha divided by the pass@1 of the same-engine alpha 1.0 cell, with its paired interval.
3. The half-gain point: on the piecewise-linear curve through (0, 0 %), (0.25, .), (0.5, .), (0.75, .), (1.0, .), the
   smallest alpha at which the rate reaches half of the alpha 1.0 rate, with its paired interval, and a flag when the
   curve is not monotone. The point at alpha 0 is the stored 0 %; if the crossing lies below 0.25 it is reported as
   "below 0.25", because no cell was sampled there.
4. Mean generated tokens per draw; draws cut at the 32,768-token cap; draws without an end-of-thinking tag; draws
   with no row.
5. Answer style of the passes (metrics 2's `answer_style` statistic): the share of passing answers with no sentence
   punctuation (no `.`, `!` or `?` in the text after the last `</think>`) and the median number of words in passing
   answers. The strict checker accepts short unpunctuated word strings, and metrics 2's ExPO passes were mostly that.
6. The same-engine alpha 1.0 rate beside the stored `s2300` rate (69.4 %, 32 draws per prompt, another engine run),
   with the paired difference. This is a description, not a gate. If the same-engine rate is below 50 % or above 90 %
   the write-up leads with that.

Stored reference points shown but never used as the denominator: `dpo` 0 % (0 of 768), `dpo` with a zero steering
vector 0.26 % (1 of 384, metrics 2), stored `s2300` 69.4 %, ExPO at gamma 3.5 70.3 % (4 per prompt, metrics 2).

## Reading rule (fixed now)

With r(0.5) and its 95 % paired interval [lo, hi]:

- **"about linear"** if 0.35 <= r(0.5) <= 0.65 and the interval lies inside 0.2 to 0.8 (lo >= 0.2 and hi <= 0.8);
- **"late switch, as in the small provers"** if r(0.5) < 0.35 and hi < 0.5;
- **"early"** if r(0.5) > 0.65 and lo > 0.5;
- otherwise **"unclear at this size"**.

Cases the rule needs but the sentence above does not cover (added here, before data):

- If the alpha 1.0 cell or the alpha 0.5 cell is incomplete (see "Cut cells"), the rule is evaluated three times: on
  the rows as sampled, with every missing draw counted as a fail, and with every missing draw counted as a pass. The
  verdict stands only if all three agree; otherwise it is "unclear at this size (cell cut)".
- If the alpha 1.0 rate is 0 %, there is no reading.
- The half-gain point, the alpha 0.25 and alpha 0.75 rates and the answer-style numbers are reported beside the
  verdict but do not change it.

What will be said with any verdict: 48 prompts x 4 draws bounds the precision (the alpha 1.0 rate has an interval of
roughly plus or minus 10 points, r(0.5) roughly plus or minus 0.1 to 0.15); this is one RL run of one model; the
prompts were selected for the pre-RL model failing, so its end of the curve is 0 % by selection; passes are checker
passes until the blinded read is done; and a weight-space path says where along the line the behaviour changes, not
that the pre-RL model "contains" the behaviour.

## Cut cells and missing draws

- Each cell's sampler runs under its own timeout of 90 minutes (5,400 s; the outer job timeout is 6,300 s). The
  expected time is 15 to 25 minutes (metrics 2's ExPO cell of 416 draws took 27 minutes). The timeout is a backstop, not
  a budget tool: nothing else stops a cell early.
- A cell is complete only when 192 graded rows exist with exactly the planned (prompt, draw index) keys, the gzip file
  is whole, the sampler's own record shows temperature 1.0, top_p 1.0, 32,768 tokens, seed 710012 and the intended
  model directory, and every prompt's token sha256 equals the one in metrics 2's ExPO cell.
- If a cell is cut anyway (timeout, crash, pod loss), its rows are graded and kept, and it is reported three ways: as
  sampled (passes over finished draws), all-fail and all-pass over every unfinished draw. A cut removes the longest
  draws, so the as-sampled rate is biased and is never reported alone. Missing draws are never dropped silently: the
  analysis prints the count per cell and refuses to call a cell complete without all 192.
- A cut cell is not resumed automatically. Re-running the sampler would fill only the missing draws, which gives long
  draws a second chance to come out short (a retry bias). A re-run is a decision for the coordinator and would be
  reported as such.
- A merged checkpoint is deleted only after its cell is complete and counted. After a cut cell it stays on disk.
- Chain rule: the chain stops if the alpha 1.0 cell is not complete (no yardstick, nothing can be read), if a download
  or a write fails its checks, or if a cell fails for any reason other than its timeout. After a merged cell that was
  cut by its timeout the chain goes on to the next alpha.

## Predictions (written before any data)

Strict-checker pass@1 on the 48 reach prompts, 4 draws each.

| | alpha 0.25 | alpha 0.5 | alpha 0.75 | alpha 1.0 (same engine) |
|---|---|---|---|---|
| orchestrator of this measurement (Fable 5.2) | 1.5 % (range 0 to 4 %) | 9 % (range 4 to 18 %) | 33 % (range 22 to 46 %) | 69 % (range 60 to 78 %) |
| coordinator (main thread), as given in the task | under 3 % | between 5 and 20 % | between 35 and 55 % | not given |

My reasoning: if each prompt's log-odds of passing moved in a straight line along the weight path, from below
0.3 % at `dpo` to about 69 % at `s2300`, the rates would be about 1.4 %, 7 % and 30 %; spread between prompts lifts the
middle values a little. ExPO in metrics 2 is consistent with that picture along the early direction (4.5 % at gamma 1,
70.3 % at gamma 3.5). The rounding described above pulls alpha 0.25 towards `dpo` and alpha 0.75 towards `s2300`.
Predicted verdict: "late switch, as in the small provers" (my probability about 0.7), then "unclear at this size"
(0.2), "about linear" (0.1), "early" (under 0.02). Predicted half-gain point: alpha 0.75 to 0.8.

## What each outcome would and would not show

- "late switch": most of the pass rate appears in the second half of the line, as it did in the small provers. It
  would not show a threshold in training time (the training path is not this line; Dan measured its straightness at
  0.10), and it would not show that the first half of the weight change is useless: it may be necessary and not
  sufficient.
- "about linear": the pass rate grows in rough proportion to the distance travelled. It would not show that each
  half of the change carries half of the "skill": pass rates are bounded and averaged over prompts with very different
  rates.
- "early": half of the change already gives most of the pass rate. With answers unread this could also be a cheap
  answer style that the checker accepts; the answer-style numbers and the later blinded read decide that.
- Any outcome is about checker passes on 48 selected prompts for one RL run.

## Budget and stopping

- One H200 (141 GB, secure cloud, $5.29 per hour). Expected 3 hours including set-up, the two downloads, three
  writes with their checks and four cells: about $16. The coordinator is asked before $21.
- A ledger gate before every cell (`tools/ledger_ok.py` of metrics 2: hours since pod creation x $5.29 + the cell's
  worst case of 90 minutes, $7.94) skips the cell and stops the chain if the total would pass $21; the coordinator
  then decides.
- Nothing sealed (textbook72, test_13_20, test_21_30, the 832 theorem sets) and no credential goes to the pod; the
  push refuses such names. The weights come from the public hub. The coordinator deletes the pod after an md5-checked
  fetch.

## Points raised with the coordinator (not decided by me)

1. Seeds: "the same seeds scheme as the ExPO reach cell" is read as "one engine seed, one request per draw". Dan's
   sampler cannot give per-draw seeds without an edit, and it is not edited.
2. Rounding to bf16 makes the merged model differ from the exact point on the line (section "Construction"). It is
   measured and reported, not corrected; a stochastic-rounding writer would be a different, unverified tool.
3. A cell in which every draw runs to the 32,768-token cap could need more than 90 minutes (192 x 32,768 = 6.3 million
   tokens; the collapsed rank-1 cell of metrics 2 ran at 600 to 800 tokens per second). Such a cell would be cut and
   reported with bounds, as written above.

---

## Amendment 1 (2026-11-28 3:42 am PT, after the audit, before any pod, weight or sample)

Everything above stands as written at 2:56 am except where a point below says "replaces". The audit is in
`discourse/audit.md` with my reply to each finding. No pod exists yet and no draw has been sampled.

**A1.1 Check 2 of "Construction" (replaces its wording).** The exact-rational verifier costs about 45 ns per element
on 4 threads (measured on serrano), so about 24 minutes for the 32.2 billion elements of one checkpoint. Check 2 is
therefore done in two parts:

- on all 707 tensors: the written tensor is re-read from disk and must equal, bit for bit, a fresh fp32
  recomputation; an element whose two inputs are equal must be unchanged; and every written element must lie
  between its two inputs. The last two tests do not depend on the writer's arithmetic.
- on the embedding, the output head, all 257 norm weights and all seven matrices of four layers chosen by a fixed
  seed: the exact-rational verifier, with a stricter criterion than metrics 2 used. "Within one bf16 step" alone
  would pass a tensor whose weights are one step apart and that was written wholly at the wrong end (audit
  finding 3). So a tensor also fails if more than 8 of its elements (or more than 1 in 10 million, if that is more)
  differ from the reference at all. Yesterday's real writes had 0 such elements in 36.7 million.

`interp_write.py --exact all` runs the verifier on everything; it is not in the chain because of its cost.

**A1.2 Added checks.** The token ids in `generation_config.json` of the two checkpoints must agree (the file is not
read by Dan's sampler, which passes its own sampling settings). With the last write, two passes that do not depend on
alpha are made once: the count of bf16 steps between the two inputs per tensor class, and the number of elements at
which the writer's arithmetic at alpha 1 fails to return `s2300` exactly. The second is recorded and flagged above
1 in a million; it is not a failure, because it says nothing against a checkpoint with alpha below 1.

**A1.3 Seeds (replaces the last two sentences of the "Seed" paragraph).** I wrote that equal seeds do not pair draws
across cells. That is wrong. Dan's requests carry no seed, so all 192 draws of a cell take their random numbers from
one generator seeded with 710012, and every cell starts that generator in the same state. Cells are therefore
coupled: the same prompt can open the same way in two cells until the weights make the streams part. The rates and
their intervals are unaffected in expectation, and the bootstrap is over prompts and paired across cells in any
case. Two consequences: the cells are not independent replications of each other, and the blinded readers should be
told that identical openings across cells are expected. Whether the coupling shows on the pod is not known (the
auditor read a later vLLM version than the pinned one).

**A1.4 Cut cells (replaces the first added case under "Reading rule").** Counting every missing draw as a fail in
both cells, or as a pass in both, does not bound a ratio. The three evaluations are: as sampled; the lowest possible
ratio (alpha 0.5 with every missing draw a fail, over alpha 1.0 with every missing draw a pass); and the highest
(the reverse). A prompt with no finished draw is outside the as-sampled rate and inside both bounds; the number of
such prompts is printed per cell.

**A1.5 The smooth reference (new, fixed now from stored numbers only).** The audit's first finding is right: the
label "late switch, as in the small provers" is also what a path with no switch at all would earn. If every
prompt's log-odds of passing moved in a straight line in alpha, from one small value at `dpo` to that prompt's
stored `s2300` rate (32 draws, half a pass added on each side), the rates would be:

| pre-RL rate assumed | alpha 0.25 | alpha 0.5 | alpha 0.75 | alpha 1.0 | r(0.5) | r(0.75) | half-gain point |
|---|---|---|---|---|---|---|---|
| 1 of 384 (the fresh `dpo` run of metrics 2) | 1.7 % | 11.4 % | 40.9 % | 68.8 % | 0.17 | 0.59 | 0.69 |
| half a pass in 769 (the stored 0 of 768) | 0.6 % | 6.2 % | 34.9 % | 68.8 % | 0.09 | 0.51 | 0.75 |

Both rows fall under the rule's "late switch" (r(0.5) below 0.35). The small provers had r(0.5) = 0.016 and
r(0.75) = 0.64 (0.31 %, 12.2 % and 19.0 %), far below this reference at 0.5. So the rule's verdict is reported under
the coordinator's label, unchanged, and a second line is reported beside it with equal weight:

- "sharper than the smooth reference" if the upper end of the interval of r(0.5) is below 0.09;
- "earlier than the smooth reference" if its lower end is above 0.17;
- "compatible with a smooth path" otherwise.

Only "sharper than the smooth reference" supports the words "as in the small provers". The figure shows the band
between the two reference curves. I ask the coordinator whether the verdict label itself should change
(for example to "below linear at the midpoint"); I have not changed it.

My prediction for this line: "compatible with a smooth path" (probability about 0.6), "sharper" (0.3), "earlier"
(0.1). My earlier predictions stand.

**A1.6 How often the rule errs at this size.** Simulated with the stored per-prompt `s2300` rates, 48 prompts x 4
draws, 300 runs per value: a true r(0.5) of 0.30 is called "about linear" in 11 % of runs; 0.35 is a coin flip
between "late switch" and "about linear"; 0.40 is called "late switch" in 15 %; 0.60 is called "early" in 17 %;
0.65 is a coin flip; 0.70 is called "about linear" in 15 %. "Unclear at this size" came back in none of the runs,
because the interval of r(0.5) is only 0.13 to 0.21 wide. The rule is the coordinator's and is not tightened; a
verdict with r(0.5) within 0.05 of 0.35 or 0.65 will be written as "near the boundary of the rule".

**A1.7 Sensitivity line.** r(alpha) is also given with the stored `s2300` rate (32 draws per prompt) as the
denominator. It is not used by the rule. It is there because the 4-draw alpha 1.0 cell carries a good part of the
noise of r.

**A1.8 Rounding.** The half-gain point is also given on the realised axis (each merged checkpoint placed at its
realised fraction instead of its nominal alpha), and the realised fraction is recorded per tensor class.

**A1.9 Schedule, disk and cost (replaces the 6,300 s in "Cut cells" and the expected cost in "Budget").** The
three writes run on the CPU while the cells sample (audit finding 2), in the order 0.5, 0.75, 0.25; each cell waits
for its own write and its checks. The order of the cells, the 90-minute sampler timeout, the checks and the rule
"delete a merged checkpoint only after its cell is complete and counted" are unchanged. Up to three merged
checkpoints can then exist at once (322 GB with the two inputs); the volume is 500 GB and the write step budgets
from file sizes, because `df` on the pod reports the whole storage cluster. A cell's outer job timeout is 9,900 s
(up to 3,600 s of waiting, 5,400 s of sampling, 900 s for grading and counting). The task's original schedule (one
merged checkpoint at a time) is kept as `make_chain.py --mode sequential`. Expected cost: about $8.5 (range $7 to
$12) in the default schedule, about $11.5 in the sequential one. The gate at $21 means a cell can start only in the
first 2 hours 28 minutes of the pod's life; it is called with the pod's own start time, rate and cap.

**A1.10 Labels.** The alpha 1.0 rows carry the model label `s2300_same_engine`, so they cannot be mistaken for
Dan's stored `s2300` rows.

**A1.11 Resume.** The cell script refuses to start when rows of an earlier attempt exist, so the no-resume rule is
enforced by the tool and not only stated.

---

## Amendment 2 (coordinator's decision, 2026-11-28 3:58 am PT, before any pod, weight or sample)

The thresholds of the reading rule are unchanged; the names of its verdicts are. r(0.5) below 0.35 with the interval's upper end below 0.5 is now called 'below proportional at the midpoint'; 0.35 to 0.65 with the interval inside 0.2 to 0.8 is 'about proportional'; above 0.65 with the lower end above 0.5 is 'above proportional'; otherwise 'unclear at this size'. None of these names says anything about a switch. The sentence 'a switch, as in the small provers' may be written only if the result is also 'sharper than the smooth reference' by A1.5's criterion; with 'compatible with a smooth path' the report says that the curve is what a gradual change in each prompt's log-odds would give. The report always prints both lines, and says 'near the boundary' within 0.05 of 0.35 or 0.65 as already recorded. My earlier prediction stands as written (alpha 0.25 under 3 %, alpha 0.5 between 5 and 20 %, alpha 0.75 between 35 and 55 %); by the smooth reference those numbers are what a gradual path gives, so I am predicting 'below proportional, compatible with a smooth path'.

### Orchestrator's notes to Amendment 2 (agent:claude, same time; not part of the coordinator's text)

- The text above is the coordinator's, verbatim. Wherever the note above this amendment uses the old names, read the
  new ones: "late switch, as in the small provers" is 'below proportional at the midpoint', "about linear" is 'about
  proportional', "early" is 'above proportional'. The earlier text is left as written, as a dated record.
- My own predictions of 2:56 am and 3:42 am stand: verdict 'below proportional at the midpoint' (probability about
  0.7), second line 'compatible with a smooth path' (about 0.6). So the coordinator and I predict the same pair.
- How the report applies it (`analyze.py`): the sentence about a switch is written only when the verdict is 'below
  proportional at the midpoint' and the second line is 'sharper than the smooth reference'. With 'earlier than the
  smooth reference' the report says the pass rate rises earlier than a gradual change would give. If the stored rates
  cannot be read, the second line says "not evaluated" and nothing is said about a switch.
- Not part of the measurement, recorded for completeness: the pod is shared with another job afterwards. The merged
  checkpoints are written under this run's own folder (`/workspace/wi/models`), the disk budget counts the whole
  volume, and a last step removes the merged checkpoints and the `s2300` this run downloaded, leaves `dpo` in place and
  writes a "GPU free" marker (`RUNBOOK.md`).
