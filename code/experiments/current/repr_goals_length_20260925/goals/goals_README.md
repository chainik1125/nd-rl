---
written_on: 2026-09-25
---

# Goals variant: the prover writes its own goal annotations (ab3/goals)

**Status: done (2026-09-25 05:00 UTC).** Results below. Full tables: report/report.md. Log: STATUS.md.

## Question

The prover is a 4-layer, d=256 GPT (3.2M parameters). It fails at long proofs:

- its attempts are too short: 6-10 lines whatever the required length;
- most failures are ordinary invalid steps around lines 4-5 (a faked conclusion is only 1-12% of failures);
- on long tests, attempts that survive die near the training maximum.

The idea tested here: its text never records what it still has to prove. If the model writes that down as it goes,
does it get better at proofs longer than it was trained on?

- **Training.** Theorems of true length 6-11.
- **Test.** Every length from 6 to 18, plus the matched control set.

## What changes

Only this: while writing a proof the model also writes short notes, `GOAL <formula>`, between proof lines. The
notes are text the model generates itself. At test time nothing is given to it except the usual prompt. The
verifier never sees the notes: they are removed before checking. The reward is still only "the proof checks".

**Where a note is written** (DESIGN.md §1 has the exact rules):

- **After a box is opened** (an assumption), the note says what the box is for: the consequent of the implication
  being proved, `F` for a proof by contradiction, or the target of a case split.
- **When the proof must prove a conjunction `A & B`** and does so by proving both parts, the note names the part
  it is working on. After the first part is done, the next note names the part still owed. This is the case the
  composition diagnosis found broken. Most long test theorems are such conjunctions: 55% at L15, rising to 95% at
  L18.
- **After a side lemma is finished**, the note names the goal the proof returns to.

**Where no note is written:**

- when the next line is forced anyway: closing a box, or combining two finished parts;
- at the start, for the theorem's conclusion, because the prompt already states it.

**What the notes cost.** 13% more proof tokens on the pretraining data, and 25% more on RL-length proofs. The
longest annotated 18-line proof with its prompt is 843 tokens, within the model's 1,024.

**Why this design.**

- **Write only when the goal changes.** Writing the goal after every line would cost more than twice as many
  tokens. Under a fixed compute budget every token costs training rounds, and writing only on a change still
  records every change of what is owed.
- **Nothing the model could not know.** The notes contain no line counts and nothing about the test theorem. A
  note occasionally names a formula that the proof proves later (a side lemma's conclusion). That is the model
  stating its own plan before carrying it out, which it has to choose anyway (DESIGN.md §1.5).

## How it is run

This is the repr/ pipeline, so the comparison with repr's abs baseline is like for like:

- **Pretraining.** The pretraining team's 039 recipe at a fixed 2,911 steps of 256 records, seeds 0, 1 and 2. The
  data are the cap-6 proofs with notes added by the deterministic annotator (`lib/goals.py`). With the same steps
  the variant sees 8.4% more tokens (8.81e7 against 8.13e7).
- **RL.** The arm-C recipe (lengthexp/rl_C.py: targets of length 6-11, fine-tune proofs of at most 11 lines) at the
  same compute budget, RL seeds 101, 102 and 103:
  - samples are verified after removing the notes;
  - the model trains on its own proofs with its own notes, uncorrected;
  - the notes cost compute, so the variant gets fewer rounds;
  - how often its notes match the canonical ones for its own proof is logged, but not rewarded.
- **Evaluation.** The lengthexp evaluation (k = 32, T = 0.8, max_new 1,024), and the same metrics as repr:
  - pass@1 and pass@4 per length;
  - the log-odds drop per extra line past 11;
  - the share of attempts at least `L_true` lines long;
  - the first-error position;
  - in-distribution checks;
  - plus what the notes look like.

Deviations from the baseline recipe, each forced by the longer text (DESIGN.md §4-5):

- the RL sampling cap is 768 tokens instead of 512;
- the held-out greedy check allows 768 new tokens instead of 512;
- one extra vocabulary token (+512 parameters, +0.008% compute per token, charged).

## Commands

From `/scratch/cowifornia/nd-rl/ab3/goals`, with `PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python`:

```
$PY tests/test_goals.py --n 20000     # annotation: hand cases, strip/verdict/determinism/tokenizer on 20k proofs per source
$PY tests/test_rl_goals.py            # RL edits (judging stripped samples, found records, relabelling, mix)
$PY build_data.py                     # annotate + validate all cap-6 / held-out / RL-found proofs -> data/, data/VALIDATION.json
$PY analysis/structure_stats.py       # design statistics: box/goal structure, token overhead per policy
$PY make_rl.py; $PY make_eval.py      # rl_goals.py / evaluate.py from rl_C.py / lengthexp evaluate.py (+ .diff files)
nohup $PY launch.py all --chains 2 > logs/launch.log 2>&1 &   # pretrain -> harness copy -> RL -> evaluation, per seed
$PY launch.py status
uv run --script analyse.py            # -> report/report.md, report/fig_main.png, report/results.json
```

## Layout

| path | contents |
|---|---|
| DESIGN.md | the design: annotation rules, justification, tokenizer, strip, pipeline edits, token and compute impact |
| lib/goals.py, lib/gtokenizer.py | annotator (agenda, display rules), strip, consistency check; abs + GOAL tokenizer |
| data/ | annotated cap-6 train and held-out files, VALIDATION.json |
| pretrain.py | repr/pretrain.py (039 recipe, fixed steps) with the goals edits |
| make_frozen.py, frozen/goals_s*/ | harness copies (ab3/frozen + goals tokenizer, stripped held-out check, annotated retention file) |
| make_rl.py, rl_goals.py, rl_goals.diff | the RL file: lengthexp/rl_C.py plus the goals edits |
| make_eval.py, evaluate.py, evaluate.diff | the evaluator: lengthexp/evaluate.py plus stripping and annotation statistics |
| launch.py | the pipeline driver (two seed chains at a time, every GPU job through the lock) |
| analyse.py, report/ | goals against repr's abs baseline (and arm C for reference) |
| analysis/ | design statistics |
| tests/ | tests and smoke outputs (tests/out/, eval_smoke/: plumbing only, not results) |

## Results

**Conclusion: self-written goal notes do not fix length generalisation.**

- **Past the training length, the notes help only a little.** Just past it they buy a small, consistent gain at
  L12-13. From L14 on both models solve nothing.
- **In distribution the variant loses about 3 points.** It pays for the notes in compute, with about 12% fewer RL
  rounds.
- **The notes break down with length.** On long conjunctions the model plans the split, but after finishing the
  first part it rarely writes down the part it still owes. The note fails to generalise with length just as the
  proof does.

All comparisons are goals against repr's abs baseline: the same 039 pretraining, arm-C RL, test sets and seeds. Each
number is the mean over 3 seeds; intervals are 95% t-intervals over the three seed pairs.

| | goals | abs |
|---|---|---|
| pass@1 at L12 | 11.4% | 9.6% |
| pass@1 at L13 | 2.2% | 0.6% (difference +1.6 points [+0.6, +2.6]) |
| pass@1 at L14-18 | 0.0% (1-2 correct samples in 9,600 at L14 and L15) | 0.0% |
| pass@1 pooled L12-14 | 5.4% | 4.1% (+1.35 points [+0.01, +2.69]; all 3 seed pairs favour goals) |
| log-odds drop per extra line past 11 | -1.94 [-2.69, -1.20] | -3.22 [-5.15, -1.28] |
| pass@1 pooled L6-11 (in distribution) | 44.4% | 47.6% (-3.2 points [-11.3, +4.8]; L7: 65.8% against 74.2%) |
| matched control, pass@1 at L12 / L13 / L14 | 8.2% / 2.3% / 0.6% | 6.8% / 0.8% / 0.0% |
| share of attempts at least `L_true` lines long, L15-18 | 0.0-0.5% | 0.1-0.3% |
| median written lines, L15-18 | 10 | 10 |
| held-out cap-6 greedy after pretraining / after RL | 98.2% / 98.7% | 98.0% / 98.6% |
| RL rounds at the same FLOPs | 51-52 | 58-59 |

- **The slope.** The paired slope difference is +1.28 log-odds per line [-0.32, +2.87]. The exact permutation p =
  0.10, which is the smallest possible with 3 + 3 runs. Only L12 and L13 carry successes, so this slope describes
  the drop from 12 to 13 lines and says nothing about longer lengths.
- **Attempt length.** The notes do not make attempts longer. The per-line failure hazard on the long sets climbs to
  0.7-1.0 by lines 12-13 for both models (report §4b, fig_hazard.png). goals fails slightly more often at lines 5-7
  and slightly less often at 12-13.

**What the model writes** (report §6 and §8b, analysis/mechanism.md):

- **The format is learned.** The notes are well formed (0.02-2% of samples have a malformed group). After
  pretraining, 99.8% of correct greedy proofs carry exactly the canonical notes.
- **Under RL the notes drift but stay truthful.** RL trains on the model's own uncorrected notes, and the share of
  exactly canonical proofs falls to 51-93% by the last round. Yet 85-100% of the notes still name a goal that really
  is pending.
- **The split is planned.** On L15-18 theorems whose conclusion is `A & B`, the model names one of the parts in
  73-79% of attempts. It derives the first part somewhat more often than abs (L16: 23% against 18%; L18: 18% against
  12%).
- **The debt is not recorded.** After deriving the first part, the next note names the part still owed in 63% of
  attempts at L12, but only 28% at L15 and 7% at L18. Otherwise:
  - there is no note before the error: 47-62%;
  - the note names the whole conclusion: 12-24%;
  - either way, a faked ANDI usually follows.
  - Neither model ever derives both parts at L15-18.
- **Faked conclusions.** On this stronger 039 base, 10-19% of L15-18 failures fake the conclusion, for goals and
  abs alike. The old frozen-pretrain arms had 1-6%.

**Caveats.**

- **Three seeds per arm.** The in-distribution loss and the L12-13 gain are each about one seed-pair interval wide.
- **Compute trade.** The variant spends its fixed compute partly on the notes: 25% more tokens per RL proof and
  about 12% fewer rounds. Some of the in-distribution loss may be that, not the notes themselves.
- **Test-set mix.** 38% of L12 and only 2% of L13 test theorems are conjunctions, so the L13 gain is not the
  conjunction split at work.
- **Token position.** The annotated proofs sit at later token positions, which slightly worsens the lengthexp
  length/position confound for this variant.
- **Fewer RL rounds, same FLOPs.** goals made 13% fewer optimizer steps and trained on the model's own noisy notes.
  A version that re-annotates found proofs canonically before fine-tuning would test a different idea (supervised
  bookkeeping) and was not run.

GPU time used: about 3.5 GPU-h, against an estimate of 3.7. Jobs lasted at most 40 min, all through the gpulock
queue.
