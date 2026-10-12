# textbook72_distract2: the 72 textbook problems with two unused premises each

Workstream `pretrain-data`. Experiment folder `experiments/pretrain-data/textbook72_distract2/` (proposal, status,
results, run logs). Built 2026-10-12 by `code/pretrain_data/textbook72_distract_release.py`. Status: **run on four
models, one seed each, then paused at Charles's request** (Dan's round-8 seeds 1 and 2 were not run).

## Summary

**Question.** Earlier screens on generator problems found that none of our models writes an accepted proof once a
prompt lists four or more premises, even when the extra premises are not needed (`docs/pretrain-data/pools.md`,
2026-10-09 and 2026-10-10). Is that a property of the generator problems, or of the models? If it is the models, the
same wall should appear on the real textbook problems.

**Test.** Take the sealed 72 textbook problems (Dmitry's 58 `textbook_dev` + 14 `textbook_train`), add exactly two
premises to each that no proof needs, and re-score the same models on the original and the modified set, paired.
Pre-registered prediction (`proposal.md`, committed before the run): if the wall holds, the problems that cross to
four or more premises lose every solve on every model (0 to 3 of them still solved), while the problems that stay at
three or fewer keep 60 to 85 % of their solved count.

**Result.** 71 of the 72 problems took two distractors; 25 of those 71 cross to four or more premises, 46 stay at
two or three. On all four models the number of crossing problems still solved after distraction is **0**:

| model | original: solved ≥ 1 of 32 | distracted: solved ≥ 1 of 32 | crossing to ≥ 4 premises (25): original → distracted | staying at ≤ 3 premises (46): original → distracted |
| --- | --- | --- | --- | --- |
| Robbie `scale-d384-p1200_s0` pretrain, 9.56M | 12 of 72 | 5 of 71 | 5 → **0** | 7 → 5 |
| Robbie `scale-d768-p1200_s0` pretrain, 38.0M | 11 of 72 | 6 of 71 | 3 → **0** | 8 → 6 |
| Dan `stage1_best12_s0_b1200` Stage-1 base, 9.56M | 25 of 71 | 12 of 71 | 8 → **0** | 17 → 12 |
| Dan `la_T1_best12_s0_r8` RL round 8, 9.56M | 49 of 71 | 28 of 71 | 17 → **0** | 32 → 28 |

The RL model loses 21 of its 49 solved textbook problems to two premises it never has to use; all 17 lost on the
crossing side, and only 4 on the staying side. The premise-count wall is a property of the models, not of the
generator problems. Full metrics are in the Results section and in `metrics/`.

## What is in this folder, and why the problems themselves are not

The 72 are the repo's sealed test set and are never committed (`CLAUDE.md`, Data integrity); a copy with two extra
premises contains every original verbatim, so it is not committed either. The folder ships everything else needed to
rebuild the modified set exactly from the sealed files, and checks the result by hash:

| file | what |
| --- | --- |
| `distractors.jsonl` | 71 rows: problem `name`, `slice` (`dev58` / `train14`), the two added premises as formula strings, and the positions they occupy in the modified premise list. No original content. |
| `apply_distractors.py` | standard-library script that inserts the distractors into the sealed prompts and verifies the canonical sha256 in `MANIFEST.json` (`5f91d124…`). Output goes to the gitignored `docs/textbook-problem/textbook72/`. |
| `examples.jsonl` | the same construction applied to four **non-sealed** textbook exercises, for the examples below |
| `metrics/` | the four models' `metrics.jsonl` as written by the evaluators, plus `summary.json` (the tables below, recomputed) |
| `MANIFEST.json` | sha256 of the sealed inputs, of the canonical modified rows, of every file here and of every checkpoint; counts; protocol |

To see the modified problems:

```
bash code/ws3/textbook72/fetch.sh                      # restores the sealed 72 (hash-checked)
uv run python data/pretrain_data/release_2026-10-12_textbook72_distract2/apply_distractors.py
# -> ok: 71 distracted problems rebuilt, canonical sha256 5f91d124…
```

## How the problems were modified

Builder: `code/pretrain_data/textbook72_distract.py` (function `distract_one`), seed 0, the same construction as the
`k12_distract` pools of the 2026-10-09 release. For each problem:

1. Parse the prompt into premises and goal. Find the minimal proof length of the original with the `minlen` search
   (bound = the reference proof length, 10 s); fall back to the reference length on timeout.
2. Draw candidate distractors: random formulas of depth ≤ 2 over the four atoms P Q R S, at most 9 tokens, that are
   not a subformula of any premise or of the goal, and that keep the premise set satisfiable (so they cannot be used
   to derive anything by explosion). Two distinct ones are kept per draw.
3. Insert each at a random position among the original premises.
4. Shortcut check: run `minlen` on the modified problem with bound = minimal length + 1 (10 s). If a proof that short
   exists, the distractors make the problem easier; redraw (up to 6 draws). A timeout counts as "unknown" and the
   problem is kept.

Under the pinned verifier every premise must be introduced by its own `PR` line, so a proof of the modified problem
is the original proof plus two lines; the original reference proof does not verify on the modified prompt (checked
on all four examples below). Counts over the 71 modified problems: shortcut check `no_shortcut` 44, `unknown` 27
(the 10 s search timed out; a shortcut would only make a problem easier, which works against the result); minimal
length found for 45, reference length used for 15, search timed out on 11. Prompts grow from 28.1 to 45.4 tokens on
average. One of the 72 could not take two satisfiable, shortcut-free distractors in six draws and is absent.

Premises per problem, before → after: 0 → 2 for 16 problems, 1 → 3 for 30, 2 → 4 for 22, 3 → 5 for 3.

### Examples

The sealed problems cannot be shown, so these are four exercises from *forall x: Calgary* (Thomas-Bolduc & Zach,
CC BY 4.0; our `candidate_v0` set, audited as not overlapping the sealed 72) put through the same builder. Prompts
are in the model's token format: `THM <premises> SEQ <goal> PRF`, `>` is →, `&` is ∧, `v` is ∨, `~` is ¬.

**0 premises → 2** (forallx-yyc-prooftfl.tex:1240). Distractors `( ( Q & R ) > P )` and `( ( Q > R ) > P )`;
reference proof 11 lines, minimal 11, no shortcut.

```
original:   THM SEQ ( ( ( P > Q ) & ( R > Q ) ) > ( ( P v R ) > Q ) ) PRF
distracted: THM ( ( Q & R ) > P ) , ( ( Q > R ) > P ) SEQ ( ( ( P > Q ) & ( R > Q ) ) > ( ( P v R ) > Q ) ) PRF
```

**1 premise → 3** (forallx-yyc-prooftfl.tex:1251). Distractors `( ( ~ R ) & S )` and `( ( P & S ) > S )`;
reference 11, minimal 11, no shortcut.

```
original:   THM ( P > Q ) SEQ ( ( ~ P ) v Q ) PRF
distracted: THM ( P > Q ) , ( ( ~ R ) & S ) , ( ( P & S ) > S ) SEQ ( ( ~ P ) v Q ) PRF
```

**2 premises → 4** (forallx-yyc-prooftfl.tex:795). Distractors `( S > ( Q > S ) )` and `( P > ( R & Q ) )`, placed
before the original premises; reference 13, minimal 13, no shortcut on the second draw (the first draw's pair was
rejected by the checks).

```
original:   THM ( ( ~ P ) > Q ) , ( P > R ) SEQ ( Q v R ) PRF
distracted: THM ( P > ( R & Q ) ) , ( S > ( Q > S ) ) , ( ( ~ P ) > Q ) , ( P > R ) SEQ ( Q v R ) PRF
```

**3 premises → 5** (forallx-yyc-prooftfl.tex:1466). Distractors `( Q > S )` and `( ( S > Q ) > Q )`, one at each
end; reference 14, minimal 14, no shortcut.

```
original:   THM ( P v Q ) , ( Q v R ) , ( ~ Q ) SEQ ( P & R ) PRF
distracted: THM ( ( S > Q ) > Q ) , ( P v Q ) , ( Q v R ) , ( ~ Q ) , ( Q > S ) SEQ ( P & R ) PRF
```

`examples.jsonl` has the full records, including the reference proofs.

## Models tested

| model | what it is | params | source | sha256 |
| --- | --- | --- | --- | --- |
| `scale-d384-p1200_s0` (pretrain) | Robbie's lean_seq pretrain, 6 layers, d 384, 8 heads, d_ff 1280 (the `p1200` run, pretrain stage); the base used by every pretrain-data screen | 9,557,760 | HF `robbiethompson2018/nd-rl-checkpoints`, snapshot `b55351ac…`, `runs/scale-d384-p1200_s0/pretrain.pt` | `447fab6c…` |
| `scale-d768-p1200_s0` (pretrain) | Robbie's larger lean_seq pretrain, 6 layers, d 768, 12 heads, d_ff 2560 | 37,989,888 | same repo and snapshot, `runs/scale-d768-p1200_s0/pretrain.pt` | `5297ea76…` |
| `stage1_best12_s0_b1200` | Dan's Stage-1 base in his `lean_staten` proof-state format, cap 12, 1,200 s on K12; the point his RL ladder starts from | 9,560,832 | `hf://buckets/dan-pandori/nd-rl`, `best-state/ckpts/bs/stage1_best12_s0_b1200.pt` | `2a2a9407…` |
| `la_T1_best12_s0_r8` | the same network after 8 rounds of Dan's T1 best-state ladder (seed 0); the strongest RL model scored on the 72 (49 of 72 on 2026-10-02) | 9,560,832 | same bucket, `best-state/ckpts/bs/ladder/la_T1_best12_s0_r8.pt` | `a992485e…` |

Full hashes are in `MANIFEST.json`. Dan's checkpoints run through his `nd-takehome` code at commit 0c492902; Robbie's
through the `nd-takehome` fork at 51604b3 plus each run's own `source/` folder.

## Protocol

- k = 32 samples per problem, temperature 1.0, seed 0, for both sets and every model. Original and modified sets are
  scored in the same job, so the comparison is paired per problem.
- Robbie's pretrains: `code/textbook_problem/eval_ckpt.py --sets textbook72,textbook72_distract2 --k 32 --temp 1.0
  --max-new 400`, free sampling in lean_seq tokens (the 2026-10-02 pretrain evaluation's settings).
- Dan's checkpoints: `code/pretrain_data/textbook72_distract_eval_dan.py`, his environment sampler
  (`state_sample.env_generate`, 48 steps, 256 tokens per action, batch 256). This is the pretrain-data screens'
  protocol; the 2026-10-02 evaluation of the same checkpoint gated proofs through Lean, so its figure (49 of 72) is
  reproduced here on the 71 under a different judge.
- Every proof is judged by the pinned `nd_verify` through `code/textbook_problem/strict_verify.py`. A problem counts
  as solved when at least one of its 32 samples verifies. The average solve rate is the mean over problems of
  verified samples / 32, with a 95 % interval over problems (1.96 × SD / √n).
- CPU only (laptop), 5 to 17 minutes per model and set. No paid compute.

## Results

Source files: `metrics/robbie_d384.metrics.jsonl`, `metrics/robbie_d768.metrics.jsonl` (per set; the bins recomputed
from the gitignored per-problem sample dumps), `metrics/dan.metrics.jsonl` (per checkpoint, with the bins);
`metrics/summary.json` holds all of it in one place. n = 71 modified problems (72 originals for Robbie's runs; on the
71 shared problems the originals' counts are the same 12 and 11, average rates 0.1052 and 0.1307). One seed per model.

Average solve rate (mean over problems of verified samples out of 32, ± 95 % interval over problems):

| model | original | distracted | ratio |
| --- | --- | --- | --- |
| Robbie d384 | 0.104 ± 0.064 (n 72) | 0.027 ± 0.030 (n 71) | 0.26 |
| Robbie d768 | 0.129 ± 0.073 (n 72) | 0.030 ± 0.030 (n 71) | 0.23 |
| Dan Stage-1 | 0.124 ± 0.061 (n 71) | 0.040 ± 0.027 (n 71) | 0.32 |
| Dan r8 s0 | 0.378 ± 0.088 (n 71) | 0.140 ± 0.061 (n 71) | 0.37 |

Problems solved at least once in 32, by the original premise count (n per bin 16 / 30 / 22 / 3), original → distracted:

| model | 0 premises (→ 2) | 1 premise (→ 3) | 2 premises (→ 4) | 3 premises (→ 5) |
| --- | --- | --- | --- | --- |
| Robbie d384 | 1 → 0 | 6 → 5 | 5 → 0 | 0 → 0 |
| Robbie d768 | 1 → 0 | 7 → 6 | 3 → 0 | 0 → 0 |
| Dan Stage-1 | 5 → 4 | 12 → 8 | 7 → 0 | 1 → 0 |
| Dan r8 s0 | 9 → 7 | 23 → 21 | 14 → 0 | 3 → 0 |

Problems solved on both the original and the distracted version (Dan's checkpoints, from `metrics/dan.metrics.jsonl`):
Stage-1 3 / 7 / 0 / 0 per bin, r8 7 / 19 / 0 / 0. So the problems that survive distraction are a subset of the
problems solved before; distraction created no new solves.

Parse-failure fraction of Robbie's samples (lean_seq output that does not parse): d384 0.324 → 0.430, d768
0.384 → 0.485. Longer prompts break the output format more often, so part of the pretrains' loss may be formatting.
Dan's sampler does not report a per-set parse-failure rate.

Pre-registered expectation versus outcome, crossing problems still solved (the decisive cell): predicted 0 to 1
(Robbie d384), 0 to 1 (d768), 0 to 2 (Dan Stage-1), 0 to 3 (Dan r8) if the wall holds; observed 0, 0, 0, 0.
Staying problems were expected to keep 60 to 85 % of their solved count; observed 71 %, 75 %, 71 %, 88 %.

## What this means

- The four-premise textbook cell (0 of 7 solved by the RL model in the 2026-10-02 evaluation) is a premise-count
  failure, not a premise-selection failure. Every textbook problem with two or more premises is one unused premise
  away from it.
- K12, the Stage-1 data behind all of these models, has no problem with more than three premises. Together with the
  generator-problem screens (0 of 71 and 0 of 180 four-plus-premise prompts solved, `docs/pretrain-data/pools.md`),
  this points at a data gap rather than a capacity gap, which is the premise-pool hypothesis the workstream queues
  first (`k12_prem4_10` and its fractions in `data/pretrain_data/release_2026-10-09/`).
- Prediction recorded for that run: a ladder on a premise-pool Stage-1 solves ≥ 40 of the 71 modified problems,
  against 28 here, without any other change.

## Caveats

- One seed per model. The paired within-model change (33 solved crossing problems → 0 across four models) is not a
  seed effect, but no between-model comparison is claimed.
- Dan's round-8 seeds 1 and 2 were not run (paused at Charles's request); the resume command is in the experiment's
  `status.md`.
- 27 of the 71 modified problems have "unknown" shortcut status (the 10 s search timed out). A shortcut can only make
  a problem easier, so it cannot produce the observed losses.
- One of the 72 is absent from the modified set; Robbie's original-set figures are over 72, the paired bins over 71.
  It is the one problem that already has four premises. Its premises are inconsistent and its reference proof is
  20 lines; the 10 s minimal-length search timed out, so the builder used 20 as the minimum, and since explosion
  gives much shorter proofs every distracted draw was flagged as a "shortcut" (replay over 18 fresh draws: 16
  shortcut, 2 timeout, 0 accepted). A false rejection by the check, not a problem that cannot take distractors.
  It would have gone from four premises to six, a crossing problem, so its absence does not affect the 0-of-25 result.
- Dan's checkpoints were judged by the pinned verifier, not by his Lean judge; the original-set count (49 of 71)
  matches the Lean-gated 2026-10-02 figure (49 of 72), so the judge did not move the number.
- Per-problem results stay in the gitignored `docs/textbook-problem/textbook72/` (sample dumps, hit counts); nothing
  here identifies a sealed problem beyond its name and slice.

## Attribution

Sealed problems: Dmitry's `textbook_dev` and `textbook_train` (branch `dmitry/cap-comparison-textbook-eval-data`).
Pretrain checkpoints: Robbie Thompson (`robbiethompson2018/nd-rl-checkpoints`). Stage-1 and RL checkpoints and the
`lean_staten` sampler: Dan Pandori (`dan-pandori/nd-rl`, `dan-pandori/nd-takehome`). Example exercises: *forall x:
Calgary*, CC BY 4.0. Distractor construction, evaluators and this release: the pretrain-data workstream (Charles,
with Claude).
