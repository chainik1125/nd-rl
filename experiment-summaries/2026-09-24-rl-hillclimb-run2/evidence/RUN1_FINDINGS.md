# What run 1 (2026-09-23) learned

Setting: same pretrained 4-layer prover, 600-second experiments, targets of true length 7-9, dev score
on a sealed length-7-9 slice at pass@32 (1,610 theorems). Full record:
`experiment-summaries/2026-09-23-rl-hillclimb-ab/README.md` in the nd-rl repo (copied to
`frozen/ref/run1_summary.md`).

## Scores

Pretrained checkpoint 7 of 1,610. Unmodified seed pipeline (4 rounds, 1,500 targets, k=32) 238.
Best after three hours of search: 401 (GEAR), 339 (orchestration). On sealed length-10 theorems the
best checkpoint solved 15-20 of 907; on length 11-14, at most 1 of 459; on the 58 textbook problems,
nothing beyond the six-line ones the pretrained model already solves.

## What moved the score

- **Many small rounds.** The seed's four big rounds lost to 400-target windows with 135 fine-tune
  steps and 15 rounds in 600 s. At equal sample counts (166k samples, no clock) the small-round recipe
  still beat the big-round recipe 367-374 to 316-320 on dev and 26 to 2 at length 10. This is the
  recipe you start from.
- **A compacting sampler** (drop finished sequences from the batch and KV cache every 16 steps,
  batch 768) cut sampling time 50 s to 30 s per round. It only matters under a clock, but every
  experiment here is under a clock.
- **Hardest-first refill of the window and temperature-1.0 training samples**: within noise of the
  above, kept because they did not hurt.
- **A dedicated exploration stream over schema-instantiated targets** (contraposition, export, De
  Morgan and the like) raised length-7 dev counts from about 45 to 110 in the orchestration arm. The
  gain depends on the first proof of a schema appearing early (331 vs 236 on two seeds of one recipe),
  then fine-tuning spreads it to other instances of the same schema. It did not transfer to the real
  textbook problems.

## What did not help (do not re-test without a written reason)

GRPO pass before the supervised step (minus 56); hindsight relabelling of by-products (about 15 per
run, a no-op); rarity weighting of targets; lighter cap-6 retention (minus 60); persistent AdamW
state across rounds (length-7 collapse); sampling batch 1024 with the library sampler (slower); adding
the 15k extra length-7-8 targets (length 9 fell); learning-rate changes; a length curriculum (length 9
up, length 7 down, net zero); larger rounds (fewer rounds lose); the k=16 wider window (more training
solves, no dev gain).

## Noise

Scoring noise between sampling seeds on one checkpoint: 1 to 15 theorems at the 400 level. Training
noise between seeds of one recipe: 16 to 95. Promotion margins in run 1 (5 theorems) were below noise,
which is why this run has a two-seed rule for BEST.

## Where the frontier stalls

Every solved theorem at length 9 and above in run 1 and in the earlier ladder study belongs to one
proof family (implication towers around AND/OR/IMPE). Theorems whose proof needs a reductio, a case
split with a box, or a tautology sub-goal stay at zero. The length-12-14 frontier set is dominated by
schema instances of exactly those shapes. The first verified proof of a new shape is what unlocks a
bin ("ignition"); everything that makes that first proof appear earlier is worth more than anything
that trains faster on shapes already solved.
