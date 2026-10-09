# Pretrain-data release 2026-10-09: candidate Stage-1 pools for the natural-deduction prover

Workstream `pretrain-data` (docs in `docs/pretrain-data/`, builders in `code/pretrain_data/`). Status at release:
**50 candidate pools built and screened without training; no pool has been trained on; no GPU spent.** This release
publishes the pools the screens single out as worth a Stage-1 run, with the screen numbers, the audit, and the
build recipes. It is a snapshot for the workstream to return to, not a result.

## What the pools are for

The RL model trained on Dan's K12 set fails the textbook problems in a pattern, and every failure lines up with
something K12 never shows: no problem has more than three premises, no box is deeper than two, no proof is longer than
twelve lines, and half of the long proofs are padded with manufactured case splits. The hypothesis is that these are
data gaps, not capacity gaps: a Stage-1 set that closes them should let the same RL recipe, unchanged, solve more real
textbook problems. The network, the RL ladder, the tokenizer and the verifier stay fixed; only the data varies.

## Format

Every pool is **K12 with a slice replaced**: K12's 155,000 records in K12's order, with 1 %, 10 % or 30 % of the rows
outside the first 1,024 (the pinned trainer's held-out validation slice) swapped for new records. The release holds
only the new records, each with `pos`, the row index it occupies, under `slices/`; `assemble.py` rebuilds each full
pool from K12 and checks the result against `MANIFEST.json`. K12 itself is Dan's and is not included: fetch it from his
bucket (`MANIFEST.json` gives the path and the sha256).

```
hf cp hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz train_k12.jsonl.gz
uv run python assemble.py --k12 train_k12.jsonl.gz --out pools --verify
```

Records are in K12's format (`prompt`, `proof`, `thm`, `key`, `n_lines`, `n_prem`, `rules`, `pool`) plus the fields
each builder adds (`slice`, `base_prompt`, `n_distract`, `shortcut_check`, `perm_of`, `family`, `joins`, `subst`,
`origin`). `key` is the curry-aware renaming class. Every new proof was verified by the pinned `nd_verify` through
`strict_verify.py`. `k12_minimal` is published as proofs by position (`slices/k12_minimal.proofs.jsonl.gz`): the
prompts are K12's and come back from the K12 file at assembly.

## The pools

| pool | what replaces the slice | slice | why it is here |
| --- | --- | --- | --- |
| `k12_prem4_10`, `_1e2`, `_1e3` | generator problems with 4–5 premises (the pinned generator's `Gen` class with `max_prem 5`; its CLI never asks for more than 3), 10 % / 1 % / 0.1 %, nested | 15,500 / 155 / 15 | the cleanest gap found: K12 and every earlier pool cap at 3 premises; the RL model fails all 7 four-premise textbook problems; the base solves none of these at 32 tries, at any fraction |
| `k12_distract10`, `_30`, `k12_minimal_distract10` | K12 problems with 1–2 unused premises added, shortcut-checked by `minlen` | 15,500 / 46,500 / 15,500 | the one slice eight RL rounds do not lift (0.63 → 0.69 on Dan's checkpoints, every other slice +0.2 to +0.4) |
| `k12_joins1`, `k12_joins10`, `k12_minimal_joins10` | problems whose minimal proof needs a rule join the hard textbook proofs use and K12 lacks (→E after ∧E, ¬E after ¬I, nested ∨E …); `slices/join_problems.jsonl.gz` is the source | 1,550 / 15,500 / 15,500 | the only pools in the relevant-and-hard quadrant on every screen (base pass@32 0.14–0.19, coverage 0.74–0.76) |
| `k12_jdist10` | join problems with 1–2 distractors | 15,500 | the two hard axes multiply: base pass@32 0.05 |
| `k12_jsubst10` | join problems with atoms replaced by depth-1 formulas | 15,500 | the join skeletons at textbook formula width |
| `k12_long16_10` | generator long mode at 13–16 lines, box depth ≤ 3, at least one hard join present | 15,500 | the only pool covering the 13+-line cells (coverage 0.77, the best of any pool); its problems are mostly solvable short |
| `k12_dial_1e1` … `_1e4` | ↔-elimination problems, a pattern K12 never shows, at 10⁻¹ … 10⁻⁴ of the pool, nested | 15,500 / 1,550 / 155 / 16 | the exposure dial: how much of a pattern Stage-1 needs for RL to ignite it |
| `k12_minimal` | every K12 proof replaced by a minimal proof where `minlen` found a shorter one (84,700 of 153,976) | 84,700 | the padding-free K12; Dan's base gives these proofs twice its K12 loss, so it has learned the padded shape |

Not released: the redundant pools (substitution, the size ladder), the far pool (long easy chains), the controls
(wrong-proof pools), the four pools that reproduce K12 or teach padding (nested ∨E, long reductio, reused lines,
nested conditional goals), the model-mined pool, the atom-count pools and the seven 2026-10-07 idea pools (not yet
screened on Dan's checkpoints). All exist under `data/pretrain_data/pools/` in the workstream and are described in
`docs/pretrain-data/pools.md`.

## Audit

Every new record is one per curry-aware renaming class and not a class of K12; every pool was audited against every
evaluation set and the sealed 72 textbook problems by renaming class (`pool_common.EvalIndex`): twins of the sealed
72 and of test-split problems were removed, twins of seed-split problems kept and tagged `seed_set_twin`. Sources short
of a slice were filled with atom permutations of their own records (`perm_of`), which the rules allow for training data.
Counts per pool are in `docs/pretrain-data/pools.md`. No textbook problem was used as a seed, template or filter.

## Screens (no training)

`screens.json` holds, per published pool: need-weighted coverage of the cells the unsolved seed-split textbook
problems use (K12 0.51); loss ratio and pass@1 / pass@32 of the slice on Robbie's frozen lean_seq base
(`scale-d384-p1200_s0`, 160 prompts × 32 samples, pinned verifier); and, where run, the same on Dan's `lean_staten`
Stage-1 base and after eight T1 ladder rounds (128 prompts × 32). One checkpoint per screen, no seeds.

| pool | coverage | Robbie base pass@32 | Dan Stage-1 pass@32 | Dan round-8 pass@32 |
| --- | --- | --- | --- | --- |
| K12 training rows | 0.51 | 0.89 | 0.98 | 1.00 |
| `k12_prem4_10` | 0.69 | 0.00 | — | — |
| `k12_distract10` | 0.69 | 0.55 | 0.63 | 0.69 |
| `k12_joins10` | 0.76 | 0.19 | 0.36 | 0.69 |
| `k12_minimal_joins10` | 0.74 | 0.14 | 0.30 | 0.55 |
| `k12_jdist10` | 0.70 | 0.05 | — | — |
| `k12_jsubst10` | 0.74 | 0.17 | — | — |
| `k12_long16_10` | 0.77 | 0.69 | — | — |
| `k12_dial_1e1` | 0.51 | 0.26 | 0.35 | 0.77 |
| `k12_minimal` (K12's problems) | 0.49 | 0.88 | 0.98 | 0.99 |

`matched_distract.json`: 200 distracted problems against their own originals on Robbie's base. Every solved distracted
problem has three or fewer premises in total; prompts with four or five are 0 of 71 even when the extra premises are
unused, while their originals are solved at 0.83. The wall is the prompt's premise count, not premise use.

## Current progress and what comes next

Done: the gap analysis of the hard textbook problems; 50 pools; three screens on all of them (data-only screens on
every pool, Robbie's base on 37, Dan's two checkpoints on 10); the pool ranking above. Not done: any Stage-1 run, any
ladder, the Dan-checkpoint re-screen of the 2026-10-09 pools (his worktree has to be re-created first), and a
minimal-proof pass over the premise and long pools, whose proofs are 99 % ∨E case splits (the generator's long-mode
style) even though their problems carry the new difficulty.

Hypotheses queued, in the order they would be run (details and predictions in the status doc and
`docs/pretrain-data/plan.md`): (1) exposure, not volume: the three premise fractions; (2) premises are a data gap, not
a capacity gap; (3) premise selection has to come from pretraining because RL does not find it; (4) proof style is a
separate lever from problem content; (5) hard axes compound. Each is a Stage-1 run plus the unchanged ladder, read per
cell on the textbook seed split and in aggregate on the sealed 72; one seed to screen, three before any claim.

## Caveats

- Nothing here has been trained on. Every number is a screen on a frozen checkpoint, one checkpoint and no seeds.
- Dan's noise-floor study puts nearly all between-run variance in the training run, not the data; the plan gates every
  claim on three seeds and a gain of at least 8 test-split problems beyond twice the baseline's seed spread.
- The pools inherit K12's generator style; the premise and long pools in particular carry padded ∨E proofs.
- Both tokenizers in use have four sentence letters (P Q R S); 26 of 91 excluded textbook exercises in one scan were
  dropped for having five to seven.
- K12 is Dan Pandori's data (`dan-pandori/nd-rl`, `cap-horizon/data/kh`). The pools are derivative works on it; the
  assembled pools must not be redistributed without that attribution.
