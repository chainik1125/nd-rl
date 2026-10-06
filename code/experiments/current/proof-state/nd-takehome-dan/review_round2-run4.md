# Review of round2-run4 (GRPO vs expert iteration at f = 0)

Reviewer session, independent of the executor. Phase 1 (this section) was written from the run brief
(`BRIEF_ROUND2.md` §Run 4), the pre-registration (log.md as committed in cb3c07e, 23:07:23 UTC, *before* the first
run-4 pod job), the code (`grpo.py`, `expert_iter.py`), the data and the raw artefacts only — no executor write-up
(`run4.md`, `numbers.md` §Run 4, `STATUS.md`, `artifacts/r4/summary*.json`, `analysis_*.log`, figures) was read
before it was committed. One caveat on blindness: `log.md` is in the phase-1 workspace, and a grep for the
pre-registration also returned the executor's two result entries (00:08 and 01:45 UTC, headline numbers only);
every number below was nevertheless recomputed from the raw files by my own code before any comparison.
All counts come from `review_run4_recount.py` and `review_run4_train.py` (own proof parser, dependency pruning,
start-index normaliser, box-depth counter and atom-renaming key — the reviewer's module `review_run5_recount.py`).
Only `nd_verify` is shared with the executor. Machine-readable outputs: `artifacts/review_r4/recount.json`,
`artifacts/review_r4/train.json`.

## Recount

### Hard constraints

| check | result |
|---|---|
| `nd_verify` unmodified | tree hash of `nd_verify/` at HEAD = `origin/main` (9437bb7); sha256 of both files identical in the repo and in the review copy |
| `artifacts/TEST_RUN_DONE` | no diff against `origin/main`; last commit ca93f83 (Sep 15 07:38, one test run) |
| evaluation files in training code | `grpo.py` reads the target pool (the RL prompts, by design), and `transfer` / `heldout` **only** inside the boundary block (pass@32 sample and greedy pass@1, never in the update). No supervised file is read at all: the only data that enters a gradient are the model's own samples, weighted by their verifier reward minus the group mean (rejected samples get negative weight; no invalid string is ever a positive example). **OK** |
| cap 6 on supervised data | the three Stage-1 sets `train_depth3_f0_{a1,a2,a3}`: 155,000 proofs each, max written length 6, 0 violations, 0 unparsable, 1,563 random proofs re-verified per set with 0 failures. GRPO adds no supervised data (see above); it trains on its own 7–11-line verified samples, which is the RL step the take-home permits |
| f = 0 | **0** depth-≥3 proofs (my predicate, dependency-pruned) in each of the three sets. The sets are distinct draws from one pool (pairwise shared proofs 47,485–47,581 of 155,000); the identical reductio count in all three (10,547) is the assembly's stratification, not a copy |
| splits disjoint by class | 0 renaming-class overlaps between each Stage-1 set and targets (1,000), transfer (500), held-out (5,000) |
| hand-written / LLM-written training proofs | none; every positive is a model sample accepted by `nd_verify` |
| pre-registration before the first pod job | R4-E1…E4 committed 23:07:23 UTC (cb3c07e). First arm's `args.json` written 23:09:19, its first found file 23:12:32 (mtimes preserved by the pull). **Met, by 2 min.** The chunked-update commit (faa87d9, 23:14:28) landed after the first two arms (`g8_a1_s0`, `g8_a2_s0`, whose `args.json` lack `lp_batch`) had started; the diff changes only how the backward pass is split (chunks of 128 sequences, zero-advantage sequences skipped), not the gradient |

### The implementation (`grpo.py`, as run)

On-policy: each step samples P prompts × G completions from the current parameters at T = 0.8 (P·G = 512:
G = 8 → 64 prompts, G = 32 → 16 prompts), rewards each completion by `nd_verify` on the prompted sequent, sets
A = r − mean(r over its group) (no std normalisation), and takes one AdamW step (lr 10⁻⁴, β = (0.9, 0.95), no weight
decay) on −Σ A·log π(completion) / (P·G·400). No KL, no clipping, no replay, no retained data. 500 steps ×
512 = 256,000 samples = 8 × 32 × 1,000 = EI's budget; the prompt order is a reshuffled cycle over the targets, so
every target receives exactly 256 samples in both algorithms. Two remarks that do not change the results: (i) with
Adam the "fixed divisor" is inert (Adam is invariant to the loss scale up to ε), and the recorded gradient norms
(0.03–0.13) never reach the clip of 1.0 — the update is Adam on the group-baselined REINFORCE direction; (ii) the
per-round `steps` lists hold the last 62 steps before each boundary, so the reported per-round reward and variance
are means over 496 of the 500 steps.

### Bookkeeping (12 arms: G ∈ {8, 32} × draws a1–a3 × seeds 0, 1)

| check | result |
|---|---|
| rounds present | 8 round-equivalents in 11 arms; **7 in `g8_a1_s0`** (round-8 files lost with pod p5; its round-8 checkpoint exists locally and in the bucket, `extra = {grpo_round: 8, step: 500}`) |
| budget | `samples` = 256,000 at the last boundary in all 11 complete arms (224,256 at round 7 of `g8_a1_s0`); boundaries at steps 62, 125, 188, 250, 312, 375, 438, 500 |
| per-round found files nested, `round` field = first file containing the proof | true for all keys in 11 arms (1,040–1,390 distinct normalised proofs each). **`g8_a2_s1`**: `found_1.jsonl` / `round_1.json` were written by a duplicate process that was killed (executor's log, 01:01); 28 of its proofs are not in `found_2`, and 144 of 1,392 keys disagree on the round. Rounds 2–8 are the surviving run's. For that arm I take round-1 from the `round` field of `found_8` (625 solved / 351 depth-3 theorems) rather than from the duplicate's files (605 / 329) |
| executor's cumulative solved counts (`round_<r>.json`) | equal to my recount at every round in all arms, except `g8_a2_s1` rounds 1–3 (the duplicate's files; the survivor's count is 20 / 3 / 1 higher) |
| duplicate records in found files | 0 |
| verification | every distinct normalised target proof re-verified against its prompt: 14,741 (GRPO, 12 arms) + 4,471 (EI, 6 arms), **0 failures**, 0 unparsable; every distinct transfer proof at the last boundary (4,587) and in the EI round-8 files (2,242): 0 failures |
| checkpoints | 12 `ckpts/r4/grpo_*_r8.pt`, all `{grpo_round: 8, step: 500}`, tokenizer `abs`, 52 tensors; all 12 in `hf://…/round2/run4/ckpts/r4/`; `artifacts/r4` synced (301 files, 01:36 UTC) |

### Base rates of the six Stage-1 draws (ignition study's coverage files `artifacts/ign/cov_depth3_f0_<draw>.s0.jsonl`, 300 targets × 2,000 samples, my predicate; every stored proof re-verified, 0 failures)

| draw | valid samples / 600k | targets solved | depth-3 hits (theorems) | depth-3 rate per sample |
|---|---:|---:|---:|---:|
| a1 s0 | 1,963 | 17 | 152 (6) | 2.5·10⁻⁴ |
| a1 s1 | 2,620 | 19 | 370 (7) | 6.2·10⁻⁴ |
| a2 s0 | 1,574 | 11 | 301 (8) | 5.0·10⁻⁴ |
| a2 s1 | 930 | 7 | 3 (2) | 5·10⁻⁶ |
| a3 s0 | 315 | 2 | 6 (1) | 1·10⁻⁵ |
| a3 s1 | 1,444 | 14 | 1,128 (12) | 1.9·10⁻³ |

All six draws have a non-zero depth-3 base rate (the pre-registration's "≥ 1.7·10⁻⁶" holds); a2 s1 and a3 s0 are
the two rare-prior draws (3 and 6 hits in 600k samples), and under EI they are the draws that ignite latest
(a2 s1: 0 / 0 / 8 / 78 depth-3 theorems at rounds 1–4).

### Results at matched sample budgets (my recount; depth-3 = ≥ 1 verified proof of the target whose dependency-pruned form has box depth ≥ 3)

EI arms are the follow-up's block-A arms on the same Stage-1 checkpoints (`artifacts/p2/ei_depth3_f0_<draw>`,
k = 32, T = 0.8, 8 rounds, retain 20,000 Stage-1 records, 600 fine-tuning steps per round); their per-round
counts use the minimum `round` per (theorem, normalised proof). `g8_a1_s0` is compared at round-equivalent 7
(224k samples) against EI's round 7.

| draw | EI: depth-3 r8 (r7) | EI: solved r8 (r7) | G = 8: depth-3 r8 (r7) | G = 8: solved r8 (r7) | G = 32: depth-3 r8 (r7) | G = 32: solved r8 (r7) | solved ratio G8 / EI, G32 / EI |
|---|---|---|---|---|---|---|---|
| a1 s0 | 335 (327) | 645 (623) | – (**478**) | – (**778**) | 440 (432) | 728 (718) | 1.25 (at r7), 1.13 |
| a1 s1 | 364 (354) | 698 (686) | 505 (501) | 819 (814) | 467 (458) | 798 (790) | 1.17, 1.14 |
| a2 s0 | 350 (341) | 589 (579) | 473 (463) | 777 (764) | 437 (424) | 737 (724) | 1.32, 1.25 |
| a2 s1 | 341 (317) | 652 (624) | 475 (470) | 799 (793) | 404 (399) | 725 (719) | 1.23, 1.11 |
| a3 s0 | 361 (348) | 605 (590) | 447 (442) | 747 (738) | 411 (409) | 710 (705) | 1.23, 1.17 |
| a3 s1 | 352 (346) | 583 (572) | 472 (465) | 780 (768) | 395 (389) | 691 (685) | 1.34, 1.19 |

Cumulative depth-3 theorems per round-equivalent (1,000 targets):

| arm | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EI a1 s0 / a1 s1 | 2 / 4 | 8 / 41 | 117 / 175 | 259 / 279 | 299 / 311 | 317 / 340 | 327 / 354 | 335 / 364 |
| EI a2 s0 / a2 s1 | 3 / 0 | 75 / 0 | 199 / 8 | 291 / 78 | 308 / 242 | 327 / 298 | 341 / 317 | 350 / 341 |
| EI a3 s0 / a3 s1 | 1 / 7 | 31 / 92 | 185 / 227 | 280 / 319 | 319 / 338 | 335 / 345 | 348 / 346 | 361 / 352 |
| G8 a1 s0 / a1 s1 | 344 / 357 | 375 / 414 | 405 / 453 | 436 / 469 | 458 / 483 | 471 / 490 | 478 / 501 | – / 505 |
| G8 a2 s0 / a2 s1 | 373 / 351 | 400 / 391 | 419 / 405 | 433 / 411 | 445 / 436 | 454 / 447 | 463 / 470 | 473 / 475 |
| G8 a3 s0 / a3 s1 | 350 / 325 | 381 / 369 | 407 / 395 | 421 / 418 | 428 / 433 | 435 / 448 | 442 / 465 | 447 / 472 |
| G32 a1 s0 / a1 s1 | 176 / 251 | 321 / 350 | 368 / 387 | 389 / 423 | 402 / 440 | 415 / 451 | 432 / 458 | 440 / 467 |
| G32 a2 s0 / a2 s1 | 283 / 124 | 342 / 309 | 364 / 351 | 372 / 366 | 395 / 376 | 415 / 389 | 424 / 399 | 437 / 404 |
| G32 a3 s0 / a3 s1 | 252 / 201 | 337 / 317 | 377 / 357 | 384 / 371 | 394 / 377 | 402 / 383 | 409 / 389 | 411 / 395 |

So: every GRPO arm crosses the run-3 review's ignition threshold (20 theorems) inside the first round-equivalent
(the first 32,000 samples, 62 updates), including the two rare-prior draws (a2 s1: 351 / 124; a3 s0: 350 / 252),
where EI needs 4 / 2 rounds. At the full budget G = 8 reaches 447–505 (0.447–0.505 of the pool), G = 32
395–467 (0.395–0.467), EI 335–364 (0.335–0.364). **The acquired sets nest:** on every draw the EI arm's
depth-3 theorems are, up to 1–3, a subset of the G = 8 arm's (EI − G8 = 1, 3, 1, 1, 3, 2; G8 − EI = 144, 144, 124,
135, 89, 122). The extra theorems are the longer ones: by oracle minimum length, a1 s0 length 9 has 30 depth-3
theorems under EI vs 94 (G8) / 69 (G32), length 10: 14 vs 33 / 27, length 11: 11 vs 25 / 24 (the 7- and 8-line
targets are at ceiling for all three). Written lengths reach 10–11 lines in the GRPO arms (8–42 proofs of ≥ 10
written lines per arm; EI: 0; EI's longest is 9). A handful of depth-4 proofs appear (0–3 per GRPO arm; EI 0–1).

Training signal and in-distribution damage (from `round_<r>.json`; the per-step values are the executor's, the
round means and minima are mine):

| arm | reward per sample r1 → r8 | groups with reward variance: step 1 / max round mean / r8 mean | held-out greedy r1 → r8 (min) | transfer greedy r8 |
|---|---|---|---|---:|
| G8 a1 s0 | 0.37 → 0.74 (r7) | 0.11 / 0.31 / 0.08 | 0.756 → 0.562 (0.562) | 0.554 |
| G8 a1 s1 | 0.42 → 0.79 | 0.06 / 0.33 / 0.05 | 0.803 → 0.644 (0.599) | 0.670 |
| G8 a2 s0 | 0.40 → 0.73 | 0.12 / 0.35 / 0.06 | 0.833 → 0.672 (0.672) | 0.592 |
| G8 a2 s1 | 0.38 → 0.76 | 0.09 / 0.31 / 0.07 | 0.767 → 0.621 (0.621) | 0.620 |
| G8 a3 s0 | 0.32 → 0.71 | 0.03 / 0.35 / 0.07 | 0.740 → **0.340** (0.340) | 0.578 |
| G8 a3 s1 | 0.35 → 0.72 | 0.05 / 0.33 / 0.09 | 0.796 → 0.531 (0.531) | 0.548 |
| G32 a1 s0 | 0.25 → 0.63 | 0.12 / 0.47 / 0.22 | 0.733 → 0.564 (0.564) | 0.534 |
| G32 a1 s1 | 0.34 → 0.70 | 0.06 / 0.49 / 0.28 | 0.846 → 0.643 (0.643) | 0.604 |
| G32 a2 s0 | 0.31 → 0.63 | 0.12 / 0.49 / 0.26 | 0.906 → 0.657 (0.657) | 0.560 |
| G32 a2 s1 | 0.24 → 0.63 | 0.06 / 0.45 / 0.21 | 0.829 → 0.610 (0.610) | 0.588 |
| G32 a3 s0 | 0.25 → 0.61 | 0.06 / 0.49 / 0.22 | 0.773 → **0.376** (0.310) | 0.516 |
| G32 a3 s1 | 0.27 → 0.61 | 0.12 / 0.51 / 0.20 | 0.731 → 0.521 (0.511) | 0.536 |

The EI arms keep held-out greedy at 0.903–0.945 at round 8 (Stage-1 level; they retain 20,000 Stage-1 records
each round) and end at transfer greedy 0.510–0.580 and per-sample acceptance 0.57–0.68 on the targets. Every GRPO
arm loses 0.2–0.6 of held-out greedy by the first boundary already and ends at 0.34–0.67; the loss is spread over
all lengths 2–6 (e.g. G8 a3 s0 at the end: 0.48 / 0.26 / 0.29 / 0.36 / 0.32 by length). Transfer greedy of the
GRPO arms (0.52–0.67) is at or above EI's.

**Transfer (never trained on).** The two bookkeepings differ: EI's `found_transfer_8` is the union of 8 × 32
samples across the evolving models, GRPO's `found_transfer_<r>` is a fresh pass@32 of the boundary model. On like
terms — GRPO's union over its 8 boundary files vs EI's union; GRPO's last boundary vs EI's round-8 samples alone:

| draw | EI union: solved / depth-3 | G8 union | G32 union | EI round-8 only | G8 last boundary | G32 last boundary |
|---|---|---|---|---|---|---|
| a1 s0 | 331 / 186 | 393 / 236 | 370 / 218 | 231 / 140 | 338 / 200 | 315 / 183 |
| a1 s1 | 337 / 183 | 422 / 264 | 406 / 249 | 224 / 126 | 381 / 229 | 353 / 202 |
| a2 s0 | 304 / 184 | 403 / 249 | 386 / 233 | 206 / 131 | 347 / 210 | 323 / 190 |
| a2 s1 | 331 / 179 | 409 / 253 | 368 / 218 | 241 / 137 | 351 / 207 | 338 / 188 |
| a3 s0 | 315 / 193 | 381 / 235 | 357 / 213 | 216 / 125 | 332 / 197 | 302 / 174 |
| a3 s1 | 307 / 189 | 382 / 236 | 347 / 206 | 230 / 138 | 330 / 192 | 313 / 183 |

The pattern generalises to the transfer pool under both algorithms, and more so under GRPO (of 500).

### Pre-registered expectations (cb3c07e, 23:07 UTC) against my values

| id | expectation | my value | verdict |
|---|---|---|---|
| R4-E1 | G = 8 acquires depth-3 in ≥ 4 / 6 arms but later and lower than EI (round-8 equivalent 0.10–0.30); G = 32 in 6 / 6 at 0.25–0.40 | G = 8: 6 / 6, all ignite in round-equivalent 1, 0.447–0.505 at the end (EI 0.335–0.364); G = 32: 6 / 6, round-equivalent 1, 0.395–0.467 | **wrong in both directions**: G = 8 is earlier and higher than EI, not later and lower; G = 32 is above the band and above EI |
| R4-E2 | the fraction of groups with reward variance starts at 0.05–0.20 and rises to ≥ 0.4 by the end in every acquiring arm | at step 1: 0.03–0.12 (as expected); it peaks inside round-equivalents 1–2 (round means 0.31–0.35 for G = 8, 0.45–0.51 for G = 32) and then **falls** to 0.05–0.09 (G = 8) / 0.20–0.28 (G = 32) at the end as the per-sample reward rises to 0.61–0.79 | **wrong** on the trajectory: the signal saturates rather than grows |
| R4-E3 | final solve rate within ±20 % of the EI arm's on the same draw | G = 8 / EI = 1.17–1.34 (5 of 6 above +20 %; a1 s0 at round 7: 1.25); G = 32 / EI = 1.11–1.25 (1 of 6 above +20 %) | **not met** for G = 8 (higher than the band), met for 5 / 6 G = 32 arms — the failure is on the upper side |
| R4-E4 | ≥ 1 of 12 arms degrades (held-out greedy < 0.5 at some boundary) | 2 of 12 (G8 a3 s0: 0.477 / 0.458 / 0.340 at r6–r8; G32 a3 s0: 0.466 at r5, 0.310 at r7, 0.376 at r8); every arm ends 0.2–0.6 below its Stage-1 level | **met** |

### Reproducibility

`python3 review_run4_recount.py` and `python3 review_run4_train.py` in the repository root regenerate every number
above from `artifacts/r4/`, `artifacts/p2/ei_depth3_f0_*`, `artifacts/ign/cov_depth3_f0_*`, `ckpts/r4/`,
`data/p2/` (outputs in `artifacts/review_r4/`; the recount takes ≈ 10 min on the VPS, the Stage-1 scan ≈ 15 min).
Checkpoint provenance was read without torch from the 12 `.pt` files' stored `extra` dicts. Not re-derived: the
greedy / pass@32 evaluations themselves (they need a GPU; I re-verified every proof the files contain but did not
re-sample), pod spend, kill-switch timing, and the bucket contents beyond the listings above.

## Compare (phase 2: `run4.md`, `numbers.md` §Round 2 — Run 4, `log.md`, `STATUS.md` §Run 4, `artifacts/r4/summary.json`, `figures/run4_grpo_vs_ei.png`, `QUESTIONS.md`)

Gate 0: expectations R4-E1…E4 committed 23:07:23 UTC, first arm started 23:09:19 (met). The results-vs-expectations
entry (log 01:45) scores E1 and E2 as wrong and E4 as met — honest; E3 is scored generously (below). The three
bookkeeping losses (one arm at round-equivalent 7, one round-1 file from a killed duplicate, one arm re-run after
its pod was deleted unpulled) are disclosed in `numbers.md` and `log.md`, and the files on disk are consistent with
those accounts (mtimes, nesting, checkpoint `extra`). The figure is the recount's numbers (two label collisions at
the right edge, cosmetic).

One correction to my phase-1 table: for `g8_a2_s1` my "files rule" counted 475 depth-3 theorems and 1,392 distinct
proofs because `found_1.jsonl` is the killed duplicate's and holds 28 proofs (3 depth-3 theorems) the surviving run
never sampled. The surviving run's count — the executor's 472 / 790 pattern proofs, which my "round field" rule
reproduces — is the right one; the same applies to that arm's written-length histogram (658 / 464, not 661 / 472).

| claim (executor) | my independent value | verdict |
|---|---|---|
| per-arm depth-3 acquisition and solved at the last round-equivalent (12 arms, `numbers.md`) | identical in all 12 arms (`g8_a2_s1` with the round-field rule); pattern-proof counts, written-length histograms, per-round acquisition and solved lists, reward and variance-fraction lists, held-out greedy, transfer pass@32 and its depth-3 count: identical | **reproduced** |
| EI references: 0.335 / 0.364 / 0.350 / 0.341 / 0.361 / 0.352, solved 645 / 698 / 589 / 652 / 605 / 583, per-round lists | identical (my predicate on my normalised proofs, min round per proof) | **reproduced** |
| means: G = 8 0.474 (0.447–0.505), G = 32 0.426 (0.395–0.467), EI 0.351 | 0.4745 / 0.4257 / 0.3505 | **reproduced** |
| "the first 32,000 samples already give 124–373 depth-3 theorems where expert iteration's first round gives 0–8" | 124–373 (the surviving `g8_a2_s1` run: 351); EI 0–7 | **reproduced** |
| "groups of 8 beat groups of 32" | on all six draws, in acquisition (+31 to +77) and in solved (+50 to +89) | **reproduced** |
| "held-out greedy falls from 0.87–0.95 to 0.34–0.67" | ends 0.340–0.672; already 0.73–0.91 at the first boundary; minimum 0.310 (`g32_a3_s0`, r7) | **reproduced** |
| R4-E2 scored "wrong — the fraction of groups with reward variance starts at 0.31–0.51 and falls; it never rises" | at *step 1* it is 0.03–0.12, i.e. the pre-registered start (0.05–0.20) was right; it rises inside the first 1–2 round-equivalents to round means of 0.31–0.51 and then falls to 0.05–0.28. The verdict "wrong" stands (it does not end ≥ 0.4), the stated trajectory does not | **verdict right, description imprecise** |
| R4-E3 "held at the top of the band (solved 691–819 vs EI 583–698)" | paired per draw, G = 8 / EI = 1.17, 1.32, 1.23, 1.23, 1.34 and 1.25 (a1 s0 at r7): 5 of 6 outside ±20 %; G = 32 / EI = 1.11–1.25, 1 of 6 outside | **mis-scored**: not met for G = 8 (GRPO solves more than the band allowed); comparing ranges instead of pairs hides it. No effect on the conclusion's direction |
| R4-E4 "held in 12 / 12 (held-out greedy 0.34–0.67 at the end)" | the pre-registered criterion was "< 0.5 at some boundary": 2 of 12 arms meet it (both a3 s0); all 12 end 0.2–0.6 below Stage-1. The expectation ("≥ 1 of 12") is met either way | **met**; the "12 / 12" uses a looser criterion than the one registered |
| `g8_a2_s1` "round_1.json … excluded from the per-round table" (`numbers.md`) | the per-round lists printed for that arm still carry the duplicate's round-1 values (329 depth-3, 605 solved, reward 0.38, variance 0.31); the surviving run's round-1 values from its own `round` field are 351 / 625 | **minor inconsistency**, one cell |
| "on-policy, binary verifier reward, group-mean baseline, fixed loss divisor, no KL, one AdamW step per batch; 512 samples per step, 500 steps = 256,000 = EI's budget" | as implemented and as run (`args.json`, `round_<r>.json` samples, `extra` of the checkpoints); each target receives exactly 256 samples under both algorithms. The divisor and the clip are inert under Adam (grad norms ≤ 0.13) | **reproduced**; the inert knobs are worth a sentence |
| "the arm is the algorithm as the sprint ran it" / "the difference must sit in the sprint's model, codec or targets" | not checkable here (no sprint code). See the verdict: the six draws are not zero-base-rate draws, so the run does not test the sprint's setting even if the algorithm matches | **unverifiable**, and the "must" overreaches |
| transfer | `run4.md` gives no transfer comparison; `numbers.md` lists GRPO's boundary pass@32 without the EI counterpart. On like terms GRPO is ahead there too (last boundary 302–381 / 174–229 depth-3 vs EI's round-8 samples 206–241 / 125–140; unions 347–422 / 206–264 vs 304–337 / 179–193) | **omission**, in GRPO's favour |
| base-model reachability | `run4.md` cites none. The six draws' Stage-1 depth-3 rates are 5·10⁻⁶ – 1.9·10⁻³ per sample (3 – 1,128 hits in 600k; 1–12 of 300 targets) — the ignition study's numbers, which the write-up should quote next to "crosses zero coverage" | **standards gap** (policy: a base-model number with every "RL solved X") |
| spend | `STATUS.md` says run 4 ≈ $1.6, `numbers.md` ≈ $2.4 (p4 / p5 shared with run 3); both far inside $50 | **inconsistent by $0.8**, not re-derived |
| bucket | `round2/run4/{artifacts/r4 (301 files), ckpts/r4 (12)}` present | **reproduced** (listing only) |

### Verdict

Every number in the write-up reproduces from the raw files with independent code, the implementation is what the
brief asked for, and the losses are disclosed. The result is solid and the direction is not in doubt: at an equal
sample budget on the same six Stage-1 draws, on-policy GRPO reaches the depth-3 pattern in the first 62 updates on
every draw and ends 0.40–0.51 against expert iteration's 0.34–0.36, solving 11–34 % more targets, at the cost of
the in-distribution model (held-out greedy 0.34–0.67 vs 0.90–0.95) — the expected price of no KL and no retained
data.

Two things the write-up says, or implies, that the data do not support:

1. **"Crosses zero coverage" is the campaign's term for f = 0 in the supervised set, not for a zero prior.** All six
   draws have a non-zero Stage-1 depth-3 rate (3 – 1,128 hits per 600k samples), and the acquired sets nest: on every
   draw the EI arm's depth-3 theorems are, to within 1–3, a subset of the GRPO arm's, and the extra 89–144 are the
   longer targets (oracle length 9–12). GRPO here is a faster and less conservative eliciter of a prior that is rare
   but present — exactly the "rare-but-known" side of the project's question — and it tells us nothing yet about
   igniting from a true zero. The run-3 / ignition-study draws with **0** hits in 600k (depth-3 a1 s5, reductio s1 /
   s2) are the ones that would test that, and they were not run under GRPO.
2. **The sprint contrast is not settled.** "GRPO saw nothing" in the sprint may be a property of a zero-prior model
   (the 85M / relative-codec draw), of its targets, or of the algorithm; this run removes only the third possibility
   *for non-zero priors*. The sentence "the difference must sit in the sprint's model, codec or targets" should be
   "…or in the base rate of its draw".

Smaller points: R4-E3 should be scored as not met for G = 8 (paired ratios, not ranges); R4-E2's description should
say the signal starts at 0.03–0.12, peaks within two round-equivalents and then decays; the one duplicate-run cell
should be replaced by the survivor's 351 / 625; the write-up should carry the six base rates and an EI transfer
column; and the two spend figures should agree.

### Next measurement

GRPO on the zero-base-rate draws, same recipe and budget: depth-3 a1 s5 (0 hits in 600k; EI never ignited by round
8 without injection), reductio s1 and s2 (0 valid samples of any kind at Stage-1), G = 8, two sampling seeds each
(≈ 6 × 15 min on one RTX 3090, ≈ $1). Expected if run 4's reading is right: 0 pattern theorems throughout (the
group-mean baseline has no gradient when every group is all-zero; the variance fraction stays 0). If any of them
ignites, GRPO differs from EI in kind, not only in rate, and the sprint's negative needs a different explanation.
A second, cheap variant that would separate "faster" from "different": EI on the same six draws with the update
applied every 512 samples (one fine-tuning step per batch on the verified samples, no retained data) — if that
matches GRPO's curves, the gain is update frequency, not the policy gradient.
