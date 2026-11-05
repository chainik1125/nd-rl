---
written_on: 2026-11-04
written_by: agent:claude
---

# Pre-registration: is the base saturated? Anneal, pretraining dose, a longer Stage 1, and RL from a better base

Written 2026-11-04, about 11:15 pm PT (07:15Z Nov 5), before any job of this folder ran. Leon
approved the experiment at about 11 pm PT, with C and D run in parallel on Runpod A40 pods.

## Why

[`cpt_control_20261105`](../cpt_control_20261105/README.md) showed the following. Continued
pretraining (CPT) of S1g3_s0 on its own Stage-1 pairs, for 312 AdamW steps (82.1M tokens, matched to
RL's trained tokens), reaches RL at large k on Charles149. Pooled over 3 seeds, in % of 149:

| k | base | CPT | RL |
|---|---|---|---|
| 1 | 32.6 | 54.2 | 70.5 |
| 1024 | 74.5 | 88.6 | 86.6 |

RL solves 0 problems that CPT does not. The paper's "RL creates capability at large k" therefore
rests on a base that may not be saturated.

Facts about S1g3_s0 (`think/runs/logs/stage1_S1g3_s0.log`, `stage1/S1g3_s0/{args,train}.json`):
- Recipe: 9.57M params; Muon on the matrices plus AdamW; lr 1e-3; 200 warmup steps; a step cosine
  to a floor of 30% of peak (`MIN_LR_FRAC` 0.3); MTP 0.3.
- Length: 24,000 steps, 4.33 epochs of 959,374 pairs, 1,330 s; code `cd089700`, seed 0.
- Val loss (7,000 held-out pairs) still drifting down at the end: about 0.075 to 0.058–0.069, noisy;
  final 0.0692.
- The checkpoint holds the model only. There is no optimizer state, and the MTP head was not saved.
  Checked: `model.pt` keys are cfg, state, tok_mode, extra.

Precedent: capability-defs J7 (cpt_control PREREG amendment 1) annealed a different base and was
lateral. It differed in base, data and length: no thinking traces, 15–18× Stage-1 length.

## Hypotheses

- **H-anneal.** CPT's gain is mostly the anneal that Stage 1 never finished (its LR stopped at 30% of
  peak), not the extra data.
- **H-undertrained.** Stage 1 was too short. More pretraining tokens keep helping, beyond what an
  anneal gives.
- **H-RL.** RL creates capability at large k over a saturated base. RL from a better base still
  adds problems at k1024 over its own starting point.
- H-anneal and H-undertrained can both hold in part. H-RL is independent of them.

## Arms (single seed, s0)

All new models are trained from S1g3's own data, `think/runs/data/S1g3.pkl` (md5 `d1f29544`). Its
`report.json` excludes textbook_dev, textbook_train, Dan's heldout and the ood_check sets.

- **A. Anneal only.** Code: `anneal.py`, Stage 1's own trainer `train.train_best` at `cd089700`,
  from S1g3_s0.
  - Muon on the matrices and AdamW on the rest, as Stage 1 (wd 0.1, token budget 18,432, grad
    clip 1, seed 0).
  - LR: a linear warmup over 50 steps to Stage 1's final values, because no optimizer state was
    saved. Those values are AdamW 3e-4 and Muon 0.3 × `MUON_LR`. Then a cosine to 0 at step 1,200,
    which is 5% of Stage 1's steps.
  - **MTP weight 0** (a deviation). Stage 1 did not save its MTP head. A fresh head (loss ≈ ln 118)
    would steer the first Muon updates of the shared trunk and output head.
  - Tokens: about 1,200 × 18.4k budget ≈ 22M padded, about 0.27× CPT's 82.1M.
  - Val loss is logged as Stage 1 logs it, before (the base) and during.
- **B. CPT dose.** Code: `circuit_diffing_20261101/cpt.py` at `32bc1da`, exactly as CPT_s0.
  - Settings: lr 1.5e-4, 10 warmup steps, cosine to 4.5e-5, AdamW betas 0.9 / 0.95, wd 0, 400
    records per step, record order of `train.finetune` seed 0. The cosine spans each arm's own
    steps.
  - **B0.25:** 78 steps, the first 31,200 records of CPT_s0's order.
  - **B4:** 1,248 steps, about 328M tokens, 3.2 epochs of the 154,630 records. **The data repeat**,
    about 3 times.
  - With CPT_s0 (1×, 312 steps) and the base (0×), this is a dose curve.
- **C. Stage 1 retrained properly.** Code: `stage1_to0.py`, which is `python -m code.thinking.train`
  at `cd089700` with one change.
  - Same settings as S1g3_s0: seed 0, so the same initialization and data order; same data,
    held-out set and recipe.
  - `--best_steps 48000`, 2× the steps (about 8.7 epochs).
  - `MIN_LR_FRAC` 0, so the cosine ends at lr 0, not at 30%.
  - Val loss is logged every 30 s, as Stage 1 logs it.
- **D. RL from a better base.** The paper's RL recipe, both stages, seed 0. Each stage's argv is
  exactly as its `args.json` records it, run on the code commit it records:
  - L2 = `grpo_think_vs_s0_L2` at `fa28f6eb`;
  - L3 = `grpo_think_vs_mb_kkl0_s0_L3` at `f35d3daf` (39 iterations, wsd lr 1.5e-4, unlikeliness
    0.25, 3-premise extras at 0.33, KL 0).
  - Only these change: `--init`, `--name`, `--out`, the pool paths, and `--kl_anchor_ref`. L2
    anchors to its own init (0.02), as the recipe does.
  - **D1:** from CPT_s0 (`circuits/ckpts/cpt/cpt_lr0.00015.pt`).
  - **D2:** from C's model, when C is done.
  - The paper's L3 started from an L2 with no rollbacks. D's L2 guard line is recorded and reported,
    not used to refuse.

## Reads

Every new model (A, B0.25, B4, C, D1 = L3 it39, D2 = L3 it39) gets the paper's exact Charles149 read:
- the same as cpt_control's `reads.py` and PREREG: `code.thinking.eval`, plain, B64, T 0.8,
  `--batch 1024 --max_action 512 --max_steps 96 --max_rej 10`;
- 16 chunks of k64 with `--seed` 0–15, merged per theorem in seed order by `hirl_queue.merge_group`
  (`cce4dfc`, the CPT reads' code);
- the set file `charles_clean.jsonl` (md5 `b104d60f`, the same file as think_hirl's);
- chunks may run on different pods. A chunk's result depends on its seed, not on its pod.
- Deviation: `LEAN_GATE_WORKERS` is min(16, the pod's CPU quota), not 16. Lean's verdicts do not
  depend on it.

Existing reads used for comparison, all in `think/runs/eval/v2/`:
- `charles149_s1g3_s0_plain_k1024` (base);
- `charles149_cpt_s1g3_s0_plain_k1024` (CPT_s0);
- `charles149_grpo_think_vs_mb_kkl0_s0_plain_k1024` (RL L3 s0).

New merged reads go to `/scratch/cowifornia/nd-rl/base_saturation/final/`, not to `eval/v2/`, so
that scripts that glob `eval/v2` do not pick them up.

## Analysis (`table.py`)

- pass@1, @5, @64 and @1024 as % of 149, using the unbiased estimator; for every model, and the
  curve.
- Paired gaps with 95% problem-bootstrap intervals (`stats.paired_diff`, 4,000 draws, one seed) for
  these pairs:
  - A − base, A − CPT;
  - B0.25 − base, CPT − B0.25, B4 − CPT;
  - C − base, C − CPT;
  - D1 − CPT, RL − D1;
  - D2 − C, D2 − RL.
- Counts of problems solved by X and not Y at k1024 (one-arm-only), for RL vs D1, D1 vs CPT,
  D2 vs C, C vs base, A vs base, A vs CPT; both directions.
- Val loss (Stage 1's held-out pairs) for A and C, from their own logs.
- Descriptive: val loss of the other models, if computed on CPU with the same function.

## Pre-registered readings (pass@1024 unless stated; single seed)

- **Unfinished anneal (H-anneal):** A − base is at least half of CPT − base, and the A − CPT
  interval includes 0. (Seed 0's CPT − base is +9.4 [+4.7, +14.8]; see the correction below.)
- **Anneal is not the story:** the A − base interval includes 0.
- **Undertrained (H-undertrained):** the B4 − CPT interval lies above 0, or C − base lies above 0
  with C at least CPT's pass@1024.
- **Dose plateau:** the B4 − CPT interval includes 0. **Turns over:** it lies below 0.
- **RL adds coverage over a better base (H-RL holds):** the D1 − CPT interval lies above 0, or the
  D2 − C interval lies above 0.
  - For each, also the count of problems D solves at k1024 and its start never does.
- **RL adds no coverage over a better base (H-RL fails):** both D1 − CPT and D2 − C intervals include
  0 (or lie below it).
  - If at the same time D − start at pass@1 lies above 0, the paper's claim should become "RL
    improves sampling efficiency (small k) over a saturated base", not "RL creates capability".
- **Seed-noise caveat (pre-stated).** These are single-seed reads, so the intervals cover problem
  sampling only. In cpt_control, RL − CPT at pass@1024 ranged +1.3 to −4.0 across 3 seeds. A
  difference of 3 points or less at pass@1024 is within that seed spread, whatever its interval
  says.

## Predictions

| hypothesis | A vs CPT | dose B0.25 → CPT → B4 | C vs base, vs CPT | D1 − CPT, D2 − C |
|---|---|---|---|---|
| H-anneal | A ≈ CPT | early plateau: B0.25 near CPT, B4 ≈ CPT | C > base, C ≈ CPT ≈ A | (independent) |
| H-undertrained | A < CPT, A − base small | climbs: B0.25 < CPT < B4 | C > base, C ≥ CPT | (independent) |
| H-RL holds | — | — | — | intervals above 0, RL-only problems at k1024 |
| H-RL fails | — | — | — | intervals include 0; pass@1 still gains |

The agent's prior, stated before the data: a mix. Some of CPT's gain is anneal (A > base), the
dose curve flattens by 1–4×, and D1 − CPT at pass@1024 is within ±3 points.

## Compute, pods and bounds

- **Pods.** Five Runpod A40s (SECURE, $0.59/h), created by the coordinator at 07:03Z. Every copy to a
  pod is the coordinator's (`runpod_copy.sh`, `runpod_copy_ckpt.sh`).
  - `rp-sat-a`: A, then its read, then C chunks 0–7.
  - `rp-sat-b025`: B0.25, then its read, then C chunks 8–15.
  - `rp-sat-b4`: B4, then its read.
  - `rp-sat-c`: C, then D2, then D2 chunks 0–7.
  - `rp-sat-d1`: D1, then its read, then D2 chunks 8–15.
- **Bounds.**
  - Every step runs under `timeout`: a read chunk 45 min; A 1 h; B 30 min + 9 s per step; C 3 h;
    each GRPO stage 6 h (line.sh's).
  - An on-pod watchdog raises STALL, OVERRUN (1.5× expected) and EXIT events. Serrano polls them.
- **No lab GPUs.** Serrano CPU only, at most 8 cores, for staging, merging and tables.
- **Estimate.** About 21 pod-hours, $12.3 (range $10.9–14.2). The critical path, C → D2 → D2's read,
  ends about 5:00–5:30 am PT.
- **Not done.** textbook72 is never copied, trained on or read. No Claude-written traces.

## Code

- Pre-registration commit: the commit that adds this file on branch `leon/base-saturation` (from
  `leon/circuit-diffing` `32bc1da`), worktree `/scratch/cowifornia/nd-rl/worktrees/base-saturation`.
- Code snapshots on the pods (`git archive`):
  - Stage 1 `cd089700`;
  - CPT `32bc1da`;
  - GRPO L2 `fa28f6eb`;
  - GRPO L3 `f35d3daf`;
  - reads `cce4dfc`.
- This folder's `anneal.py` and `stage1_to0.py` are copied into the `cd089700` snapshot.

## Correction (11:40 pm PT, 07:40Z, before any read of a new model)

Two factual fixes to the text above, made before any result. The design and the readings are
unchanged.
- The H-anneal reading first quoted CPT − base as "+13.5 for s0". `table.py` on the existing seed-0
  reads gives +9.4 [+4.7, +14.8] at pass@1024: base 78.5%, CPT 87.9%, RL 89.3%. The reading compares
  against seed 0's own CPT − base, because every arm here is seed 0.
- The header first gave the writing time as 11:35 pm. The file was committed at 11:18 pm
  (`994790c`); the jobs launched at 11:18–11:19 pm.
