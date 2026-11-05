---
written_on: 2026-11-04
written_by: agent:claude
---

# Pre-registration: a continued-pretraining (CPT) control for the paper's base-vs-RL comparison

Written 2026-11-04, about 5:25 pm PT (01:25Z Nov 5), before any job of this folder ran.

**Why.** Stage 3 of
[`circuit_diffing_20261101`](../circuit_diffing_20261101/log.md) measured CPT_s0. CPT_s0 is
S1g3_s0 trained a further 312 AdamW steps on its own Stage-1 pairs, with no RL. On the circuit
G problems it solves 29.6% against the base's 2.0% and the L3 end's 67.1%.

Stage 1 ended un-annealed: a cosine to a floor of 30% of peak. Both CPT and GRPO add a low-LR
AdamW phase that ends below that floor. So the paper's base-vs-RL gap may partly be an anneal the
base never had. Leon's standing rule on fair comparisons applies: the coordinator asked for this
control on 2026-11-04.

## Arms

All three arms come from the same base seed s.

- **Base.** `S1g3_s{0,1,2}`, the paper's existing reads `charles149_s1g3_s{s}_plain_k1024`.
- **RL.** The recipe's L3 end, `grpo_think_vs_mb_kkl0_s{s}_L3` it39, read as
  `charles149_grpo_think_vs_mb_kkl0_s{s}_plain_k1024`.
  - RL seed s starts from base s with `grpo --seed s` in both stages, L2 and L3.
  - So the RL seeds vary the base and RL's own sampling and data order.
- **CPT.** `circuit_diffing_20261101/cpt.py` from base s, with lr 1.5e-4, 312 steps, 400 records
  per step, a 10-step warmup and a cosine to 30%. Weight decay is 0, AdamW betas 0.9 / 0.95.
  - It trains on 82.1M tokens: 0.98× the L3 chain's trained tokens and the same number of AdamW
    steps. Its R is 0.186 against RL's 0.289, which includes sampling.
  - Every CPT seed uses the record order of `train.finetune` seed 0: the same 124,800 records in
    the same order.
  - So the CPT seeds vary only the base, plus GPU nondeterminism, not the data order.
  - CPT_s0 already exists: `circuits/ckpts/cpt/cpt_lr0.00015.pt`, sha256 `6f418716…`.
  - CPT_s1 and CPT_s2 are new: `cpt_lr0.00015_s{1,2}.pt`.

## Reads

The paper's exact read on Charles149, at k1024:
- `code.thinking.eval` from the `think_hirl` clone (branch `leon/high-rl-reads`, which has the
  same `code/thinking` as the read commits `d0cb16d` and `e9e30cf`);
- the set `think_hirl/sets/charles149/charles_clean.jsonl`;
- flags: `--k 64 --batch 1024 --budgets 64 --arms plain --temperature 0.8 --max_action 512
  --max_steps 96 --max_rej 10`;
- 16 chunks with `--seed` 0–15, merged per theorem in seed order by `hirl_queue.merge_group`.

Outputs are copied to serrano as
`/scratch/cowifornia/nd-rl/think/runs/eval/v2/charles149_cpt_s1g3_s{s}_plain_k1024/`.

The one deviation is `LEAN_GATE_WORKERS` 8 instead of 16, for the 12-core limit. Lean's verdicts
do not depend on it.

## Analysis (`table.py`)

- pass@1, pass@5, pass@64 and pass@1024, as % of the 149 problems: the unbiased estimator,
  averaged over problems and then seeds.
- Gaps CPT − base, RL − CPT and RL − base, with 95% intervals from `stats.paired_diff`
  (`paired_seeds=True`, 4,000 draws; problems resampled, seeds resampled in pairs), as in
  `high_rl_passk_20261030/seeds3.py`. Per seed, and pooled over seeds 0–2.
- The share of RL's gap that CPT matches, (CPT − base) / (RL − base). A point estimate.
- At k1024, per seed and pooled over 3 × 1,024 attempts: problems solved only by RL and not by
  CPT, and only by CPT and not by RL. The same against the base.

## Pre-registered readings (pooled seeds, pass@1024 and pass@1)

- **RL's gain survives the control** if the RL − CPT interval lies above 0 at pass@1024.
- **A large part of the paper's gap is an anneal effect** if CPT − base is at least half of
  RL − base at a given k. The paper's comparison should then use CPT as the baseline, or report
  both.
- **The control explains little** if CPT − base's interval includes 0 at pass@1024.

## Order and bounds

1. Read CPT_s0 (16 chunks).
2. Train CPT_s1, then read it.
3. Train CPT_s2, then read it.

- Every chunk has `timeout 1800`; each training run has `timeout 3600`. A watchdog and the GPU
  guard run as in Stage 3. One GPU (ynez 2); at most 12 cores.
- The deadline is 3 am PT, with 7 am PT as the hard stop. Steps that cannot finish by then are
  marked "not run". The t1320s250 k512 read runs only if time remains, and is not part of this
  pre-registration.

## Amendment 1: the J7 precedent and the G selection

Written 2026-11-04, about 6:30 pm PT, before any job of this folder ran. Requested by the
coordinator. No design change.

**Precedent.** J7 of capability-defs, 2026-10-05:
[summary](../../../../experiment-summaries/2026-10-05-capability-defs-compute-matched-budget-shrinks-equal-k-creations-rl-still-ahead-lem-teachable/README.md)
and [`REPORT.md`](../../../../docs/project_strategy/capabilities/REPORT.md).
- **Setup.** Robbie's best-cap12 `pend` was trained further on K12 with the LR re-warmed to
  3e-4 and decayed by cosine to 3e-5, for 28,054–33,535 steps (about 82% of the EI r8 ladder's
  GPU time).
- **Base schedule.** `pend`'s pretraining also used a cosine to a 30% floor
  (`MIN_LR_FRAC = 0.3`). So it was un-annealed in the same way, and J7 annealed it.
- **Result.** Solved@256 of 322 is a lateral move: 220 / 215 / 219 against `pend`'s 232 / 236 /
  234 and r8's 286 / 286 / 293. It gains 16–19 theorems and loses 31–40.
- **What it implies.** The hypothesis "the un-annealed Stage 1 is why CPT gains" is one that J7
  argues against. It stays a hypothesis here.
- **Differences from this control.**
  - base: Robbie's `pend` against our S1g3;
  - data: K12 against S1g3's pairs;
  - eval: 322 theorems at k256 against Charles149 at k1024;
  - length: J7 ran tens of thousands of steps, 15–18× Stage 1's time, against CPT's 312 steps
    (0.81 epoch, R 0.186), about 90× shorter relative to Stage 1.

**G selection.** The 80 circuit G problems were selected with the paper's earlier reads: the base's
Charles149 k1024 read (chunk seeds 0–15), and its 13–20 k8 + k4 reads (`problems.py`).
- Stage 2's base 2.0% on G is a fresh read of 32 attempts at sampler seed 7, independent of those
  draws. It is not biased low by the selection. Regression to the mean is visible but small: the
  13–20 G problems, selected at 0 of 12, read 2.7% fresh.
- CPT's 29.6% comes from the same fresh setting.
- The pass@k table of this folder covers the whole Charles149 set, with no selection.
