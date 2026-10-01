---
written_on: 2026-10-01
written_by: agent:claude
---

# Capability benchmarks for the base and RL provers (ndbench)

Status: first full pass complete and audited (2026-10-01). Single checkpoint per model, single
sampling seed.

Question: beyond pass@1 on a fixed set, what did RL change about the small ND prover? Does it
compose solved parts, extend to arbitrary length within a narrow theorem family, handle larger
fillers in known laws, finish partial proofs, learn new patterns more cheaply, and does the
change show up in likelihoods, steering vectors, weight interpolation and linear probes?

Working folder: [`code/experiments/current/benchmarking_20260930/`](../../code/experiments/current/benchmarking_20260930/README.md)
(plan, interp spec, schedules, configs). Package: [`code/ndbench/`](../../code/ndbench/).
Branch `leon/benchmarking`.

## Configuration

- **Models** (`configs/models.yaml`, sha256 prefixes checked at load):
  - base: `ab2/frozen/pretrain.pt` (`f4409e4f82d8922f`), pretrained on proofs of at most 6 lines.
  - rl: run-2 RL champion `ab2/monitor/final/champion2_n060_fullpool/final.pt`
    (`548a9e343c71675f`).
  - armC: length-experiment arm C seed 101 `ab3/lengthexp/runs/C_s101/final.pt`
    (`9ec083a966662fdc`), RL on true lengths 6-11.
- **Judge**: `nd_v2` = vendored `nd_verify` plus a forward-citation check (the upstream verifier
  accepts circular proofs; the fix is proposed for main in PR #22).
- **Sampling**: temperature 0.8 unless stated; per-item `max_new` = min(1024 - prompt,
  max(160, 3 x reference tokens + 32)). Unbiased pass@k; item-level bootstrap CIs.
- **Suites** (out/suites): textbook_dev 58, textbook_law 202, gen_ctrl 60, parts_pool 1,139,
  skill_pool 11,524 (legacy-joined), families 2,472, laws 1,755, padding v2 740, glue 792,
  glue_parts_renamed 1,317, prefix 1,039; interp eval suites dev 720 / sealed 1,240 / k128 150.
- **Outputs**: `/scratch/cowifornia/nd-rl/bench/out/` (store, logs, quarantine). One-page
  summary: `out/dash_summary/summary.html`; detail pages: `out/dash_v2/`.
- **Compute**: about 6.9 GPU-h on the ledger in total, including the audit reruns.
- **Audit**: two independent read-only audits on 2026-10-01 (benchmark evals; EDL and interp).
  Their confirmed fixes and five approved reruns are in the commits after `d7ed980`.

## Observations

Percentages are of the stated item set. "Observed" lines are measurements; interpretations are
marked.

- **Textbook pass@k (Yue-style).** textbook_dev, 58 theorems, k up to 4096 for all three models:
  base 4.1% pass@1 / 17.2% pass@4096, rl 5.5% / 22.4%, armC 9.3% / 13.8%. rl is above base at
  every k in the point estimate, but about 40% of bootstrap resamples show a crossover, so the
  ordering is not robust at 58 items. armC gains nothing past k=256. textbook_law (202): base
  0.0%, rl 1.4% pass@1 / 4.5% pass@1024.
- **Restricted families.** Every family collapses within a few sizes. On theorems absent from
  training (the smallest sizes are mostly training theorems), base has no horizon in any family;
  rl reaches n*=5 on imp_nest (about 11 written lines) and n*=3 on and_tower; armC n*=4 on
  imp_nest. With training theorems included base reaches n*=2 on three families: 68% of base's
  family successes are on seen theorems (21.5% for rl).
- **Glue (compositionality).** R = joined pass@1 / product of the parts' pass@1, with each part
  sampled as it appears inside the join (atoms renamed): AND base 0.002 / rl 0.031, CHAIN
  0.24 / 0.50 (4 and 34 items), CASE 0 / 0, BOX 0.61 / 0.83. Renaming a part's atoms lowers rl's
  pass rate by about 3 points beyond regression to the mean.
- **Skills.** On the 5,066 items both have, base solves 0.0% in every skill group; rl 25.3% with
  no special skill, 0.05% when a case split is needed, 2.4% when classical. armC vs base on 6,458
  lengthexp items: 13.6 / 0.0 / 7.7% vs 1.4 / 0.0 / 0.3%. No model solves case-split proofs.
- **Law x substitution size.** Pooled over 8 schemas (seen-in-training instances excluded): base
  peaks at 16.5% (s=2), rl at 11.8% (s=1); both are near 0% by s=4. base scores 0% on 6 of 8
  schemas.
- **Free lines (padding).** At 8 written lines rl solves 7.7% of padded easy theorems vs 33.7% of
  15 natural gen_ctrl theorems (which come from rl's hill-climb selection set, so optimistic).
- **Prefix completion.** With 1 line left rl finishes 48.7% vs base 14.2%; with 3 left 12.1% vs
  0.0%. First-failure hazard (forced lines excluded from the risk set): rl 32% at line 8, 37% at
  line 10; base 87% at line 5.
- **EDL (excess description length of learning a new class online).** Six learning rates
  (0 = no update, 1e-5 ... 1e-3), chosen by online code length, plus the ln 6 selection cost,
  2 data orders, n = 32 or 64 per class. rl's EDL per example is lower than base's on 9/9
  classes with no overlap between orders (e.g. imp_nest_n8 51.0 vs 7.4, gen_L10 8.2 vs 1.3
  nats). But rl also starts with lower test loss on 9/9, and EDL tracks starting loss (Spearman
  0.94 over class x model); per nat of test loss that training removes, rl costs as much as or
  more than base on 7/9. Null control (heldout_cap6): base picks lr 0 (EDL 0.27/example), rl
  1e-5 (0.19). Interpretation: lower EDL here mostly reflects rl starting closer, not rl learning
  faster.
- **Likelihood.** On the 43 gen_ctrl theorems rl solves and base never does, base assigns about
  7.0 nats per line less than rl to neutral minimal proofs and about 13.0 to rl's own proofs.
  Marginalising over valid line orders changes log p by 0.000 nats/line in median; 93% of
  neutral proofs have one valid order.
- **Steering (delta at proof positions only, after the fix).** Adding the RL - base mean
  activation difference to base never beats unsteered base (best of 20 settings 0.14% vs 0.17%).
  Subtracting it from rl costs 2.0 points (23.5% to 21.5%; sealed set 19.0% to 16.7%). Before the
  fix delta was also added to the prompt and the same comparison showed 7.5 points.
- **Weight interpolation.** On a 150-theorem subset, pass@1 rises from 0% (base) to 24.9% (rl),
  mostly between alpha 0.5 and 0.75; alpha 1.25 gives 6.6% pass@1 but 49.3% pass@128; alpha 1.5
  gives 0%. Sealed set alone: 0.31% at 0.5, 12.2% at 0.75, 19.0% at 1, 3.4% at 1.25.
- **Probes.** next_rule accuracy at the deepest layer is 83.7% (base) and 84.9% (rl), about 11
  points above a position-only lookup (line index + premise count, 73.0%); the two models are
  within 1.2 points at every layer. lines_to_go R2 0.82 vs 0.74 from the line index alone.

Interpretation (not tested directly): RL's gains concentrate in things base cannot do at all on
unseen theorems (longer chains in imp_nest/and_tower, finishing proofs, composing boxes), while
case splits stay at zero for every model. Base's likelihood of even neutral proofs of RL-only
theorems is very low, which fits "RL reached proofs base would not sample" better than "RL
re-weighted proofs base already found". Linear probes do not separate the two models.

## Limitations and next step

- One checkpoint per model and one sampling seed; textbook_dev has 58 items, several glue and
  family cells have under 10 items.
- dev L8-11 (gen_ctrl, interp_eval_dev, part of the k128 subset) was the RL hill-climb's
  selection metric, so rl numbers on it are optimistic; the sealed set is the clean comparison.
- skill_pool results are legacy imports judged by the legacy pipeline (the audit re-judged all
  1.47M benchmark samples with nd_v2 and found no mismatch).
- EDL uses 2 data orders, so its uncertainty is the spread between them.
- Rerunning: [`code/ndbench/README.md`](../../code/ndbench/README.md) lists each benchmark's GPU
  cost. The cpu tier (redraw, re-analyse) needs no GPU, and the cheap tier costs about 14 GPU-min
  per model.
- Next: run Dan's models through the same adapters (`code/ndbench/adapters/`), and decide which of
  the 12 benchmarks to keep as the standing capability panel.
