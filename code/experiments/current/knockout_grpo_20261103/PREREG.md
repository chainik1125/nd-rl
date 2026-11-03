---
written_on: 2026-11-03
written_by: agent:claude
---

# Pre-registration: how much GRPO does it take to bring back proof by contradiction?

Written 2026-11-03 at about 06:15 UTC (Nov 2, 10:15 pm PT), before any GPU job of this
experiment. Its sha256 and the time are recorded in [`log.md`](log.md). Branch
`leon/knockout-grpo` (from `leon/high-rl-paper` at `50d0336`), worktree
`/scratch/cowifornia/nd-rl/worktrees/knockout-grpo`; artifacts under
`/scratch/cowifornia/nd-rl/knockout_grpo/`.

This replicates Dan's create-or-elicit / coe-dose01 dose-response chart ("RL compute ÷
pretraining compute needed to solve 14 of 106 strict σ-covered held-out theorems", origin/dan
`docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/`) with Leon's GRPO + thinking-trace
pipeline instead of T1 expert iteration. Leon asked for it ("can you rerun this properly to match
dans"); it is for the midterm report's section "How much RL is needed to recover rare or fully
excluded proof strategies?".

## What was seen before writing this

- Dan's write-ups, code and data (below), and his chart's numbers.
- The σ shares of K12, S1g3 and the three new corpora (CPU, below).
- The existing GRPO runs' metrics (time and R per iteration). No model of this experiment has
  been trained or sampled, and no model has been read on the 106.

## Dan's definitions, used unchanged

Code: read-only clone of `dan-pandori/nd-takehome`, `/scratch/cowifornia/nd-rl/ext/nd-takehome-ps`
(branch `dan_proof-strategies` at `e6bfe0ea`, which carries create-or-elicit's `coe/`). sha256
prefixes: `coe/coe_sigma.py` 165222fa, `coe/coe_assemble.py` 7e0afacc, `coe/coe_dose.py`
cd32ff82, `patterns.py` e1b9f8f1, `gen.py` 60b702c1.

- **σ** = a DN step citing a NEGI line (after following reiteration), in the written or the
  dependency-pruned ND proof (`coe_sigma.flags`: `sigma_strict` / `sigma_chain`). Dan's dose
  predicate is `sigma_strict` in either form.
- **σ⁺** = a DN step citing any derived line (not PR / AS), in either form (`coe_assemble.knock`).
  The knockout removes σ⁺. The "lemma route" (`coe_assemble.lemma_route`) is kept out of every
  refill.
- Positive control: on K12 the classifier gives Dan's 16,703 σ and 16,961 σ⁺ (10.94 %) records.
- **The 106.** create-or-elicit's strict σ-covered held-out members
  (`data/coe/sigma_heldout_strict.txt`, rows from `transfer_track832.jsonl`;
  `sets/strict106.jsonl`, sha256 d8c19c44…): classical-only (not G4ip-provable) members of the
  six classical families' held-out (transfer) halves plus `la_transfer_916`, whose sequent has no
  ¬¬ subformula. No training set here contains them: 0 renaming-class hits (canon_key, canon2,
  ndbench pkey) in the GRPO pools L2 / L3 / L3_3p / L2_3p and the canary set; K12 has 0 (Dan's
  check) and the refill excludes them. None is a textbook72 theorem.
- **k = 32**, Dan's per-round transfer sample.
- **Bar.** The median, over the full-exposure bases, of the count solved before any RL update.
- **Crossing.** The first RL compute at which the count reaches the bar.

## Departures from Dan (agreed with the coordinator)

1. GRPO (Leon's primary recipe) instead of T1 expert iteration; no replay of pretraining data.
2. The thinking model (`lean_thinke`, programmatic search traces, effort budgets 0 / 64 / 256).
3. The pretraining corpus is S1g3 (one search trace per K12 theorem), with Dan's removal and refill
   applied to it (below), not K12.
4. Compute is our R rule (below); Dan's ladders are recomputed with it.
5. y is read from model checkpoints (below), not from the ladder's own per-round samples.

## Dose corpora (built and verified before this registration)

`assemble.py`, `refill_pool.py`, `sigma.py`; reports in `OUT/corpora/plan_report.json`,
`OUT/refill/candidates_report.json`, `OUT/sigma/*.json`.

- **S1g3** (full): 154,630 records; σ⁺ 3,838 (**2.48 %**), σ 3,711 (2.40 %), lemma route 2.
- **ko**: remove every S1g3 record whose theorem Dan's knockout removed from K12 (16,775 of his
  16,961 are in S1g3) and every record whose own search proof is σ⁺ or lemma-route (3,840):
  17,980 removed (14,140 Dan only, 2,635 both, 1,205 ours only). Refill in place, per K12
  generator length (Dan's matching key): first Dan's own K12⁻ refill theorems of that length whose
  search proof is σ⁺- and lemma-free (16,609), then his eligible pool (`coe_assemble.consider`,
  re-run here with his helpers, plus our exclusion sets: S1g3's DEFAULT_EXCLUDE + textbook72
  train, transfer_track832, read1054, fam_extra) for the shortfall (1,371). Seed 0.
- **d01 / d1**: Dan's `coe_dose.py` rule on ko: round(d × 154,630) = 155 / 1,546 removed S1g3
  records with σ in their search proof, per generator length in the removed σ records' mix, each
  replacing a refill record of the same length (seed 0, independent draws).
- **Traces.** Kept and dose records keep their S1g3 traces byte for byte. Refill traces come from
  the same generator (`searchtrace.py` at `79749ce`), which reproduces S1g3's e256 / e64 / plain
  records byte for byte on the first 2,990 K12 theorems. No trace is written by a language model.
- **Pickles** (`data.py` at `cd089700`, S1g3's command: `--graded 0:0.2 64:e64:0.4 256:e256:0.4
  --no-lean --extra-exclude textbook_train`, which rebuilds S1g3.pkl byte for byte, md5
  d1f29544): ko b4a7cb34, d01 2c836acb, d1 7455bd09.
- **Verified σ shares** (Dan's classifier on each corpus's proofs): ko 0 σ⁺ (0.000 %), d01 155
  (0.100 %), d1 1,546 (1.000 %); S1g3 2.48 %. Pair tokens per epoch: S1g3 1.017e8, ko 9.86e7
  (−3.0 %), d01 9.88e7, d1 1.008e8.

## Bases

`code.thinking.train` at `cd089700` (S1g3's commit; train.py is unchanged to HEAD), the S1g3
command: `--data PKL --heldout bench/dan_ckpts/data/heldout.jsonl --seed S --best_steps 24000`
(recipe best, 24,000 steps, token budget 18,432, MTP 0.3), seeds 0–2, on ko, d01, d1: 9 bases.
The full arm's bases are the existing `S1g3_s{0,1,2}`.

## GRPO (Leon's primary recipe, continued as ki78 was)

- **L2**: `grpo.py` at `fa28f6eb` (the full arm's L2 commit) with `grpo_think_vs_s{S}_L2`'s
  args.json, init and KL-anchor reference = the arm's own base: 39 iterations, 256 prompts × 16,
  lr 1.5e-4, pool `pools/curriculum/L2/pool.jsonl`, canary, rollbacks, think-ban stop, KL anchor
  0.02, logical arm with --max_rej 0, rej_penalty 0.1, effort 0,64,256, T 1.0.
- **L3**: `grpo.py` at `f35d3daf` with `grpo_think_vs_mb_kkl0_s{S}_L3`'s args (the merged recipe
  with KL anchor 0: WSD, decay 20 % sqrt to 0.3×, unlikeliness 0.25, L3_3p extra targets at
  0.33), from the arm's own L2 it39, with `--iters T_L3` and `--save_every T_L3`, as ki78 changed
  only `iters` and `save_every` (ki78 kept KL 0.02; we keep the primary recipe's 0).
- **Iteration budget.** One Dan ladder's RL GPU time, ≈ 15 GPU-h on an A40 (create-or-elicit
  `compute.md`, "rl" 13.5–16.3 h per 16-round ladder). T_L3 is fixed once, for every arm, from
  the pilot: T_L3 = round((15 h − the L2 time) / the L3 seconds per iteration), the median over
  the pilot machines. The value is written to `log.md` before the bulk launch. Arms are matched on
  iterations (attempts), as Dan's arms were matched on protocol.
- **Checkpoints.** The only code change: `--snap_iters` (`snap_iters_*.patch`), model-only
  checkpoints after listed iterations, written after the iteration's own saves, touching no
  training state, RNG or rollback target. Grid: cumulative iteration c (L2 1–39, then 39 + L3 it)
  in {round(10^(0.05 j))} ∪ {35, 70, 105, …}, i.e. about every 0.05 in log10 R (R grows about
  linearly in c) plus the attempt-matched "Dan rounds" (35 iterations = 143,360 attempts, one
  EI round's 143,840).
- **Stop rule.** A run stops at c = 39 + T_L3, or earlier once the latest evaluated checkpoint is
  at or above the bar and at least 1.1 × the R of the first checkpoint at or above the bar since
  the last one below it. A think-ban stop (exit 3) ends a run there ("not reached by R_stop"). A
  machine crash is resumed from the latest rolling checkpoint (`--resume`), logged.
- **Full arm.** Full seeds whose base is below the bar are rerun with exactly this protocol (L2 +
  long L3, same code and patch); seeds at or above it are "at the bar before RL" and need no run.
  Their existing kkl0 checkpoints are read for the by-product panel only.
- **Optional, last**: one knockout seed continued to 2× the budget, if GPU budget is left.

## Measurement

- **Read** (the GRPO models' standard plain read): `code.thinking.eval` at `f35d3daf` (identical
  eval / sample code to the reads commit `38e12690`), `--sets strict106.jsonl --budgets 64 --arms
  plain --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10`, Lean-judged. n = 64
  attempts per theorem at each checkpoint (two seeds of k 32); n = 256 at each base.
- **Primary y**: E32 = Σ over the 106 of the unbiased pass@32 estimate 1 − C(n−c, 32) / C(n, 32),
  the expected number solved by a k 32 draw; reported as a count and as % of 106.
- **Bar** = median of E32 over the three full bases S1g3_s0–2 (n = 256).
- **Crossing R\*** = the first R at which E32 ≥ bar, interpolated linearly in log10 R between the
  bracketing checkpoints (E32 linear in log R). If never: "not reached by R_max". A base at or
  above the bar: "at the bar before RL" (hollow circle).
- **Secondary (Dan-matched cumulative y)**: Dan rounds r = 1, 2, …: round 1 = the base, round r
  = the checkpoint at c = 35 (r − 1); y_r = theorems solved by any round ≤ r's first k 32 block;
  bar_D = median over the full bases of their single-block count; crossing = first round with
  y_r ≥ bar_D.
- **R** = (6 · trained tokens + 2 · sampled tokens) / (6 · Stage-1 tokens), summed over L2 and L3
  `metrics.jsonl` rows up to the checkpoint (rolled-back iterations count); Stage-1 tokens =
  epochs × (state + completion tokens per epoch) of the arm's own base and pickle
  (`high_rl_passk_20261030`). Prefill, old-log-prob, KL-anchor and canary passes are excluded.
  GPU-hours (GRPO process wall time) are reported beside it.
- **Dan's ladders on our axis**: his per-round tokens (registry rows / round files) with the same
  rule: 6 · fine-tune tokens + 2 · sampled tokens, no prefill, over 6 · his Stage-1 tokens; and
  the map from his "5.6–6.1×" to our R.

## Predictions (written before any model was read)

Probabilities are mine. "Reach" = R\* within the iteration budget.

- **P1, bar.** The full bases' E32 median is between 10 and 50 of 106 (80 %).
- **P2, knockout bases.** Each ko base's E32 is below 2 (85 %); the three ko bases are below every
  full base (90 %).
- **P3, 1 %.** 3 / 3 d1 seeds reach the bar (75 %), with median R\* ≤ 0.3 (60 %).
- **P4, 0.1 %.** ≥ 2 / 3 d01 seeds reach the bar (65 %); median R\* above d1's (70 %, if both
  reach).
- **P5, knockout.** ≤ 1 / 3 ko seeds reaches the bar within the budget (60 %). Unlike Dan's EI,
  about 40 % of the L2 / L3 pool is classical-only, so GRPO rewards σ directly and often; I put
  P(≥ 2 / 3 ko seeds reach) at 30 %.
- **P6, graded.** If ko does not reach, the medians order 1 % < 0.1 % < ko (60 %).
- **P7, compute.** Under our rule, Dan's 16-round ladders are below his own accounting's 5.6–6.1×
  (85 %), because his includes state prefill.

## Reporting

Observed results, interpretation and open hypotheses are kept apart. Every count is also given as
% of 106. The figure follows Dan's layout (`code/ndbench/figures/style.py`), with a by-product
panel of E32 against R per arm and seed. Misses of these predictions are reported.
