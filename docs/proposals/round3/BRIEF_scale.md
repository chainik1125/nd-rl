# Run brief: round3-run4a / round3-run4b — does "or nothing" survive model size?

Two executors on the second host, in parallel, each with its own run id, branch, budget and
pre-registration: **run4a** (reductio, branch `dan_round3-run4a`) and **run4b** (depth-3,
branch `dan_round3-run4b`), both from `origin/dan_novelty` or its merged successor. Role:
executor. `AGENT_POLICY.md` governs; designs are suggestions. Read first: `run5.md`,
`review_round2-run5.md`, `ignition.md`, `review_ignition.md`, and §4 of
`~/nd-rl/docs/proposals/2026-09-18-proposals-round-3.md`. The two executors do not share
pods or files; each uploads its own bucket directory.

## Question

Every result so far is on a 3.2M-parameter model (4 layers, d = 256). At ≈ 25M (8 layers,
d = 512) and ≈ 85M (12 layers, d = 768; the sprint's size), trained on the same cap-6 f = 0
sets: how often does a Stage-1 draw generalise the pattern before RL, at what rate, and how
much does expert iteration add beyond the base's reach — targets acquired that the base
does not reach at pass@10⁴ — and does that fraction grow or shrink with size?

## Why it matters

Clause (2)'s "or nothing" is observed only on zero-rate draws. If larger bases are never
zero-rate, the clause stops applying at scale and the SPAR question's answer is "RL elicits;
bigger bases have more to elicit". If the EI-only fraction grows with size, that is the first
evidence in this setting of RL composing more as the base gets stronger — the opposite of the
elicitation reading. The campaign's next-step list named scale as the untested axis.

## Gate 0: `preregistration/round3-run4a.md` / `round3-run4b.md`

Each executor commits its own file before its first pod, with the expectations for its
pattern.

### Pre-registered expectations (numbers)

- **E1 non-zero draws (≥ 1 pattern hit in 600k pre-RL samples).** run4a reductio: **1–2 of 3**
  at 25M, **3 of 3** at 85M (4 / 11 at 3.2M). run4b depth-3: **3 of 3** at both sizes
  (10 / 16 at 3.2M).
- **E2 per-sample base rates** rise by ≥ 10× per size step at the median over draws.
- **E3 zero-rate draws** at any size stay at 0 on the required pool after 8 rounds (nothing to
  train on).
- **E4 EI-only fraction** (acquired required targets not base-reachable at 10⁴ ÷ acquired).
  run4a: falls with size, from run 5's 45 / 51 = 0.88 at 3.2M to **≤ 0.5 at 85M**, while the
  8-line stratum is acquired by ≥ 2 of 3 85M draws within 8 rounds. run4b: plateau
  acquisition rises above the 3.2M's 0.30–0.36 to **≥ 0.5** at 85M on the required@8 pool,
  and every 85M draw ignites by round 3.
- **E5 model quality gate.** Held-out greedy (5,000 cap-6 theorems) ≥ 0.90 at 25M and ≥ 0.93
  at 85M (3.2M: 0.86–0.89); if a size misses the gate after one hyperparameter retry
  (lr 3e-4 → 1e-4, steps 6000 → 12000), report it as untrained rather than tuning further.

### What would falsify the standing rule

- E4 reversed — the EI-only fraction grows with size — kills the elicitation framing at
  scale; RL would be adding more, not less, as the base improves.
- E1 with reductio still zero-rate in ≥ 1 of 3 draws at 85M keeps "or nothing" alive at the
  sprint's size; with 3 / 3 non-zero at 85M and E3 vacuous, the clause is restated as a
  small-model regime.
- E3 violated (a zero-rate large draw ignites on the required pool) contradicts "never trains
  ⇒ never ignites" — check the arm's training logs before believing it.

## Design (suggestion)

**Sizes.** `train.py --mode abs --cap 6 --bs 128` with `--n_layer 8 --d 512 --n_head 8`
(≈ 25M) and `--n_layer 12 --d 768 --n_head 12` (≈ 85M); lr 3e-4 (min 3e-5), warmup 500,
steps 6000 first; seeds 0–2 per size. Sets: run4a `train_reductio_f0.jsonl`; run4b
`train_depth3_f0_a1.jsonl`. Held-out greedy at the end of Stage-1 is the E5 gate. Print the
parameter count. The 3.2M row comes from existing files (run 5 for reductio; the ignition
study for depth-3) and is not re-run.

**Pools.** run4a: `data/p2/targets_reductio_req.jsonl` (300) + `transfer_reductio_req.jsonl`.
run4b: `data/r3_1/targets_depth3_req8.jsonl` from run 1 if committed, else the 300 shortest
of `targets_depth3.jsonl` (disclose; then the EI-only count is still defined but "required"
is not).

**Per draw.** Pre-RL `coverage.py --k 2000 --temperature 0.8` over all 300 targets (600k
samples; batch 1024 at 85M); classify zero / non-zero. EI `expert_iter.py --k 32 --rounds 8
--temperature 0.8 --retain 20000` (batch reduced as needed at 85M; `--ft_steps` and
`--ft_lr` scaled down to 3e-5 for 85M — record what was used). Frozen control `--no_train`
at equal attempts. For every igniting draw, pass@10⁴ on the required targets from its Stage-1
checkpoint (3·10⁶ samples) for the EI-only count; for non-igniting draws the pre-RL sample
suffices.

**Report** per size × draw: parameters, held-out greedy, pre-RL hits and rate, ignition
round, acquisition per stratum (run4a 7 / 8 / 9 / 10 lines; run4b required@8 with the
9-line-alternative field), frozen, base-reachable at 10⁴, EI-only fraction; the 3.2M row
alongside.

## Necessity

run4a inherits run 5's reviewer-certified reductio pool. run4b inherits run 1's required@8
certificate (restricted search fails at bound 8; the 9–10-line flat alternative recorded per
target) or discloses the fallback pool as not required.

## Controls and seeds

Three Stage-1 seeds per size and pattern (6 new models per executor); frozen at equal
attempts; the 3.2M row from existing reviewed files. Differences quoted only across ≥ 2 seeds.

## Budget and stop rule

A100 80 GB pods at $1.59/h (two per executor). Per executor: Stage-1 3 × 25M × ≈ 10 min +
3 × 85M × ≈ 25 min ≈ 2 h; pre-RL 6 × 600k samples ≈ 6 × 0.5 h (85M slower) ≈ 3 h; EI 6 arms
× ≈ 0.5–1 h ≈ 4.5 h; frozen 6 × 0.3 h ≈ 2 h; pass@10⁴ for ≈ 4 igniting draws × 1.5 h ≈ 6 h;
total ≈ 17 A100-hours ≈ **$27** per executor; ceiling $50 each. Stop at $45 or when the 85M
cell has 3 draws through 8 rounds; if short, drop the 25M pass@10⁴ runs first, then the 25M
frozen arms. Kill switch re-armed; pods deleted when pulled. Check `rpbalance` before each
pod; both executors share the balance.

## Deliverables

Per executor: `preregistration/round3-run4{a,b}.md`; `run4{a,b}.md` (≤ 400 words + figures:
non-zero fraction and EI-only fraction vs parameters, with the 3.2M point); `numbers.md`
§Round 3 run 4{a,b}; `artifacts/r3_4{a,b}/summary.json`; `STATUS.md` `RUN3-4{a,b} DONE <UTC>`;
bucket `hf://buckets/dan-pandori/nd-rl/round3-run4{a,b}/{artifacts,ckpts,data}`;
`touch ~/runs/round3-run4{a,b}/executor.done`.
