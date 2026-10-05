---
written_on: 2026-10-03
written_by: agent:claude
---

# mcts-a: PUCT search with a learned value does not beat sampling on group C at matched wall clock (gate FAIL); at the end of pretraining it helps on the long pool

Status: completed and reviewed 2026-10-03. No hard-constraint violation and nothing quarantined. **MCTS-A GATE: FAIL**,
recomputed by the reviewer, and proposal 22's pre-registered falsifier fires (Δ ≤ 0 on 3 of 3 seeds). All 90 solved
cells, the paired differences and the exploratory 10× read reproduce on the reviewer's own code. 6,700 / 6,700 search
proofs and 2,333 / 2,333 re-checked sampled proofs pass Lean 4 core, and the negative controls fail. The reviewer asks
for six rewordings, applied below. One sentence is **not supported as stated**: "a value trained on that policy's
rollouts cannot point to [group C's rare steps]". It is a hypothesis; nothing measured it. Consequence: `mcts-b`
(MCTS as the RL expert) is gated on this result and should not run as specified.

Question (proposal 22, Phases 0 + A): at equal wall clock on the same GPU, does PUCT tree search over proof states,
guided by a value head learned on the frozen trunk, solve group-C theorems (solved by neither the base nor EI r8 in
`trajectory`) that k 256 sampling from the same frozen policy does not? Proposal:
[`docs/proposals/2026-10-02-mcts.md`](../../docs/proposals/2026-10-02-mcts.md); brief
[`BRIEF_mcts-a.md`](../../docs/proposals/state-env/BRIEF_mcts-a.md).

## Configuration

- **Repository.** Fork `dan-pandori/nd-takehome`, branch `dan_mcts-a`.
  - Pre-registration `preregistration/mcts-a.md`, commit `339d0876` (Fri 23:13:52Z), 20 s before the first pod (mc-0,
    23:14:12Z). Its header says "~23:50"; `log.md` corrects it.
  - Addendum 1 (tuning grid widened to K 2 / 4 × T 0.8 / 1.0): `9dee9bef` (00:12:42Z), before any evaluation read-out.
  - Bug fix `a6a108f6` (00:31Z): `mcts.DEFAULT['max_action']` was 64, not the pre-registered 256. Runs made with it are
    in `artifacts/mcts/invalid_maxaction64/` and are not used.
  - Gate commit `01341eca` (02:52Z). Addendum 2 (exploratory 10× read) `a9a0452a` (04:29Z), before running it. DONE
    `e288991e` (05:27Z).
  - Review [`review_mcts-a.md`](https://github.com/dan-pandori/nd-takehome/blob/5e70afbd/review_mcts-a.md):
    [`b6f8b2be`](https://github.com/dan-pandori/nd-takehome/commit/b6f8b2be) §Recount (phase 1; blindness was partial,
    disclosed: the gate line was in a commit subject and in Addendum 2),
    [`5e70afbd`](https://github.com/dan-pandori/nd-takehome/commit/5e70afbd) §Compare / §Verdict. Reviewer scripts in
    `review/mcts-a-recount/` (own ND → Lean translator, own pruner, own bootstrap).
- **Timeline.** Executor Fri 2026-10-02 23:01 – Sat 05:27 UTC; reviewer 05:28–05:42.
- **Model for every number:** `trajectory`'s best-cap12 seeds s0–s2: `best_model.ALiBiGPT` 6 × 384, **9,560,832
  params**, `lean_staten`, trained from scratch on K12 (155,000 generator proofs, cap 12). Both checkpoints frozen
  throughout; no policy training.
  - **pend** = `stage1_best12_s{S}_b1200.pt` (end of pretraining);
  - **r8** = `la_T1_best12_s{S}_r8.pt` (T1 EI ladder, round 8), from `hf://buckets/dan-pandori/nd-rl/trajectory/ckpts/tj/`.
- **Built** (Phase 0): `mcts.py` (PUCT over proof states, AlphaProof-style c(s), state-deduplicated samples,
  progressive sampling, many trees per GPU batch) and `value_head.py` (MLP on the frozen trunk: solvable logit plus
  γ-discounted steps-to-go). Value data: rollouts k 16, T 1.0 on all 4,495 `rl_targets` + 1,500 K12 theorems, per
  checkpoint (`mcts_value_data.py`, `mcts_value_train.py`; 30 epochs, batch 4,096).
- **Arms** (each search job's budget is the matching sampling job's wall clock on the same pool, checkpoint and A40):
  - **sample:** `state_eval.py`, k 256, T 0.8, `max_action` 512, `max_steps` 96, batch 2,048, sample seed 2 (fresh).
  - **prior:** PUCT, prior only, K 2 / T 1.0.
  - **value:** PUCT + value, K 8 / T 1.5. Search `max_action` 256, depth ≤ 96.
  - Tuning on `tune200` (184 held-out `rl_targets` with `L_true` ≥ 9; s0 pend only) chose the configs by the
    pre-registered rule. **The prior arm's winner (K 2) is one of the configs Addendum 1 added on the strength of a
    number (105 / 184) that the `max_action` 64 bug produced**; at 256 the original K 8 / T 1.0 prior config solves
    143 / 184, above sampling's 140. The value arm's winner (K 8 / T 1.5) is a pre-registered config, so the gate is
    unaffected, and no evaluation pool was read before the change. *[reviewed: item 5]*
- **Pools:** group C per seed (36 / 35 / 28, rebuilt exactly by the reviewer from `trajectory`'s k 256 files),
  textbook72, holdout250, rrQ100 (100 of `transfer_long_rr600` `L_true` 13–16, `random.Random(20261002)`), long2 (21).
  Splits are disjoint by renaming class; K12 shares one textbook72 theorem (inherited, in no group-C set).
- **Checker: Lean alone** (`lean_gate.gate` on the literal `lean_seq` text). No pre-2026-09-27 number is compared.
- **Gate (pre-registered):** Δ_s = value − sample on group C at r8; PASS iff Δ_s ≥ 3 and Δ_s > |sample −
  `trajectory`'s seed-1 count| on ≥ 2 of 3 seeds.
- **Pods:** mc-0 … mc-5, NVIDIA A40 at $0.49/h, **17.67 pod-hours, $8.66** of a $15 / 30 h budget (`podbudget
  mcts-a`; `~/podhours.log`). Pre-registration estimate ≈ $6–8 for ≈ 12–14 GPU-hours. All pods deleted.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/mcts-a/`: `ckpts/mcts/value_s{S}_{ck}.pt`; `artifacts/mcts/`
  (per-theorem `.jsonl` read-outs, logs, `vdata/`); `data/mcts/`.
- **Reproduce.** `python3 mcts_analysis.py` → `artifacts/mcts/analysis.md`, `summary.json`;
  `mcts_termsize.py` → `termsize.json`. Executor's figure: `figures/mcts_a.png` on the fork branch (not copied; the
  review checked its "still climbing" reading only weakly).

## Observations

Observed results, in the reviewer's wording where it differs from the executor's. Every number is on the models above,
judged by Lean alone.

**Solved, s0 / s1 / s2** (group C n = 36 / 35 / 28):

| pool | ck | sample | PUCT prior | PUCT + value |
|---|---|---|---|---|
| **group C (gate)** | r8 | 4 / 2 / 4 | 2 / 1 / 3 | 2 / 2 / 3 |
| group C | pend | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 1 / 1 |
| textbook72 | r8 | 49 / 50 / 55 | 44 / 44 / 54 | 47 / 49 / 57 |
| textbook72 | pend | 32 / 32 / 36 | 29 / 34 / 34 | 35 / 37 / 37 |
| holdout250 | r8 | 239 / 236 / 239 | 239 / 236 / 240 | 240 / 239 / 240 |
| holdout250 | pend | 199 / 207 / 204 | 215 / 212 / 212 | 206 / 216 / 203 |
| rrQ100 | r8 | 96 / 93 / 92 | 96 / 94 / 88 | 96 / 96 / 90 |
| rrQ100 | pend | 59 / 65 / 60 | 76 / 69 / 66 | 78 / 75 / 74 |
| long2 | r8 | 21 / 19 / 19 | 20 / 14 / 17 | 21 / 18 / 19 |
| long2 | pend | 8 / 5 / 4 | 9 / 3 / 5 | 12 / 4 / 6 |

**Gate: FAIL.** Δ_s = −2 / 0 / −1; `trajectory`'s seed-1 counts 3 / 1 / 1. No seed passes, and the falsifier
(Δ ≤ 0 on 3 / 3) fires. Paired, mean over seeds with a stratified-bootstrap 95 % interval over theorems:
value − sample **−1.00 [−3.00, +1.33]**, prior − sample −1.33 [−3.67, +1.00]. The arms solve *different* theorems:
the union of the three on C at r8 is 7 / 3 / 5, and on s0 value and sampling share none. At ≈ 0.1 solves per theorem,
which theorems fall is dominated by draw noise; the pre-registered per-seed MDD (≈ 6–7) is the right frame.

**Exploratory 10× read (Addendum 2; not part of the gate).** Group C at r8, sampling k 2,560 (seed 3) against PUCT +
value at that read's wall clock: **10 / 7 / 5 vs 9 / 5 / 7** (value-only 1 / 3 / 3, sample-only 2 / 5 / 1), mean
−0.33 [−2.67, +2.0]. Level. Addendum 2 gave ≈ 25 % to "value ≥ 3 ahead on ≥ 2 seeds"; observed on 0.

**Where search helps: the end-of-pretraining checkpoint on the long pool.**
- pend rrQ100: value − sample **+19 / +10 / +14** (mean +14.3, CI +9.7 to +19.3, executor's; the reviewer reproduced
  the per-seed values but did not re-bootstrap this pool). That is ≈ 44 % of what 8 EI rounds add on this pool by
  sampling (+32.3).
- **Of that +14.3, the value adds +5.3 [+0.7, +10.0] over the prior-only search**; the rest is search without the
  value. On pend holdout250 the value arm is *below* prior (value − prior −4.7 [−9.3, −0.3]); there most of the gain
  is search itself (prior − sample +9.7). *[reviewed: item 4]*
- pend textbook72 value − sample +3.0 [0, +6.0]; pend long2 +1.7 (−1 on s1). "Still climbing" at the end of the
  budget reproduces only weakly (solves in the last 10 % of budget: 1 / 7 / 3).
- **At r8, PUCT + value is within ±3 of sampling on every pool** (holdout250 and rrQ100 reach +3 on s1). The prior-only
  arm is not: it is −5 / −6 on textbook72, −5 on long2 (s1) and −4 on rrQ100 (s2), outside the pre-registered ±3–4.
  *[reviewed: item 1]*

**Value calibration.** Held-out tenth of `rl_targets` + K12 by name hash: AUC pend 0.770 / 0.744 / 0.765, r8 0.816 /
0.791 / 0.794; steps-to-go Spearman 0.69–0.83; Brier below the constant on 6 / 6. **The pre-registered AUC 0.80–0.92
was missed on 5 of 6 heads.** The held-out theorems are `rl_targets`, on which r8 was EI-trained, and 89 % of the r8
value-data rollouts are accepted (42–47 % at pend). This calibration therefore does not speak to group-C states, where
sampling's per-attempt success rate is ≈ 0.05 %. The reviewer recomputed s0 pend and s0 r8 on CPU (AUC and Brier
identical, Spearman within 0.0006). *[reviewed: item 2]*

**Search on group C barely terminates.** At r8, summed over seeds, the search arms sent 6 (prior) and 7 (value) proofs
to Lean in 367 s per arm, all accepted; sampling sent up to 5,264 finished attempts in the same time. Where search
did find a group-C proof, its hardest step (lowest prior, the pre-registered definition) sat at depth ≥ 2 in 8 / 9
value solves, at median prior rank 2. (`numbers.md`'s search-statistics table defines the hardest step by lowest
log π instead, which deviates from the pre-registration; under log π the root's name token wins on r8 rrQ100.)
*[reviewed: item 6]*

**Proof length.** Searched proofs are not shorter. Against sampling's shortest proof per theorem (a comparison that
favours sampling), search is +0.2 to +4.0 term-size nodes longer on average, most on r8 long2 (+2.8 to +4.0) and
rrQ100 (+2.3 to +2.9). The two or three group-C overlaps go the other way, too few to read.

**Lean re-check (reviewer).** Every solved tree's stored literal text: prior 3,303 / 3,303, value 3,376 / 3,376, x10
21 / 21. Sampling stores ND strings, so one proof per solved theorem per job plus every group-C proof went through the
reviewer's own translator: 2,269 / 2,269 (x10 64 / 64). Controls: 0 / 300 recorded rejects and 0 / 200 goal-to-`False`
swaps accepted. No forbidden tactic; axioms ⊆ {propext, Classical.choice, Quot.sound}. 99 % of the 97,704 counted
sampled proofs were not individually re-checked.

**Compute** (per-job files; summed over 3 seeds; matched on wall clock, so GPU-seconds = wall clock per job on one A40):

| item | GPU-s | generated tokens | Lean checks | GPU util |
|---|---|---|---|---|
| read-outs, sample (r8 / pend) | 5,281 / 2,895 | 109.5 M / 70.8 M | ≤ 268,861 / ≤ 226,224 finished attempts (upper bound; the gate de-duplicates) | 63 % |
| read-outs, PUCT prior | 5,283 / 2,898 | 41.5 M / 25.0 M | 1,203 / 968 | 53 % |
| read-outs, PUCT + value | 5,260 / 2,900 | 63.8 M / 41.4 M | 1,242 / 991 | 72 % |
| tuning (17 runs × 466 s, s0 pend) | 7,922 | — | — | — |
| value data (6 checkpoints, 95,920 rollouts each) | 8,929 | 111.3 M | 456,677 | — |
| exploratory 10× (sample + value) | 7,210 | — | — | — |

- No arm exceeds 1.25× its comparator: search / sample wall 0.907–1.005 (≤ 0.84 % overrun); search actions and tokens
  0.21–1.10× sampling's. Search is the lower-throughput arm at matched time, not a bigger spender.
- Value-head training: 1,710–3,360 steps per head, 0.23–0.46 M states. No policy training.
- Peak memory: search ≤ 10.7 GB in the read-outs, **13.2 GB counting the x10 run**; sampling ≤ 29.9 GB.
  *[reviewed: item 6]*

**Expectations against outcome** (pre-registration, scored by the reviewer): r8 group-C value 4–8 (point 5) → 2 / 2 / 3,
**missed on all 3 seeds**; gate ≈ 45 % PASS, central +2 to +4 → FAIL, −2 / 0 / −1; value AUC 0.80–0.92 → missed on 5
of 6; prior within ±4 of sample on textbook72 → −5 / −6 on s0, s1; pend rrQ100 gain 0–10 → +19 / +10 / +14, above the
range on 2 seeds (favourable); sampling GPU util 70–95 % → 63 %. Hits: pend group C, prior vs sample on C within ±2,
Brier, steps-to-go Spearman, hard-step depth and rank.

**Interpretation, labelled as such.** The executor reads it as: search pays where the frozen policy is unreliable over
many steps — the gap EI already closes — and not where the needed step is rare. The reviewer finds this supported on
rrQ100 only (pend gains elsewhere are mixed). The further reading that on group C "a value trained on that policy's
rollouts cannot point to" the rare steps is **a hypothesis, not a measurement**: no value prediction on group-C states
exists in the artefacts. The indirect support is that the r8 value was trained and calibrated almost entirely on
states the policy already solves. *[reviewed: item 3]*

## Limitations and next step

- **Statistical power on group C is low.** At ≈ 0.1 solve probability per theorem the per-seed MDD is ≈ 6–7, and the
  arms' solved sets barely overlap. The gate's FAIL is a failure to find a large effect, consistent with no effect; it
  does not bound a small one.
- **Tuning.** The prior arm's config came from the grid extension motivated by an invalid (max_action 64) number. Tuning
  ran on one seed and one checkpoint (s0 pend).
- **Truncation above policy.** Sampling attempts ending `truncated` (an action > 512 tokens): mean 0.19 %, above 0.1 %
  in 15 of 30 read-out jobs (max 0.67 %, s0 pend rrQ100). This is `trajectory`'s protocol, held on purpose; such
  actions are likely runaway decodes. Search truncation at 256: prior mean 0.03 % (max 0.27 %), value ≤ 0.03 %.
- **Records.** Registry `ckpt_md5` is null for 5 of 6 checkpoints (the x10 logs print the r8 md5s). The per-theorem
  `gen_tokens` field in the search jsonl is padded (`sampled × max_action`); run-level totals are real tokens. Same-pod
  placement per (seed, checkpoint) is by design, not verified per job.
- A duplicate s0 pend draw (`dup_s0_pend_mc0/`) is quarantined; its counts equal the counted copy's.
- **Next step (reviewer).**
  1. **Score the r8 value heads on group-C states directly**: from the x10 sampling read (≈ 253k attempts, ≈ 21
     accepted proofs across seeds), AUC of on-track vs off-track partial proofs, per seed. AUC ≈ 0.5 confirms the
     executor's reading; a clearly higher AUC points at the search budget, not the value, and would justify a value
     trained on group-C-like data.
  2. Any rerun of the gate needs several independent draws per arm, or a hard pool of ≥ 150 theorems, before calling a
     search-vs-sampling difference on group C.
  3. `mcts-b` (Phase B) is gated on this run; by its brief it does not create pods when the gate fails.
