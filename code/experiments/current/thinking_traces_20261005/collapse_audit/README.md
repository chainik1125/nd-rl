---
written_on: 2026-10-08
written_by: agent:claude
---

# Seed-0 thinking GRPO: why 13–20 accuracy collapsed at it48–50

**Run:** `runs/restart3_grpofix/grpo_T1v1e` (seed 0, effort levels, lr 1.5e-4, dual clip 3,
`--overlong fail`; code `8cc9a2c` on `leon/thinking`). Control: `grpo_T1v1e_s1` (seed 1, same
recipe). The audit is read-only, CPU only (8 threads, `nice`). Probe scripts are in this folder; large
outputs are in `/scratch/cowifornia/nd-rl/think/runs/collapse_audit_tf/`. Nothing was trained,
evaluated or committed.

## Summary

**What broke (observed).**
- Seed 0's 13–20 failures after it39 are mostly runaway generation, not wrong proofs.
- From it49 an action runs to the 512-token cap without `<eos>` in 31–78% of B256 attempts, and
  in 5–36% of B0 attempts. At it39 it was 0.8% and 0.4%; seed 1 never exceeds 9.1%.
- At it49 the run-on text is `: : : : ...`, the colon token repeated.

**The mechanism (observed by teacher forcing).**
- Between it39 and it50 the policy learned to put probability on `:` wherever it is unsure.
  That probability grows steeply with the number of hypothesis lines (`n : F`) in the context.
  On 13–20 reference steps it is:

| hypothesis lines in context | it39 | it49 | it50 |
|---|---|---|---|
| 10–11 | 0.001% | 2.5% | 22% |
| 14–16 | 0.045% | 38% | 84% |

- Each sampled `:` adds to that count, so the effect feeds itself.
- Hard12 states almost never reach that length (53 of about 35,000 scored positions have 12 or
  more lines). The RL data barely constrain the region where it bites.

**After it50 (observed).**
- The attractor reached in-distribution contexts at it50: the it51 batch fell to 69.2% pass@1.
- RL then suppressed `:` within two iterations, but overshot. Legitimate `:` inside think lines
  went from ≈0 to 0.37–0.66 nats of NLL.
- Long-context generation stayed broken, now running into `<nl>` and `<no>` instead of `:`.
  Hard12 recovered to 94.5%; 13–20 stayed at 3.7–6.4% pass@1 at B256 through it78.

**Why seed 0 and not seed 1 (partly open).**
- Both Stage-1 inits start equally far from the attractor.
- Seed 0's realised updates over it39→it49 carried a consistent component along "raise `:` in
  long contexts" (cosine +0.043, first-order +4.1 nats). Seed 1's updates had the same size and
  were orthogonal to it (−0.006).
- On-policy gradients of CPU proxy batches do not reproduce that component. The loss term that
  drove it is not identified.

**Ruled out:** curriculum or batch-composition shift; checkpoint or eval artefacts; the old
scoring bug; step cap; a single-token logit bump. The weight change is distributed over all six
blocks.

**Confidence.**
- High that the collapse is this long-context attractor, created by seed 0's updates in it40–50
  and never repaired because the pool has no long states.
- Moderate on the overshoot explaining the persistence.
- Low on which loss term produced the drift.

**Proposals.** A GPU restart from it39 with per-iteration dumps and a cheap long-context canary
(§5). As fixes: an alarm that works under `--overlong fail`, the canary, and a KL anchor or a
small share of long proofs in the pool (§5).

## 1. What happened (observed)

### 1.1 Outcome mix on 13–20 (986 valid items, k = 8, stored eval rows; `eval_rows.py`)

Share of all attempts. "Truncated" means an action ran to the eval's 512-token cap without `<eos>`.
"Formula" and "unbound" are environment rejections of a malformed formula and an unknown name.

| checkpoint | budget | pass@1 | truncated | formula | unbound | done, Lean-rejected |
|---|---|---|---|---|---|---|
| s0 it39 | B256 | 11.9% | 0.8% | 0.3% | 31.1% | 34.4% |
| s0 it49 | B256 | 3.3% | **78.1%** | 5.3% | 1.3% | 5.0% |
| s0 it52 | B256 | 6.3% | **30.9%** | 25.6% | 3.0% | 8.0% |
| s0 it60 | B256 | 3.7% | **76.9%** | 4.2% | 0.9% | 5.4% |
| s0 it70 | B256 | 6.4% | **60.8%** | 5.2% | 0.5% | 6.8% |
| s0 it78 | B256 | 6.2% | **46.2%** | 9.1% | 0.8% | 6.4% |
| s0 it39 | B0 | 5.8% | 0.4% | 0.9% | 17.6% | 24.0% |
| s0 it49 | B0 | 3.4% | 12.7% | 24.9% | 12.5% | 10.7% |
| s0 it52 | B0 | 4.2% | 5.5% | 33.8% | 7.0% | 10.4% |
| s0 it60 | B0 | 5.0% | 35.8% | 14.9% | 7.3% | 15.1% |
| s0 it78 | B0 | 6.0% | 13.9% | 24.4% | 6.1% | 14.9% |
| s1 it39 | B256 | 7.9% | 1.3% | 1.4% | 21.3% | 28.7% |
| s1 it50 | B256 | 10.6% | 1.1% | 19.4% | 16.3% | 21.0% |
| s1 it55 | B256 | 8.5% | 9.1% | 12.3% | 13.1% | 23.5% |
| s1 it78 | B256 | 10.1% | 1.0% | 16.7% | 20.3% | 22.4% |
| s1 it78 | B0 | 4.7% | 0.0% | 7.7% | 10.2% | 21.3% |
| no-think s0 it78 | B0 | 10.6% | 0.0% | 0.4% | 14.4% | 24.1% |

Step-cap failures are 0.00–0.03% everywhere. Full outcome mixes for every checkpoint are in
`eval_rows.json`.

- Truncation is new in seed 0 after it39. Across it49–it78 it is the dominant failure, at 31–78% of B256
  attempts, against at most 9.1% at any seed-1 checkpoint. B0 is hit too, at 5.5–35.8%.
- It starts early in the proof. The accepted steps before the truncated draw peak at 5–7, and the
  share is 72–83% of B256 attempts at every reference length from 13 to 20 (it49; 69–85% at it60).
  So it is not a step index beyond 12.
- In distribution it is rare but not absent. On hard12_test B256, truncation is 0.13% at it39 and
  1.60% at it78 in seed 0, against 0.11% and 0.04% in seed 1.

### 1.2 What a failing attempt looks like (`sample_truncated.py`, CPU, eval settings)

At **it49**, 10 of 12 B256 attempts on the four items with the most it49 truncations end
truncated. The think block runs to its 320-token cap with colon salad (`⊢ False := : ⊢ ( ¬ Q ) ⊢ (
¬ Q ) : ⊢ Q <nl> : ( : : ( ¬ Q : : ( ¬ Q ) : ...`), and the action is then 512 tokens of
`: : : : ...`. At B0 there is no think block, but the same token takes over the action:
`have n29 : : ( R : : : : ( S : : : ; <eos>` (fail:S:formula) and `have n26 : ( Q ∧ ( ¬ Q ) ) := : ;
<eos>` (fail:S:unbound).

At **it78**, the run-on content is mostly think-syntax tokens instead: `<nl>`, `<no>`, `:=`,
`have`, and stray `</think>` inside actions. A think block breaks off inside a formula (`⊢ ( R ∨
R ) := ⊢ ( <nl> ⊢ ( ( <nl> <nl> <nl>`), and the action continues
`<nl> <nl> <no> := <nl> <no> n20 <nl> ...` to the cap. In 15 attempts: 6 truncated, 4 "action head", 2 formula,
1 "have name", 1 colon and 1 solved.

### 1.3 Training metrics (seed 0 vs seed 1, `metrics.jsonl`)

| iter | s0 clip-low | s0 neg_r3 | s0 ratio max | s0 batch pass@1 (B0) | s1 clip-low |
|---|---|---|---|---|---|
| 40–45 | 0.0016–0.0021 | ≤0.0001 | 7–23 | 87.8–91.0% | 0.0014–0.0019 |
| 46 | 0.0019 | 0.0002 | 10 | 91.4% (86.5%) | 0.0018 |
| 47 | 0.0029 | 0.0005 | 23 | 89.8% (82.5%) | 0.0020 |
| 48 | 0.0066 | 0.0014 | 58 | 90.0% (84.1%) | 0.0019 |
| 49 | 0.0142 | 0.0023 | 281 | 87.8% (87.1%) | 0.0024 |
| 50 | 0.0197 | 0.0043 | 901 | 87.7% (83.6%) | 0.0030 |
| 51 | 0.0079 | 0.0010 | 120 | **69.2% (59.3%)** | 0.0021 |
| 52 | 0.0141 | 0.0016 | 2349 | 86.7% (80.6%) | 0.0022 |
| 53 | 0.0032 | 0.0002 | 15 | 80.7% (71.6%) | 0.0025 |

- Batch pass@1 at iteration t is sampled with the policy left after iteration t − 1.
- Within-iteration policy movement grew for four iterations at constant lr, with mean grad norms of
  0.02–0.04 (max 0.09 at it52).
  The policy left after it50 failed 31% of in-distribution attempts (it51 batch). Two iterations
  later the batch was back at 86.7%.
- Curriculum and batch composition did not move. Over it40–50, skip rate stayed at 45–56% (s1: 46–51%) and the
  per-budget learnability means drifted by less than 0.01 per iteration. Think tokens per attempt stayed at 78–98.
  The skip-rate dips at it51 (29%) and it53 (30%) come after the damaging update.
- **Monitoring blind spot.** `overlong_frac` counts masked attempts, so under `--overlong fail` it
  is identically 0 and the overlong alarm can never fire. The run logged 0.0% throughout, while
  a CPU resample of the policy that sampled the it51 batch truncates 13.5% of B256 attempts (§2.3).

## 2. Mechanism

### 2.1 A `:` attractor that grows with the number of hypothesis lines (teacher forcing; `teacher_force.py`, `tf_length.py`)

**Setup.**
- Reference proofs of 200 valid 13–20 items (3,518 steps) and 200 hard12_test items (2,146 steps), rendered into
  (state, action) pairs by the Stage-1 data path (`progtrace.trace_for` → `data.trace_pairs`, with
  progtrace/1 think blocks).
- Names are shifted by a per-item base in [0, 32], as the sampler does.
- Each step is scored in two layouts: B0 (`state <e0> <think> </think>` → action) and B256
  (`state <e256> <think>` → think, then action).
- Scoring uses the sampler's masked distribution at T = 1.

**Measure.** P(`:`) at action positions whose reference token is *not* `:`, binned by the number of
`:` tokens in the decoder input. In B0 that count is the number of hypothesis lines in the state.

B0 layout, 13–20 references (number of scored positions in brackets on the first row):

| checkpoint | <8 `:` [27,361] | 8–9 [12,165] | 10–11 [6,396] | 12–13 [2,042] | 14–16 [283] |
|---|---|---|---|---|---|
| s0 it39 | 0.000% | 0.000% | 0.001% | 0.004% | 0.045% |
| s0 it49 | ≤0.003% | 0.13% | 2.5% | 13.3% | 38.0% |
| s0 it50 | ≤0.37% | 4.6% | 22.3% | 56.9% | 84.4% |
| s0 it52 | 0.000% | 0.000% | 0.002% | 0.030% | 0.31% |
| s0 it78 | 0.000% | 0.001% | 0.001% | 0.004% | 0.008% |
| s1 it49 | 0.000% | 0.000% | 0.000% | 0.001% | 0.002% |

The same measure on hard12 references:

| checkpoint | <8 `:` | 8–9 | 10–11 [1,430] | 12–13 [53] |
|---|---|---|---|---|
| s0 it49 | ≤0.006% | 0.025% | 0.20% | 1.7% |
| s0 it50 | 0.16–0.49% | 2.2% | 8.9% | 16.9% |
| s0 it52, it78; s1 it49 | ≤0.001% | ≤0.001% | 0.000% | ≤0.001% |

Hard12 states almost never hold 12 or more hypothesis lines: 53 of about 35,000 scored positions do.

**Observed.**
- Between it39 and it50, seed 0 learned to put probability on `:` wherever it is unsure. That
  probability rises steeply with the number of hypothesis lines in context.
- It also rises with the colons a think block adds. B256 (13–20) at it49 reaches 27% in the
  14–16 bin. At it49, `:` is the argmax at 1.0% of all non-`:` action positions on 13–20, against
  0.003% on hard12.
- Seed 1 never shows it: at most 0.003% (B0) and 0.006% (B256) in any bin at it39, it49, it52 or it78.
- Apart from `:`, the two seeds' teacher-forced NLLs are alike (`tf_report.json`).
  - At it49 seed 0's action NLL rises only in the classes where `:` takes mass: `:=`, `(`, names and
    rule words.
  - `n`-name NLL (B0) rises from 0.20 to 0.28 nats at it49, then is back at 0.20 at it52 and 0.24 at
    it78. Seed 1 stays at 0.29 throughout. So high-numbered names are not the persistent problem.

**Interpretation.**
- It is an attractor because the trigger is the number of `:` in context. Every sampled `:` adds one, so a
  stray `:` makes the next one more likely, and the attempt ends in `: : : ...` until the action cap.
- B256 is hit harder than B0 (78% against 13% truncated at it49) because a think block adds colons
  (`n38 : R ⊢ ...`, `fun ( n : A )`) on top of a long state.
- Long proofs are hit because only they put the model past about 10 hypothesis lines.

### 2.2 The weights: distributed, and pushed there by the updates (`param_deltas.py`, `patch_probe.py`, `update_probe.py`)

**Measure.** All three probes use the 990 13–20 reference steps with at least 10 `:` in the state
(B0 layout). The colon score is c = mean log P(`:`) at non-`:` targets on those steps.

- **Not one token's logit.** Patching the `:` row of the head or the embedding moves P(`:`) by
  under 0.5 pp in either direction (it39 ↔ it50).
  - No tensor group transplanted from it50 into it39 produces it: at most 0.046%, against 32.4% at
    it50. Groups are the embedding, the head, `ln_f`, and each block's attention and MLP.
  - Reverting any single group of it50 to it39 or it52 only partly repairs it. The best single group
    (block-0 attention from it52; block-2 attention from it39) leaves 19–21%; most leave 22–32%.
  - The change is spread over all six blocks and acts jointly.
  - Raw row deltas agree. `:` is the most-moved embedding row and the second most-moved head row in
    both alarm windows (it49→50, it50→52), but those rows alone do nothing.
- **First order, not a cliff.** Along θ_A + α(θ_B − θ_A), c rises almost linearly with α.

| window | c at A → B | first-order g·d | cos(g, d) | ‖d‖ | P(`:`) |
|---|---|---|---|---|---|
| s0 it39 → it49 | −15.38 → −8.63 (+6.75) | +4.09 | +0.043 | 1.970 | 0.003% → 6.2% |
| s0 it49 → it50 | −8.63 → −4.17 (+4.46) | +4.73 | +0.065 | 0.464 | 6.2% → 32.4% |
| s0 it50 → it52 | −4.17 → −16.25 (−12.07) | −11.22 | −0.083 | 0.860 | 32.4% → 0.018% |
| s1 it39 → it49 | −15.93 → −15.89 (+0.03) | −0.44 | −0.006 | 1.971 | 0.000% → 0.000% |
| s1 it49 → it50 | −15.89 → −15.94 (−0.04) | −0.07 | −0.005 | 0.335 | 0.000% → 0.000% |

**Observed.**
- Both seeds moved the same distance over it39→it49: ‖d‖ is 1.970 against 1.971, as AdamW's
  roughly sign-sized steps make likely.
- Seed 0's steps had a consistent component along "raise `:` in long contexts", and seed 1's were
  orthogonal to it.
- The single it50 update was 39% longer than seed 1's it50 update and more aligned. It added
  4.5 nats, two-thirds of what the previous ten iterations added together.
- Once it showed up in distribution, the it51–52 updates removed it faster than it was built:
  −12.1 nats in two iterations.

### 2.3 What the training batches contained (`colon_loops.py`, CPU resample of the pool)

192 attempts per checkpoint: 48 rl_hard12 targets × 4, B256, GRPO's sampling (T 1.0,
max_action 256, max_steps 48). The reward is the environment verdict.

| checkpoint | attempts with a run of ≥ 8 `:` | of those, finished | env-done | truncated |
|---|---|---|---|---|
| s0 it39 | 0.0% | — | 97.4% | 0% |
| s0 it49 | 1.6% | 0 | 96.9% | 2.1% |
| s0 it50 | 13.5% | 0 | 76.6% | 13.5% |
| s1 it49 | 0.0% | — | 97.9% | 0% |

`:`-runs appear in the training distribution only once the attractor is already strong (it49–50).
They always fail, so they are never rewarded. Successful attempts' think blocks are 0.8–0.9% `:` at
it39, it49 and it50 alike. So "degenerate thinks reinforced by success" is ruled out.

### 2.3b Which loss component? Not resolved on CPU (`grad_decomp.py`)

**Setup.**
- A proxy GRPO batch per checkpoint: 32 (target, budget) pairs from rl_hard12 × 16 attempts,
  with GRPO's sampling settings. The reward is the environment verdict, the advantage is Dr. GRPO,
  and items come from `grpo.group_items`.
- The on-policy loss gradient (`grpo.policy_loss`, r = 1) is split by token into
  positive- or negative-advantage × think or action.
- Each part is projected on g, the gradient of the colon score at that checkpoint.
- "Raw" is the plain descent step. "Sign-like" divides each coordinate by the batch's total
  |gradient|, a crude stand-in for Adam's preconditioner. Positive means the step would raise `:`.

| checkpoint | A>0 think | A>0 action | A<0 think | A<0 action | total raw | total sign-like |
|---|---|---|---|---|---|---|
| s0 it39 | +11.7k | −7.2k | −0.3k | −10.2k | −0.25 | −6.0k |
| s0 it49 | +7.7k | −12.0k | −87.9k | +49.0k | −5.02 | −43.1k |
| s1 it49 | −47.0k | +41.8k | −51.8k | +58.9k | +0.08 | +1.9k |

(sign-like projections; the s0 it49 batch held 4 truncated `:`-loop attempts, which make up
most of its A<0 think part.)

**Observed.**
- On these proxy batches, the on-policy gradient does not reproduce the realised drift. At
  s0 it39 and it49 its total would *lower* `:` in long contexts.
- The total is a small residue of large, opposing parts, and its sign differs between checkpoints
  and seeds.
- Positive-advantage think tokens are the only part that raises `:` in both seed-0 checkpoints
  and lowers it in seed 1.

**Interpretation.** This is suggestive, not established. One 32-group batch is far smaller than
the 10 × 256 groups that built the drift. Adam's momentum and history-based preconditioner are not
modelled, and neither are the clipped off-policy minibatches 2–4 (clipping was 2% of tokens at
it49–50). The direction of the realised updates is a measurement (§2.2). Which tokens drove it
is open (§6).

### 2.4 After it52: the symptom is gone, the long-context damage is not

**Observed.**
- **Clean contexts recover, mostly.** By it52 the `:` attractor at clean teacher-forced contexts is
  gone (table in §2.1), and hard12 is back (it52 batch 86.7%; it78 hard12_test 94.5%).
- **Some long-context loss remains.** Seed 0 stays worse than seed 1 at it78 in the longest
  contexts (`tf_bins.py`):

| 13–20 references at it78 | seed 0 | seed 1 |
|---|---|---|
| B0 action NLL, ≥14 `:` in input | 0.48 nats | 0.36 nats |
| B256 action NLL, ≥14 `:` | 0.29 nats | 0.16 nats |
| B256 think NLL | 0.40 nats | 0.22 nats |

- **Free running is still broken.** Truncation on 13–20 stays at 31–77% of B256 attempts
  (it52–78). The run-on tokens are now think syntax: `<nl>`, `<no>`, `:=` and `</think>` (§1.2),
  where the it49 runs were `:`.
- **Other error types.** On hard12, s0 it78 truncates 1.6% of B256 attempts against 0.04% for
  seed 1. On 13–20 B256 it gains "action head" errors at 11.0% (0.1% at it39).

**The repair overshot (observed, `tf_ratio.py`, `teacher_force.py report`).**
- Over it50→it52, legitimate `:` tokens *inside think blocks* took 35% of the log-prob the update
  removed on 13–20 B256 references and 42% on hard12. `<nl>` took 15–16%.
- Think-block NLL on legitimate `:` targets:

| set | s0 it39 | s0 it52 | s0 it78 | s1 it39–it78 |
|---|---|---|---|---|
| 13–20 references | 0.0008 nats | 0.37 nats | 0.66 nats | ≤0.009 nats |
| hard12 references | 0.0002 nats | 0.39 nats | 0.24 nats | ≤0.001 nats |

- In actions, `:` stays cheap (0.003 nats at it52 and it78).
- The it78 samples break exactly there: `have n38 <nl> <nl> <no>`, and `have n22 <nl> have n22 : (
  exact <nl>`, where a `:` must follow the name.

**Interpretation.**
- RL repaired the `:` logit where its own samples exposed it, by suppressing `:` in think lines
  generally.
- That broke think-line structure in long contexts, which now runs into `<nl>` and `<no>` and
  spills into the action.
- With no state beyond about 12 hypothesis lines in the pool, nothing in training measures or
  repairs that region.
- The colon score also creeps back up after it60. In ≥12-colon states it is −13.84 at it78 against
  −15.57 for seed 1, a 12× higher P(`:`).

### 2.5 Seed 1 and the other runs

- **Seed 0's Stage-1 init is not more prone.** In ≥12-colon states the colon score is −15.40 for
  seed 0 and −14.60 for seed 1 at it0.
  - By it39 seed 0 had drifted up to −14.54, a P(`:`) 18× seed 1's (0.0092% against 0.0005%).
  - It then jumped to −5.45 at it49 (16.3%).
  - Seed 1 stayed between −15.6 and −14.6 through it78.
- **Seed 1 has its own milder drift that does not cascade.** Its 13–20 formula errors rise from
  1.4% to 16.7% (B256, it39→it78).
  - Sampled failures put out-of-scope low names in formula positions (`have n38 : ( ( n3 n2 ;`,
    `have n13 : n7 n5 ;`). Its top wrong argmax under teacher forcing is `n1`, 13–14% of misses.
  - Each such error ends the attempt locally. Seed 0's favourite wrong token is also its own
    trigger, so it compounds into runaway actions.
- **Other GRPO thinking runs show related long-context errors.** All are seed-0 runs at lr 1.5e-4. The one-budget
  runs start from T1v1_s0; the graded run starts from T1v1g4_s0. All gain long-context errors on
  13–20 by it78:

| run (B256) | truncated, it39 → it78 | formula errors, it39 → it78 |
|---|---|---|
| one budget | 0.0% → 4.7% | 1.0% → 14.3% |
| one budget + token cost | 0.1% → 0.0% | 2.4% → 23.1% |
| graded + LCPO | 1.5% → 12.4% | 0.1% → 10.2% |

- **EI does not drift.** EI thinking stays at 0.0–0.5% truncated and 0.1–0.9% formula errors.

## 3. Hypotheses

| | hypothesis | verdict | evidence |
|---|---|---|---|
| H1 | curriculum or batch-composition shift at it40–50 | **ruled out as the trigger** | §1.3: skip rate 45–56%, learnability means move <0.01 per iteration, per-budget pass and think length match seed 1's ranges; the only outliers (it51, it53) follow the it50 update |
| H2 | action or format drift that only bites in long proofs | **confirmed, in a specific form** | runaway actions (truncation 31–78% of B256, 5–36% of B0) from a context-dependent token attractor (§2.1); not step cap (≤0.03%), not "exact not last" (it falls to 0.0%), not high-numbered names or step index > 12 as such (truncation peaks at accepted step 5–7; name NLL moves with `:` only) |
| H3 | think-block degeneration in long contexts | **confirmed as the B256 route, not the whole story** | think blocks fill to the 320 cap with `:` salad (attempts with a force-closed B256 block: 7.6% at it39 → 78.5% at it49), but B0 truncates and mis-parses too (12.7% truncated, 24.9% formula at it49) |
| H4 | one large update on negative-advantage tokens pushed down a needed token | **partly: large updates yes, push-down no** | the attractor was built by updates (§2.2, first-order, 4.5 nats in the it50 update alone), but it *raises* `:` rather than lowering a needed token, it is spread over all blocks, and it accumulated over it40–50 (cos +0.043 over ten iterations) rather than in one step |
| H5 | artefacts (checkpoints, eval settings) | **ruled out** | alarm checkpoints carry `grpo_iter` 49/50/52 and are written after the iteration's optimizer steps (`grpo.py` saves after the update loop); every 13–20 eval used the same args (k 8, T 0.8, max_action 512, max_steps 96, line forcing); `sample.py`, `eval.py`, `ckpt.py` and the vendored env are unchanged between the eval commits `38b7c4b`, `cc1d448` and HEAD; the CPU probe reproduces the truncation from the it49 and it78 checkpoints |
| H6 (new) | the failure lives where the RL pool has no support, so RL cannot see or repair it | **confirmed** | the attractor scales with hypothesis lines in context; hard12 states almost never exceed 11 (§2.1); RL removed it at it51–52 once it reached in-distribution contexts but long-context generation stayed broken to it78 (§2.4) |
| H7 (new) | the scoring bug of 7b082b7 again | **ruled out** | seed-0 think banned mass 2.8e-5–2.1e-4 over all 78 iterations (alarm threshold 1e-3); `nonfinite` 0 in every iteration |
| H8 (new) | the overlong alarm should have caught it | **a monitoring bug** | `overlong_frac` counts *masked* attempts; with `--overlong fail` it is always 0, so the 5% overlong alarm cannot fire. A CPU resample of the it50 policy (the it51 sampler) truncates 13.5% of B256 attempts (§2.3) |

## 4. Most likely cause, and how sure

**Cause.** The likely cause is unconstrained drift of the policy in long contexts, a region the
RL pool never shows it.

1. **Build-up (it40–50).** Over roughly it40–50 (part of it before it39, see §2.5), seed 0's GRPO
   updates moved the model along a direction that raises `:` wherever the model is unsure.
   - The effect scales with the number of hypothesis lines in context.
   - Its cost on hard12 was close to zero until it50, because hard12 states rarely reach 10 or
     more lines.
   - A `:` in the context is also its own trigger, so on 13–20 states it is a runaway attractor.
     In the free-running sampler that means endless `:` until the action cap.
2. **In-distribution failure and repair (it50–52).** When the drift finally reached
   in-distribution contexts (it50; the it51 batch fell to 69.2%), RL removed it in two iterations.
   It did so by suppressing `:` inside think lines broadly, an overshoot that left long-context
   think lines broken.
3. **No recovery (it52–78).** Since then nothing in the pool measures long contexts, so 13–20
   never recovered while hard12 rose to 94.5%.

The clip-low spike was not the cause. It is a symptom of the drift's last, fastest phase (it47–50)
and of the repair (it52).

**Confidence.**

| claim | confidence | basis |
|---|---|---|
| The 13–20 collapse is this attractor and its aftermath, not curriculum, artefact or scoring | high | directly measured: eval rows, CPU samples, teacher forcing, weight interpolation; seed 1 is a clean negative control |
| It persists because the pool lacks long states and the repair overshot | moderate | the overshoot and the it78 failure text are observed; that they *cause* the persistent truncation is an interpretation |
| Why seed 0's updates aligned with the `:` direction | low, open | on-policy proxy gradients on CPU do not reproduce it (§2.3b); candidates in §6 |

## 5. Proposed confirmation experiment and fix

### Confirmation (GPU, not run)

**E1. Restart seed 0 from it39 for 16 iterations, three arms.**
- **Arms:**
  - (a) the same settings;
  - (b) lr 7.5e-5;
  - (c) the same settings plus the long-context KL anchor below.
- **Logged every iteration:**
  - the colon score and long-state NLL canary (below; seconds on GPU);
  - a checkpoint;
  - the full training batch (prompts, generated ids, masks, advantages, old log-probs);
  - the AdamW state (the run did not save optimizer state, so the restart re-runs warmup with fresh
    moments; this is a known deviation).
- **At the end:** 13–20 B0 and B256 evals.
- **Predictions:**
  - If the drift is a property of seed 0's it39 policy under this recipe, arm (a) raises the colon
    score again by several nats by the it49-equivalent. The dumped batches and Adam state then allow
    an exact decomposition of the realised update by token class, which this audit could not do on
    CPU.
  - If (a) does not drift, the it40–50 trajectory was a chance event, and the fix is the guards
    below rather than a bias correction.
  - (c) should stay flat in every case.
- **Cost:** about 2 GPU-min per iteration × 16 ≈ 35 GPU-min per arm, plus about 15 GPU-min per
  13–20 eval. About 2.5 GPU-h for three arms.

**E2. No GPU.**
- Run the canary below over the existing checkpoints of the other thinking variants
  (`grpo_T1v1_b256*`, `grpo_T1v1g4e*`), generalised to "the most-probable wrong token's mass".
  That tests whether their it78 formula and truncation rise is the same kind of drift.
- About 1 CPU-min per checkpoint with this folder's scripts.

### Fix

1. **Monitoring (do first; low risk).**
   - **Repair the overlong alarm.** Log the share of attempts that a cap ended (truncated, step
     cap, context) whatever `--overlong` is, and alarm on it, per budget.
   - **Add a long-context canary.** Every iteration, teacher-force about 1,000 reference steps from
     long proofs. Draw them from a generator pool that is disjoint from the sealed 13–20 and 21–30
     sets (`pools/long_cands.jsonl` and `long2_cands.jsonl` need an exclusion check first), and
     restrict to states with 10 or more hypothesis lines. Log the action NLL and the largest mean
     probability any single non-target token gets.
     - In ≥12-line states, a threshold of 5× the it0 P(`:`) would have fired by it39: 0.0092%
       against 0.0011%, 8×. Seed 1 never exceeds 0.0018% in this sweep, so it would not fire there.
     - The 6.2% at it49 would have fired any threshold.
   - **Make clip-low relative.** Alarm at 2× the running median: it48's 0.0066 is 3.5× the it40–47
     median (0.0019), one iteration before the absolute 1% alarm.
   - **Roll back on alarm.** Restore the last rolling checkpoint and halve lr, instead of only
     saving.
2. **Prevention (needs E1 to confirm).**
   - **A KL anchor on long contexts.** Add β · KL(π_ref ‖ π_θ), with π_ref the it0 or Stage-1
     model, on the canary's teacher-forced long-state steps at each update. The penalty sits
     exactly where RL has no signal. It costs one extra forward on about 1–2k steps per minibatch.
   - **Alternatively, put long proofs in the pool.** Give 10–20% of groups to 13–16-line generated
     proofs, so the attractor is penalised where it appears. This changes what "out of
     distribution" means for the 13–20 read; 21–30 would become the OOD tier.
   - **Lower lr.** Use 5e-5, the script's default (the run used 3× that). The drift was first-order
     in the update, so this slows it by about 3× and buys canary time. It does not remove it.
   - **Not recommended:** a sampler-side repetition ban. It hides the symptom, and the attractor
     already moved from `:` to `<nl>` and `<no>` once.

## 6. Open questions

- **Which loss component carried the `:` direction?**
  - Positive-advantage think tokens are the only part that raises `:` in both seed-0 proxy batches
    and lowers it in seed 1's (§2.3b).
  - Seed 0's think blocks are also longer than seed 1's over it40–50: about 85, 130 and 140 think
    tokens per attempt at B16, B64 and B256, against about 65, 90 and 100.
  - A one-batch CPU proxy cannot resolve this; E1's dumps can.
- **Is the it78 `<nl>`/`<no>` runaway caused by the `:` overshoot?** The two co-occur and the break
  points match, but this is not proven. It could be a separate long-context failure of the same
  drift.
- **Would the 13–20 gain have come back with more iterations?** The colon score creeps back up
  after it60 (§2.4), which suggests not without a fix.

## 7. Files and checks

**Probe scripts** (this folder; run from the repo root with `ND_OFFLINE=1 code/.venv/bin/python -m
code.experiments.current.thinking_traces_20261005.collapse_audit.<name>`):

| script | what it does |
|---|---|
| `eval_rows.py` | outcome mix of the stored eval rows → `eval_rows.json` |
| `sample_truncated.py` | prints failing draws (CPU sampler, eval settings) |
| `colon_loops.py` | `:`-run frequency at training settings |
| `param_deltas.py` | per-tensor and per-row weight changes |
| `teacher_force.py` | `score` writes `tf_<ckpt>.npz` to `/scratch/cowifornia/nd-rl/think/runs/collapse_audit_tf/`; `report` writes `tf_report.json` |
| `tf_length.py`, `tf_bins.py`, `tf_ratio.py` | binned and paired reads of those arrays |
| `patch_probe.py` | tensor-group transplant and repair |
| `update_probe.py` | first-order projection and interpolation |
| `grad_decomp.py` | proxy-batch gradient decomposition; log in the scratch dir |

**Checks run.**
- Every number above comes from these scripts or from stored metrics and eval rows.
- Teacher-forced checkpoints: s0 it0 is in the colon-score sweep only; full arrays exist for s0 it39,
  49, 50, 52 and 78, and s1 it39, 49, 52 and 78.
- `make -C code check` was not run. No library code was changed.

**Limits.**
- CPU samples are small (12–192 attempts each) and are meant to show the text, not to estimate rates.
- The teacher-forced think blocks are progtrace/1 traces, not the RL'd model's own thinking.
- The proxy batches use the environment verdict instead of Lean as reward.
