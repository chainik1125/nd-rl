---
written_on: 2026-10-07
written_by: agent:claude
---

# Experiment queue: what is running, queued, held and done

This is the single tracking list for the thinking-traces runs. Update it whenever a job starts, finishes or
changes; Git keeps the history. All paths are under `/scratch/cowifornia/nd-rl/think/runs/` and all times are PDT.

**Matched budget.** One EI round is 5,000 × 32 = 160k attempts, and 39 GRPO iterations are 256 × 16 × 39 =
160k. So EI r1 ↔ GRPO it39 and EI r2 ↔ GRPO it78.

## Running (updated 2026-10-10, 9:30 am PDT)

**Goal since the 2026-10-09 group meeting:** a better GRPO recipe. Leon's answers (2026-10-09, 4:30 pm): GRPO +
verifier = verifier-guided rollouts (redraw steps a Lean-sound check rejects) plus a penalty on each rejected draft;
curriculum 10–12 lines, then 13–15; GRPO on both the thinking (S1g3) and no-think (T0S) models; priorities: GRPO vs
EI in a fair setup, what each change buys, then the model-chosen think budget.

All `grpo_v2/line.sh`, seed 0 unless noted; reads as in the v2 section below.

| job | GPU | ETA | why |
|---|---|---|---|
| `grpo_think_vs`: GRPO + verifier without redraws, thinking | 4 | ~1:30 pm | the verifier redesign (7:09 am entry). One canary rollback at L2 it 3 (P(':') 29× base): back to it 0, lr × 0.5 for the rest of L2 |
| `grpo_nothink_vs`: the same, no thinking | 1 | ~11:30 am | as above; no alarm so far |
| `ei_think`: plain EI, thinking | 0 | ~4 pm | the matched counterpart of plain `grpo_think` |
| `ei_nothink_v` seed 1: EI + verifier, no thinking | 2 | ~2:30 pm | second seed |
| `ei_think` seed 1: plain EI, thinking | 3 | ~7 pm | second seed of the thinking plain pair (`grpo_think` has seeds 0 and 1) |
| `grpo_nothink` then `ei_nothink`, seed 1 | 1, after `grpo_nothink_vs` | ~7:30 pm | second seed of the no-think plain pair |

## Queued: GRPO recipe v2 (`grpo_v2/line.sh`, branch `leon/grpo-v2`)

Launch about 9 pm once the curriculum pools (`pools/curriculum/L2`, `L3`) pass `grpo_v2/preflight.py` and the
review fixes are in (step cap 96 for GRPO and EI, renaming-class preflight, accepted-token normalisation of the
loss). Stages: L2 (10–12 lines) then L3 (13–15), 39 GRPO iterations or one k-32 EI round each. Reads after L3:
13–20 (the 16–20 part is out of distribution) k 16, textbook72 k 64, Charles 173 (report the clean 149) k 64,
hand_proved_14 k 64, 1,000 of 21–30 k 8; plain arm for all, guided arm on 13–20, textbook72 and Charles.
About 78 GPU-hours, so two waves on five GPUs.

| wave | GPU queue | arms |
|---|---|---|
| 1 (~9 pm → ~4–5 am) | one per GPU | seed 0: `grpo_think_v`, `ei_think_v`, `grpo_think`, `grpo_nothink_v`, `ei_nothink_v` |
| 2 (~4–5 am → ~noon) | same GPUs | `ei_think` seed 0; seed 1 of `grpo_think_v`, `ei_think_v`, `grpo_nothink_v`, `ei_nothink_v` |

Then: the model-chosen think budget (`leon/thinking-auto`) on top of the winning recipe.

**Change at 12:46 am, 2026-10-10 (proposed to Leon at 12:25 am; no reply in 30 minutes).** The first finished v2 run,
`grpo_nothink_v` seed 0, collapsed under plain sampling: 13–20 pass@1 4.5% (guided read 19.0%) against 24% for EI
no-think r2; textbook72 50.1% (guided 67.3%); Charles149 40.2%. Training action entropy fell 0.22 → 0.02 while
about 1.7 steps per attempt were still rejected and silently redrawn. Reading (not tested): with redraws an invalid
step no longer fails the attempt, so the only pressure against it is the 0.1 penalty. So the two queued seed-1
GRPO + verifier runs were replaced: GPU 3 runs `grpo_nothink` seed 0 (plain GRPO, no-think, curriculum: the
control that separates the verifier from the curriculum) after `grpo_think` seed 0; GPU 1 runs `grpo_think` seed 1
after `grpo_think_v` seed 0. EI arms unchanged. The verifier recipe needs a redesign before more GPU time.

**4:40 am, 2026-10-10.** The control answered it: plain `grpo_nothink` seed 0 reaches 13–20 pass@1 28.7% (verifier
arm 4.4%), textbook72 68.1% (50.1%), Charles149 59.6% (40.2%), so the verifier recipe, not the curriculum, causes the
collapse. Fairness control queued without asking: `ei_nothink` seed 0 (plain EI, its matched counterpart) on GPU 3
after `grpo_nothink`'s reads (about 5 am → 1 pm). `ei_think` seed 0 stays on GPU 0 after `ei_nothink_v` seed 0
(about 7:20 am → 4 pm).
**Open, needs Leon:** `grpo_think_s1_L3` spent its 3 rollbacks on iterations 1–3 on the think-ban-mass alarm (mass
0.0011–0.0013 against the fixed 1e-3 threshold, already above it at the stage-2 checkpoint; canary flat: NLL 0.2275
vs base 0.2290) and trains at lr × 0.125 from iteration 4. Seed 0 never fired it. Proposed: make that alarm log-only
(canary, grad-norm and clip alarms keep rolling back) and rerun seed-1 L3 from `grpo_think_s1_L2_it39.pt`, about 3 h
on GPU 1. Not done: auto mode refused the change as weakening a safeguard. `grpo_think_v_s0_L3` also lost 2
rollbacks to it (lr × 0.25 from iteration 19). Since then seed 1 read level with seed 0 (13–20 29.4 vs 29.1,
textbook72 72.2 vs 72.3), so the rerun is dropped from the proposal.

**7:09 am, 2026-10-10 (proposed to Leon at 6:40 am; no reply in 30 minutes).** EI + verifier holds up under plain
sampling (13–20 pass@1 29.5 / 32.4 thinking seeds 0 / 1, 28.1 no-think) where GRPO + verifier (redraw) does not
(14.9 / 4.4), so the redesign keeps the checker but drops the redraws: arms `grpo_think_vs` / `grpo_nothink_vs`
(logical arm with `--max_rej 0`: the first rejected step ends the attempt and takes the 0.1 penalty; everything
else as `grpo_*_v`; CPU smoke of both passed). Seed 0: thinking on GPU 4 from 7:09 am (about 1:30 pm), no-think
on GPU 1 after `grpo_think` seed 1's reads (about 7:50 am → 11:30 am). Alternatives left for Leon: redraws with a
10× penalty, or half of each group sampled plainly.

**8:50 am, 2026-10-10.** First fair GRPO vs EI pair (no thinking, seed 0, both plain, same curriculum and attempts),
GRPO − EI: 13–20 pass@1 +1.6 [−0.4, +3.6], pass@5 −3.1 [−5.8, −0.4]; textbook72 +4.5 [−2.1, +11.3]; Charles149
+5.2 [−0.2, +10.6]. The verifier helps EI a little (EI + verifier − EI, Charles149 +3.2 [+0.0, +6.4], 13–20
+1.0 [−0.3, +2.2]). Both no-redraw GRPO runs train cleanly through L2 (training action entropy 0.039 at it 30,
as plain GRPO; no collapse). Tables: [matched_eval/v2_results.md](matched_eval/v2_results.md),
[confusion_tb72.md](matched_eval/confusion_tb72.md), [confusion_charles149.md](matched_eval/confusion_charles149.md).

**9:25 am, 2026-10-10 (proposed to Leon at about 8:50 am; no reply in 30 minutes).** Second seeds of the plain
pairs: GPU 3 runs `ei_think` seed 1 (about 10 h, so the thinking GRPO − EI comparison has two paired seeds as GRPO
does); GPU 1 runs `grpo_nothink` then `ei_nothink`, seed 1, after `grpo_nothink_vs`'s reads (about 11:30 am →
7:30 pm). Pre-flight passed. GPUs 4, 2 and 0 (free about 1:30, 2:30 and 4 pm) stay open for the steps that follow
the no-redraw reads.

**10:05 am, 2026-10-10.** `grpo_nothink_vs` (no redraws) reads: 13–20 pass@1 19.9% (plain GRPO 28.7%, EI 27.2%,
redraw arm 4.4%), textbook72 65.5% (68.1, 63.5), Charles149 57.2% (59.6, 54.4). The 13–20 gap is all on 3-premise
problems: on 1–2 premises the arm is the best one (46.4% vs plain GRPO 41.9%, EI 44.8%), on 3 premises it falls to
12.7% (25.0, 22.5), 43% of those attempts ending with `S:PR position` at the third premise line (premise lines must
come first, as h1, h2, h3 in order); the redraw arm 64%; plain GRPO and EI 0%. The v2 pools have no 3-premise
problem (L2 0–2, L3 0–2; v1's rl_hard12 had 42%), while 13–20 is 78% 3-premise, Charles149 19%, textbook72 5%.
Plain GRPO and EI do not need them (3-premise pass@1 v1 vs v2: GRPO thinking 25.4 vs 25.3, EI 26.7 vs 27.4);
only the verifier arms lose the third premise line. Why is open; the guided read (about 10:30 am) logs the
checker's rejection causes. `grpo_think_v` shows no PR-position failures. **Proposed to Leon:** rebuild L2 and L3
with about a third 3-premise problems (same lengths and pre-flight), then rerun the no-redraw arm with plain GRPO and
EI on the new pools.

**11:10 am, 2026-10-10.** `grpo_think_vs` (thinking, no redraws) seed 0, 13–20 plain: pass@1 35.2% against plain GRPO
29.1% (+6.1 [+4.8, +7.4], seed 0 paired) and EI + verifier 29.5% (+5.7 [+4.0, +7.4]); pass@5 58.6% (+6.2, +5.1);
16–20 lines +7.1 and +6.5. No PR-position failures (3-premise problems 31.6% vs 25.3 and 27.4), so the third-premise
failure is specific to the no-think model. One seed; problem-level intervals only. Scoping of the 3-premise options
(subagent, read-only): pfgen never builds > 2 premises; test_13_20 is lemma composites (`test_21_30/build_composites.py`),
3 premises by construction; no non-test pool has certified 3-premise rows at 10–15 lines. Options A (composites from
pieces no test composite uses), B (pfgen change), C (rewrite pool rows: goal antecedents become premises).
**11:15 am (proposed at about 10:45 am; no reply in 30 minutes):** C's build started (CPU only; subagent;
`runs/pools/curriculum_3p/`). **Proposed at 11:15 am:** seed 1 of `grpo_think_vs` on GPU 4 after seed 0's reads; the
no-think mechanism test (`grpo_nothink_vs` + C rows at a third of each iteration, `--targets_extra`) on GPU 2 after
`ei_nothink_v` seed 1. A waits for Leon's yes (composite family in training; reuse of unused builder pieces).

## Proposed (not run; waiting for Leon, 30-min auto-run rule applies)

From [RL_RESULTS.md](RL_RESULTS.md#proposals-not-run-need-leons-call):
1. **Second seeds of the graded arm and of one budget + token cost.** About 3 GPU-h each, plus 1.5 GPU-h of evals.
   Seed spread is currently as large as the differences between variants.
2. **Restart the seed-0 effort run from it39**, once with the same settings and once at half the learning rate,
   to see whether the it48 clip-low spike and the 13–20 drop recur.

## Done 2026-10-07 (write-up: [RL_RESULTS.md](RL_RESULTS.md); tables: [matched_eval/final_tables.md](matched_eval/final_tables.md))

- **Matched EI vs GRPO** (r1/r2 vs it39/it78, both arms, seed 0) on hard12_test, textbook72 and t1320 (986 valid
  items).
- **Seed-1 GRPO pair**, plus the seed-1 13–20 learning curve (it10–75).
- **Single-budget thinking GRPO** (`grpo_T1v1_b256`) and the token-cost variant (`grpo_T1v1_b256_lp02`), both to
  it78.
- **Budget forcing (s1-style).** Forcing any minimum think length lowers accuracy.
- **Locating the 13–20 drop in the seed-0 effort run.** The gain is gone by it49 and never recovers.
- **Graded-length traces** (`graded_traces/`), Stage-1 T1v1g4, and GRPO from it, plain and with LCPO. All evals
  are done.
- **LCPO reward** implemented (`grpo.py`, 4608e60).
- **Process fix:** eval queues now count an eval as done only when every budget's rows file exists
  (`followup_20261007/lib.sh` `evdone`). `summary.json` is rewritten after each budget, so it is not enough.

## Held

- **hard12r** (≤12-line targets that stay hard after RL). Pilot done, 0.3 GPU-h; see
  [pools/hard12r/README.md](pools/hard12r/README.md).
  - RL'd no-think GRPO (it35) solves 7–8 of 8 on 88.5% of fresh ≤12-line generator candidates, and on 85.9%
    of hard12's spare in-band targets. Only 2.7% of fresh candidates land at 1–2 of 8.
  - Structural features barely move this (3–9% at ≤2/8 across goal types, depth and length).
  - So ≤12-line headroom after RL is close to exhausted. Any test split must be an unfiltered random sample.
- **A harder pool selected by model-independent features.** Revisit on the same condition.
- **EI with smaller rounds** (k 8 per target, 40k attempts per round, a fine-tune after each). This gives a
  checkpoint every 10 GRPO iterations, so EI and GRPO can be compared below one full round. Decide after the
  r1 vs it39 evals: if EI is already near ceiling at r1, the finer curve adds little. Cost: about 4 h no-think
  and 6 h thinking, on two GPUs.

## Done (details in the linked notes)

- Stage 1 (4 arms × 3 seeds), goal-stack and filler controls, and the hard12 seed control: see
  [STAGE1_STATS.md](STAGE1_STATS.md) and [`stage1_stats/`](stage1_stats/).
- Adaptivity across RL checkpoints: [ADAPTIVITY.md](ADAPTIVITY.md).
- GRPO scoring bug (fixed in 7b082b7) and overlong-masking drift: [GRPO_COLLAPSE_AUDIT.md](GRPO_COLLAPSE_AUDIT.md).
- EI on hard12: both arms stopped on 2026-10-07 at Leon's call (stop early). Checkpoints r1–r4 are kept (r5 for
  no-think).
- The 21–30-line sealed test set: [test_21_30/README.md](test_21_30/README.md).
