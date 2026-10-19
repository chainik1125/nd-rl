---
written_on: 2026-10-08
written_by: agent:claude
---

# Overnight log, 2026-10-08 (Leon asleep; decisions delegated)

Leon (about 1:15 am PDT): "you have my permission to start experiments and make decisions on your own … when I get
back tell me what you chose in a tldr and a summary of what worked and what didn't."

Times are PDT. Each entry is a decision or a result, with its reason.

## Decisions

- **1:20 am.** Gap analysis done ([gap_analysis/README.md](gap_analysis/README.md)). One unit-test theorem was a
  renaming of a textbook72 problem, so I replaced it. textbook72 must never be trained or tuned on, even in a test
  string.
- **1:25 am.** Started building the proof-first hard 7–12-line generator (`hard712/`) from gap-analysis targets
  T1–T8.
  - The gates: the shortest proof must be 7–12 lines and within 1 line of the constructed proof; the targeted
    structure must be necessary; Lean; exclusion against every set.
  - The sealed test split is an unfiltered random sample, drawn before any model-based filtering.

- **2:10 am. Plan review in** ([PLAN_CRITIQUE.md](PLAN_CRITIQUE.md); Opus plus four research agents, about 70 papers).
  I adopted:
  - **Step validity first.** One invalid step ends 60–92% of failed attempts on textbook 7–12s and on 13–20. So I ran
    the review's first experiment: read the RL'd checkpoints with the `logical` arm, which redraws a step the
    environment rejects, and compare with plain sampling at matched tokens.
    - If logical − plain is at least +5 pp at 2,000 tokens per problem, step validity is the binding constraint, and
      the trace work re-centres on checking the policy's own candidates.
    - If it is at most +1 pp, the search-trace plan goes ahead unchanged.
  - **Ladder:** drop e16 (default 0/64/256). e1024 is gated on evidence of need; none is observed so far.
  - **Search traces get a matched no-think control (T0-S)** trained on the same found proofs, so proof-style changes
    don't masquerade as a thinking gain. They also get CPU go/no-go gates, the key one being the share of genuine
    dead ends.
  - **Model-decided effort is deferred, not dropped (Leon asked for it).**
    - An oracle that picks the best level per problem beats the best fixed level by only −1.4 to +2.2 pp pass@1, but
      with 30–65% fewer tokens.
    - A free verify-and-escalate cascade (e0 → e64 → e256, stop at the first Lean-verified proof) matches the oracle
      above 2,000 tokens per problem.
    - The per-step `min_level` labels are still being built; auto-mode training waits on the step-validity result.
  - **Distractor premises dropped.** Textbook problems have none, and the pool already has 1.2 redundant premises
    per problem.
  - **Noted for RL:**
    - the pool's signal is gone: 62–74% of groups were skipped at it78;
    - add a coverage-preserving objective, because GRPO leaves 3.5× as many hard12 test targets at 0/16 as EI;
    - the composition pipeline's 13–15-line composites are in band 18–37% of the time;
    - a STaR-style arm could use the 105k verified EI attempts on disk.
  - **A new textbook-like test set is needed,** because dev58 is now a design set. Candidate: `transfer_long`'s 282
    textbook-schema theorems, which have 0 renaming overlaps with textbook72; schema overlap still to check.
- **2:15 am. GPU 0:** logical arm on 13–20 for graded GRPO, GRPO no-think seeds 0 and 1, GRPO effort seed 1, and EI
  thinking r2 (about 3 h serial). The textbook72 reads follow on GPUs 1–4 once the EI round 3–4 evals finish.

- **2:40 am. Honest-search generator done** ([search_traces/README.md](search_traces/README.md), `79749ce`).
  - **Correct:** 99.76% of K12 proofs found, 0 checker disagreements, Lean 600/600. No reference leakage: the head of
    a line predicts nothing beyond the majority baseline, against +16 pp for v1. Ex falso is 1.6% of dead ends,
    against 71%.
  - **Shallow on K12:** genuine dead ends in 12.9% of blocks (v1 13.4%), and e0's first guess is already viable in
    90.2% of steps. The uncapped record is p99 291 tokens, with 0 blocks over 1,024. So **no e1024**, and e16 stays
    dropped (it is the minimal level in 0.65% of steps).
  - **Decision:** this is cheap, so train it anyway as a clean test of "honest vs staged traces", with a matched
    control:
    - S1g3 (search traces, 0/64/256 at 20/40/40);
    - T0S (no-think on the same proofs);
    - 2 seeds each, on GPUs 1–4 after the EI evals, then reads on hard12 / textbook72 / 13–20.

    Comparisons: S1g3 vs the staged graded T1v1g4; S1g3 vs T0S (the thinking effect); T0S vs T0 (the proof-style
    effect, since the search proofs are on average 1.75 steps shorter than the K12 references).
  - The generator is not ready for harder theorems yet: 12% fail on 13–20 because of drift in reductio contexts. It
    needs a completeness fallback before it runs on hard712.

## Results

- **21–30-line sealed set (2,767 theorems, k 8), final.** pass@1 / pass@8, 95% intervals; RL'd checkpoints at it78
  or r2.

  | model | pass@1 | pass@8 | tokens / attempt |
  |---|---|---|---|
  | Stage-1 T0 | 0.0 | 0.0 | 227 |
  | EI no-think r2 | 0.2 [0.1, 0.3] | 1.0 [0.7, 1.4] | 314 |
  | GRPO no-think s0 / s1 | 0.2 / 0.3 | 0.9 / 0.8 | 300 |
  | **EI thinking r2** | **1.2 [1.0, 1.5]** | **5.2 [4.4, 6.1]** | 1,691 |
  | GRPO effort s0 / s1 | 0.2 / 0.5 | 0.7 / 2.4 | 750 |
  | GRPO token cost | 0.3 | 1.4 | 531 |
  | GRPO graded | 0.8 [0.6, 1.0] | 3.9 [3.3, 4.7] | 1,144 |

  - Paired differences:
    - EI thinking − GRPO graded: +0.4 [+0.2, +0.7] pass@1, +1.3 [+0.4, +2.2] pass@8.
    - Thinking − no-think: EI +1.0 / +4.2; GRPO graded vs no-think +0.6 / +3.1; GRPO effort seed 1 +0.2 / +1.6.
  - **Read:** on the longest proofs EI with thinking is the best model. Thinking is worth more than the choice of RL
    algorithm, and the models that think longest do best.
  - **Interpretation, not tested:** GRPO's shrinking of think blocks may cost it on long horizons.
- **2:00 am.** GPUs 1–4: EI rounds 3 and 4 (thinking at B 256, no-think at B 0) on 13–20 and 21–30, to see whether EI
  keeps climbing out of distribution. These are cheap evals (about 1 h), with no matched GRPO partner beyond it78.
  GPU 0 is kept free for the collapse-audit confirmation.
- **EI keeps improving out of distribution with more rounds; EI without thinking stalls.** 13–20 (986), pass@1 /
  pass@5:

  | round | EI no-think | EI thinking (B 256) |
  |---|---|---|
  | r1 | 5.1 / 12.4 | 6.7 / 19.1 |
  | r2 | 7.7 / 16.8 | 11.4 / 28.6 |
  | r3 | 7.6 / 17.0 | 13.8 / 32.9 |
  | r4 | 8.5 / 19.0 | **16.1 / 38.4** |

  - 21–30, EI thinking: pass@1 / pass@8 of 1.2 / 5.2 (r2), 1.6 / 7.4 (r3) and 1.7 / 7.1 (r4).
  - EI thinking r4 − GRPO graded it78 on 13–20: +4.2 [+2.8, +5.7] pass@1, +10.5 [+7.8, +13.1] pass@5. But r4 is 2×
    GRPO's attempts. GRPO's own 13–20 curve was flat from it50 to it78 (seed 1, about 9–10%).
  - **Decision (3:00 am):** continue the best GRPO run (graded, seed 0) from it78 for 78 more iterations, to match EI
    r4's attempts. `grpo.py` has no optimizer resume, so this is weights only, with a fresh optimizer and curriculum,
    seed 100. It runs on GPU 0 after the logical reads, then reads 13–20, 21–30, hard12 and textbook72.
- **Step validity is a binding constraint (logical arm, pre-registered rule ≥ +5 pp at 2,000 tokens per problem).**
  On 13–20 (986), logical (invalid steps redrawn) − plain:

  | model | pass@1 per attempt | pass@5 | at 1,000 tokens / problem | at 2,000 | at 4,000 |
  |---|---|---|---|---|---|
  | GRPO no-think s0 it78 | +8.5 [+7.3, +9.8] | +11.5 | +7.4 | **+8.6 [+6.9, +10.5]** | +12.1 (clamped) |
  | GRPO graded it78 | +9.1 [+8.1, +10.3] | +16.0 | +9.1 (clamped) | **+4.7 [+3.7, +5.9]** | +6.9 |

  - Absolute, logical arm: graded 21.1% pass@1 and 43.9% pass@5; no-think 19.0% / 28.8%.
  - Read: the rule is met for no-think and is borderline for graded thinking. GRPO effort seed 1, GRPO no-think seed 1
    and EI thinking r2 are pending.
- **About 3:06 am: serrano crashed again; back at 2:08 pm.** Losses and recovery:
  - **Lost:**
    - the four Stage-1 runs (at step about 13.7k of 24k; `train.py` cannot resume, moved to
      `stage1/_crashed_20261008/`);
    - the logical read of EI thinking r2;
    - the GRPO continuation (not yet started);
    - three agents mid-task.
  - **Relaunched at 2:25 pm:** the same Stage-1 runs on GPUs 1–4, and the logical read then the GRPO continuation
    on GPU 0. The agents were resumed from their saved transcripts; their on-disk work survived.
- **Collapse audit (near final, [collapse_audit/README.md](collapse_audit/README.md)).**
  - **Observed:** seed 0's 13–20 failures after it39 are runaway actions, `: : : …` to the 512-token cap.
  - **Mechanism:** between it39 and it50 the policy put probability on `:` wherever it was unsure, and that
    probability grows steeply with the number of hypothesis lines in context. At 14–16 lines it went from 0.045% at
    it39 to 38% at it49 and 84% at it50. Each sampled `:` adds a line, so the effect feeds itself.
  - **Why the pool didn't stop it:** hard12 states almost never reach that length.
  - **After it50:** RL suppressed `:` in short contexts but long-context generation stayed broken, now running away
    on `<nl>` and `<no>` instead.
  - **Seed 1** never developed this. Its updates were orthogonal to the attractor direction.
  - **Ruled out:** curriculum shift, artefacts, the old scoring bug.
  - **Proposed fix:**
    - a long-context canary plus relative clip-low alarms with rollback;
    - a KL anchor on long-context states, or 10–20% long proofs in the pool;
    - lr 5e-5, the script default (the runs used 1.5e-4, 3×).
  - **Confirmation:** E1, a 3-arm restart from it39, about 2.5 GPU-h.
- **2:45 pm. Checked `transfer_long`'s textbook part as a fresh test set: rejected.** Its 282 "textbook" theorems are
  substitution instances of only 6 classic schemas: De Morgan nand-to-or, both distributivity directions and their
  converses, the negated conditional, and Peirce. Those are the structures the dev58-derived generator targets, so the
  set overlaps the design set at the schema level.
  - **Open for Leon:** a genuinely new textbook-style test set is needed, for example the exercise sequents of an
    openly licensed textbook (forall x: Calgary is CC BY 4.0), transcribed as statements only and checked against
    textbook72 classes.
- **Decided: the next RL experiment is EI-based.** EI with thinking scales out of distribution and supports `--arm
  logical`; a round takes about 1.7 GPU-h.
  - Design, 2 × 2: pool (rl_hard12 vs hard712) × sampling (plain vs logical). rl_hard12 with plain sampling already
    exists (`ei_T1v1e` r1–r4), so 3 new runs of 3 rounds each.
  - Starts when the hard712 pool is ready and GPUs free up.
- **Started:** an agent implementing the audit's GRPO safeguards, opt-in:
  - capped-attempt alarm, long-context canary, relative clip-low alarm, rollback on alarm;
  - optimizer-state resume, a long-context KL anchor, batch dumps and a long-proof target mix;
  - the E1 script.

## Afternoon, 2026-10-08 (Leon back)

- **Stage 1 on honest-search proofs is a large win, mostly from proof style.** 2 seeds each, pass@1 with 95%
  intervals:

  | Stage-1 model | hard12 | textbook72 | 13–20 (986) |
  |---|---|---|---|
  | old T0, no think | 42.4 | 17.2 | 0.8 |
  | old staged thinking, graded e256 | 48.7 | 17.1 | 0.7 |
  | **T0S** (no think, search proofs) | **68.4** | **31.6** | **4.7** |
  | **S1g3** (honest-search thinking, B 256) | **75.9** | **41.8** | **7.3** |

  - Proof style, T0S − T0: +26.0 [+22.8, +29.4] hard12, +14.4 [+8.0, +21.4] textbook72, +3.9 [+2.4, +5.5] 13–20.
  - Thinking, S1g3 − T0S: +7.5 [+2.3, +12.6] hard12, +10.3 [+3.7, +17.2] textbook72; +2.6 [−0.7, +6.0] on 13–20,
    not resolved.
  - **Mechanism:** Lean rejections of environment-accepted proofs fall from 36.9% to 14.0% of attempts on hard12,
    and from 55.4% to 17.8% on textbook72. The search builds Lean-clean proofs step by step through the
    environment, so the model learns proofs that check.
  - **Decision (pre-registered rule passed):** honest-search traces and proofs are the default Stage 1 from now on.
- **GRPO vs EI at matched attempts: EI wins.** GRPO graded continued to it117 (≙ EI r3; weights only, fresh
  optimizer) reads 5.2% / 12.8% on 13–20, against 11.9% / 27.9% at it78. EI thinking r3 reads 13.8% / 32.9%.
  - EI r3 − GRPO it117: +8.6 [+7.2, +10.0] pass@1, +20.1 [+17.3, +22.6] pass@5.
  - hard12 stayed at 93.9%. The long-proof failures turned into format errors: empty terms 22.9%, formulas 12.9%,
    box heads 10.1%. This is the same long-context drift the audit describes, in another form.
  - **Decision:** EI is the RL method. E1 stays deferred.
  - My canary relaunch did not take effect (the first launch's training process was already running), so this run
    has no canary.
- **Queue trimmed at Leon's prompt** (see EXPERIMENT_QUEUE.md): GRPO continuation +78 → +39; EI 2×2 → a 1-round
  pilot; E1 deferred; textbook72 logical reads dropped.
- **`ei.py` now accepts an init that already has effort tokens** (it copies it as r0); tested.
- **EI pilot launched (about 5:00 pm PDT), 1 round each, rl_hard12 pool:**
  - GPU 0: from S1g3_s0, plain sampling;
  - GPU 1: from T0S_s0, plain sampling (control);
  - GPU 2: from S1g3_s0, logical sampling.

  Then reads on 13–20, hard12 and textbook72.
  - **Rules:**
    - thinking vs no-think under EI on 13–20 pass@1;
    - logical-sampling EI vs plain-sampling EI, read with plain sampling: extend if ≥ +2 pp;
    - extend arms that improve on their Stage-1 init by ≥ 2 pp.
- **Size sweep:** the 35.6M pair (S1g3 and T0S) is training on GPUs 3–4.
- **5:30 pm. The hard712 pool is ready** ([hard712/README.md](hard712/README.md), `edc223d`).
  - 23,092 problems: pool 21,592, sealed test_1k and dev_500, both drawn uniformly before any model filtering.
  - 90% have a shortest proof of ≥ 9 lines (rl_hard12: 8%); 45% need reductio or a nested negation box (2%);
    Lean 100%; 0 exclusion hits.
  - **Short of target:** 25k problems; biconditionals 0.4%; no-premise 8.8%; case splits 24.8%.
  - **Queued (GPU 0, after the EI pilot):** read the hard712 dev split with EI thinking r4, GRPO graded it78, GRPO
    no-think it78, EI no-think r4, and Stage-1 S1g3 and T0S, to check the difficulty band.
    - Rule: RL on hard712 goes ahead if the best current model's pass@1 on dev is between 10% and 70% (signal), or
      above that with pass@8 < 95%.
- **8:20 pm. Size sweep result: stop at 35.6M (pre-registered rule not met).** Stage 1, honest-search thinking (S1g3)
  vs its no-think control (T0S), pass@1:

  | | 9.6M (2 seeds) | 35.6M (seed 0) |
  |---|---|---|
  | thinking gain, 13–20 (986) | +2.6 [−0.7, +6.0] (seed 0: +3.8) | +1.5 [+0.2, +2.6] |
  | env-rejected steps, 13–20, think / control | 71.4% / 82.0% | 73.5% / 78.8% |
  | thinking gain, hard12 | +7.5 (seed 0: +12.2) | +13.8 |
  | thinking gain, textbook72 | +10.3 | +8.1 |

  - Read: a 3.7× larger model does not raise the out-of-distribution thinking gain or cut invalid steps. Size is
    not the binding constraint at Stage 1. Caveat: one seed, and the same data and step count (24k).
  - The 35.6M evals OOMed at batch 1024 and were re-run at batch 256 (`followup_20261007/evqb.sh`).
- **EI pilot, no-think arm (T0S → EI r1 on rl_hard12):** 13–20 pass@1 20.9% raw (46.7% solved within 8); hard12
  89.0%; textbook72 52.4%. Every earlier model is below this. The new models solve 95–99.6% of the rl_hard12 pool
  in round 1, so the pool is spent for them.
- **8:35 pm. EI pilot (round 1 from the new Stage 1, rl_hard12), 95% intervals.** 13–20 on the 986 items:

  | EI r1 from | 13–20 pass@1 / pass@5 | textbook72 pass@1 | hard12 pass@1 |
  |---|---|---|---|
  | T0S (no think) | 20.8 / 40.8 | 52.4 | 89.0 |
  | S1g3, B 256 | 22.0 / 45.0 | 57.6 | 90.0 |
  | S1g3, logical sampling in EI, B 256 | 22.4 / — | 56.7 | 89.3 |

  - **Compared with the old best EI:**
    - new EI thinking r1 − old EI thinking r4, on 13–20: +5.9 [+4.0, +7.8];
    - new EI thinking r1 − old EI thinking r2, on textbook72: +26.0 [+18.3, +33.9].
  - **Thinking − no-think under EI:**
    - 13–20: +1.3 [−0.8, +3.3] pass@1, +4.2 [+1.0, +7.5] pass@5, +2.3 [+0.3, +4.3] at B 64;
    - textbook72: +5.2, not resolved;
    - hard12: +1.1, not resolved.

    Proof style now carries most of the gain; thinking adds a smaller, mainly pass@5, edge.
  - **Logical sampling inside EI:** +0.3 [−1.0, +1.4], so that arm is dropped (rule: extend only at ≥ +2 pp).
- **hard712 dev difficulty (k 8, pass@1 / solved within 8).** In band, so RL on hard712 goes ahead.
  - EI thinking r4: 39.7 / 75.8
  - S1g3: 37.7 / 72.2
  - GRPO graded it78: 35.9 / 60.0
  - T0S: 30.2 / 59.8
  - EI no-think r4: 28.3 / 54.0
  - GRPO no-think it78: 25.0 / 37.4
- **8:40 pm. EI round 2 launched, 2 × 2: pool × thinking**, each from its pilot r1 checkpoint, one round:
  - pools: a fixed random 5,000 of the hard712 pool, vs continuing on rl_hard12 (control);
  - models: S1g3 thinking vs T0S no-think;
  - after each: reads on 13–20, hard712 dev, textbook72 and hard12. The r1 models are also read on hard712 dev
    (GPU 4) as the starting point.
  - **Rule:** hard712 becomes the RL pool if its arm beats the rl_hard12 control by ≥ 2 pp on 13–20 pass@1, or on
    hard712 dev, for either model.
- **Midnight. EI round 2, pool × thinking** (95% intervals; 13–20 on the 986 items; thinking at B 256 unless
  noted).

  | arm | 13–20 pass@1 / pass@5 | hard712 dev pass@1 | textbook72 pass@1 |
  |---|---|---|---|
  | thinking r1 (start) | 22.0 / 45.0 | 61.2 | 57.6 |
  | thinking r2 on the old pool | 26.2 / 49.5 | 63.0 | 58.2 |
  | **thinking r2 on hard712** | **27.6 / 50.6** | **77.6** | **65.4** |
  | thinking r2 on hard712, B 64 | **30.7 / 55.4** | 76.8 | **67.0** |
  | no-think r2 on the old pool | 23.6 / 45.1 | 51.3 | 53.6 |
  | no-think r2 on hard712 | 24.3 / 46.5 | 74.7 | 62.8 |

  - **hard712 − old pool:**
    - 13–20: +1.4 [+0.0, +2.7] thinking, +0.7 [−0.6, +2.0] no-think;
    - textbook72: +7.2 [+3.9, +10.8] thinking, +9.3 [+3.8, +15.2] no-think (caveat: dev58 informed hard712's
      design);
    - hard712 dev: +14.6 thinking, +23.4 no-think (in distribution).
  - **Thinking − no-think, both on hard712, on 13–20:** +3.2 [+1.1, +5.5] pass@1 and +4.1 pass@5 at B 256; +6.4
    [+4.3, +8.6] pass@1 and +8.9 pass@5 at B 64. Up from +1.3 after r1, so harder training data widens thinking's
    out-of-distribution advantage.
  - **Decision:** hard712 is the RL pool, with thinking as the main line. **Next:** round 3 for both models on a
    fresh, disjoint 5k of hard712 (`pool_s5k_b`), GPUs 2–3.
- **Fair GRPO runs (Leon: "do a fair run, this shouldn't be a question"),** safeguards on (canary + rollback, KL anchor
  β 0.02 to the init, lr 1.5e-4), 39 iterations each:
  - (A) from EI r1 on the same hard712 5k (a), against EI r2 on hard712;
  - (B) from Stage-1 S1g3 on rl_hard12, against EI r1 (re-tests "EI beats GRPO" with the new init).
  - Running on GPUs 4 and 0.
- **New memory:** fairness controls run without asking.

## 2026-10-09: scope reset by Leon. The question is uplift from the thinking traces at matched compute; EI is only the baseline

- Dropped every experiment that does not compare the thinking-trace model with the EI no-think baseline (see
  EXPERIMENT_QUEUE.md).
- **Matched-compute result (13–20, thinking B 64 − baseline, pass@k at equal sampled tokens per problem):**

  | pair | per attempt | 1k | 2k | 4k | 8k | 16k |
  |---|---|---|---|---|---|---|
  | r1, seed 0 | +1.3 | −11.1 | −9.6 | −7.2 | −4.7 | −2.3 |
  | r2, seed 0 | +5.9 | −8.5 | −6.1 | −3.9 | −2.5 | −1.6 |
  | r2, seed 1 | +6.8 | −6.5 | −3.4 | +0.7 | **+4.6** | **+7.6** |
  | r3, seed 0 | +2.9 | −13.9 | −12.6 | −10.7 | −8.8 | −7.1 |

  - textbook72: thinking ahead at ≥ 4k tokens in both seeds (seed 0 +3.5 at 8k; seed 1 +5.4 to +7.5).
  - hard712 dev: behind at low compute, tied from 2k up.
  - **Verdict so far:** clear per-attempt uplift, but no robust uplift at matched compute on 13–20.
- **Next:** the thinking model's own B 0 curve (adaptive budget), the seed-1 round-3 pair, and the figure and note.
