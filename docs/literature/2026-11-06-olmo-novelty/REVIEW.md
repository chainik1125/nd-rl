---
written_on: 2026-11-07
written_by: "agent:claude (executor, run olmo-novelty)"
papers:
  - yueDoesReinforcementLearning2025
  - liuProRLProlongedReinforcement2025
  - huScalingLLMReinforcement2025
  - huBroRLScalingReinforcement2025
  - wuInvisibleLeashWhy2025
  - matsutaniRLSqueezesSFT2025
  - yangWhatOutcomeRL2026
  - yangWhenPassCrossovers2026
  - olmoOlmo32025
  - pyatkinGeneralizingVerifiableInstruction2025
  - zhangReplayFailuresSuccesses2025
  - weiVerifierInducedSupportReshaping2026
  - yoonDiagnosingCorrectingReward2026
  - sunBaseRolloutsRL2026
  - suSciIFBenchmarkingScientific2026
  - agrawalGEPAReflectivePrompt2025
  - zhangWhenRewardSuite2026
  - helffLLMsGamingVerifiers2026
  - wangBaseModelsCan2026
  - wenReinforcementLearningVerifiable2025
  - tasaltiDoesCoTPasskReally2026
  - dragoiBeyondPasskBreadthDepth2025
  - xuPhantomGainsAuditing2026
  - chengRevisitingReinforcementLearning2025
  - zhaiDoesRLExpand2026
  - yuanUnderstandingDiversityCollapse2026
  - zhouWhenRLVRShrinks2026
  - tangRingZeroScalingZero2026
  - bangeraRLVRSharpensPencil2026
  - bharadwajTracingEvalAwarenessEmergence2026
---

# How novel are our Olmo create-vs-elicit results? (run `olmo-novelty`, 2026-11-07)

Models (`claims_C1-C8.md`): `dpo` = Olmo-3-32B-Think-DPO (32 B; base → Think SFT → DPO); `s0100` / `s0750` /
`s2300` = Olmo-3.1-32B-Think RL steps; C1 = Olmo-3-1025-7B base vs Olmo-3.1-7B-RL-Zero-Math. No proofs are judged.
200 sources screened (176 new to the earlier reviews), 23 in depth, then a fresh search against every "novel" label.
Bracketed ids are ledger quotes checked in their sources (748 / 748; 0 / 735 negative controls). "Hand" = blinded
Claude subagent readers plus the executor's confirmation (C1: the reviewer's six reader subagents), not humans.

## Bottom line

1. **Novel as far as we found, on instruction following (IF) only:** C3's per-prompt IFBench reach at k = 256 (≈ 20 %
   of 218 prompts, RL side confirmed on fresh draws, hand-gated) and C4's gaming audit of the successes that decide
   it. Neighbours: aggregate IFBench pass@k to 1,024 [IF-36], per-prompt IF coverage at k = 32 [IF-47], lone-win
   audits in physics [IL-99], a Fail→Pass quality drop at k ≈ 1 [IF-60].
2. **C1 replicates Yue et al.'s reach result under RL's checker** (10 of 17 resolutions sound by hand) and disputes
   the Olmo 3 report's pass@32 reading [EX-1] on the 3.1 successor run. "Nothing lost" is weak: a locked 1–3 % minority
   [EX-15, EX-16] predicts ≈ 1–3 reverse problems among 83 (our arithmetic), though its 7B pairs show no supported loss
   [EX-14].
3. **C5's token-cap half and C8 test 1 replicate the finishing mechanism** (the base "closes most of the finishing gap itself" [IL-96];
   here 1 of 3 geometry reach prompts). The brevity hint carries cue elicitation [GG-56] and IFBench prompt
   optimisation [NV-7] to reach prompts.
4. **ProRL's "largest where weakest" holds on our prompts** (ρ = −0.50, `s2300` vs `dpo`, IFBench strict pass@1) but
   is weaker than a random-pairing null (−0.66): it cannot separate creation from sharpening.
5. **The Invisible-Leash objection is partly answered** (§2): format tolerance is open (loose grading credits `dpo` on
   25 of 49 strict-reach prompts, unread); 10 prompts survive 1,024 draws (4 hint-elicitable); Leash's Reasoning Gym
   budgets above 1,024 are untested.

## 1. Novelty matrix

**A. The closest papers' setups.** S / F / G / C = symmetric budgets, format tolerance, guessing control,
contamination control (✓ yes, ~ partly, ✗ no or not stated, – n/a). RL share = RL / pretraining FLOPs from
rl-compute-share's `compute_table.csv` (n.c. = not computed).

| paper | start (size) | RL share | domain | max k | grading | S F G C |
|---|---|---|---|---|---|---|
| Yue 2504.13837 | Qwen2.5 / Llama-3.1 bases, released RL models | 10⁻⁵–3 × 10⁻² (own runs ≤ 2 × 10⁻⁴) | maths, code, vision | 1,024 | answer; hardest solves read [GG-49] | ✓ ✗ ✓ [GG-47] ✗ |
| ProRL v1 2505.24864 | R1-Distill-Qwen-1.5B | 2.6 × 10⁻³–1.2 × 10⁻² | 5 domains incl. IF | 256 | verifiers; IF continuous [PR-22] | ~ ✗ ✗ ✗ |
| Invisible Leash 2507.14843 | ProRL v1 / v2; 7–14B distilled | as ProRL | re-test | 8,192 maths; 2,048 other; 16,384 RG [IL-12] | checkers; RG extraction widened [IL-36] | ✓ ~ ✗ ✗ |
| What Outcome RL Buys 2610.34894 | 20 RL models vs thinking bases, 1.5–32B | n.c. | physics, maths | 8 (64) [IL-93] | matcher, judge, readers | ✓ ✓ ~ ~ |
| Wang 2610.06851 | Olmo 3 7B base vs RL-Zero-Math | 4.6 × 10⁻⁴–2.3 × 10⁻³ | maths | 16 | Math-Verify | ✓ ~ ✗ ✗ |
| CoT-pass@k 2506.14245 | Qwen2.5-32B → DAPO (+ R1-Distill-7B → AceReason, code) | 2.4 × 10⁻⁴–3.0 × 10⁻² | maths, code | 1,024 [GG-17] | answer + LLM judge [GG-12] | ✓ ✗ ✓ ~ |
| HiR 2512.23457 | Qwen3-4B-Instruct | n.c. | IF | 1,024 [IF-36] | IF verifiers | ~ ✗ – ~ |
| Wei 2608.00220 | Qwen3-8B-Base, Qwen2.5-Math-7B | n.c. | maths- or IF-RLVR | 32 per prompt [IF-47] | IF checkers; LLM audit | ✓ ✗ – ✗ |
| Yoon 2610.19525 | Olmo 3 DPO→RLVR pairs (32B Think, presumably the 750-step release) | 1.8 × 10⁻³–7.4 × 10⁻³ (32B) | IF | 1 | IFBench loose + LLM quality | ✓ ✓ – ✗ |

**Ours:** 7B base (C1; RL 1.5–3.2 × 10⁻³, 2,800 steps) and the 32B SFT + DPO start (3 × 10⁻³–2 × 10⁻² FLOPs, 14 % of
GPU-hours; 2,300 steps); start budget k 256–512 (1,024 on 10 prompts); RL's checker, final answer, hand; S ✓ F ~
(loose not read) G ✓ (C1) C ✓.

**B. Claims against them.**

| claim | closest prior: relation | ours alone |
|---|---|---|
| C1 RL-Zero AIME | Yue: **replicates** reach (on most problems "the base model can produce at least one correct CoT" [EX-12]), **contradicts** on narrowing, other model and data (their pass@256 falls with steps [DT-108]). What Outcome RL Buys: replicates (87 of 121 lone wins recovered [IL-101]). Olmo 3 report: ours **contradicts** its pass@32 inference (3.1 successor, RL's checker; the small-k gain reproduces) [EX-1, EX-2]. Wang: extends (same 7B base vs the Olmo 3 RL-Zero release; a two-token cue lifts MATH-500 pass@1 42 → 78 %, RL 75 % [GG-56]) | hand-read large-k resolutions; reverse 0 at k ≤ 512 |
| C2 32B Think AIME | ProRL's maths "Diminish" [PR-55], What Outcome RL Buys (reach ≤ 2.1 pp [IL-98]): replicate; truncation "can create, mask, or reverse" a crossover [IL-154] (our candidate: 43 % of `dpo` tries capped) | the 32B SFT + DPO start; three grades |
| C3 IFBench reach | HiR: replicates direction (aggregate [IF-34]; values figure only); ProRL IFEval: extends (continuous score; ≈ +0.03 at k 128, figure); Wei: extends (maths-only RL *lowers* per-prompt IF best@32 [IF-45]); Leash: contradicts in kind, outside IF (ProRL v2 +48 / −175 [IL-17]); Guru [DT-40, DT-41], Olmo 3 (+4 AIME, +20 IFBench [DT-16]): domain dependence replicates | per-prompt reach at k 256, confirmed, hand-gated; ≈ 20 % RL-only |
| C4 gaming | lone-win audits in physics (3.5 % hold up [IL-99]; robust cells "about three fifths are answer form" [IL-105]); human-ruled audit of code-RL false positives, 47.57 % wrong [EX-11]; Wei 73.5 % shortcuts [IF-55]; Yoon: Fail→Pass prompts lose judged quality (k ≈ 1, own runs) [IF-60]; IFBench over-optimisation [IF-18]: phenomenon replicated in IF, audit design outside it | gaming among reach-deciding IF successes; reach re-counted |
| C5 brevity, token cap | What Outcome RL Buys [IL-95, IL-96], crossovers [IL-154], IF-RLVR answers of 210 tokens [IF-22], SciIF (RL "reduces preambles and meta-commentary" [NV-13]): **replicate** | the IF-reach version |
| C6 failures substantive | ProRL's base writes \boxed{} not the tags [PR-66]; after Leash's fixes it recovers dice [IL-32], partly boxnet [IL-25]: those zeros were partly format. SciIF's pre-RL IFEval failures are "conversational packaging and elaboration" [NV-12], like our IFEval (C5): no IFBench precedent | blind failure reading on IFBench reach prompts (93 %; unreviewed) |
| C7 methods | resampling nulls (a frozen model "expands" 7 of 25 AIME problems [GG-71]); guessing filters [GG-47]; Cover@τ, no reasoning check [GG-10]; public-checkpoint pass@256 [DT-88]; Olmo 3.1 checkpoints for behaviour [DT-97]: **extends** | the combination; reach at three public steps of a 32B run |
| C8 olmo-iffu | test 1 replicates finishing [IL-96]; test 2 extends cue elicitation [GG-56] and GEPA's IFBench prompts (38.61; GRPO 35.88, start 36.90 [NV-7, NV-8]); test 3: first per-prompt larger-k test on IF (HiR's aggregate reaches 1,024) | tests 2–4 on reach prompts |

## 2. ProRL side by side

| | ProRL v1 (+ v2 blog, BroRL) | ours |
|---|---|---|
| RL | R1-Distill-Qwen-1.5B [PR-27]; > 2k steps; KL + resets; 10k IF prompts, continuous reward [PR-11]; BroRL reports pass@1 only [PR-105] | 32B SFT + DPO; 2,300 steps of OlmoRL without KL; maths, code, IF, chat |
| largest k | 256 samples; claims at 128 [EX-8, PR-31] | 256 / 512 / 1,024 for the start |
| "base fails entirely" | some Reasoning Gym tasks at k ≤ 256 [PR-44] | 49 strict / 42 hand IFBench prompts at 256 |
| "largest where weakest" | "significant negative correlation" [PR-32]; no coefficient in the text (the legend's R² = 0.879 is image-only) | ρ = −0.50, weaker than the null below |

**ProRL's correlation on our data** (`correlation.py`): `dpo` has 16 fixed draws per prompt, RL 8; strict grade.
x = `dpo` pass@1 on half its draws; gain = RL pass@1 minus `dpo` on the other half.

| unit (n), k | model | Spearman ρ [95 %] | same draws | random-pairing null |
|---|---|---|---|---|
| IFBench prompt (218), 1 — **P1** | `s2300` | **−0.50 [−0.60, −0.40]** | −0.54 | −0.66 |
| IFBench constraint type (47), 8 | `s2300` | −0.80 [−0.90, −0.62] | −0.88 | – |
| IFBench prompt, 1 | `s0750` / `s0100` | −0.25 / +0.01 | −0.27 / −0.03 | −0.62 / −0.63 |
| IFEval prompt (150), 1 | `s2300` | −0.65 [−0.78, −0.50] | −0.79 | −0.75 |

*Observed:* the mean `s2300` gain is 0.54 where `dpo`'s half-sample rate is 0 and 0.00 above 0.75; 69 % of the summed
gain is on the 107 prompts where `dpo` has 0 of 16. *Our reading:* shuffling RL's rates across prompts (RL unrelated to
the start) gives a stronger negative ρ, because a difference of bounded rates falls with the start unless RL tracks it
closely (`s0100`: ρ(x, RL) = +0.88, ρ = +0.01). At `s2300` RL's rate rises with the start's (+0.50), so the observed ρ
is weaker than the null. A maths re-analysis of ProRL fits a log-odds slope instead (0.84 [PR-118]).

**Does IFBench reach survive an Invisible-Leash-style objection?** That objection re-grades the base format-tolerantly
and raises k: Leash samples 4,096–8,192 on maths, 1,024–2,048 on other non-maths sets, and up to 16,384 on Reasoning
Gym, the one place it also clarifies prompts and widens extraction [IL-12, IL-36]. It tests no IF (its one language
set is typos [IL-66]); nor does RL Squeezes, which reads RL on maths and code as "probability mass redistribution"
[IL-82]. Ours:
- **Format tolerance:** IFEval's loose grade credits `dpo` on 25 of the 49 strict-reach prompts (≥ 2 times on 19; all
  256 draws on 2, suggesting voided constraints). None was read: **open**.
- **Larger k:** 1,024 `dpo` draws on 10 prompts give 0 genuine (≤ 2.9 × 10⁻³ per draw), Leash's non-maths budget; 4 of
  the 10 are hint-elicitable, 33 of the 43 hand-reach prompts are untested, and k > 1,024 is untried.
- **Prompt format**, the analogue of Leash's clarified prompts: the 60-word hint elicits 12 of 42 IFBench reach
  prompts; 2 survive both the hint and 1,024 draws.

## 3. The other questions

- **Q3, an IF boundary at large k:** refuted in the aggregate (HiR, IFBench to 1,024 [IF-36]; ProRL, IFEval to 256
  [IF-76]; confirmed per prompt at k ≥ 256 (the closest per-prompt study stops at k = 32
  [IF-47]).
- **Q4, domains within one run:** precedented: Guru's 7B AIME curve crosses its base while Zebra expands [DT-40,
  DT-41]; Zhai's agent RL gains +4 on trained tasks, none on MATH-500 [DT-26–DT-28]; Olmo 3's scores move +4 on AIME,
  +20 on IFBench [DT-16].
- **Q5, gaming and grading:** no audit of gaming among reach-deciding successes on an IF task; Cover@τ makes "models that rely on random guessing degrade rapidly"
  [GG-2] but checks no reasoning [GG-10]. CoT-pass@k uses an LLM judge [GG-12], which in an audit left 71 % of wrong
  answers accepted [GG-26].
- **Q6, trajectories:** step-resolved large-k curves exist (diversity collapse, 38 entries vs 152 losses [DT-65];
  Ring-Zero's pass@1024 rises, then plateaus [DT-80]; an endpoint study adds an early stop [DT-72, DT-74]). OLMo-2-1B's
  public RL checkpoints were read at pass@256 [DT-88]. Olmo 3.1 Think 32B's 46 `step_*` branches [DT-1, DT-2] have
  been used for behaviour [DT-97]; Ai2's own pass@K stops at 32 [EX-2]; we found no outside reach test.

## 4. Claims ranked

1. **Novel** (as far as found; on IF only): C3, per-prompt confirmed reach at k = 256; C4, gaming among the
   reach-deciding successes; C6, the failure reading (unreviewed).
2. **Extends:** C8 tests 2–4; C7, including reach tests at three public steps of a 32B run (a 1B precedent [DT-88]).
3. **Replicates:** C1's reach result; C2 (no maths reach); C5 and C8 test 1.
4. **Contradicted, weakly:** C1's "nothing lost", by narrowing on other starts and data (83 problems; the 7B pairs show
   no supported loss [EX-14]). C1 in turn contradicts the Olmo 3 report's inference [EX-1].

## 5. Five objections a reviewer would raise

1. ***"256 is not 'cannot'."*** The project's standard (capabilities REPORT §1) matches the budget to RL's compute:
   Olmo 3.1 Think 32B's RL used ≈ 1.4 × 10⁵ H100-hours (rl-compute-share), and one `dpo` IFBench draw costs ≈ 3.0
   H200-seconds (`log.md` 2026-11-05 15:32), so K ≈ 8 × 10⁵ draws per prompt over 218 prompts (order of magnitude;
   mixed GPU types). *Answer:* we claim only "not reached at 256", never "created".
2. ***Format and prompt artefacts*** (Invisible Leash). *Answer:* partly; see §2.
3. ***"Hand" is model-read.*** LLM judges can be unreliable [GG-25, GG-26]. *Answer:* blinded, rule-based, confirmed,
   and reproduced within ± 2 by an independent model review; a human sample is missing.
4. ***The ProRL correlation is an artefact.*** *Answer:* agreed; we report it with its null and do not rest on it.
5. ***RL saw constraints like IFBench's.*** Olmo's RL IF constraints are "sampled from IFEval … and IFBench-Train"
   [EX-10]; IFBench's are held out [IF-1, IF-8]. *Answer:* partly: 20 of 42 hand-reach binding constraints have no RL
   analogue (executor's map).

## 6. Draft related-work section (≈ 600 words)

> **Create or elicit.** Yue et al. [@yueDoesReinforcementLearning2025] made equal-k pass@k the standard test: RLVR
> models win at small k, and base models overtake them as k grows. Answer-graded re-tests agree the base catches up; paired inference
> certifies few crossovers (one of nine public checkpoints, narrowly; none at 7B), and token limits alone can create
> or hide one [@yangWhenPassCrossovers2026]. At budgets of 1,024–8,192, ProRL v2 gains 48
> questions and loses 175 [@wuInvisibleLeashWhy2025]; losses outnumber entries along training
> [@yuanUnderstandingDiversityCollapse2026; @zhouWhenRLVRShrinks2026]. Across 20 released pairs, where a base often runs out of tokens, most of RL's gain is
> finishing within the budget, and reach adds at most 2.1 points [@yangWhatOutcomeRL2026]. Our Olmo 3.1 7B RL-Zero result
> agrees under RL's checker: within 512 samples the base is credited on every RL-only AIME solve, and 10 of 17
> resolutions read as sound, which disputes the Olmo 3 report's reading of a pass@32 gain as going "beyond its
> initial capabilities" [@olmoOlmo32025].
>
> **Claims of expansion.** ProRL [@liuProRLProlongedReinforcement2025] reports gains "where base models fail
> entirely", largest where the base is weakest. Its zero-base tasks are Reasoning Gym puzzles, where the paper notes
> the base answers in `\boxed{}` instead of the required tags; with clarified prompts and wider extraction, Invisible
> Leash's base reaches full coverage on dice, while ProRL keeps a lead on boxnet. Other expansions are
> domain-specific: Zebra puzzles but not AIME within one 7B run [@chengRevisitingReinforcementLearning2025]. One 32B RL
> run gives us one inconclusive AIME candidate but 42–49 IFBench prompts (≈ 20 %) the DPO start never solves in 256
> samples. ProRL's correlation holds on our prompts (ρ = −0.50), but a random-pairing null produces it too.
>
> **Instruction following.** IF-RLVR improves generalisation to unseen constraints but over-optimises toward the
> constraint [@pyatkinGeneralizingVerifiableInstruction2025]. On aggregate IFBench pass@k up to k = 1,024, an IF-RL
> method stays above its start [@zhangReplayFailuresSuccesses2025]; per prompt at k = 32, maths-only RLVR shrinks IF
> coverage [@weiVerifierInducedSupportReshaping2026]. Checker exploitation is documented: an LLM judge finds 73.5 % of
> an IF-RLVR checkpoint's verifier passes are shortcuts [@weiVerifierInducedSupportReshaping2026], Olmo 3's released
> DPO→RLVR checkpoints gain constraint satisfaction but lose judged quality [@yoonDiagnosingCorrectingReward2026], and
> RLVR-trained models pass rule-induction verifiers by enumeration [@helffLLMsGamingVerifiers2026]. We count reach per
> prompt, confirm the RL side on fresh draws, and have model readers read nearly every deciding success blind: about a
> third of `s2300`'s first-checked strict IFBench successes are gamed.
>
> **Measurement.** At large k, guessing is filtered with answer-only probes [@yueDoesReinforcementLearning2025],
> reliability thresholds [@dragoiBeyondPasskBreadthDepth2025] or LLM-judged chains
> [@wenReinforcementLearningVerifiable2025], though such judges accept corrupted chains
> [@tasaltiDoesCoTPasskReally2026]. Lone-win audits find most RL-only wins do not hold up [@yangWhatOutcomeRL2026]; a
> human-ruled audit finds about half of the rewarded false positives in code RL genuinely wrong [@zhangWhenRewardSuite2026];
> resampling a frozen model shows "expansions" at k = 128 can be luck [@xuPhantomGainsAuditing2026]. Cues match
> RL's gain and optimised prompts beat RL [@wangBaseModelsCan2026; @agrawalGEPAReflectivePrompt2025]. Like What
> Outcome RL Buys we read deciding successes blind, grade twice, test the reverse and lengthen the budget (65k); we
> add a one-line brevity hint, which elicits 16 of 48 hinted IF reach prompts.
>
> **Trajectories.** ProRL compares one unnamed intermediate checkpoint. A lab's public RL checkpoints have been read at
> pass@256 for a 1B model [@bangeraRLVRSharpensPencil2026], and Olmo 3.1's for behaviour
> [@bharadwajTracingEvalAwarenessEmergence2026]. We test reach against the DPO start (256 samples) at Olmo 3.1 Think
> 32B's public steps 100, 750 and 2,300 (8 draws each, more on candidates): step 750 is already confirmed on 26 of 43
> IFBench reach prompts.

## 7. Pre-registered expectations vs outcomes

Scored against `preregistration/olmo-novelty.md` (664aaa93). **Correlation.** C-1 to C-6 held. C-7 missed narrowly: the per-type ρ was −0.803, against a predicted [−0.80, −0.20].

**Literature.** L-1 **failed** by its own falsifier (HiR's aggregate pass@k at k ≥ 256); its per-prompt clause held.
L-2 held (no coefficient in the text; the base recovers fully on dice, partly on boxnet). L-3 and L-4 held (IF gaming audits
exist, none at reach; physics has lone-win audits). L-5 partly held ("extends"); its 0.6 clause failed
(OLMo-2-1B, pass@256). L-6 partly held: C3 and C4 are novel only on IF and per prompt; C5 ranks "replicates", not "extends". The
fresh search (25 sources, 24 new) knocked down no further label.
