---
written_on: 2026-11-07
written_by: agent:claude
status: proposals 39–44, written at Dan's request 2026-11-07. Dan rates 39, 40, 41 and 44 high interest. Dan approved 39 ($40) and 40 ($60) on 2026-11-07: 39 ran as `olmo-rlzero-if` on agent1 on 2026-11-08 ($28.74 at the GPU rate, $29.34 billed; reviewed 2026-11-08, see the result under proposal 39); 40 was queued as `olmo-transplant` on agent2 until the RunPod balance reached $95 (the $35 floor plus its budget; the balance was $93.38), and ran on 2026-11-08 after a top-up ($17.82 at the GPU rate, $17.95 billed; reviewed 2026-11-08, see the result under proposal 40). Dan approved 41 on 2026-11-08 (launched as `olmo-ifladder` on agent2 at $80, brief `followups/BRIEF_olmo-ifladder.md`; it reuses `olmo-iffu`'s brevity-hint draws as rung 1 and adds a control LoRA). 44 remains a strong next candidate; 42 and 43 are medium.
papers:
  - yueDoesReinforcementLearning2025
  - liuProRLProlongedReinforcement2025
---

# Proposals 39–44: follow-ups that test the Olmo claims

## Where these come from

Four reviewed runs (`olmo-rlzero-pilot`, `olmo-think32b`, `olmo-ifomega`, `olmo-iffu`) and the literature positioning
(`olmo-novelty`) support six claims. They are agent:claude's synthesis, written with Dan on 2026-11-07, and most are
directional:

1. **Domain dependence within one RL run.** On Olmo 3.1 Think 32B, RL barely moves AIME (pass@1 0.716 → 0.746, no
   reach). It moves IFBench a lot (0.290 → 0.683): about 20 % of prompts are out of the DPO start's reach at 256
   samples, and 10 of 10 tested survive 1,024.
2. **More RL rescued creation in one domain, not the other.** At ≈ 10⁻² of pretraining FLOPs (14 % of GPU-hours;
   ProRL's band, about 100× Yue's median), math still looks like elicitation. IF kept gaining from step 750 to 2,300.
3. **Part of the IF "reach" is elicitable disposition or gaming.** About 36 % of RL's first-read checker wins are
   gamed, and a 60-word hint makes the DPO start succeed on 16 of 48 reach prompts.
4. **The residue looks like learned search.** RL writes 7k–55k characters of thinking to produce answers of 4–90
   words. It may have learned to spend its reasoning on satisfying the constraints.
5. **Measurement choices flip conclusions.** Large k credits guesses on integer answers, lenient checkers credit
   gaming, and asymmetric confirmation manufactures "the base can do it".
6. **Reach at fixed k may be rare-event amplification.** The σ organism amplifies a 0.1 % trace but not a zero one.
   Olmo's IF reach is consistent with a per-sample rate below 1 / 1,024.

Each proposal below names the claim it tests and what either outcome would mean. Costs use this project's measured
prices: ≈ $0.6–1.0 per M generated tokens on one H200 at 32B; 7B is cheaper. All are projections, to be re-costed by
each run's stage 0.

| # | proposal | tests | cost | interest |
|---|---|---|---|---|
| 39 | RL-Zero-IF from the pretrained base (7B) | 1 (removes the SFT confound) | ≈ $25–40 | **high**: start now |
| 40 | Thinking transplant and budget ablation (32B) | 4 | ≈ $30–60 | **high**: start now |
| 41 | Disposition vs skill: an elicitation ladder, then teachability (32B) | 3 | ≈ $50–80 | **high**: next |
| 42 | Probing the DPO start's support on surviving reach prompts (32B) | 6 | ≈ $60–90 | medium: after 41 |
| 43 | Toy law: RL compute needed against the base's per-attempt σ rate | 6 (toy side) | ≈ $25–50 | medium: paper value |
| 44 | Mapping the domain split on 32B (LiveCodeBench, ZebraLogic) | 1, 2 | ≈ $150–250 | **high**: next |

## Proposal 39: does RL straight from the pretrained base create IF reach? (`olmo-rlzero-if`)

**Claim tested:** 1. On the 32B model, math and IF differ in their domain *and* in what SFT supplied before RL. Math
was largely saturated by DeepSeek R1 distillation, so "IF creates, math doesn't" could mean the domain matters or that
SFT didn't saturate IF. The 2×2 grid shows the gap:

| | math | IF |
|---|---|---|
| pretrained base → RL (7B RL-Zero) | done: no reach (`olmo-rlzero-pilot`) | **this proposal** |
| base → SFT → DPO → RL (32B Think) | done: no reach (`olmo-think32b`) | done: reach (`olmo-ifomega`) |

**Design:**
- **Models:** `allenai/Olmo-3-1025-7B` (base) against `allenai/Olmo-3-7B-RL-Zero-IF` (RL straight from that base on
  `Dolci-RL-Zero-IF-7B`, 13,179 prompts; 19 checkpoints, steps 100–1,900, plus `main`) at about three steps (100, 500,
  final).
- **Parity:** use the RL-Zero-IF recipe's template and decoding for every model. The like-for-like math run is
  `Olmo-3-7B-RL-Zero-Math` (Olmo 3, 19 checkpoints). The pilot used the 3.1 math run, so record the difference.
- **Data and grading:** IFBench (all 300) and IFEval (the 150-prompt subset `olmo-ifomega` used). Grade with the
  official strict checkers, plus the same hand gate (gamed / genuine) on every deciding success. Run a contamination
  scan against `Dolci-RL-Zero-IF-7B`.
- **Protocol:** `olmo-ifomega`'s, symmetric: the base at k = 16, RL checkpoints at k = 8, the base extended to a
  second genuine success up to 256 on reach candidates, and the same rule for reverse candidates.
- **Format check, required:** classify the base's failures on reach prompts (no usable answer or rambling
  continuation; ignored the constraint; attempted but far; near miss; broke another constraint). A pretrained base can
  fail IF prompts just by not answering cleanly, and that would make reach a formatting effect.
- **Optional:** `Olmo-3-7B-RL-Zero-Math`'s final checkpoint on IFBench. Does math RL alone move IF reach?

**Outcomes:**
- **Reach from the base, against no math reach from the same base:** the domain effect is real without SFT, and holds
  at ≈ 10⁻³ RL share. This strongly supports claim 1.
- **No IF reach from the base:** the 32B IF reach needs something SFT installed (most plausibly long chain-of-thought
  search). This supports claim 4 and makes 40 central.
- **Either way:** the gamed share for RL from the base, against 32B's 36 %.

**Cost and risk:** ≈ $25–40, re-costed at stage 0. The risk is the format confound, handled by the required check.
**Run brief:** [followups/BRIEF_olmo-rlzero-if.md](followups/BRIEF_olmo-rlzero-if.md).

**Result (reviewed 2026-11-08):** RL straight from the 7B base mostly learned to game the IF checkers: 94 % of
`main`'s first-read passes on reach candidates are gamed, and `main` is hand-confirmed on 10 of 164 candidates. The
pre-registered verdicts stand (by hand IFBench Reach on 5 prompts, IFEval inconclusive; under strict Reach on both), but
every hand-reach prompt is an exact-format prompt, the base answers 3 of the 6 genuinely when it reads `<think>` as
text, and by the pre-registered format rule 5 of 6 are reach a format repair would remove. "Robust" reach is not
supported, and SFT as the cause of the IF/math split only as "consistent with". Summary:
[olmo-rlzero-if](../../experiment-summaries/2026-11-08-olmo-rlzero-if-rl-zero-games-if-checkers-94pct-ifbench-hand-reach-5-format-prompts-ifeval-inconclusive/README.md).

## Proposal 40: is RL's new IF capability in its thinking? (`olmo-transplant`)

**Claim tested:** 4. RL learned a search procedure in its chain of thought, not a better final-answer skill.

**Design:**
- **Prompts:** IFBench hand-reach prompts (seeded ≈ 20, stratified by constraint family, including the 6 that
  survived 1,024 DPO samples and were not hint-elicitable in `olmo-iffu`), plus ≈ 10 IFBench prompts both models solve,
  as calibration.
- **Thinking traces are reused, not regenerated.** `olmo-ifomega` stored every sample's full text, thinking included,
  with token ids (bucket `olmo-ifomega/artifacts/olmo_ifomega/samples/*.jsonl.gz`). Use `s2300`'s hand-labelled
  genuine successes and the DPO start's failures. Both models share the Olmo 3 tokenizer and chat template, so a trace
  can be inserted as an assistant prefix ending in `</think>`, and only the final answer is sampled.
- **Conditions** (8 answer samples per prompt × trace × condition):

  | | thinking supplied | model that writes the answer |
  |---|---|---|
  | T1, transplant | RL's genuine thinking | DPO |
  | T2, self | RL's genuine thinking | RL (expected to be high) |
  | T3, reverse | DPO's failed thinking | RL |
  | T0, none | empty | each model |
  | B, budget | RL's thinking cut at N ∈ {0, 1k, 2k, 4k, 8k, 16k tokens, full}, then `</think>` | each model |

  Because B uses prefixes of *successful* traces, also run fresh budget-forced RL generations at two or three budgets
  on a few prompts.
- **Grading:** strict checker plus the hand gate on every success.

**Outcomes:**
- **T1 ≈ T2:** the capability is in the thinking, and the DPO model can execute RL's plan. This supports claim 4.
- **T1 ≪ T2 but T3 high:** what RL changed is answer writing, not search.
- **The B curve** shows how much search the win needs. RL's advantage vanishing at small N supports search; flat
  curves suggest a disposition.

**Cost:** ≈ $30–60. Most of the compute is prefill of long traces; decodes are short.
**Run brief:** [followups/BRIEF_olmo-transplant.md](followups/BRIEF_olmo-transplant.md).

**Result (reviewed 2026-11-08):** handed the RL model's own successful thinking on 41 IFBench reach prompts, the DPO
start writes a genuine answer 0.14 of the time against the RL model's 0.92 (R1 = 0.15). Without that thinking the RL
model scores 0.06–0.09 and the DPO start 0.00, so the thinking carries most of RL's gain, not all of it, and it is not
sufficient. Pre-registered readings: "partly" (only through its "neither" clause) and "graded"; "in the thinking",
"answer writing" (T3 0.06), "needs long search" and "flat" fail. The forced `</think>` does not cap the RL model's thinking. Summary:
[olmo-transplant](../../experiment-summaries/2026-11-08-olmo-transplant-dpo-given-rl-thinking-0p14-vs-rl-0p92-genuine-on-41-ifbench-reach-prompts-readings-partly-and-graded/README.md).

## Proposal 41: disposition or skill? An elicitation ladder, then teachability (`olmo-ifladder`)

**Claim tested:** 3. How much of the IF reach can be elicited from the DPO start without RL, and by what?

**Design:**
- **The ladder:** on the 43 hand-reach IFBench prompts and 6 IFEval prompts, sample the DPO model under prompt
  interventions of rising strength, each pre-registered with fixed wording:
  1. the brevity hint (`olmo-iffu`'s, as the baseline);
  2. "check each constraint before you answer";
  3. a procedure hint per constraint family (for example "first list candidate words A–Z");
  4. few-shot: two genuine RL answers to *other* prompts with the same constraint type.

  Then a tiny LoRA fine-tune of the DPO model on ≈ 16 genuine RL trajectories from other prompts (thinking included),
  evaluated on the held-out reach prompts.
- **Grading:** strict checker and hand gate; report each prompt's first intervention that elicits ≥ 2 genuine.

**Outcomes:**
- **Prompting recovers it:** disposition.
- **Only the 16-example fine-tune recovers it:** "teachable, not latent", in capability-defs' terms.
- **Neither:** the strongest candidates for RL-created capability. They feed 42.

**Cost:** ≈ $50–80, with the LoRA on one H200. **Interest:** high. It turns claim 3 from shares into a
per-prompt taxonomy.

## Proposal 42: probing the DPO start's support on surviving reach prompts (`olmo-support`)

**Claim tested:** 6. Is "not reached at 1,024" a per-sample rate of ≈ 10⁻⁴–10⁻⁶, or truly zero?

**Design:** on the prompts that survive 41, or `olmo-iffu`'s 6 non-elicitable survivors if 41 has not run:
- **Likelihood:** the DPO model's log-likelihood of RL's genuine trajectories, per token and as a whole, against its
  own successes on calibration prompts.
- **Prefix-seeded sampling:** give the DPO model the first x % of RL's thinking (x = 10, 25, 50, 75) and let it
  continue thinking and answer, to find where its support breaks. This is the backward-algorithm idea from
  Go-Explore and POPE.
- **Deeper sampling:** extend the DPO model to 4,096 samples on 4 prompts, for a binomial bound.

**Outcomes:** a per-prompt rate, or an upper bound, and the point at which support breaks. That turns reach into a
number comparable with the σ organism's ≈ 10⁻⁶ per-attempt rates.

**Cost:** ≈ $60–90. **Interest:** medium. It is most informative after 41 removes the elicitable prompts.

## Proposal 43: a toy law, RL compute needed against the base's per-attempt rate (`coe-baserate`)

**Claim tested:** 6, from the toy side. The prediction is RL compute to acquire ∝ log(1 / p), where p is the base's
per-attempt rate on the capability.

**Design:** create-or-elicit's next-steps item 1b. Measure each base's per-attempt σ-member rate (full, 1 %, 0.1 %;
bounded for the knockout) at large k, then fit it against the RL ÷ pretraining compute each arm needed. Those values
are already in `docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/`.

**Outcome:** a quantitative relation that makes the midterm figure predictive, and a principled extrapolation for what
an Olmo-scale base would need.

**Cost:** ≈ $25–50 (create-or-elicit's estimate, at its measured attempt prices). **Interest:** medium. It is the
best-value item for the paper's toy story.

## Proposal 44: mapping the domain split on 32B (`olmo-domains`)

**Claims tested:** 1 and 2, broadly. Does reach track how saturated a domain was before RL?

**Design:**
- **Same protocol:** `olmo-ifomega`'s symmetric protocol on the same DPO start and RL checkpoints (steps 750 and
  2,300).
- **New domains:**
  - **LiveCodeBench** problems released after the model's training-data cutoff, graded by unit tests (no guessing);
    the open-instruct code verifier or LCB's own harness;
  - **ZebraLogic**, whose answers are structured grids and exact.
- **Analysis:** combine these with AIME, OMEGA, IFEval and IFBench, then plot the reach share against DPO's pre-RL
  pass@k per domain, and against each domain's share of RL data.
- **Interpretation:** read it with 39's result. If 39 shows IF reach without SFT, domain differences here are more
  likely intrinsic to the domain.

**Outcome:** the most convincing form of the domain-split claim. It also shows whether code, a domain with
unforgeable grading, behaves like math or like IF.

**Cost:** ≈ $150–250. Code responses are long; re-cost after a stage-0 length measurement. **Interest:** high.

## Later proposals

- Proposal 45 (`olmo-answeronly`), written 2026-11-09: [the transplant with the answer-writing habit removed](2026-11-09-olmo-transplant-answer-only.md).
