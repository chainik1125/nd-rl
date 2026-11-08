# Run brief: olmo-ifladder — can prompting or a tiny fine-tune give the DPO start RL's instruction-following reach?

Run id `olmo-ifladder`. Repository `~/work/olmo-ifladder` (branch `dan_olmo-ifladder` from the fork's
`origin/dan_olmo-transplant`, made by `newrun`). That branch carries `olmo-ifomega`'s pipeline, hand-gate rules and
stored samples, `olmo-iffu`'s hint code and results, and `olmo-transplant`'s chat-template code. Role: executor.
`AGENT_POLICY.md` governs. Proposal 41 (nd-rl `docs/proposals/2026-11-07-olmo-followups.md`). Authorised by Dan
2026-11-08.

- **Budget:** **$80 / 50 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $5) measures draw lengths and throughput for the ladder prompts and a LoRA training step.
  Re-cost, and pause and ask in `QUESTIONS.md` if the projection exceeds $72.
- **If over budget, cut in this order:**
  1. the `s2300` sanity arm;
  2. run rungs 3 and 4 only on prompts that no earlier rung elicited;
  3. LoRA evaluation draws from 64 to 32 per prompt.

  Keep the control LoRA.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

On Olmo 3.1 Think 32B, about 20 % of IFBench prompts are out of the DPO start's reach at 256 samples (`olmo-ifomega`).
`olmo-iffu` showed that one line, "Answer in at most 60 words.", makes `dpo` succeed (≥ 2 genuine of 64) on 16 of the 48
reach prompts it tested. `olmo-transplant` found that `dpo` mostly fails even when handed RL's successful thinking
(0.14 against RL's 0.92).

The question is how much of the rest can be had **without RL**, and by what:
- stronger prompting;
- or a few examples of RL's own work, via a small fine-tune.

The goal is a per-prompt classification: disposition, teachable, or neither. Dan's note: the middle rungs are close to
spoon-feeding and matter less in themselves. The outputs that matter most are:
- the **"neither"** set, which feeds Proposal 42;
- the **fine-tune result**: `dpo` cannot use RL's thinking handed to it in context, but can 16 training examples teach
  it to produce that kind of thinking on new prompts?

## Design (pre-register your version, with exact wording, before any ladder draw)

1. **Prompts:**
   - the 48 instruction-following reach prompts `olmo-iffu` tested: 42 IFBench (the review's 43 hand-reach prompts
     less `ifbench_281`) and 6 IFEval;
   - plus about 10 calibration prompts both models solve. These check that no intervention or fine-tune breaks
     ordinary instruction following.
2. **The ladder.** Sample `dpo` under prompt interventions of rising strength, appended to the user turn. Grade against
   the original prompt, as `olmo-iffu` did.
   - **Rung 1, the brevity hint:** reuse `olmo-iffu`'s tests 2 and 2c (64 draws each); do not re-sample.
   - **Rung 2:** rung 1 + "Before answering, check each constraint against your draft."
   - **Rung 3:** rung 2 + a *procedure hint* per constraint family, for example "first list candidate words for each
     letter A–Z". Write every family's hint from the constraint's description alone, before seeing `dpo`'s outputs
     under it. A hint must state a general method, never content that answers the prompt; the reviewer checks this.
   - **Rung 4:** rung 3 + few-shot. Two genuine `s2300` answers (after `</think>`) to *other* IFBench prompts with the
     same constraint type, given as prior user/assistant turns.
   - **The 60-word line:** on prompts where it conflicts with a constraint (a minimum length, or `ifbench_281`-like
     content), drop it from every rung. Pre-register that list.
   - **Draws:** 64 per prompt per rung, at `olmo-iffu`'s decoding and response cap.
   - **The `s2300` sanity arm:** 16 per prompt at rungs 2–4, checking whether an intervention breaks the RL model.
3. **Teachability, by LoRA fine-tunes of `dpo`:**
   - **Main:** about 16 hand-labelled-genuine `s2300` trajectories, thinking included. Source them from the stored
     `olmo-ifomega` samples on IFBench prompts **outside** the 48, family-matched to the reach prompts as far as
     possible. No training prompt may share its base request text with an evaluation prompt; check and report this.
   - **Control:** the same recipe on the same number of `dpo`'s *own* genuine successes on the same training prompts.
     It separates "RL's trajectories teach something" from "any light fine-tune on IF data moves it".
   - **Hyperparameters:** pre-register rank, learning rate, epochs and sequence cap. Do not tune them on the evaluation
     prompts; use the calibration prompts plus a held-out slice of training-pool prompts.
   - **Evaluation:** both LoRAs on all 48 prompts without hints, 64 draws each, and on the calibration prompts. Report
     thinking and answer lengths against `dpo` and `s2300`.
4. **Grading:**
   - the strict checker (primary) and loose (secondary);
   - the hand gate on every deciding success (`olmo_ifomega/handcheck_rule_if.md` plus `olmo-iffu`'s addendum;
     blinded reader sub-agents, confirmed by you);
   - the gamed share at each rung and for each LoRA. "Check each constraint" may raise gaming.

## Pre-registered readings (per prompt; "elicited" = ≥ 2 genuine of 64, as in `olmo-iffu`)

- **Prompt-elicitable at rung r:** the first rung that elicits it.
- **Teachable from RL's trajectories:** no rung elicits it; the main LoRA does and the control does not.
- **Teachable by any light fine-tune:** both LoRAs elicit it.
- **Neither:** no rung and neither LoRA. These are the strongest candidates for RL-created capability at this scale.
  Flag the ones that also survived 1,024 unhinted draws in `olmo-iffu`.

Report the counts by constraint family. Separately, report the four `olmo-iffu` prompts that survive 1,024 draws yet are
hint-elicitable.

## Deliverables

- `run_olmo_ifladder.md` (≤ 400 words + one figure: the per-prompt ladder, with prompts as rows and rungs plus the
  LoRAs as columns); `numbers.md` § olmo-ifladder; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words.
- `olmo_ifladder/`, containing:
  - code, the exact intervention wordings, the LoRA recipe and training manifest (which stored samples, their hashes,
    and the overlap check);
  - `results.csv` and the hand-check table;
  - the LoRA adapters and every sample's text, in the bucket.
- A `STATUS.md` line `OLMO-IFLADDER DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-ifladder/`,
  delete pods, and `touch ~/runs/olmo-ifladder/executor.done`.

**For the reviewer:**
- Check that hints carry no answer content and were written before the outputs.
- Check the training/evaluation overlap.
- Re-grade the results, and repeat the hand gate blind on the deciding successes.
- Recompute the per-prompt classification.

**Rules:**
- Do not contact anyone.
- Weights stay on pods; adapters may go to the bucket.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
