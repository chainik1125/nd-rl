# Run brief: olmo-answeronly — when the DPO start fails on RL's thinking, is it only how it writes the answer?

Run id `olmo-answeronly`. Repository `~/work/olmo-answeronly` (branch `dan_olmo-answeronly` from the fork's
`origin/dan_olmo-ifladder`, made by `newrun`). That branch carries `olmo-ifomega`'s pipeline, hand-gate rules and
stored samples, `olmo-iffu`'s hint code, `olmo-transplant`'s trace manifest, insertion and grading code, and
`olmo-ifladder`'s LoRA path. Role: executor. `AGENT_POLICY.md` governs. Proposal 45 (nd-rl
`docs/proposals/2026-11-09-olmo-transplant-answer-only.md`; read it first). Authorised by Dan 2026-11-09, up to $30.

- **Budget:** **$30 / 20 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $3) measures prefill-heavy throughput for one condition on 5 prompts. Re-cost, and pause
  and ask in `QUESTIONS.md` if the projection for the required design exceeds $27.
- **If over budget, cut in this order:** the optional LoRA arm; condition A2; the calibration prompts for A4.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

`olmo-transplant` gave the DPO start (`dpo`) the RL model's (`s2300`) full genuine thinking on 41 IFBench reach
prompts and let it write only the answer: genuine 0.14, against `s2300`'s 0.92 on the same thinking. `dpo`'s answers
have a median of 126 words against 26; 28 % contain RL's drafted answer verbatim and most of those fail by adding
words; only 4.7 % copy it exactly.

Is the gap a writing habit that one instruction removes, or can `dpo` not use RL's thinking at all?

## Design (pre-register your version before any pod)

1. **Prompts and traces:** exactly `olmo-transplant`'s: the 41 reach prompts, the 115 stored `s2300` genuine traces
   (`traces.csv`), the 10 calibration prompts and their traces. Insert by stored token ids. Do not regenerate
   thinking.
2. **Baselines:** reuse `olmo-transplant`'s T1 and T2 rows. Do not re-sample them.
3. **Conditions:** each is the transplant plus one change, with both writers (`dpo`, `s2300`), 8 answers per
   prompt × trace, at `olmo-transplant`'s decoding and seeds scheme (new seeds, registered).

   | | change |
   |---|---|
   | A1, instruction | append to the user turn: "After thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation." |
   | A2, brevity | append `olmo-iffu`'s hint verbatim: "Answer in at most 60 words." |
   | A3, closing thought | append to the end of the inserted thinking, before `</think>`: "I will now output exactly the final draft above and nothing else." |
   | A4, prefill | after `</think>\n\n`, supply the first 5 tokens of the trace's own final answer; the writer continues |

   - Fix the wordings above; do not tune them. If a wording cannot be inserted cleanly (template boundary), record the
     exact tokens used in the pre-registration.
   - A4 is graded on the prefill plus the continuation. Report how often the writer stops where the draft stops.
   - Answers that re-open thinking (contain `</think>`): report rates both ways, as the `olmo-transplant` review did.
4. **Optional arm, only if the required design is projected under $22.** `olmo-ifladder`'s `lora_main` (a LoRA of
   `dpo` on 16 genuine `s2300` trajectories), through vLLM's LoRA path:
   - `lora_main` as writer on RL's thinking (T1 with the LoRA), on the reach prompts that are not in its training
     manifest's source prompts;
   - plain `dpo` as writer on `lora_main`'s own successful thinking from `olmo-ifladder`'s stored evaluation draws
     (up to 4 hand-genuine traces per prompt).

   `olmo-ifladder` was unreviewed when this brief was written. Say so wherever this arm is reported, and check its
   review's verdict before relying on its labels.
5. **Grading:** the strict IFBench checker, plus the blinded hand gate with `olmo-transplant`'s rules, read caps and
   identity inheritance. Report strict and genuine, and the exact-copy and near-copy shares (word 3-gram Jaccard
   ≥ 0.8) against the trace's final answer.
6. **Analysis:** per condition and writer, rates with a 95 % prompt bootstrap; R = `dpo` ÷ `s2300` within a condition,
   on prompts where the condition keeps `s2300` ≥ 0.75; paired differences from T1 by prompt; by constraint family.

## Pre-registered readings (state these thresholds before data)

Choose the best of A1–A3 on the **calibration** prompts, not on the reach prompts, and report all three.

- **R ≥ 0.75: a habit.** With the thinking in hand, `dpo` lacks only output discipline.
- **R ≤ 0.40: a skill.** Even told to, `dpo` cannot turn RL's thinking into the answer.
- **Between:** report it by constraint family.
- **A4 high (R ≥ 0.75) while A1–A3 are ≤ 0.40:** the habit sits in the first few tokens.
- **A condition that breaks `s2300`** (its rate falls below 0.75 on more than half the prompts) is uninformative;
  say so and do not read it.

Also register an expected value and range for each of A1–A4 for `dpo` before sampling.

## Deliverables

- `run_olmo_answeronly.md` (≤ 400 words + one figure: the condition bars for both writers beside T1 / T2);
  `numbers.md` § olmo-answeronly; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words.
- `olmo_answeronly/`: code, the exact inserted wordings and token ids, `results.csv`, the hand-check table, and every
  sample's text in the bucket.
- A `STATUS.md` line `OLMO-ANSWERONLY DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-answeronly/`,
  delete pods, and `touch ~/runs/olmo-answeronly/executor.done`.

**For the reviewer:**
- Check that each inserted prefix equals the stored ids plus exactly the registered change.
- Check the template boundary on a sample of each condition.
- Re-grade the results, and repeat the hand gate blind on the deciding successes.
- Recompute R and the readings.

**Rules:**
- Do not contact anyone.
- Weights stay on pods.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
