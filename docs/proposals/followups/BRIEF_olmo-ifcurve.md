# Run brief: olmo-ifcurve — how does instruction following change over the course of Olmo 3.1 Think 32B's RL run?

Run id `olmo-ifcurve`. Repository `~/work/olmo-ifcurve` (branch `dan_olmo-ifcurve` from the fork's
`origin/dan_olmo-ifladder`, made by `newrun`).

That branch carries:
- `olmo-ifomega`'s pipeline, hand-gate rules and stored samples;
- `olmo-iffu`'s results and rule addendum;
- `olmo-transplant`'s code;
- `olmo-ifladder`'s per-prompt classification (`olmo_ifladder/classify.csv`).

Role: executor. `AGENT_POLICY.md` governs. Proposal 46 (nd-rl `docs/proposals/2026-11-20-olmo-rl-curve.md`; read it
first). Authorised by Dan 2026-11-20, up to $120.

- **Budget:** **$120 / 60 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $5) measures draw lengths and throughput at two checkpoints (`step_0050` and
  `step_1600`). Re-cost, and pause and ask in `QUESTIONS.md` if the projection for the core arms exceeds $108.
- **Core arms:** A, B (with its extension), C and D.
- **Optional arms:** F (AIME), then E (Olmo 3 7B Think), then B+ (densify). Add them in that order only if the
  re-costed core leaves room.
- **If the core is over budget, cut in this order:**
  1. arm A from 8 to 4 draws at the new checkpoints;
  2. checkpoints 350 and 1,100;
  3. arm B's extension from 128 to 64.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

Every comparison so far on Olmo 3.1 Think 32B used two or three points of one RL run. AI2 released 46 checkpoints of
it (`step_0050` … `step_2300`, every 50). This run measures the trajectory:
- IFBench pass@1 and pass@8 by RL step (Dmitry's question);
- when each of the 48 instruction-following reach prompts becomes solvable, and whether gradually or abruptly
  (claim 6);
- whether the prompts that prompting alone recovers in `olmo-ifladder` are reached earlier than the teachable and
  "neither" ones (habit before skill);
- whether the gamed share of checker wins rises or falls over training.

## Design (pre-register your version before any new draw)

1. **Model and checkpoints:**
   - `allenai/Olmo-3.1-32B-Think` at `step_0050`, `0100`, `0200`, `0350`, `0500`, `0750`, `1100`, `1600` and `2300`
     (`main`; confirm it equals `step_2300`);
   - plus the DPO start `allenai/Olmo-3-32B-Think-DPO` as step 0.

   Record each revision's commit hash. Use `olmo-ifomega`'s decoding: the Think RL settings, with a 32,768-token cap.
2. **Reuse:** inventory the stored samples for `dpo`, `s0100`, `s0750` and `s2300` (`olmo-ifomega`'s fixed phase and
   extensions, and `olmo-iffu`'s `s0750` draws on the reach prompts). Reuse them only where the template, decoding and
   prompt text are identical. Say which you reused and which you re-sampled.
3. **Arm A, aggregate curve:** the 218 IFBench prompts `olmo-ifomega` kept after its SFT exclusion, 8 draws per prompt
   per checkpoint.
   - Report strict pass@1 and pass@8 (unbiased estimator), with prompt-bootstrap intervals.
   - Report overall and per constraint family.
4. **Arm B, reach emergence:** `olmo-ifladder`'s 48 reach prompts (42 IFBench, 6 IFEval), 32 draws per prompt per
   checkpoint.
   - At each prompt's first checkpoint with a genuine success, extend the checkpoint before it to 128 draws.
   - Report each prompt's genuine rate by step, its first step with ≥ 1 genuine, and its first with ≥ 2.
5. **Arm C, gaming over training:**
   - the hand gate on every deciding success in arm B;
   - plus a seeded sample of up to 40 strict successes per checkpoint from arm A.

   Use the blinded reader sub-agents, confirmed by you, under `olmo_ifomega/handcheck_rule_if.md` and `olmo-iffu`'s
   addendum. Report the gamed share by step.
6. **Arm D, lengths:** median thinking and answer length (tokens and words) by step, for all prompts and for the reach
   prompts.
7. **Optional arms:**
   - **F:** AIME 2025–26 (`olmo-think32b`'s 59 problems) at the new checkpoints, 8 draws each, graded as in that run.
   - **E:** `allenai/Olmo-3-7B-Think` and its DPO start, arm A only, at every 100th step, 4 draws each.
   - **B+:** for a seeded 10 reach prompts, every 50-step checkpoint between the last one with zero successes and the
     first with a rate ≥ 0.25.

## Pre-registered readings (from the proposal; keep the thresholds)

- **Timing by class.** Use `olmo-ifladder`'s classes: prompt-recoverable, teachable-only and "neither" (the 7:
  `ifeval_3478` and IFBench 3, 67, 166, 188, 268 and 284). Report `ifbench_154` under both of that run's rules.
  - **"Habit first":** prompt-recoverable median first-genuine step < teachable median ≤ "neither" median, and the
    prompt-recoverable vs rest difference has a prompt-bootstrap 90 % interval above 0.
  - **"No order":** otherwise.

  Prompts never reached count as step > 2,300.
- **Gradual or abrupt,** per reached prompt:
  - **gradual:** at least two sampled checkpoints lie between the first with any genuine success and the first with a
    rate ≥ 0.25;
  - **abrupt:** the prompt goes from 0 of 128 to ≥ 0.25 at the next sampled checkpoint.

  Report the counts and the earliest detectable rates.
- **Gaming.** Compare the gamed share among strict successes at steps ≤ 200 with steps ≥ 1,100.
  - **"Gaming first":** early exceeds late by ≥ 0.2.
  - **"Gaming grows":** late exceeds early by ≥ 0.2.
  - **"Flat":** otherwise.
- **Brevity.** Find the step by which median answer length is within 20 % of its step-2,300 value, against the step
  by which half the reach prompts have ≥ 2 genuine. **"Brevity first"** if the former is earlier.
- **Aggregate shape:** widening (pass@1 and pass@8 rise together) or sharpening (pass@1 rises toward a flat pass@8).
  State the criterion before the data.

## Deliverables

- `run_olmo_ifcurve.md` (≤ 400 words + one figure);
  - Panel 1: arm A's pass@1 and pass@8 against RL step.
  - Panel 2: arm B as a heatmap, reach prompts × steps, coloured by genuine rate and grouped by `olmo-ifladder` class.
- `numbers.md` § olmo-ifcurve; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words.
- `olmo_ifcurve/`: code, the checkpoint manifest with hashes, the reuse inventory, `results.csv`, the per-prompt
  emergence table, the hand-check table, and every sample's text in the bucket.
- A `STATUS.md` line `OLMO-IFCURVE DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-ifcurve/`, delete
  pods, and `touch ~/runs/olmo-ifcurve/executor.done`.

**For the reviewer:**
- Check the checkpoint hashes and the reuse inventory (identical template, decoding and prompts).
- Re-grade the results, and repeat the hand gate blind on the deciding successes and on a sample of arm C.
- Recompute the readings, including the class medians and their bootstrap intervals.

**Rules:**
- Do not contact anyone.
- Weights stay on pods.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
