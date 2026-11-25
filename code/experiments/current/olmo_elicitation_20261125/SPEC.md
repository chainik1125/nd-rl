---
written_on: 2026-11-25
written_by: agent:claude (orchestrator)
---

# Olmo elicitation curves: spec

Leon's goal (2026-11-25): (1) Olmo at different checkpoints, (2) elicitation measures on each
checkpoint, "particularly edl", graphed as curves over the checkpoints. First recreate Dan's setup.

This file is the contract between the orchestrator and the build agents. `PREREG.md` (written before
any pod result) fixes the readings. `log.md` records what ran.

## What exists already (Dan, public)

- Code: `dan-pandori/nd-takehome`, one branch per run. Local clone (read-only, pinned):
  `/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome` at `8e744a89` (branch `dan_olmo-ifcurve`).
- Samples and grades: public bucket, fetched with his `fetch_artifacts.py` (standard library only).
- Model: `allenai/Olmo-3.1-32B-Think`, revisions `step_XXXX`; RL starts from
  `allenai/Olmo-3-32B-Think-DPO` (`main`). 64.5 GB of bf16 weights per checkpoint.
- Run `olmo-ifcurve`: nine snapshots (dpo = step 0, 50, 100, 200, 500, 750, 1100, 1600, 2300) on 218
  "arm A" prompts (pass@1/2/4/8) and 48 "reach" prompts (32 tries per snapshot; 42 of the 48 are also
  in arm A). Sampler `olmo_ifladder/sample.py` (vLLM 0.11.0, T 1.0, top_p 1.0, 32768 max tokens,
  `olmo_thinker` template), grader `olmo_ifomega/grade.py` (official strict IF checkers).
- Earlier runs measured elicitation at the start only: `olmo-iffu` (60-word hint), `olmo-ifladder`
  (instruction ladder, LoRA on 16 trajectories of the final model: `lora.py`, r 16, alpha 32,
  lr 1e-4, 4 epochs), `olmo-transplant` and `olmo-answeronly` (final model's thinking pasted in).
- Caveat from `olmo-weights`' review: steps 150 and 200 lie on an abandoned branch of the RL run.
  We keep step 200 to line up with his curve and mark it.

## Checkpoints

`dpo` (0), `s0050`, `s0100`, `s0200`*, `s0500`, `s0750`, `s1100`, `s1600`, `s2300`. Revisions and
weight digests: Dan's `olmo_ifcurve/checkpoints.csv`. Every job records the commit it loaded.

## Stage 0: recreate Dan's setup (CPU on serrano, then one pod)

0a. Fetch his stored samples and grades, re-grade with his grader, reproduce `passk.csv` (pass@1/2/4/8
    per checkpoint, 218 prompts) and the reach-prompt table. Report any difference.
0b. Our figure of his curve (pass@1, pass@8 and the reach prompts' solved share per checkpoint), drawn
    through `code/ndbench/figures/style.py`.
0c. On a pod: his `setup.sh` versions, his sampler unchanged, a fresh replication draw at two
    checkpoints (`dpo`, `s2300`): the 218 arm-A prompts, 2 draws each, new seed. Pass rule fixed in
    `PREREG.md`.

## Stage 1: EDL at each checkpoint (the main new measurement)

Excess description length as in `code/ndbench/edl.py` (Donoway et al., arXiv 2601.04728):
`EDL(n) = sum_{i<=n} l(theta_{i-1}; x_i, y_i) - n * L_test(theta_n)`, online LoRA fine-tuning.

- **Data.** `y` = a stored strict-pass response (thinking and answer, the stored token ids) of the
  final model `s2300` to prompt `x`. One response per prompt. Prompts where a hand label says "gamed"
  are excluded; a hand-labelled "genuine" draw is preferred. Three disjoint prompt sets:
  `train` (n = 64, arm-A prompts that are not reach prompts), `test` (m = 32, same rule),
  `reach` (the 48 reach prompts that have a genuine `s2300` response). A response-token cap bounds the
  cost (set by the orchestrator after the agent reports the length distribution).
- **Loss.** NLL in nats of the response tokens given the templated prompt (Dan's `lora.py` encoding,
  unchanged). Stored per example and per segment: `think` (up to and including `</think>`) and
  `answer` (the rest, EOS included), with token counts.
- **Learner.** PEFT LoRA on Dan's target modules, r 16, alpha 32, dropout 0 (his 0.05 would make the
  online score depend on a random mask), AdamW (0.9, 0.999), weight decay 0, grad clip 1.0, batch 1,
  constant learning rate, one step per example. bf16 weights, fp32 LoRA.
- **Protocol** per (checkpoint, lr, order): score example i in eval mode, then one training step on
  it, for i = 1..64. `L_test` on `test` at n' in {0, 8, 16, 32, 64}; `L_reach` on `reach` at
  n' in {0, 64}.
- **Grid.** lr in {3e-5, 1e-4, 3e-4} (a pilot at `dpo` may shift it by one step; the grid is then
  frozen for all checkpoints). Orders: 2 at `dpo`, `s0750`, `s2300`; 1 elsewhere.
  `EDL* = min over lr of mean-over-orders EDL(64) + ln(number of lrs)`.
- **Readouts per checkpoint**, each as a curve over RL steps, total and by segment:
  `L_0` (nats per token of the final model's successful responses before any fine-tuning: the
  likelihood curve), `EDL*` (nats and nats per response token), the shape of EDL per token over n'
  (falling throughout, or rising first), `L_64`, and the share of the gap to `s2300` that 64 examples
  close: `(L_0(t) - L_64(t)) / (L_0(t) - L_0(s2300))`.
- **Null control.** `s2300` fine-tuned on its own responses: EDL should be near zero.

## Stage 2: sampling arms Dan ran only at the start, repeated at every checkpoint

- **T (transplant).** His `olmo-answeronly` arms on the 41 transplant prompts: the final model's
  thinking pasted in, with and without the "reply with only your final answer" instruction;
  16 draws; strict checker. Cheap (answers only).
- **H (hint).** "Answer in at most 60 words" on the 48 reach prompts, 8 draws; strict checker.
  Plain rates at the same checkpoints come from Dan's stored arm B.
- **L (LoRA-elicited reach), optional, only if budget remains.** The 48 reach prompts sampled through
  the stage-1 adapter of each checkpoint (best lr, order 0), 8 draws.

Checker passes are reported as checker passes. Genuine-versus-gamed reading is not part of the first
pass and is said so wherever a sampling arm is shown.

## Order on a pod, per checkpoint

download -> stage 1 (EDL) -> T -> H -> (L) -> results pulled to serrano and md5-checked -> weights
deleted. One checkpoint's results never wait for another's.

## Budget and rules

- Runpod balance 2026-11-25 02:15 PT: $133.66. Cap for this experiment: $110. H200 community
  $3.59/h (secure $5.29/h).
- Only the orchestrator creates and deletes pods. Agents never do, and never start a job on a pod
  that is not listed for them.
- At most one lab GPU in total (smoke tests of the EDL runner on a small model; never share a GPU).
- No sealed ND set goes to any pod (none is needed here).
- Every long job has a timeout and a watchdog line; every pod job writes a heartbeat.
- Results as percentages of the prompt set; pass@1 and pass@8 lead; figures through
  `code/ndbench/figures/style.py` (5.5 in wide, serif 7-8 pt, PDF/SVG/PNG).

## Where things go

- Code and notes: this folder (`olmo_elicitation_20261125/`), branch `leon/olmo-elicitation`.
- Data and results: `/scratch/cowifornia/nd-rl/olmo_elicit/` (`ext/` Dan's clone and fetched samples,
  `pool/` EDL data, `runs/<checkpoint>/`, `figures/`).
