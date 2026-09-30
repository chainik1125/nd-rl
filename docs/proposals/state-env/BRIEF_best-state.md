# Run brief: best-state — Robbie's network and pretraining recipe in the proof-state format, cap 6 and cap 12

Run id `best-state`. Repository `~/work/best-state` (branch `dan_best-state` from the fork's `origin/dan`, made by
`newrun`). Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$60**, ceiling
**120 pod-hours**. Authorised by Dan 2026-09-30: "I'd like to run this experiment today". Proposal:
nd-rl `docs/proposals/2026-09-30-best-network-proof-state.md`, which is the question.

Read first:
- The proposal.
- In nd-rl, branch `origin/robbie-experiments`, directory `code/experiments/current/factorial_20260929/`:
  `README.md`, `pretrain_best.py` (the recipe; built on the fork's `GPT`), and `harness_combo.py` (how it is called).
  The incumbent's lineage is in `code/experiments/current/autoresearch/`. Reuse Robbie's code; don't rewrite it.
  Credit it in the write-up. Do not contact Robbie or anyone else.
- The `state-cap12`, `state-env`, `long-pool-2` and `textbook72` summaries in `~/nd-rl/experiment-summaries/`.
- `STATE_ENV.md`, `state_train.py`, `state_ladder_ei.py`.

**Code not on `dan`:** `lpool_reread.py` and `pod/sc12/` (including `sn_seed.sh`, the SN-cap12 recipe) are on
`origin/dan_state-cap12`. The textbook72 problems and their read-out are on `origin/dan_textbook72`: reuse its
eval-only copy and its manifest rule.

## Build (first, and fast: the port should take hours, not a day)

`state_train.py --recipe best`, or a sibling script, trains Robbie's architecture and optimizer on (state,
action) data:
- the network: 6 × 384, Peri-LN, ALiBi / NoPE heads;
- the optimizer: Muon on the hidden matrices, AdamW for the rest;
- MTP at weight 0.3, and token-budget batches.

Keep the state trainer's data path, loss masking (action tokens only) and checkpoint format, so
`state_sample`, `state_ladder_ei` and `lpool_reread` load the result unchanged. Test on CPU:
- a checkpoint round-trip;
- one sampled episode;
- the loss decreases on a small slice.

Then one GPU pilot of ≤ 20 min.

## Stage-1 budget

Robbie's recipe trains for 300 s on whole-proof sequences. State data is different: more, shorter sequences.
- **Pilot:** seed 0 at cap 6, at 300 s and at 1,200 s. Pick the budget by validation loss and held-out greedy on
  the validation split, never on textbook72 or a transfer pool.
- **State the rule before the pilot,** and use one budget for every new model.

## Data locations

The two training sets are not on `dan` and not in its manifest. They are in the bucket:
- `hf://buckets/dan-pandori/nd-rl/state-env/data/p2/train_depth3_f0_a1.jsonl`: the cap-6 control set the SN-v2
  models were trained on, 133.7 MB. Identical copies are under `lean-format/` and `round3-run4b/`.
- `hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz`: K12, the SN-cap12 set.

Check both against the record counts in the `state-env` and `state-cap12` summaries (155,000 each) before
training.

## Arms

| cell | data | seeds |
|---|---|---|
| best-cap6 | the cap-6 control set (`data/p2/train_depth3_f0_a1.jsonl`, the SN-v2 cap-6 set) | 3 |
| best-cap12 | K12 (`data/kh/train_k12.jsonl`, the SN-cap12 set) | 3 |

- **Inherited comparators, not re-trained:** SN-v2 cap-6 frozen and T1 s0 / s1 (`state-env`); SN-cap12 frozen
  and T1 s0–s3 (`state-cap12`). Use the same seeds' read-outs where they already exist on the same protocol,
  and read the rest here with identical settings.
- **T1:** exactly `state-cap12`'s protocol (`state_ladder_ei.py`, 8 rounds, k 32, T 0.8, `rl_targets.jsonl`,
  replay from the cell's own data), one pod per ladder, all six in parallel.
- **Batch:** the largest that fits. Report peak memory.

## Read-outs (frozen as soon as Stage-1 finishes, T1 at the end)

- **textbook72**, pass@256, T 0.8, `max_steps` 96, the same code and settings as the `textbook72` run. Report
  dev58, train14, all 72, and by `reference_lines`.
- **Robbie's dev metric:** the dev half of the ladder transfer pool as his harness defines it (1,108 theorems),
  64 samples at T 0.8, solved at L_true ≥ 7. Also his **holdout250** at pass@256, if that pool is on his branch
  (`score_holdout.py --pool holdout250`).
- **Long pools:** `rr600` L_true 13–16 and `transfer_long2`, k 256.
- **Held-out greedy.**
- **Compute:** a per-arm table of GPU-seconds, generated tokens and training tokens. The comparison is recipe
  against recipe, not compute-matched; say so.

## Pre-registration (before the first pod)

For each new cell, the expected textbook72 IQM (frozen and T1) and dev metric, and the sign of
(best − ours) at cap 6 and at cap 12. Include a falsifier for "Robbie's recipe helps in the state format", and
the MDD at 3 vs 2 (cap 6) and 3 vs 4 (cap 12) seeds from `NOISE_FLOOR.md`, or say why it does not apply.

## Timing

Dan wants this today.
- Post the frozen read-outs in `STATUS.md` as soon as they exist, marked unreviewed: they answer the pretrained
  half.
- Then start the six ladders at once.
- If the ladders' projected wall-clock passes 24 h, say so in `QUESTIONS.md`. Propose a cut (for example 6
  rounds) and continue on your default.

## Deliverables

- The recipe port, merged into the fork's `dan` after review.
- `preregistration/best-state.md`, `run_best_state.md` (≤ 400 words + a figure: textbook72 and dev metric by
  cell), `numbers.md` § best-state, `log.md`, and `STATUS.md` line `BEST-STATE DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/best-state/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/best-state/executor.done`.

Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
