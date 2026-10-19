# Run brief: fast-stage1 — make Stage-1 training use the GPU (proposal 15, item 1)

Run id `fast-stage1`. Repository `~/work/fast-stage1` (branch `dan_fast-stage1` from the fork's
`origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` (nd-rl canonical) governs. Budget
**$5**, ceiling **10 pod-hours**. Authorised by Dan 2026-09-28 as part of proposal 15's queue; Claude
quota may throttle this run, which is accepted. Read first: proposal 15
(`~/nd-rl/docs/proposals/2026-09-28-repo-and-method-improvements.md`, §1), `train.py`, `model.py`,
`lean_tok.py` (`shift_abs`, the random name offset), `NOISE_FLOOR.md`, the `stage1-dynamics` summary.

## Measured starting point (2026-09-28)

A 6,000-step Stage-1 run (batch 128) takes 1,288 s on an A40, ≈ 215 ms per step. Sequences average 102
tokens (p99 198), so ≈ 61k useful tokens/s — ≈ 1.2 TFLOP/s of model compute, **0.78 %** of the A40's bf16
dense peak. `train.py` builds each batch in Python one row at a time (`torch.tensor` per row, `.to(dev)`),
pads every batch to its longest row (≈ 1.9× compute per useful token), does not compile the model. The
whole run is ≈ 78 M tokens.

## Goal

The same models, ≥ 5× faster wall-clock (aim for 10×), and several seeds per GPU at once.

## Suggested changes (your call; say what you did and why)

Pre-tokenise the training set once to GPU-resident tensors; sample batches and apply the random name
offset on the GPU (the augmentation's distribution must be unchanged — test it); bucket by length or pack
sequences (with the attention and loss masking that packing needs); `torch.compile`; fused AdamW; sweep
batch size with the learning rate scaled; then run N seeds concurrently per GPU (separate processes, or a
batched ensemble) and measure the aggregate throughput curve. Keep the old path behind a flag until the
acceptance test passes; `train.py`'s CLI and checkpoint format stay compatible.

## Acceptance test (pre-register it in `preregistration/fast-stage1.md` before the first pod)

Equivalence, not just speed. At equal training tokens, new recipe vs old, **≥ 4 Stage-1 seeds each**, on
the control set `data/p2/train_depth3_f0_a1.jsonl`: per-length validation loss (the opt-in logging `stage1-dynamics` added to `train.py`) and
held-out greedy accuracy per length bin must agree within `NOISE_FLOOR.md`'s minimum detectable
difference at that n. If the batch size changes, say what "the same recipe" means (tokens, epochs, final
learning rate) and justify it. Report wall-clock per model, useful tokens/s, the fraction of bf16 peak,
padding waste, and seeds-per-GPU aggregate throughput.

## Deliverables

The training changes, merged into the fork's `dan` **after** the acceptance test passes (the reviewer
follows; a quarantine verdict means revert); `FAST_STAGE1.md` at the repo root — how to train one model
and N seeds per GPU, measured speeds, the equivalence table; `preregistration/fast-stage1.md`;
`run_fast_stage1.md` (≤ 300 words); `numbers.md` § fast-stage1; `log.md`; `STATUS.md` line
`FAST-STAGE1 DONE <UTC>`; bucket `hf://buckets/dan-pandori/nd-rl/fast-stage1/artifacts`; pods deleted;
`touch ~/runs/fast-stage1/executor.done`.
