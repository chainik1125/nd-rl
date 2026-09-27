# Autoresearch loop for the pretrain workstream

Started 2026-09-22 (agent:claude, at Robbie's request). Rules in `program.md`; the proposal it
implements is `docs/proposals/2026-09-20-pretrain-workstream-constraints.md` with Robbie's
changes: 10 minutes total, metric = dev theorems solved at true length >= 7, holdout ignored
during the loop, single seed, no noise threshold.

## Why `pretrain_poc/pipeline.py` was split into `harness.py` + `pretrain.py`

`pipeline.py` was one script doing pretrain, RL and eval with the clock inside the same file.
For an agent loop that is unusable: the agent must be able to change the pretraining freely
while the timer, the parameter check, the RL protocol and the eval stay out of its reach.
So:

- `harness.py` (frozen) = everything from `pipeline.py` except `stage_pretrain`, plus a
  SIGALRM at 600 s, a pretrain-budget overrun check, the param-cap check, a sha256 of both
  files in every summary, and `status` in {ok, crash, timeout}. The frozen dev eval is dropped
  (it cost 65 s of the 600) and the metric is `ge[7]` instead of `ge[9]`.
- `pretrain.py` (agent-editable) = `stage_pretrain` + `train_timed`, taking the raw records
  and a budget and returning `(model, tok, stats)`.
- `pod.py` keeps one pod alive across experiments (`up` / `run` / `down`) instead of renting
  one per run; boot plus clone was ~3 minutes of the ~11 per run.

`pipeline.py` itself is untouched and still runs; nothing in the loop imports it.

## Files

| File | Editable by the agent | Purpose |
| --- | --- | --- |
| `harness.py` | no | clock, param cap, RL, eval, summary.json |
| `pretrain.py` | yes | the experiment |
| `remote.sh` | no | pod-side runner, pins the fork commit |
| `pod.py` | no | RunPod lifecycle and artifact transfer |
| `program.md` | no | the agent's instructions |
| `params.py`, `record.py`, `table.py` | no | pre-register (params.json + patch), record results.json, regenerate RESULTS.md |
| `pools.py`, `pools/<fork>.json` | no | pool sizes and proof-length histograms per fork commit, copied into every params.json |
| `results/<NNN>_<slug>/` | written by the agent via the scripts | `params.json` + `pretrain.patch` before the run, `results.json` after |
| `holdout.py`, `score_holdout.py`, `score_holdout.sh` | no | offline: score HF `final.pt` checkpoints on the transfer holdout across many pods, write `results/<run>/holdout.json` (the dashboard's pink dots; the loop never reads it) |

Artifacts land in `artifacts/autoresearch/<id>/` (gitignored):

- `pretrain.pt`: weights immediately after valid pretraining, before evaluation or RL.
- `rl_round_01.pt`, etc.: weights after each completed RL round.
- `final.pt`: weights after the final evaluation succeeds.
- `source/`: exact `pretrain.py`, `harness.py`, and `remote.sh` used for the run, including
  discarded variants. `remote.sh` identifies the required external fork commit.
- `summary.json`, logs, found proofs, and development evaluation outputs.

Checkpoints contain `state`, `tok_mode`, stage, seed, and source hashes. Reconstruct the model
and tokenizer using the saved source and pinned fork, then load `state` with
`torch.load(path, map_location='cpu', weights_only=True)`. These are weights for evaluation
or new training, not exact training-resume snapshots (optimizer/RNG state is not included).
Checkpoint writes are atomic; completed stages survive a later timeout. Saving is charged
to the overall run budget. Existing runs cannot recover pretraining weights retroactively.

For a checkpoint export without RL or evaluation, pass `--pretrain-only` to `pod.py run`.
The summary records `stage: pretrain_only`, `status: ok`, and a null evaluation metric.

The public checkpoint repository is
[robbiethompson2018/nd-rl-checkpoints](https://huggingface.co/robbiethompson2018/nd-rl-checkpoints).
Checkpoints live there, not on this disk. After pulling a run, `pod.py run` uploads
`artifacts/autoresearch/RUN_ID/` to `runs/RUN_ID` and deletes each local `.pt` whose sha256
matches HF's copy; summaries, jsonl files, and `source/` stay local. A failed upload keeps the
checkpoints and says so in the driver log; so does a push cut off by a killed session.
`pod.py push [RUN_ID...]` retries (without names: every run directory untouched for 10
minutes), and `run_idea.py` runs it before each experiment. Pushing needs the `hf` CLI:
`uv tool install huggingface_hub`, then `hf auth login` with a write token.
To score or inspect a checkpoint, download it:

```bash
hf download robbiethompson2018/nd-rl-checkpoints --include 'runs/RUN_ID/*' --local-dir checkpoints
```

Preserve the checkpoint's stage in any sharing message: `final.pt` includes RL; `pretrain.pt`
does not.

## Keep rule: what was considered and not adopted (2026-09-22)

Seed noise at the 20-minute budget is sd ~44 on a mean of ~110 (`SWEEP_s20.md`), so a plain
ratchet will sometimes keep a lucky draw. Two fixes were designed and rejected as too
complicated for the value; Robbie chose "3 seeds per idea, keep if the mean is higher, accept
the noise". Written down in case the loop visibly stalls on a lucky bar:

- **Running-mean bar.** The bar is the mean over every seed ever run on the current
  `pretrain.py`, fed by control replicates (every 4th experiment and after every keep). Costs
  ~20% of GPU time, converges to the true mean of the kept config.
- **Two-stage 6 vs 6.** An idea runs 3 seeds; if its mean beats the bar, run 3 more and keep
  only if the 6-seed mean still beats the 6-seed bar. No controls needed, promoted ideas carry
  their own 6 seeds. With sd 44, ~1 in 5 null ideas still gets kept at a +3 threshold.

`run_idea.py` already merges any seed list, so either can be added later without touching the
harness.
