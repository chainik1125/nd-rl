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
| `results/*.json` | written by the agent | prediction before, outcome after, one per experiment |

Artifacts (`summary.json`, `final.pt`, logs, found proofs) land in `artifacts/autoresearch/<id>/`
(gitignored). `final.pt` is a state_dict plus tokenizer mode, so the holdout half of the
transfer pool can be scored later with the matching `pretrain.py` from git.
