---
written_on: 2026-09-28
written_by: agent:claude
---

# c003 relaunch: pretraining on the pod

First launch (21:44 PDT 2026-09-27) never started training: uploading 186's 38 MB `pretrain.pt`
to each pod stalled at 20-29 MB after 25 min (8.5 KB/s, then 0 B/s). This Mac's uplink was
saturated by ~6 concurrent `hf upload` processes from the autoresearch loops, each with 8 fixed
Xet connections (`pod.py` sets `HF_XET_FIXED_UPLOAD_CONCURRENCY=8`); a fresh 5 MB ssh upload did
not finish in 5 min. Pods terminated after 41-45 min (~$1.06, no training).

Relaunch (22:25 PDT) with `launch.py --pretrain-on-pod`: each pod runs 186-claude's `pretrain.py`
(sha f8451342..., the same file that produced the c001/c002 weights) for its 300 s budget instead
of loading the saved checkpoint, so only ~60 KB of code crosses the uplink. RL config, seeds,
targets and eval are as registered.

Consequence for the comparison with c002: the weights are no longer identical, so pretraining
noise enters. 185 vs 186 (same recipe) gave 3-seed means 684 vs 680, so expect roughly +-10 on the
mean from this. It also makes c003 an end-to-end run: total_secs includes the 300 s pretraining,
so it can be read directly against the harness's 2,400 s budget.

HF pushes are deferred for all combined-pipeline runs from here (`pod.py push` later), because
pod.py pushes before terminating the pod and a slow uplink left pods idling.
