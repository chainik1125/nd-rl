---
written_on: 2026-11-28
written_by: agent:claude (orchestrator of this measurement, Fable 5.2)
---

# Log: weight interpolation between `dpo` and `s2300`

Times are PT. Newest entry last.

## 2026-11-28 2:56 am: pre-registration written

`PREREG.md`, before any pod, weight or sample. Inputs built on serrano by `make_inputs.py`
(`/scratch/cowifornia/nd-rl/olmo_elicit/interp/inputs/` and `refs/`): the 48 reach prompt ids, 4 draws per prompt,
the prompt token hashes of metrics 2's ExPO reach cell, and the stored per-prompt rates.

## 2026-11-28 3:15 am: independent audit

One fresh auditor (Opus 5.5), not given my conclusions. 15 findings in `discourse/audit.md`, Turn 1.

## 2026-11-28 3:42 am: Amendment 1 and replies to the audit

`PREREG.md` Amendment 1 and `discourse/audit.md` Turn 2. Main changes: a pre-registered smooth reference beside the
reading rule (a path with no switch also earns the rule's lowest verdict, which was then named after a switch); a
stricter exact check plus a bit-for-bit recompute of all 707 tensors; crossed bounds for cut cells; the corrected
statement on seeds; writes overlapped with sampling; 500 GB volume and a file-size disk budget.

## 2026-11-28 3:46 am: preparation finished (no pod, no GPU, nothing sampled, nothing committed)

Code, tests (52 pass) and `RUNBOOK.md` are in this folder; ruff check and format are clean. Not yet measured: the
speed of the write check on the pod. Waiting for the main thread to create the pod.

## 2026-11-28 4:01 am: coordinator's decisions applied (still no pod, CPU only)

- Verdict names changed by the coordinator (`PREREG.md` Amendment 2, 3:58 am PT, the coordinator's text verbatim): "below
  proportional at the midpoint", "about proportional", "above proportional", "unclear at this size". The report
  always prints the rule's verdict and the second line against the smooth reference, and names a switch only with
  "sharper than the smooth reference". Both predictions (his and mine) are now recorded as "below proportional at
  the midpoint" with "compatible with a smooth path".
- Schedule: the coordinator chose the default (writes on the CPU while cells sample). I fall back to
  `--mode sequential` if `wi_pod.sh podcheck` says so (memory limit under 96 GB or unreadable, or fewer than 345 GB
  of the volume unused); the choice made on the pod will be written here.
- Shared pod: merged checkpoints go to `/workspace/wi/models`; the disk budget counts everything under
  `/workspace`; `pod/wi_download.sh` records whether `dpo` and `s2300` existed before this run;
  `pod/wi_release.sh` (through `wi_pod.sh release`, only after an md5-checked fetch) removes this run's merged
  checkpoints and the `s2300` it downloaded, leaves `/workspace/models/dpo`, and writes `/workspace/wi/GPU_FREE`.
  Dan's fetch script can only write to `/workspace/models`, so the two downloads are there and not under
  `/workspace/wi`.
- minCudaVersion 12.8 and the 500 GB volume are accepted.
