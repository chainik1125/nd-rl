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

## 2026-11-28 4:09 am: pod up, operation started

Code as committed: `938f9017` on `leon/olmo-elicitation` (recorded in `$WI_DATA/run_meta.json`). Pod `yvq82k1iewzheh`
(`nd-rl-olmo-night-1128`, 1 x H200, secure US-CO-1, $5.29 per hour), created 2026-11-28T12:08:23Z (4:08 am PT), fresh,
no prior spend; alias `rp-wi`. Push at 12:09:59Z: 24 files, md5 equal on both sides. Set-up started 12:10Z.

`wi_pod.sh podcheck`: container memory limit 251 GB, CPU quota 13.6 cores (128 visible), 0 GB on the volume, GPU
empty. Result: `MODE lanes`, so the default schedule is used (writes on the CPU with 8 threads while cells sample).
Chains written with the pod's start time and a $21 cap: `$WI_DATA/chains/chain_prep.txt`, `chain_cells.txt`.

## 2026-11-28 5:28 am: run finished, fetched, pod released

Timeline (PT; the pod's logs are UTC, 8 hours ahead):

- 4:08:23 pod created. 4:10 push of 24 files. 4:11:38 set-up done (vLLM 0.11.0, torch 2.8.0+cu128, transformers
  4.57.1; the same versions and the same two code commits as yesterday's pod; grader self-test and GPU test passed).
- 4:12 both chains started. `s2300` downloaded and sha256-checked by 4:16:04, `dpo` by 4:20:46, preflight 4:20:48.
- Cells (sampling start to end; load, generation): alpha 1.0 4:20:49 to 4:38:05 (94 s, 935 s); alpha 0.5 4:38:08 to
  4:53:54 (117 s, 823 s); alpha 0.75 4:54:00 to 5:10:05 (92 s, 867 s); alpha 0.25 5:10:11 to 5:22:29 (84 s, 647 s).
  About 1,460 to 1,560 generated tokens per second. The GPU was never idle between cells: every merged checkpoint
  was written and checked before its cell's turn.
- Writes on the CPU with 8 threads, beside a sampling cell (write, then check): alpha 0.5 225 s + 362 s, done 4:30:39;
  alpha 0.75 213 s + 431 s, done 4:41:29; alpha 0.25 with the census 339 s + 872 s, done 5:01:45.
- Each merged checkpoint was deleted on the pod after its cell was counted complete (4:53:59, 5:10:11, 5:22:36).
- Fetches: each cell and each write report right after it ended (md5 equal on both sides each time); final fetch
  5:23:46, 141 files md5-equal (`$WI_DATA/fetch_rp-wi.md5`). My own md5 of the eight per-draw files on serrano agrees
  with that list.
- 5:24:06 release: `/workspace/models/s2300` removed (this run had downloaded it), `/workspace/models/dpo` left in
  place (64,476,938,345 bytes, commit `97604023`, 14 shards sha256-checked at download), tokenizer left in place,
  GPU memory 0 MiB, marker `/workspace/wi/GPU_FREE` written. The pod itself was not stopped or changed by me.
- Pod time of this run: 4:08:23 to 5:24:06 = 1.26 hours, $6.68 at $5.29 per hour. The watchers on serrano logged
  no stall, no overrun and no missing status, and have exited.

Result (full text in `PREREG.md`, section "Result"): pass@1 1.0 % (0.0 to 2.6) at alpha 0.25, 13.5 % (7.3 to 20.3) at
0.5, 53.1 % (42.7 to 63.5) at 0.75, 69.3 % (59.4 to 78.6) at 1.0; all cells complete, no draw cut. r(0.5) = 19.5 %
(11.1 to 29.2): "below proportional at the midpoint" and "compatible with a smooth path". Half-gain point alpha 0.63
(0.60 to 0.67). The write checks found no unequal element. The merged checkpoints sit at 0.143, 0.421 and 0.856 along
the line (bf16 rounding), which the result section states beside the verdict.

Two things done outside the RUNBOOK's list, both read-only: a polling helper in the session scratchpad that waits for
a job's end with `wi_pod.sh status`, and a trial run of `analyze.py` on three cells into a scratch folder at 5:11 am
(recorded in the result section as a deviation). No file of the code was changed after `938f9017`; the files changed
since that commit are `PREREG.md` (result section appended) and `log.md` (two entries).
