---
written_on: 2026-11-27
written_by: agent:claude (build, Sonnet 5.5)
---

# Pod tooling for the three Olmo measurements (2026-11-27)

Serrano side drives a pod through an ssh alias `rp-<name>` (entry in `~/.ssh/config`); it never creates or deletes a pod
and never prints a key. Pod root `/workspace/m2` (data in `runs/<job>/`), weights in `/workspace/models/<ckpt>`
(shared with the 2026-11-25 run). Every command takes `--dry-run`. Copy of the code in this folder: `X` =
`code/experiments/current/olmo_metrics2_20261127`.

    bash X/pod/m2_pod.sh setup    rp-a              # push + Dan's pod set-up (~10 min, idempotent), waits for SETUP_DONE
    bash X/pod/m2_pod.sh push     rp-a              # code, inputs/ and the EDL pool; md5 on both sides, refuses on mismatch
    bash X/pod/m2_pod.sh download rp-a s0100 [--verify-sha]   # job download_s0100 (own lock); pinned commit, verified
    bash X/pod/m2_pod.sh job      rp-a temp_dpo_T06 14400 [--force] [--nogpu] -- <command...>
    bash X/pod/m2_pod.sh status   rp-a [job ...]    # state, heartbeat age, progress, GPU MiB, disk GB, newest file
    bash X/pod/m2_pod.sh fetch    rp-a temp_dpo_T06 # -> $M2_DATA/runs/<job>/, FETCH_OK | FETCH_PARTIAL | FETCH_MISMATCH
    bash X/pod/m2_pod.sh stop     rp-a temp_dpo_T06 # SIGTERM to that job's recorded process groups only
    bash X/pod/m2_watch.sh rp-a temp_dpo_T06 90     # logs STALL / OVERRUN / EXIT to $M2_DATA/logs/watch_rp-a.log; kills nothing

`M2_DATA` = `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2`. Inputs the designs put in `$M2_DATA/inputs/` are pushed to
`/workspace/m2/inputs/`. A push refuses (nothing sent) if a staged file name contains `textbook72`, `tb72`, `test_13_20`,
`test_21_30` or looks like a credential.

## A sampling job

The command runs with cwd `/workspace/m2`. Sampling goes through the wrapper (Dan's sampler stays unedited), grading
follows inside the same job:

    bash X/pod/m2_pod.sh job rp-a temp_dpo_T06 14400 -- bash -c '
      mkdir -p runs/temp_dpo_T06/out &&
      python3 sample_wrap.py --dan-root /workspace/nd-takehome --temperature 0.6 --top-p 1.0 \
        --sidecar runs/temp_dpo_T06/out/wrap.json -- \
        --model /workspace/models/dpo --tokenizer /workspace/models/dpo --name dpo --rev <commit> \
        --problems <problems.jsonl> --pids @inputs/pids_armA.txt --n 2 --chunk 2 --seed 1 --job temp_dpo_T06 \
        --out runs/temp_dpo_T06/out/samples.jsonl.gz --meta runs/temp_dpo_T06/out/meta.jsonl &&
      bash tools/grade_job.sh runs/temp_dpo_T06/out/samples.jsonl.gz runs/temp_dpo_T06/out/graded.jsonl <problems.jsonl>'

`sample_wrap.py [--dan-root D] [--temperature F] [--top-p F] [--patch mod.py] [--sidecar f.json] -- <Dan's argv>`:
sets Dan's module constants `TEMPERATURE` / `TOP_P` (read at call time in `submit` and in his meta dict), asserts after the
run that the last line of his `--meta` file has the overridden values, writes the sidecar (T, top_p, patch path and
sha256, sha256 of his `sample.py`, his argv, start / end UTC). `--patch` module: `apply(args)` is called right before
vLLM's `LLM()` is constructed (the wrapper replaces `vllm.LLM` by a factory); `args` is a SimpleNamespace with
`dan_argv`, `llm_kwargs` (what Dan passes to `LLM()`, read only), `temperature`, `top_p`, `dan_root`, `sidecar`, `patch`.

## Chains

    bash X/pod/m2_pod.sh chain rp-a main1 chain.txt     # detached; jobs in order on the pod
    bash X/pod/m2_pod.sh status rp-a                    # also lists chain:<name> (state, progress k/n, rcs=job:rc,...)
    bash X/pod/m2_pod.sh stop rp-a --chain main1        # no further job starts; the current job is stopped

chain.txt, one job per line (blank lines and `#` comments ignored; the command is shell text):

    <job-name> <timeout-seconds> <continue-codes, e.g. 0 or 0,11> [--nogpu] -- <command...>

The chain stops at the first job whose exit code is not in its list (state `runs/_chains/<name>/state/FAILED`,
`rcs` has each job's rc, log in `log/chain.log`); a job whose `state/DONE` exists is skipped, so re-submitting the chain
resumes it. `--nogpu` jobs take `download.lock`, so they can run beside a GPU job.

## What the pod runner guarantees (`pod/m2_job.sh`)

One GPU job at a time (`flock` on `/workspace/m2/gpu.lock`; a second start exits 3 with `LOCKED` in
`runs/<job>/logs/runner.log`); a job with `state/DONE` is refused (exit 4) unless `--force`; GPU jobs wait up to `GPU_WAIT` s (120) for every GPU's used memory to fall below 2,000 MiB, else exit 5 with
`state/FAILED rc=5 GPU_BUSY`; `timeout --kill-after=60`
around the command; heartbeat every 60 s in `state/heartbeat` (`ts job newest newest_mtime newest_size gpu_mem gpu_util`);
`state/{RUNNING,DONE,FAILED}` (FAILED holds `rc=`, plus `stopped` or `timeout`); `logs/job.log` is the command's output. After the command exits, any member left in its process group (a vLLM engine
core) gets SIGTERM and, after `KILL_GRACE` s (10), SIGKILL. `stop` signals the command's group (SIGTERM, then SIGKILL after
`STOP_GRACE` 20 s); the runner itself is not signalled and writes FAILED with `stopped`.
Setup: `pod/m2_setup.sh` runs a copy of the 2026-11-25 `oe_setup.sh` (pushed unedited as `pod/oe_setup.sh`) after
`sed 's#/workspace/oe#/workspace/m2#g'` into `meta/oe_setup.m2.sh` (diff kept in `meta/oe_setup.sed.diff`).

## Grading

`tools/grade_job.sh <samples.jsonl.gz> <graded.jsonl> [<problems>]` calls Dan's `olmo_ifomega/grade.py` (default problems
file: his `olmo_ifomega/problems.jsonl`). On the pod it uses `python3` and `/workspace/nd-takehome`; on serrano it uses
`olmo_elicit/venv_grade` and his pinned clone.

## Tests (CPU, serrano)

    /scratch/cowifornia/venvs/olmo-edl/bin/python -m pytest -q X/tests
