---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M3 steer, Sonnet 5.5)
---

# M3 (steer): pod steps in order

Design: `discourse/steer.md` Turn 5 section F with Turn 6 d1-d7 and the orchestrator's decisions (every step is a separate
job started by hand; draw counts and cuts are command-line parameters; split-half stop = exit code 42; E0' stays at 384
draws and is shared with M1). `X` = `code/experiments/current/olmo_metrics2_20261127`, `A` = alias `rp-<name>`.
`M2_DATA` = `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2`. Pod root `/workspace/m2`; the job command runs there.

## Before the first pod step

1. **Push needs `steer/`.** `m2_pod.sh stage()` copies only top-level `*.py|*.sh|*.md` of `X`, `pod/`, `tools/`, `inputs/`
   and the pool. It does NOT copy `X/steer/`. One line is needed in `stage()` after the `tools` line (I did not edit
   `pod/m2_pod.sh`): `[[ -d $X/steer ]] && { mkdir -p "$ST/steer"; cp -rp "$X/steer/." "$ST/steer/"; }`
   (then `find "$ST" -name __pycache__ ...` already cleans caches). Without it `steer/` must be rsynced by hand.
2. **Inputs** (built on serrano, already present): `$M2_DATA/inputs/steer/` = `manifest.json`, `texts.jsonl`,
   `preflight_seqs.json`, `dpo_baseline.json`, `pids_dev_guard.txt`, `pids_e0p_reach.txt`, `pids_e1.txt`,
   `pids_e2_s{1,2,3}.txt`, `pids_params.json`. Rebuild: `python -I X/steer/steer_prep.py build` (reads Dan's clone and the pool
   read only; 15-45 s; identical output on a rerun, tested). Draw counts are parameters of the pid files:
   `python -I X/steer/steer_prep.py pids --out $M2_DATA/inputs/steer --manifest $M2_DATA/inputs/steer/manifest.json --reach-n 8 --held-n 2 --dev-n 2 [--e0-reach-n 4]`
   (pure python; it can also run on the pod). Cut values: `--held-n 1`, `--dev-n 1`. The E0' cut `--e0-reach-n 4` is NOT used (decision 3).
3. `bash X/pod/m2_pod.sh setup A`, `push A`, `download A s2300 --verify-sha` after `dpo` (download `dpo` first; both are 64.5 GB;
   volume >= 200 GB). Pod needs `python3` with torch 2.8, transformers 4.57.1, vllm 0.11.0 (Dan's set-up).

Below, `P=/workspace/models`, `IN=inputs/steer`, `DAN=/workspace/nd-takehome`, `PR=$DAN/olmo_ifomega/problems.jsonl`.
Dpo revision for the sampler's `--rev`: commit `97604023f2602a7dfe6d31b1fc229985e0002022` (`olmo_ifcurve/checkpoints.csv`).

## Steps (each `bash X/pod/m2_pod.sh job A <job> <timeout-s> -- <command>`)

| # | job | what | timeout | expected | GPU |
|---|---|---|---|---|---|
| 1 | `m3_p1` | HF dpo: activation sums on V32 texts, unhooked dpo NLL | 2400 | 12-20 min (load 2-10, 0.5 M tokens) | yes |
| 2 | `download_s2300` | already a `m2_pod.sh download` job | - | 1-3 min | no |
| 3 | `m3_p2` | HF s2300: vectors, D_rev, s2300 NLL, 10 necessity forwards | 3600 | 15-25 min | yes |
| 4 | `m3_p3` | HF dpo: timing + shrink, sweep, selection, manipulation check, split-half, pre-flight refs | 6000 | 25-60 min (3,000 tok/s guess; the plan caps the projection at $3) | yes |
| 5 | `m3_pf` | vLLM pre-flight, one process per mode, then summarize | 3600 | 20-35 min | yes |
| 6 | `m3_dev` | dev guard, chosen setting | 2400 | 12-20 min | yes |
| 7 | `m3_e0p` | E0': zero vector, 48 reach x 8 | 4500 | 25-40 min | yes |
| 8 | `m3_e1` | E1: chosen vector, 48 reach x 8 + 80 held-out x 2 | 6000 | 35-60 min | yes |
| 9 | `m3_trigger` | E2 trigger from graded draws (pod CPU) | 300 | < 1 min | no (`--nogpu`) |
| 10 | `m3_e2_s1`, `s2`, `s3` | only if step 9 printed `E2_TRIGGER=1` | 4500 each | 12-25 min each | yes |

Check-ins every 30 min per job; `m2_watch.sh` logs STALL / OVERRUN / EXIT. The watcher's OVERRUN is 1.6x the expected time.

### 1. `m3_p1`
```
bash X/pod/m2_pod.sh job A m3_p1 2400 -- bash -c 'mkdir -p runs/m3_p1/out && python3 steer/steer_hf.py p1 --model /workspace/models/dpo --texts inputs/steer/texts.jsonl --out runs/m3_p1/out'
```
Writes `acts_dpo.pt`, `nll_dpo.json`, `p1_report_dpo.json`. Then `m2_pod.sh fetch A m3_p1` is not needed (stays on the pod).

### 3. `m3_p2` (after `download A s2300`)
```
bash X/pod/m2_pod.sh job A m3_p2 3600 -- bash -c 'mkdir -p runs/m3_p2/out && python3 steer/steer_hf.py p2 --model /workspace/models/s2300 --texts inputs/steer/texts.jsonl --p1 runs/m3_p1/out --out runs/m3_p2/out'
```
Writes `vectors.pt` (V_resp, V_prompt, V_resp_s2300text, per-prompt arrays), `group_means.pt`, `nll_s2300.json`,
`necessity.json`, `p2_report.json` (D_rev, vector norms, cosines, split-half per grid layer, top-5 shares). Fetch
`p2_report.json` and `vectors.pt` (about 100 MB) to serrano afterwards; delete the `s2300` weights on the pod.

### 4. `m3_p3`
```
bash X/pod/m2_pod.sh job A m3_p3 6000 -- bash -c 'mkdir -p runs/m3_p3/out && python3 steer/steer_hf.py p3 --model /workspace/models/dpo --texts inputs/steer/texts.jsonl --seqs inputs/steer/preflight_seqs.json --p1 runs/m3_p1/out --p2 runs/m3_p2/out --out runs/m3_p3/out --budget-usd 3.0 --usd-per-h 5.29'
```
Log lines to watch: `P3_TIMING tokens_per_s=... plan=... projected_usd=...` (the shrink rule, in the fixed order alpha 0.5,
layer 40, V_prompt row), `P3_GRID ...` per setting, `P3_CHOSEN <type> layer=.. alpha=.. no_admissible=.. at_max_alpha=..`.
Outputs: `plan.json`, `p3_report.json` (grid, D_rev, g with intervals and bands on total / thinking / answer, R at alpha = 1,
scramble g, damage rises, split-half), `chosen.json`, `vecs/{chosen,chosen_half,zero,scramble2001..2003}.pt` (final alpha*v with
sha256), `preflight_refs.json`.
**Exit code 42** = `STOP_SPLIT_HALF c=... converted=... c_top5_removed=... converted_top5_removed=...`: everything is already on
disk (state `FAILED rc=42`); you decide whether to go on. Any other non-zero code is a failure.

### 5. `m3_pf` (pre-flight; abort the pod run on FAIL)
```
bash X/pod/m2_pod.sh job A m3_pf 3600 -- bash -c 'set -e; O=runs/m3_pf/out; mkdir -p $O; for m in stock zero chosen scramble2001; do python3 steer/steer_preflight.py run --mode $m --model /workspace/models/dpo --p3 runs/m3_p3/out --out $O; done; python3 steer/steer_preflight.py run --mode chosen --eager --model /workspace/models/dpo --p3 runs/m3_p3/out --out $O; python3 steer/steer_preflight.py summarize --p3 runs/m3_p3/out --dir $O --out $O/preflight_report.json'
```
`PREFLIGHT_PASS` / `PREFLIGHT_FAIL` is the last line; exit code 1 on FAIL. If the forked engine process does not see the patch
(`stock` vs `chosen` identical, or the `*_steer_active` checks fail), rerun with the in-process fallback by prefixing
`STEER_INPROC=1 ` to the python commands (sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`). The alpha/2 fallback of the dev guard is checked
only on demand: `... run --mode chosen_half ...` then `summarize` again.

### 6. `m3_dev` (dev guard, one setting; 64 dev prompts, `pids_dev_guard.txt`, 12,000-token cap)
```
bash X/pod/m2_pod.sh job A m3_dev 2400 -- bash -c 'set -e; O=runs/m3_dev/out; mkdir -p $O; export STEER_LAYER=$(python3 -c "import json;print(json.load(open(\"runs/m3_p3/out/chosen.json\"))[\"layer\"])") STEER_VEC=runs/m3_p3/out/vecs/chosen.pt; python3 sample_wrap.py --dan-root /workspace/nd-takehome --patch steer/steer_patch.py --sidecar $O/wrap.json -- --model /workspace/models/dpo --tokenizer /workspace/models/dpo --name dpo --rev 97604023f2602a7dfe6d31b1fc229985e0002022 --problems /workspace/nd-takehome/olmo_ifomega/problems.jsonl --pids @inputs/steer/pids_dev_guard.txt --n 2 --chunk 2 --max_tokens 12000 --seed 3000 --job m3_dev --out $O/samples.jsonl.gz --meta $O/meta.jsonl; bash tools/grade_job.sh $O/samples.jsonl.gz $O/graded.jsonl /workspace/nd-takehome/olmo_ifomega/problems.jsonl'
```
Guard (serrano, `steer_analyze.py` writes `dev_guard` with the verdict; or by hand from `graded.jsonl`): cap hits <= 10 %, mean
length 0.7-1.3 x the stored dpo draws capped at 12,000, pass@1 not more than 10 points below. On failure halve alpha once: use
`vecs/chosen_half.pt` (the pre-flight references for it exist; run `m3_pf` mode `chosen_half` first), rerun as job `m3_dev_half`; a second
failure stops the run. Draw counts: `--dev-n 1` makes `pids_dev_guard.txt` 64 x 1 (cut 2); then use `--n 1`.

### 7. `m3_e0p` (E0': fresh unsteered dpo, zero vector through the patched engine; shared with M1)
```
bash X/pod/m2_pod.sh job A m3_e0p 4500 -- bash -c 'set -e; O=runs/m3_e0p/out; mkdir -p $O; export STEER_LAYER=$(python3 -c "import json;print(json.load(open(\"runs/m3_p3/out/chosen.json\"))[\"layer\"])") STEER_VEC=runs/m3_p3/out/vecs/zero.pt; python3 sample_wrap.py --dan-root /workspace/nd-takehome --patch steer/steer_patch.py --sidecar $O/wrap.json -- --model /workspace/models/dpo --tokenizer /workspace/models/dpo --name dpo --rev 97604023f2602a7dfe6d31b1fc229985e0002022 --problems /workspace/nd-takehome/olmo_ifomega/problems.jsonl --pids @inputs/steer/pids_e0p_reach.txt --n 8 --chunk 8 --seed 1000 --job m3_e0p --out $O/samples.jsonl.gz --meta $O/meta.jsonl; bash tools/grade_job.sh $O/samples.jsonl.gz $O/graded.jsonl /workspace/nd-takehome/olmo_ifomega/problems.jsonl'
```
Outputs for both analyses: `runs/m3_e0p/out/{samples.jsonl.gz,graded.jsonl,meta.jsonl,wrap.json}` on the pod, fetched to
`$M2_DATA/runs/m3_e0p/` by `m2_pod.sh fetch A m3_e0p`; M1 reads that folder (T 1.0, top_p 1.0, 32,768 tokens, 48 reach x 8).
The zero vector keeps the patched code path (also the end-to-end alpha = 0 check): `wrap.json.steer.json` and
`wrap.json.steer_active.jsonl` hold the STEER_ACTIVE record.

### 8. `m3_e1`
Same as step 7 with `STEER_VEC=runs/m3_p3/out/vecs/chosen.pt`, `--pids @inputs/steer/pids_e1.txt`, `--seed 1001`, `--job m3_e1`,
output `runs/m3_e1/out` (the pid file carries 8 per reach prompt and 2 per held-out prompt; `--n 8` is only the default).
Held-out draws cut: regenerate the pid file with `--held-n 1`.

### 9. `m3_trigger` (pod CPU, after grading of E1)
```
bash X/pod/m2_pod.sh job A m3_trigger 300 --nogpu -- python3 steer/steer_trigger.py --e1 runs/m3_e1/out/graded.jsonl --e0p runs/m3_e0p/out/graded.jsonl --inputs inputs/steer --out runs/m3_e1/out/e2_trigger.json
```
Prints `E2_TRIGGER=1` or `E2_TRIGGER=0` plus the JSON (reach passes E1 - s x E0', s = E1 draws / E0' draws; held-out
|difference to the stored dpo pass@1| in points). Exit 2 = an input was empty.

### 10. `m3_e2_s1` .. `s3` (only if `E2_TRIGGER=1`)
Step 7 pattern with `STEER_VEC=runs/m3_p3/out/vecs/scramble200k.pt` (k = 1, 2, 3 -> `scramble2001`, `scramble2002`,
`scramble2003`), `--pids @inputs/steer/pids_e2_s${k}.txt`, `--seed 300${k}`, `--job m3_e2_s${k}`, output `runs/m3_e2_s${k}/out`.
Draws per scramble come from the pid file (reach split 3 / 3 / 2; held-out prompt i goes to scramble (i mod 3) + 1). The
analysis names the cells `e2_s1..3`; copy or name the fetched folders accordingly.

## After the pod
`m2_pod.sh fetch` each job; put `graded.jsonl` of every cell in `$M2_DATA/runs/steer/<cell>/` as `dev_guard`, `e0p`, `e1`,
`e2_s1..3`, then
```
python -I X/steer/steer_analyze.py --runs $M2_DATA/runs/steer --inputs $M2_DATA/inputs/steer --p2 <p2 dir> --p3 <p3 dir> --out <out dir>
```
(tables CSV/JSON; `reach_pass1.{pdf,svg,png}` through `ndbench/figures/style.py`). `steer_trigger.py` and `steer_analyze.py` are pure
python and use the same pass definition (Dan's strict grade).

## Cost cuts (decided by hand before each step)
Cut order under a $15 cap (Turn 6): E2 conditional (default), dev guard 64 x 1 (`--dev-n 1`), held-out 80 x 1 (`--held-n 1`), NLL grid
shrink (`--budget-usd` lower; the shrink order is fixed in code), E0' 192 draws (NOT used: M1 shares E0').
