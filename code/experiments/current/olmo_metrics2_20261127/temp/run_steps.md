---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M1, Claude Sonnet 5.5)
---

# M1 (`temp`): pod steps, exact commands

Design: `discourse/temp.md` Turn 5 Part 3 with the Turn 6 fixes. Every step below is one job the orchestrator starts by
hand. `X` = `code/experiments/current/olmo_metrics2_20261127`; alias `rp-a` is a placeholder for the pod's ssh alias.
Serrano side runs one process at a time, `nice -n 5`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; no
weights on serrano. Python on serrano: `/scratch/cowifornia/venvs/olmo-edl/bin/python -I`.

Pod root `/workspace/m2`, job cwd `/workspace/m2`; Dan's clone `/workspace/nd-takehome` (pinned, commit 8e744a89); weights
`/workspace/models/<ckpt>`. `m2_pod.sh push` copies `X/*.py|sh|md`, `X/pod/`, `X/tools/` and
`$M2_DATA/inputs/` (so `inputs/temp/`) to the pod. Inputs written by `tools/temp_make_inputs.py`: `inputs/temp/pids_C1_k8.txt`
(736 draws), `pids_C1_k4.txt` (544), `pids_reach_k8.txt` (384), `pids_reach_k4.txt` (192), `pids_armA_n2.txt` (436),
`stored_rates.json` (detector rates on Dan's stored draws, serrano only).

## 0. Before any pod step (serrano, CPU)

    cd X && /scratch/cowifornia/venvs/olmo-edl/bin/python -m pytest -q tests/test_temp_*.py      # 50 passed on 2026-11-27
    sha256sum -c temp/FROZEN_repeat_flag.sha256     # the detector is frozen: no change after any data exists
    bash X/pod/m2_pod.sh push rp-a                  # code, tools and inputs/temp/ (md5 both sides)
    bash X/pod/m2_pod.sh setup rp-a                 # once per pod
    bash X/pod/m2_pod.sh download rp-a dpo --verify-sha ;  bash X/pod/m2_pod.sh download rp-a s0500   # s2300 only for C3

## 1. C1 (always): dpo, T 0.6 / top_p 0.95, arm A x2 + reach x8 in ONE invocation (one model load)

Draw counts are parameters: `--pids @inputs/temp/pids_C1_k8.txt` (736 draws) or `pids_C1_k4.txt` (544; the cap-saving cut, reach
at 4 draws). `--n 2` is the default for pids without `:n` and is never used here. `--chunk 1`: one row per finished draw.
Plan for the guard: 736 draws x 5,100 tokens (544 draws for k4). Expected minutes for the watcher: 55 (k4: 45); job timeout
8400 s. The stats line format of vLLM 0.11.0 is unverified: after the first minute of generation look at
`runs/temp_dpo_C1/logs/job.log.guard` or `status`; if the guard logs `BLIND` the wall-clock fallback is in force.

    bash X/pod/m2_pod.sh job rp-a temp_dpo_C1 8400 -- bash -c '
      set -u; J=temp_dpo_C1; O=runs/$J/out; mkdir -p $O
      python3 sample_wrap.py --dan-root /workspace/nd-takehome --temperature 0.6 --top-p 0.95 --sidecar $O/wrap.json -- \
        --model /workspace/models/dpo --tokenizer /workspace/models/dpo --name dpo --rev main \
        --problems /workspace/nd-takehome/olmo_ifomega/problems.jsonl --pids @inputs/temp/pids_C1_k8.txt --n 2 --chunk 1 \
        --seed 6101 --job $J --out $O/samples.jsonl.gz --meta $O/meta.jsonl &
      SP=$!
      python3 tools/temp_inflight_guard.py --log runs/$J/logs/job.log --samples $O/samples.jsonl.gz \
        --plan-draws 736 --plan-tokens-per-draw 5100 --target-pid $SP --out $O/guard.log &
      GP=$!
      wait $SP; rc=$?; kill $GP 2>/dev/null
      bash tools/grade_job.sh $O/samples.jsonl.gz $O/graded.jsonl /workspace/nd-takehome/olmo_ifomega/problems.jsonl; grc=$?
      python3 tools/temp_repeat_flag.py $O/samples.jsonl.gz > $O/repeat_flag.txt
      exit $(( rc != 0 ? rc : grc ))'
    GROW_STALL=2400 bash X/pod/m2_watch.sh rp-a temp_dpo_C1 55 &
    bash X/pod/m2_pod.sh fetch rp-a temp_dpo_C1       # -> $M2_DATA/runs/temp_dpo_C1/ ; wait for FETCH_OK

(`;` after `wait`, not `&&`: after a guard abort the rows that exist are still graded; an aborted cell is reported as
aborted with its unfinished draws counted as failures. `GROW_STALL=2400` because with FINAL_ONLY output the first draw can take
about 14 minutes to finish.)

## 2. C1 verdicts (serrano, after fetch; decides nothing)

    cd /tmp && nice -n 5 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /scratch/cowifornia/venvs/olmo-edl/bin/python -I X/tools/temp_triggers.py \
      --c1-graded $M2_DATA/runs/temp_dpo_C1/out/graded.jsonl --c1-samples $M2_DATA/runs/temp_dpo_C1/out/samples.jsonl.gz \
      --c1b-draws 8 --out $M2_DATA/runs/temp_dpo_C1/triggers.json
      # add, when M3's E0' exists and is graded:  --e0-graded <E0' graded.jsonl> --e0-sidecar <wrap.json> \
      #   --e0-preflight <preflight result json> --template-sha eba6e269f669706e5c788e370160dad953137f3fa7014fa69d03c7b3ad9f0e72
      # (E0' is used iff accepted when C1 is graded; unverifiable fields need --allow-unverified; else the 0.13 % proxy)

Trigger A fires => ask for C3 + C4 (about $6.3); trigger B fires => C2 (about $2.9). The script prints, the orchestrator decides.
`--c1b-draws 4` when the k4 file was used.

## 3. S5R (always): s0500, T 0.6 / top_p 0.95, reach x8 (k4 file for 4)

Plan 384 draws x 5,640 tokens; watcher 35 min (k4: 25); timeout 5400 s.

    bash X/pod/m2_pod.sh job rp-a temp_s0500_C1 5400 -- bash -c '
      set -u; J=temp_s0500_C1; O=runs/$J/out; mkdir -p $O
      python3 sample_wrap.py --dan-root /workspace/nd-takehome --temperature 0.6 --top-p 0.95 --sidecar $O/wrap.json -- \
        --model /workspace/models/s0500 --tokenizer /workspace/models/s0500 --name s0500 --rev step_0500 \
        --problems /workspace/nd-takehome/olmo_ifomega/problems.jsonl --pids @inputs/temp/pids_reach_k8.txt --n 8 --chunk 1 \
        --seed 6102 --job $J --out $O/samples.jsonl.gz --meta $O/meta.jsonl &
      SP=$!
      python3 tools/temp_inflight_guard.py --log runs/$J/logs/job.log --samples $O/samples.jsonl.gz \
        --plan-draws 384 --plan-tokens-per-draw 5640 --target-pid $SP --out $O/guard.log &
      GP=$!
      wait $SP; rc=$?; kill $GP 2>/dev/null
      bash tools/grade_job.sh $O/samples.jsonl.gz $O/graded.jsonl /workspace/nd-takehome/olmo_ifomega/problems.jsonl; grc=$?
      python3 tools/temp_repeat_flag.py $O/samples.jsonl.gz > $O/repeat_flag.txt
      exit $(( rc != 0 ? rc : grc ))'
    GROW_STALL=2400 bash X/pod/m2_watch.sh rp-a temp_s0500_C1 35 &
    bash X/pod/m2_pod.sh fetch rp-a temp_s0500_C1

## 4. PF: prefill likelihood check (its own HF job, after C1 is fetched; next to M3's HF dpo session)

Manifest on serrano (CPU, tokenizer files only; about 2-3 min; 8 dpo comparators, or `--dpo-n 4` for the 4/4/4 fallback whose
gate threshold 0.35 is applied automatically):

    cd /tmp && nice -n 5 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /scratch/cowifornia/venvs/olmo-edl/bin/python -I -c "
    import sys; sys.path.insert(0,'X/tools'); import temp_pf_select as s
    sys.exit(s.main(['--tok-dir','/scratch/cowifornia/nd-rl/olmo_elicit/tok/dpo','--out','$M2_DATA/inputs/temp/pf_manifest',
      '--c1-samples','$M2_DATA/runs/temp_dpo_C1/out/samples.jsonl.gz','--dpo-n','8','--s2300-n','4','--c1-n','4']))"
    mv $M2_DATA/inputs/temp/pf_manifest.summary.json $M2_DATA/runs/temp_dpo_C1/pf_manifest.summary.json
    bash X/pod/m2_pod.sh push rp-a                  # the manifest (about 25 MB) goes to inputs/temp/ on the pod
    # (a dry run on the stored groups only: 47 prompts, 188 s2300 + 376 dpo sequences; one prompt has no s2300 pass and is dropped)

Pod job (expected 20-35 min incl. load; timeout 5400 s; sliding-window pass criterion: window 4096 on 48 sliding layers,
`--check-sliding` exits 4 otherwise; the 3 self-tested sequences must be identical twice and finite, else exit 5):

    bash X/pod/m2_pod.sh job rp-a temp_pf 5400 -- bash -c '
      set -u; O=runs/temp_pf/out; mkdir -p $O
      python3 -c "import transformers, torch; print(\"transformers\", transformers.__version__, \"torch\", torch.__version__)"
      python3 tools/hf_score.py --model /workspace/models/dpo --manifest inputs/temp/pf_manifest.jsonl \
        --out $O/scores.jsonl --dtype bfloat16 --chunk 2048 --selftest-n 3 --check-sliding'
    bash X/pod/m2_watch.sh rp-a temp_pf 30 &
    bash X/pod/m2_pod.sh fetch rp-a temp_pf

(transformers 4.57.1 is what the tool was tested with; the first line of the job log must show it. The PF job holds the GPU
lock, so it queues behind nothing and blocks M3's session while it runs: start it when the GPU is free.) Statistics and verdict:

    nice -n 5 env OMP_NUM_THREADS=1 /scratch/cowifornia/venvs/olmo-edl/bin/python -I X/tools/temp_pf_stats.py \
      --scores $M2_DATA/runs/temp_pf/out/scores.jsonl --manifest $M2_DATA/inputs/temp/pf_manifest.jsonl \
      --out $M2_DATA/runs/temp_pf/pf_stats.json                 # fallback 4 comparators: the threshold switches by itself

## 5. Conditional cells (orchestrator decides after step 2; each is its own job)

C3 (trigger A): s2300, T 0.6 / top_p 0.95, arm A x2; plan 436 x 6,900; watcher 45; timeout 6600; seed 6103; download s2300 first.
C4 (trigger A): dpo, T 0.6 / top_p **1.0**, arm A x2; plan 436 x 5,100; watcher 35; timeout 5400; seed 6104.
C2 (trigger B): dpo, T 0.8 / top_p 1.0, arm A x2; plan 436 x 5,100; watcher 35; timeout 5400; seed 6105.
All three are the section 1 job with these substitutions (and the `--pids @inputs/temp/pids_armA_n2.txt --n 2`):

    # C3: J=temp_s2300_C1  --model/--tokenizer /workspace/models/s2300 --name s2300 --rev step_2300 --temperature 0.6 --top-p 0.95 --seed 6103 --plan-draws 436 --plan-tokens-per-draw 6900
    # C4: J=temp_dpo_C4    --model/--tokenizer /workspace/models/dpo   --name dpo   --rev main      --temperature 0.6 --top-p 1.0  --seed 6104 --plan-draws 436 --plan-tokens-per-draw 5100
    # C2: J=temp_dpo_C2    --model/--tokenizer /workspace/models/dpo   --name dpo   --rev main      --temperature 0.8 --top-p 1.0  --seed 6105 --plan-draws 436 --plan-tokens-per-draw 5100

No LoRA anywhere: no `--lora` is ever passed. `wrap.json` records `dan_argv`, T, top_p and the sha256 of Dan's `sample.py`.

## 6. Analysis and figure (serrano, CPU, after the cells are fetched)

    nice -n 5 env OMP_NUM_THREADS=1 /scratch/cowifornia/venvs/olmo-edl/bin/python -I X/temp/analyze.py \
      --c1 $M2_DATA/runs/temp_dpo_C1/out/graded.jsonl --s5r $M2_DATA/runs/temp_s0500_C1/out/graded.jsonl \
      [--c3 .../temp_s2300_C1/out/graded.jsonl] [--c4 .../temp_dpo_C4/out/graded.jsonl] [--c2 .../temp_dpo_C2/out/graded.jsonl] \
      [--e0 <E0' graded.jsonl> --e0-accepted] --c1b-draws 8 --out-dir $M2_DATA/runs/temp_analysis
    nice -n 5 env OMP_NUM_THREADS=1 /scratch/cowifornia/venvs/olmo-edl/bin/python -I X/temp/fig_temp.py \
      --summary $M2_DATA/runs/temp_analysis/summary.json --out $M2_DATA/runs/temp_analysis/fig_temp
