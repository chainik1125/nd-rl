---
written_on: 2026-11-27
written_by: agent:claude (build, Sonnet 5.5)
---

# M2 (rank1): the pod steps, exactly as they run (design: `discourse/rank1.md` Turn 5 section B with Turn 6 D1-D8)

Conventions. `X` = `code/experiments/current/olmo_metrics2_20261127`; `P` = `bash X/pod/m2_pod.sh`; `A` = the pod alias (`rp-<name>`).
Pod cwd for every job is `/workspace/m2`; weights are `/workspace/models/<ckpt>` (`dpo`, `s0100`, `s0500`, `s2300` from `P download`);
the code is at `rank1/` (pushed by `P push`, which stages X/rank1; orchestrator edit 2026-11-27: the copy in `inputs/rank1/code/` is no longer used by the commands), inputs at `inputs/rank1/`; outputs go to `runs/<job>/out/`.
Dan's sampler is used unchanged through `sample_wrap.py`; one engine load per cell (per-pid draw counts in `spec_<cell>.txt`, so H x 2
and R x 4 are one call), `--chunk 1` (Turn 6 D1), T 1.0 and top_p 1.0 (Dan's constants, no override), grading inside the job.
Every script prints a verdict and exits with a code; you start the next job. Chains (`chain_*.txt`) are for sequences that need no decision.

Serrano side, once and after any edit of `rank1/*.py`:

    nice -n 5 bash X/rank1/stage_inputs.sh          # copies rank1/*.py to $M2_DATA/inputs/rank1/code/ (push stages inputs/)
    nice -n 5 python3 -I X/rank1/make_inputs.py     # (already run) pids_D/H/R, refs.json, seeds.json, spec_*.txt, MANIFEST.json
    P push A                                        # md5-checked both sides

`make_inputs.py` was run on 2026-11-27: D 64, H 112, R 48 prompts; stored pass@1 (% of prompts) dpo D 41.8 / H 32.5 / R 0.0,
s0500 60.2 / 44.4 / 4.5, s2300 78.3 / 64.5 / 69.4. Problems file for sampling and grading: `/workspace/nd-takehome/olmo_ifomega/problems.jsonl`
(grade_job's default; confirm once that all 224 pids are in it: the sampler fails at start-up otherwise).

## Timeouts (Turn 6 D2): 600 s + max(1.5 x planned_tokens / 1,400, planned_tokens / 1,400 + 900 s)

| job | planned draws | planned tokens | timeout s | expect (min, for the watcher) |
|---|---|---|---|---|
| W | | | 5,400 | 30-45 |
| build | | | 1,800 | 3 |
| gate | | | 2,400 | 10 |
| A1 | 416 | 2.87 M | 3,675 | 40 |
| A2 | 208 | 1.2 M | 2,357 | 20 |
| A1u | 320 | 1.6 M | 2,643 | 25 |
| write_g46, write_g35 (`--nogpu`) | | | 3,600 each | 8-15 each |
| Xdev46, Xdev35 | 128 each | 0.88 M | 2,129 | 15-20 |
| Xheld | 416 | 2.87 M | 3,675 | 40 |

A cell killed by its timeout keeps its rows; draws without a row count as bad in the collapse rule (`collapsed` = |capped OR no end-of-thinking OR no row|
/ planned > 5 %, `common.collapse_stats`); in pass@1 they score as fails and the all-pass bound is printed next to it (`analyze.py`).
Never use `pkill`; `P stop A <job>` signals the recorded process group. The GPU must be free before the next job (the runner checks).

## Step 0. Downloads (shared with M1 / M3)

`P download A s0100` is M2's only extra download (dpo, s0500, s2300 are downloaded for M1 / M3). Nothing in M2 deletes s0100 (keep until W's outputs are
fetched). Disk: dpo, s0100, s0500, s2300 + two ExPO checkpoints = 6 x 64.5 GB = 387 GB, +64.5 GB if E runs, + about 25 GB environment: volume >= 450 GB.

## Step 1. W (weight-space pass)

    P job A W 5400 -- python3 rank1/wpass.py --models-dir /workspace/models --out runs/W/out

Verdict lines at the end of `runs/W/logs/job.log`; `runs/W/out/w_band.txt` holds `signal` | `noisy` | `stop`; `w_summary.json` has the weighted median of
sigma_1 / null, the share of ||Delta_2300||^2 in matrices below 1.1, alpha, cos(Delta_500, Delta_2300), gamma_proj, distance ratios for gamma 3.17 / 3.5 / 4.6,
the persistence cosines (A3 trigger: `persistence_cos_u_d500_d2300_weighted_median` >= 0.3), and every exact-SVD comparison.

| exit | meaning | next step |
|---|---|---|
| 0, band `signal` (ratio >= 1.5) | signal-dominated | step 2 (chain `chain_w_build_gate.txt` does steps 1-3) |
| 0, band `noisy` (1.1-1.5) | noisy top direction, implied cosine 0.58-0.84 (iid map); cells run and are labelled "noisy" in the write-up | step 2 |
| 20 | STOP: ratio < 1.1, sigma_1 is the null's | skip A1, A1u, A2 and the gate; go to step 5 (ExPO writes), then step 7 |
| 2 | exact-SVD check failed (> 0.5 % on sigma_1 or the null) | rerun once: `P job A W2 5400 -- python3 rank1/wpass.py --models-dir /workspace/models --out runs/W2/out --iters 12`; if exit 2 again: `... --out runs/W3/out --exact-fallback` (exact sigma_1 and null for every Delta_2300 on the GPU, about +20-40 min) and continue with that output; use its directory in step 2 |
| 3 | input error (missing checkpoint, `--iters` < 6) | fix and rerun (`--force`) |

## Step 2. Build adapters (CPU work on the pod; own job)

    P job A build 1800 -- python3 rank1/build_adapters.py --w-dir runs/W/out --out-root runs/build/out/adapters

Writes `A1 A1u A2 k16 gate_m{1,4,16,64}_{r1,r16}` (PEFT format; r = lora_alpha = 1, balanced sqrt(c) split; k16 = unscaled top-16 truncation, r = lora_alpha = 16),
each with `MANIFEST.json` (sha256, c per matrix, A2 seeds). k16 is built so the conditional rank-16 cell needs no second SVD. Offline checks of the writer
are in `tests/test_rank1_adapter.py` (including a PEFT load and merge on a tiny Olmo3).

## Step 3. vLLM-only gate (one eager engine)

    P job A gate 2400 -- python3 rank1/lora_gate.py --model /workspace/models/dpo --adapters runs/build/out/adapters --pool /workspace/m2/pool --out runs/gate/out/gate.json

| exit | meaning | next step |
|---|---|---|
| 0 | consistency (r1 = r16 within 1 %, per-token corr >= 0.99, base within 0.5 %) and applied (A1, A1u, A2 change nats, repeat exactly) | step 4 |
| 30 | consistency failed | no adapter sampling; ExPO still runs (step 5 onward); report; any fix waits for the orchestrator |
| 31 | underpowered: m = 1, 4, 16, 64 all give < 5 % change of base nats | same as 30 (report the gate as "underpowered") |
| 32 | applied check failed (A1, A1u or A2 not applied or not deterministic) | same as 30 |
| 3 | vllm import / input error | fix, `--force` rerun |

## Step 4. A1 first, then the release rule

    P job A A1 3675 -- bash -c '
      mkdir -p runs/A1/out &&
      python3 sample_wrap.py --dan-root /workspace/nd-takehome --sidecar runs/A1/out/wrap.json -- \
        --model /workspace/models/dpo --tokenizer /workspace/models/dpo --name dpo --rev 97604023f2602a7dfe6d31b1fc229985e0002022 \
        --problems /workspace/nd-takehome/olmo_ifomega/problems.jsonl --pids @inputs/rank1/spec_A1.txt --n 1 --chunk 1 --seed 710001 \
        --job A1 --out runs/A1/out/samples.jsonl.gz --meta runs/A1/out/meta.jsonl \
        --lora runs/build/out/adapters/A1 --lora_name A1 --lora_rank 16 &&
      bash tools/grade_job.sh runs/A1/out/samples.jsonl.gz runs/A1/out/graded.jsonl /workspace/nd-takehome/olmo_ifomega/problems.jsonl'

(`--rev` is the dpo commit of `checkpoints.csv`; the other cells differ only in the job name, `--pids @spec_<cell>.txt`, `--seed`, `--lora*` and the timeout.)
Then `P fetch A A1` and, on serrano (one core):

    nice -n 5 python3 -I X/rank1/rules.py release --a1 $M2_DATA/runs/A1/out/graded.jsonl --refs $M2_DATA/inputs/rank1/refs.json \
        --h $M2_DATA/inputs/rank1/pids_H.txt --r $M2_DATA/inputs/rank1/pids_R.txt --n-h 2 --n-r 4 --out $M2_DATA/runs/A1/release.json

| exit | meaning | next step |
|---|---|---|
| 0 | A1 success, partial or collapse (not a clean null) | run A2 then A1u (below) |
| 10 | clean null (point estimate f < 0.25 on both H and R, not collapsed) | skip A1u and A2 (A1u is then "inferred null", not measured); go to step 5/7; one conditional cell only if budget remains (none fits under \$15) |

A2: job `A2`, timeout 2,357, `--pids @inputs/rank1/spec_A2.txt --seed 710003 --lora runs/build/out/adapters/A2 --lora_name A2`.
A1u: job `A1u`, timeout 2,643, `--pids @inputs/rank1/spec_A1u.txt --seed 710002 --lora runs/build/out/adapters/A1u --lora_name A1u`.
A collapsed A1 (exit 0 from `rules.py`, `collapse.collapsed` true in `release.json`) is a result; A1u and A2 then tell whether the norm alone breaks the model.

## Step 5. ExPO writes (CPU / disk; `--nogpu`, run in the background while adapter cells sample)

    P job A write_g46 3600 --nogpu -- python3 rank1/expo_write.py --dpo /workspace/models/dpo --s0500 /workspace/models/s0500 --gamma 4.6 --out /workspace/models/expo_g46
    P job A write_g35 3600 --nogpu -- python3 rank1/expo_write.py --dpo /workspace/models/dpo --s0500 /workspace/models/s0500 --gamma 3.5 --out /workspace/models/expo_g35

The writer refuses (exit 5) unless 2 x 64.5 GB + 20 GB (149 GB) is free under the output directory (Turn 6 D7); exit 4 = a verification tensor differs from the fp64 recompute by
> 1 bf16 ulp; exit 0 = written, `MANIFEST.json` has the sha256 of every file. Run them one after the other (`chain_expo_writes.txt`); they use `download.lock`, not the GPU lock.
(If disk is short, write g46 first and delete it after its dev cell, per the orchestrator; the dev order is 4.6 then 3.5.)

## Step 6/7. ExPO dev (D 64 x 2 each), the gamma rule, held-out

    P job A Xdev46 2129 -- bash -c '... same as A1 with --model /workspace/models/expo_g46 --tokenizer /workspace/models/expo_g46 --name expo_g46 --rev expo_g46 \
        --pids @inputs/rank1/spec_Xdev.txt --seed 710010 --job Xdev46 (no --lora flags) ...'
    P job A Xdev35 2129 -- (same with expo_g35, --seed 710011, --job Xdev35)

Fetch both, then on serrano:

    nice -n 5 python3 -I X/rank1/rules.py gamma --g46 $M2_DATA/runs/Xdev46/out/graded.jsonl --g35 $M2_DATA/runs/Xdev35/out/graded.jsonl \
        --d $M2_DATA/inputs/rank1/pids_D.txt --n-draws 2 --out $M2_DATA/runs/expo_choice.json

| exit | meaning | next step |
|---|---|---|
| 0 | chose gamma 3.5 (no lead of more than one paired SE, or 4.6 collapsed) | delete `/workspace/models/expo_g46` if still present; held-out on `expo_g35` |
| 1 | chose gamma 4.6 (lead > 1 paired SE and not collapsed) | delete `/workspace/models/expo_g35`; held-out on `expo_g46` |

If the `collapse46` and `collapse35` fields in `expo_choice.json` are both `collapsed: true`: no held-out cell; report (any other gamma only on the orchestrator's go).
Held-out: job `Xheld`, timeout 3,675, model = the chosen directory, `--pids @inputs/rank1/spec_Xheld.txt --seed 710012`. If a dev job is killed by its timeout the
rule above still runs on the rows written (missing draws count as fails and as bad draws).

## Step 8. Analysis (serrano, one core, `nice -n 5`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`)

    nice -n 5 /scratch/cowifornia/venvs/olmo-edl/bin/python -I X/rank1/analyze.py --cells X/rank1/cells_template.json \
        --inputs $M2_DATA/inputs/rank1 --out $M2_DATA/runs/rank1_analysis

Per cell and set: pass@1 (% of prompts, planned denominator) with the all-pass bound, "any pass", paired joint prompt-bootstrap intervals against stored dpo and s2300,
f = (p - p_dpo) / (p_s2300 - p_dpo), Cai's ratio beside it, f' for ExPO, collapse counts; the figure `rank1_recovered_fraction.{pdf,svg,png}` (5.5 in, `style.py`).
A2 is read against stored dpo (zero-point check of the LoRA engine).

## Cost and cut order (Turn 5 B.7 / Turn 6 D6; the success / partial / collapse branch)

\$15.65 in Turn 5, +\$0.45 for the third ExPO engine load, W 30-45 min instead of 19 (+\$1-2.3): about \$16.6-18 before cuts. Cut order under the \$15 cap, applied before a cell whose
planned cost would cross it: (1) Xdev at D 32 x 2 (edit `spec_Xdev.txt`: first 32 pids, `:2`); (2) A1u at H x 1 + R x 2 (edit `spec_A1u.txt`); (3) Xheld H x 2 -> H x 1
(edit `spec_Xheld.txt`); (4) last, gamma 3.5 pre-committed with no dev cell (skip g46 write and Xdev; the write-up says ExPO is untuned). Never cut A1's H x 2 + R x 4, A2's R x 2, W or the gate.
In the clean-null branch the cost is about \$11.75 plus the extra loads; no conditional cell (rank-16, E) fits under the \$15 cap.
Hard stop at \$15 cumulative M2 spend; beyond it only on the orchestrator's go.

## Conditional cells (the orchestrator decides from the results; both paths are built)

- rank-16: adapter `runs/build/out/adapters/k16` (unscaled top-16 truncation, r = lora_alpha = 16); job `K16`, `--lora_name K16 --lora_rank 16`, spec = `spec_A1u.txt` to start (H x 2 + R x 2) or `spec_A1.txt`.
- E (the lm_head / embeddings / norms reference, written checkpoint only): `python3 rank1/hybrid_write.py --dpo /workspace/models/dpo --s2300 /workspace/models/s2300 --out /workspace/models/hybrid_e`
  as a `--nogpu` job (timeout 3,600); the 448 matrices from s2300, the other 259 tensors from dpo, bit-for-bit, manifest with sha256; exit 5 = not enough disk (needs 64.5 GB + 20 GB), exit 4 = verification failed.
  Sample it like Xheld with `--model /workspace/models/hybrid_e --name hybrid_e --seed 710020` and a spec of R x 4 + H x 1 (`spec_E.txt` is not pre-made; take the lines of `spec_Xheld.txt` with `:4` for R and `:1` for H).
- A3 + A4 (AlphaRL) only if A1 >= 50 % of G on H and R and `persistence_cos_u_d500_d2300_weighted_median` >= 0.3; not built.
