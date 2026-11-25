---
written_on: 2026-11-25
written_by: agent:claude
---

# olmo-weights: across Olmo 3's released checkpoints (32B and 7B, pretraining to RL), the stage switches show and the change per step follows the learning rate (per-stage medians 0.57–0.87 × LR at 1,000-step spacing, pretraining through SFT); RL moves the weights 0.36 % (32B) / 0.08 % (7B) of ‖θ‖ in total, 31–185× less than SFT, and leaves 65 % / 88 % of them within one bf16 step of DPO; RL's 8–9 × LR per step holds only at its released 25–50-step spacing (0.7–1.1 × LR over its first 1,000 steps), and as a property of RL it is not supported; olmo-ifladder's LoRA points along RL's change; the 32B SFT `main` is a soup of two runs, and the 32B RL checkpoints at steps 150–200 lie on an abandoned branch

Status: completed 2026-11-25 (executor 04:27–07:15 UTC; pre-registration committed 04:48:59; DONE 07:14:59) and
reviewed 2026-11-25 (§Recount, phase 1, 07:58:04 UTC; §Comparison and §Verdict 08:12:20 UTC). **No hard-constraint
violation, nothing quarantined** (the review: "No quarantine."). The review's §Verdict has no one-word verdict. It
lists what stands, six items that must be reworded, and one claim under "Not supported": "RL's large per-step change
relative to its LR as a property of RL". That claim is the run doc's headline ("RL breaks this pattern … 8–9 × the
LR"), whose "numbers reproduce; the interpretation is not supported". What stands:
- the manifest and every pre-registered hash check, and the step and learning-rate sources;
- every per-tensor statistic ("which reproduces from the raw weights": a seeded sample recomputed on a pod), and the
  distance, straightness, unchanged-fraction and module tables;
- the expectation hits E2, E3, E5, G3 and G6, and the misses E1, E4, G4, G5 and G7;
- the checkpoint findings (the SFT `main` soup, now bit for bit on the tested tensors; stage 3 from the stage-2 merge;
  five misfiled RL-Zero names), and the compute and cost.

Elsewhere the review adds: "G2, scored as pre-registered (native spacing), is a hit."

This summary uses the review's wording throughout. The plain-summary copy marks the items that apply to it *[review]*.
The review adds a finding of its own: the 32B RL checkpoints `step_0150` and `step_0200` lie on a branch the RL run
abandoned, so olmo-ifcurve's step-200 point is off the lineage of steps 250–2,300 (marked in that run's summary; see
below).

The pipeline checkpoints are AI2's releases: bf16 safetensors, pretrained, none trained in this project (the two
LoRA adapters were trained by this project's olmo-ifladder run). Labels follow the executor's manifest
(`olmo_weights/manifest/checkpoints.csv`: label → repo, branch, commit):
- **32B** (32.23 B parameters, 707 tensors in 14 shards): `allenai/Olmo-3-1125-32B` (stage 1 `s1`; stage-2
  ingredients `s2i1`, `s2i2` and their merge `s2merge`; stage 3 `s3` and its merge `s3merge` = base `main`) →
  `allenai/Olmo-3-32B-Think-SFT` (the two learning-rate runs `sft1e-4`, `sft5e-5`) → `allenai/Olmo-3-32B-Think-DPO`
  (`dpo`, final only) → `allenai/Olmo-3.1-32B-Think` (`rl`, steps 50–2,300).
- **7B** (7.30 B parameters, 355 tensors in 3 shards): `allenai/Olmo-3-1025-7B` (`s1`, `s2`, `s3`; base `main` =
  `stage3-step11921`) → `allenai/Olmo-3-7B-Think-SFT` (`sft`, `sftmain`) → `allenai/Olmo-3-7B-Think-DPO` (`dpo`) →
  `allenai/Olmo-3-7B-Think` (`rl`, `rlmain`), plus `allenai/Olmo-3-7B-RL-Zero-{IF,Math,Code}` (`zif`, `zmath`, `zcode`:
  RL from base `main`).
- **LoRA:** olmo-ifladder's `lora_main` and `lora_ctrl` (rank 16, α 32, on q / k / v / o / gate / up / down of all 64
  layers of `Olmo-3-32B-Think-DPO` `main`, commit `97604023`; sha256 `9921a760…` / `4cf06004…`, matching that run's
  MANIFEST).

Nothing was generated, graded or trained: no text, no Lean, no `nd_verify`. Every statistic is a deterministic function
of the released weights. The one random element is the LoRA comparison's rank-16 baseline.

Review: [`review_olmo-weights.md`](https://github.com/dan-pandori/nd-takehome/blob/e8d33db7/review_olmo-weights.md)
(fork `dan-pandori/nd-takehome`, branch `dan_olmo-weights`):
- §Recount, phase 1: [`0ffe5c51`](https://github.com/dan-pandori/nd-takehome/commit/0ffe5c51), 07:58:04 UTC;
- §Comparison and §Verdict: [`e8d33db7`](https://github.com/dan-pandori/nd-takehome/commit/e8d33db7), 08:12:20 UTC;
- kit: `review/olmo-weights-recount/` (scripts `rv_*.py`, outputs in `out/` and `pod_out/`, its Hub API responses in
  `out/hub_api_responses.tgz`). The kit has no README. The review's §R8 "Reproduce" lists the phase-1 commands;
  phase 2 adds `rv_phase2_checks.py` and `rv_spacing.py` (outputs `out/rv_phase2_checks.txt`, `out/rv_spacing.txt`).

Run files at the DONE commit [`c13a307a`](https://github.com/dan-pandori/nd-takehome/commit/c13a307a), unchanged at
`e8d33db7`:
- `run_olmo_weights.md`, `numbers.md` § olmo-weights, `log.md` § olmo-weights, `STATUS.md`;
- `olmo_weights/`:
  - code: `list_refs.py`, `fetch_trees.py`, `hash_checks.py`, `check_headers.py`, `manifest.py`, `lr.py`, `engine.py`,
    `selftest.py`, `realcheck.py`, `restage.py`, `dl_bench.py`, `agg.py`, `analyze.py`, `report_tables.py`,
    `figures.py`, `compute_rows.py`;
  - `manifest/`: `checkpoints.csv`, `jobs.json`, `stage1_locations.json`, `refs.json`, `hash_checks.json`, `trees/`,
    `index/` (all at the pre-registration commit), and `headers/` (added with the results, `7ef655dd`);
  - `steps_lr.md` with its `sources/`, `README.md`, `plain_summary.md`;
- `artifacts/olmo_weights/`:
  - `stats/<job>/` (per-tensor statistics, tracked);
  - `tables/`: `stages.csv`, `chains.csv`, `intervals.csv`, `modules.csv`, `layers.csv`, `lora*.csv`, `tests.json`,
    `expectations.json`, `status.json`, `summary.md`;
  - `figures/` (Figures 1–7), `stage0/`, `restage/`, `logs/`;
- `pod/olmo_weights/` (pod scripts);
- the pre-registration `preregistration/olmo-weights.md`:
  [`a92b8cb7`](https://github.com/dan-pandori/nd-takehome/commit/a92b8cb7), 04:48:59 UTC. That is before the first pod
  request (04:55:50 in `~/pods.log`) and before any weight download. It was not amended.

Brief: [`BRIEF_olmo-weights.md`](../../docs/proposals/followups/BRIEF_olmo-weights.md). Proposal 47:
[`2026-11-24-olmo-weight-trajectory.md`](../../docs/proposals/2026-11-24-olmo-weight-trajectory.md), written at Dan's
request on 2026-11-24. Dan chose the add-ons and approved it at $50: "If this experiment is sub 50$ we don't need to
think much harder about optimization". The low-rank analysis is paused. The behavioural counterpart is Proposal 48,
[`2026-11-24-olmo-function-distance.md`](../../docs/proposals/2026-11-24-olmo-function-distance.md), not approved or
launched. The run builds on:
- [olmo-ifcurve](../2026-11-21-olmo-ifcurve-ifbench-pass1-0p29-to-0p68-sharpening-from-1100-no-abrupt-reach-switch-timing-no-order/README.md):
  its IFBench pass@1 / pass@8 and first-genuine steps, overlaid in Figure 5;
- [olmo-ifladder](../2026-11-09-olmo-ifladder-lora-on-16-rl-trajectories-elicits-40-41-of-48-if-reach-prompts-control-0/README.md):
  the LoRA adapters `lora_main` and `lora_ctrl`.

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **Figure 1**, unchanged: [`fig1_32B.png`](fig1_32B.png) and [`fig1_7B.png`](fig1_7B.png).
  - The review checked its normalisation and stage marks against the table (the brief's third reviewer check), and
    they match. Its nit: in the 32B panel, the soup's "merge (0 steps)" label sits on the DPO point.
  - *[librarian]* Applied to the figure, the review's findings mean two things. The RL points are per step at the
    released 25–50-step spacing, which the review finds inflates the rate (the bottom panel's RL points, median 8–9 ×
    LR). The 32B RL section draws steps 150 and 200 in label order, although they lie on the abandoned branch.

Question (the brief): "Plot how much the weights change between consecutive released checkpoints across the whole
Olmo 3 pipeline: pretraining stage 1, mid-training stage 2, long-context stage 3, SFT, DPO and RL. Do it for 32B (the
pipeline our runs use) and for 7B. Dan wants a deliberately simple picture that makes the switches between kinds of
training obvious." The add-ons Dan chose:
- distance from each stage's start, with straightness and consecutive cosines;
- per-module and per-layer breakdowns;
- the fraction of weights unchanged or within 1 bf16 ulp;
- the olmo-ifcurve overlay;
- the comparison with olmo-ifladder's LoRA;
- 7B's RL-from-base runs against its RL from DPO.

The low-rank analysis (paused by Dan) and behavioural KL (Proposal 48) are not in this run.

## Configuration

- **Checkpoints** (pre-registered; `checkpoints.csv` holds 453 checkpoints with branch, commit and every shard's LFS
  sha256 and size; 13 jobs, 469 member downloads of 3,244 files, 15.17 TB):

  | | 32B | 7B |
  |---|---|---|
  | stage 1 | 25 locations as adjacent pairs (s, s + 1,000), which include step 0 and the final step 656,000: 49 checkpoints | 30 locations as pairs (from step 0), plus the final step 1,413,814: 59 |
  | stage 2 | ingredient 1: all 24; ingredient 2: every 4th (4,000 … 23,842; 6); the merge | all 49 regular branches |
  | stage 3 | all 12, plus the merge (= base `main`) | all 13 (11,921 = base `main`) |
  | SFT | all 22 (the 1e-4 and 5e-5 runs) | all 43, plus `main` |
  | DPO | final | final |
  | RL | all 46 (50–2,300) | all 55 (25–1,375), plus `main` |
  | RL from the base | – | IF 19 + `main`, Math 19 + `main`, Code 29 (its `main` = `step_300` by hash) |

  - **Stage-1 locations,** fixed before any download (`manifest.py:stage1_locations`). Early targets are 0 and
    1,000 · 2^k up to 64,000, each mapped to the nearest released pair start in log distance. The rest are spread
    evenly. 32B: 0, 1,000, 5,000, 8,000, 16,000, 32,000, 64,000, 97,000 … 655,000. 7B: 0, 1,000, 2,000, 4,000, 8,000
    … 1,412,000. The review's own implementation of the rule reproduces both lists exactly.
  - **Left out:** the byte-identical `-reshard32` copy of the 32B stage-3 merge; the 7B
    `stage2-step47684-mix-*-from-2T-ckpt` ablations (another shard layout, off the pipeline path);
    `allenai/Olmo-3-32B-Think`, whose 15 checkpoints equal 3.1's steps 50–750; and every `main` byte-identical to a
    branch.
- **Steps and learning rates** (`olmo_weights/steps_lr.md`; sources copied to `olmo_weights/sources/`):
  - sources: OLMo-core `3c1a108b` (the official OLMo3 scripts and `scheduler.py`), `79a184cf` and `38f66526` (the SFT
    scripts), open-instruct `fb18a8e8` (the `olmo3/*.sh` scripts) and the Olmo 3 report (arXiv 2512.13961v2);
  - steps per interval come from the branch names. RL takes 1 optimizer step per RL step (1 minibatch, 1 epoch). DPO
    counts as one jump of pairs ÷ 128: 1,562.5 steps for 32B, 1,171.9 for 7B;
  - peak LRs: 6e-4 (32B stage 1, cosine to 10 %), 3e-4 (7B stage 1), 2.0712e-4 (the scripts' stage-2 and stage-3
    peak, linear to 0; for 7B, `steps_lr.md` and `lr.py` use the report's 2.074e-4), 1e-4 and 5e-5 (32B's two SFT
    runs; 7B 5e-5), 7e-8 / 8e-8 (DPO, 32B / 7B), 2e-6 / 1e-6 (RL, 32B / 7B, constant).
- **Statistics** (`engine.py`, on pods):
  - streamed shard file by shard file with hf_xet, each file's sha256 checked against the manifest, whole shard files
    cached as bf16 on the GPU;
  - float32 from the bf16 values (exact differences), sums in float64;
  - per interval and per tensor: ‖Δ‖², ‖θ_a‖², ‖θ_b‖², numel, the bitwise-equal count and the count with |Δ| ≤ 1 ulp of
    the earlier bf16 value; dot products of each change with the previous three; the distance from and dot products
    with the stage start; Gram matrices for the merge and soup tests; the LoRA dot products.
  - Measures: r = ‖θ_b − θ_a‖ / ‖θ_a‖ over all parameters; per step = r ÷ optimizer steps; ÷ LR, the mean LR over the
    interval; straightness S = end distance ÷ path length over the native-spacing checkpoints.
- **LoRA** (32B): ΔW = 2 · B·A for each of the 448 targeted matrices. Its cosine with θ_t − θ_DPO over those matrices,
  and the norm ratio, at every RL checkpoint. Random baseline: 8 rank-16 matrices per targeted matrix (seeded by tensor
  name), with Gaussian factors scaled to ‖ΔW_main‖.
- **Stage 0** (one A40, `ow-s0`): hf_xet reached 1.06 GB/s at 8 files in flight. The engine self-test passed on CPU and
  GPU, with positive and negative controls, and a numpy cross-check from raw safetensors bytes passed on 8 tensors
  (5 of 7B `s1` 1,000 → 2,000 and 3 of 32B `s1` 0 → 1,000). The first engine took 385 s on a 32B pair because it re-read
  shard files from disk; a GPU cache of whole shard files fixed that (0.88 GB/s through the engine). The executor's
  re-cost: ≈ $3.8 on A40s (≈ $4.3 with stage 0); the review's §R8: ≈ $3.7. Both are far below the $45 pause line. No cut
  was made, and the full pre-registered manifest ran.
- **Deviations declared in the pre-registration:**
  - GPU pods instead of CPU pods, declared as conditional on stage 0 ("A GPU pod is used if stage 0 shows it is no more
    expensive per TB"). Stage 0 measured CPU statistics 22× slower on the same pod, and A40s were used (`log.md`);
  - hf_xet instead of hf_transfer (huggingface_hub 1.x dropped hf_transfer);
  - shard-major streaming, which keeps less than one checkpoint per job on disk;
  - 49 regular 7B stage-2 branches, not 52, and 13 32B stage-3 checkpoints, not 14;
  - the added tests (DPO start, stage-3 start, merge residuals, duplicate names, `main` equalities).
- **During the run:** `5e-5-step8000` uses another shard layout than its own index file describes (549 of 707 tensors
  in another file). `restage.py` re-sharded it byte for byte, the engine took it pre-staged, and job `p32_sft` moved to
  a sixth pod (`ow-f`). A header check of all 453 checkpoints found no other such checkpoint.
- **Pods and cost:**
  - stage 0: one A40 SECURE (`ow-s0`, 05:19–05:51 UTC, 0.54 h, $0.32), after a create that was undone at once
    (04:55:50, `UNDONE` in `~/pods.log`; *[librarian]* billed $0.01) and an RTX 4000 Ada request that never got a
    machine (0.36 h, logged at $0.10);
  - main run: six 1 × A40 SECURE pods at $0.59/h, 05:51–07:01 UTC, one job at a time per pod:

    | pod | jobs |
    |---|---|
    | `ow-a` | b32_s1 |
    | `ow-b` | p32_rl |
    | `ow-c` | b32_s2, b32_s3 |
    | `ow-d` | b7_s1, b7_s2 (after its p32_sft attempt stopped on the layout check) |
    | `ow-e` | b7_s3, p7_sft, p7_rl, z7_if, z7_math, z7_code |
    | `ow-f` | p32_sft |

  - total **6.56 pod-hours, $3.77 of the $50 budget** (`podbudget`; the review's §R8). The proposal estimated
    $10–25;
  - the reviewer's A40 pod `owr`: 07:25–07:53 UTC, 0.476 h (`~/podhours.log`), $0.28, registered under
    `olmo-weights`;
  - *[librarian]* RunPod billed $4.27 with disk for the run and its review (executor pods $3.95, reviewer pod $0.32;
    complete, and equal to the balance drop); see the [spend ledger](../../docs/infra/spend.md).
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/olmo-weights/` (*[librarian]* 1,981 files, 49.7 MB, no
  `.safetensors`; the review counted the same 1,981 files and no `.safetensors`):
  - `artifacts/olmo_weights/`: the per-tensor statistics (`stats/<job>/shard_<k>/{iv,lag,st,gram,lora}.csv.gz`,
    `files.csv`, `summary.json`), tables, figures, stage 0, the restage record and logs;
  - `olmo_weights/`: code, manifest and sources;
  - `pod/olmo_weights/`: pod scripts.

  The registry rows are in `hf://buckets/dan-pandori/nd-rl/registry/olmo-weights/` (*[librarian]* one file, 91 rows,
  the same 55,707 bytes as the run worktree's `artifacts/olmo-weights/registry/` copy). No weights were stored.
- **Recompute** (fork, repository root):
  1. the manifest, on the VPS from the Hub API: `python3 olmo_weights/list_refs.py`; `fetch_trees.py <list file>`
     (`manifest/fetch_nonstage1.txt`, `fetch_missing.txt`); `hash_checks.py`; `manifest.py`; and
     `check_headers.py <label> …`, which reads `checkpoints.csv`;
  2. per job, on an A40 pod: `pod/olmo_weights/run_jobs.sh <job,…> 8 <disk cap GB>`, which runs
     `olmo_weights/engine.py --job <job> --device cuda`; `run_p32_sft.sh` for the re-staged job;
  3. on the VPS: `python3 olmo_weights/analyze.py`, `report_tables.py` and `figures.py`, then
     `ND_RUN_ID=olmo-weights python3 olmo_weights/compute_rows.py`.

  The review's phase-1 commands are in its §R8 "Reproduce": `rv_manifest_check.py`, `rv_recount.py`, `rv_soup.py`,
  `rv_files.py`, `rv_compute.py`, `rv_lora_breakdown.py` and `rv_compare.py`, with `rv_job.sh`, `rv_merge_check.py`
  and `rv_merge3.py` on the pod. Phase 2 adds `rv_phase2_checks.py` and `rv_spacing.py`.

## Observations

### What the review checked (its own code; phase 1 before reading the write-ups, phase 2 after)

- **Manifest and hashes (R1).** It fetched the Hub API itself (stdlib `urllib`). For all 453 checkpoints, every
  manifest branch points at the manifest commit, and all 3,141 shard files match in name, size and LFS sha256 (0
  mismatches; 14.69 TB of unique checkpoints). Every pre-registered hash claim holds.
- **Sources (R2).** It downloaded every cited source at its pinned commit: "All 14 files the executor copied into
  `olmo_weights/sources/` are byte-identical to upstream", and the paper's sha256 matches. *[librarian]* `sources/`
  holds 20 scripts and the paper's sha256; the review's comparison covered 14 of the scripts. Every number in
  `steps_lr.md` that the analysis uses matches its source, with two caveats:
  - minor: the 7B long-context script at `3c1a108b` has `max_duration = 5e12` tokens and `hard_stop = 597046`, copied
    from pretrain-1. So "linear to 0 at 11,921" rests on Table 35 (50 B tokens) and the released last step, not on
    the cited script lines;
  - substantive: the row "SFT released `main` = `5e-5-step8000`, LR there ≈ 1.33e-5" is wrong. Those weights are the
    soup of the two runs' final checkpoints, as the report says (§4.2.2; Table 47: "1.0e-4 souped with 5.0e-5").
- **On-pod recompute from the raw weights (R3; the brief's first reviewer check).** One A40, 07:25–07:53 UTC.
  - Its code is independent of `engine.py`: its own safetensors parser and byte reader; bf16 → float32 by bit shift,
    then float64; the ulp from the bf16 bit increment; explicit 2 · B·A products. A synthetic self-test with planted
    changes and a negative control caught a bug in the review's own code (`torch.ldexp` in float64 on CUDA). The
    engine's float32 `ldexp` checked exact on 50 M values.
  - Sample: `random.Random(20261125)` drew 2 intervals from 7B jobs and 2 from 32B jobs. Three fixed tasks were added:
    the 32B RL endpoint (for the LoRA cosine), the triple around the re-staged `32B/sft5e-5/8000` (read from its
    original Hub files), and that checkpoint's interval to DPO. 20 checkpoints, 890 GB, 192 files, all sha256 equal to
    the manifest.
  - Against the brief's 1e-4: every per-tensor squared norm and offset matches to ≤ 2.8e-8 relative, every lag-1 and
    offset cosine to ≤ 1.7e-9 absolute, the per-matrix LoRA dot products to ≤ 3.3e-5 relative (≤ 1.1e-9 as cosines)
    and the aggregated LoRA cosines to 1e-10 relative. Every equal and ≤ 1-ulp count matches exactly on every tensor.

    | task | ‖Δ‖ / ‖θ_a‖ (review = executor) | ≤ 1-ulp fraction | lag-1 cosine (or LoRA cosine) |
    |---|---|---|---|
    | 7B `s1` 5,000 → 8,000 (seeded) | 0.4378624 | 0.0122104 | 0.1413144 |
    | 7B `rl` 1,125 → 1,150 (seeded) | 2.43670e-4 | 0.9877033 | −0.2027348 |
    | 32B `rl` 1,350 → 1,400 (seeded) | 9.89354e-4 | 0.9509689 | −0.2360272 |
    | 32B `sft1e-4` 10,000 → 10,790 (seeded) | 3.86911e-3 | 0.6649127 | −0.0549384 |
    | 32B `sft5e-5` 8,000 → 9,000 (original layout) | 6.04283e-2 | 0.0757183 | −0.9686390 |
    | 32B `sft5e-5/8000` → `dpo` | 3.76498e-4 | 0.9393507 | – |
    | 32B `dpo` → `rl/2300`, all tensors and 448 LoRA matrices | ‖θ − θ_DPO‖² 357.6076 | 0.6519792 | `lora_main` 0.00524973, `lora_ctrl` 0.000186404 |

- **Merges (R3),** on the tested tensors (lm_head and layer-63 `down_proj`, 655 M values per merge). The stage-2
  merge is bit for bit round_bf16 of the mean of the two ingredients' finals. The SFT `main` (`5e-5-step8000`) is bit
  for bit round_bf16((`sft5e-5/10790` + `sft1e-4/10790`) / 2). The stage-3 merge is 88.07 % bit-identical to the
  rounded mean of `s3/10000`, `11000` and `11921`: 11.8 % differ by 1 ulp and 0.09 % by more, and the residual is 1.22×
  the pure-rounding one. So it is "the equal-thirds mean up to rounding", consistent with averaging higher-precision
  copies, which the released bf16 files cannot show.
- **Logs and completeness (R4).** All 3,244 expected member × shard files are logged once, with status ok and the
  manifest sha256, size and commit. The 14 pre-staged files match `restage.json`, whose 14 originals match the
  manifest. 15.110 TB came from the Hub in the
  jobs, plus 64.5 GB for the restage. Every interval covers all 707 / 355 tensors, with 0 non-finite values.
- **Phase 2.** "All 452 rows of `tables/intervals.csv`" agree with the review's to ≤ 4.7e-10 relative (r) and
  ≤ 5e-11 (lag-1 cosine), and the 5,424 module rows both tables share to ≤ 5e-10. *[librarian]* The table has 460
  rows; the other 8 (the bridged 7,000 → 9,000 interval, the soup row, test pairs and excluded RL-Zero rows) are not
  in the review's interval file. The stage, chain, test and LoRA values the review recomputed agree to the printed
  precision. The only difference it names is the 7B stage-1 row: 0.744 on the executor's 32 intervals against 0.725
  on the review's 31. The review says the executor's 32 include the 3,000 → 4,000 gap and the 814-step last interval;
  *[librarian]* the review's 31 include the gap too, so the difference is the 814-step interval. Its 7B stage-2 and
  stage-3 values (0.674, 0.659) also differ from the table's 0.673 / 0.658 in the third decimal: it divides by the
  scripts' 2.0712e-4, the executor by the report's 2.074e-4.
- **Blindness.** Phase 1 ran in `~/review/olmo-weights/`, "the copy with the executor's write-ups removed" (the run
  doc, `numbers.md`, `STATUS.md`, the plain summary and `tables/summary.md`). *[librarian]* The copy still held
  `log.md` with its olmo-weights section, `olmo_weights/README.md`, the CSV and JSON tables, the figures and the
  registry rows. The review says it
  did not read these, nor `~/runs/olmo-weights/*table*.md`, before its §Recount commit. It discloses one exposure:
  at session start (07:16) its auto-loaded memory index carried the executor's one-line summary ("streaming 15 TB …
  for $3.77 …, Hub branches misfiled (32B SFT soup under 5e-5-step8000, RL-Zero name collisions; check headers +
  tensor sets), RL ≈ 8–9×LR per step but cancels (S 0.07–0.10)"). It did not open the memory file, and it notes that
  the hint "may still have steered my attention to the `5e-5-step8000` branch".
  - *[librarian]* The librarian parsed the reviewer's transcript: 123 main-session tool calls before 07:58:04, and no
    sub-agents. They fit the disclosure:
    - the executor's memory note appears only as a path in a `find` listing (07:16:33) and was never opened;
    - none of these was read: the executor's write-ups, tables, figures, registry rows and transcript, or the result
      files in `~/runs/olmo-weights/` (`analyze_full.log`, `compute_table.md`, `expect_table.md`);
    - the review read its inputs (brief, pre-registration, code, manifest, `steps_lr.md`, statistics, job summaries
      and logs, stage-0 logs, the restage record) and two general tooling memory notes (`pod-tooling-gotchas.md`,
      `driver-turn-end.md`);
    - executor results reached phase 1 only through the memory-index line, the commit subjects, and inputs and job
      records: the stage-0 logs, job summaries and logs, `hash_checks.json`, the cost (the `~/podhours.log` rows and
      `podbudget` at 07:18:34, and again through `rv_compute.py` at 07:45:32 for §R8; $3.77), and the executor's
      layout finding. That
      finding reached it through `run_p32_sft.sh`'s comment ("its 14 files use another tensor-to-file layout than the
      other 23 members", 07:20:57), the restage record (07:20:42) and the header listing
      `manifest/headers/32B_sft5e-5_8000.json` (07:43:27). No table, figure or write-up was opened.

    None of the executor's distinctive values (0.744, 1.29, 0.393, 0.0665, 9.86×, w = 0.499996, "7 / 12",
    "0.57–0.87", "15.2 TB", $214.36) occurs in a tool result before the commit. The matches for 0.801, 0.924, 0.00525
    and 0.263 are outputs of the review's own code. The phase-1 text's own S_SFT 0.393 (the bridged chord, R5) appears
    in no earlier tool output; it follows from chord norms the review's code printed before the commit.

### Change per step and the learning rate (Figure 1)

Per stage: medians over the within-stage intervals (`tables/stages.csv`). The review reproduces the table; it gives
0.725 for the 7B stage-1 row (see above), and *[librarian]* 0.674 / 0.659 for 7B stages 2 and 3 (another LR constant).

| model | stage | intervals | steps / interval | rel. change / step | mean LR | per step ÷ LR | bitwise unchanged | ≤ 1 ulp | cos(consecutive) |
|---|---|---|---|---|---|---|---|---|---|
| 32B | stage 1 (sampled pairs) | 25 | 1,000 | 2.1e-4 | 4.01e-4 | 0.569 | 0.008 | 0.025 | −0.125 |
| 32B | stage 2, ingredient 1 | 23 | 1,000 | 8.51e-5 | 9.85e-5 | 0.87 | 0.018 | 0.053 | −0.061 |
| 32B | stage 2, ingredient 2 (every 4th) | 5 | 4,000 | 3.51e-5 | 8.55e-5 | 0.41 | 0.011 | 0.032 | −0.113 |
| 32B | stage 3 | 11 | 1,000 | 8.15e-5 | 9.58e-5 | 0.851 | 0.019 | 0.055 | −0.066 |
| 32B | SFT 5e-5 (the soup name excluded) | 9 | 1,000 | 2.09e-5 | 2.53e-5 | 0.823 | 0.067 | 0.181 | −0.017 |
| 32B | SFT 1e-4 | 10 | 1,000 | 3.67e-5 | 4.58e-5 | 0.803 | 0.044 | 0.119 | −0.037 |
| 32B | RL (3.1 Think) | 45 | 50 | 1.62e-5 | 2e-6 | 8.12 | 0.801 | 0.952 | −0.221 |
| 7B | stage 1 (sampled pairs) | 32 | 1,000 | 1.09e-4 | 1.70e-4 | 0.744 | 0.015 | 0.045 | −0.109 |
| 7B | stage 2 | 48 | 1,000 | 6.61e-5 | 9.87e-5 | 0.673 | 0.021 | 0.060 | −0.098 |
| 7B | stage 3 | 12 | 1,000 | 5.73e-5 | 8.71e-5 | 0.658 | 0.025 | 0.074 | −0.094 |
| 7B | SFT | 42 | 1,000 | 1.67e-5 | 2.54e-5 | 0.658 | 0.086 | 0.213 | −0.057 |
| 7B | DPO (one interval) | 1 | 1,172 | 3.25e-7 | 4e-8 | 8.13 | 0.793 | 0.916 | +0.000 |
| 7B | RL (Think) | 54 | 25 | 8.94e-6 | 1e-6 | 8.94 | 0.924 | 0.985 | −0.198 |
| 7B | RL-Zero IF | 15 | 100 | 5.2e-6 | 1e-6 | 5.2 | 0.819 | 0.950 | −0.260 |
| 7B | RL-Zero Math | 18 | 100 | 4.74e-6 | 1e-6 | 4.74 | 0.827 | 0.957 | −0.247 |
| 7B | RL-Zero Code | 25 | 100 | 5.04e-6 | 1e-6 | 5.04 | 0.841 | 0.965 | −0.244 |

- **Pretraining → SFT** (reproduces). At 1,000-step spacing the median change per step is 0.57–0.87 × the LR in every
  stage of both models. The review's recount: 32B 0.569 / 0.870 / 0.851 / 0.83 (5e-5; *[librarian]* on the
  pre-registered chain) / 0.80 (1e-4); 7B 0.725 / 0.674 / 0.659 / 0.658.
- **Stage switches** (G3, reproduces). All four stage 1 → 2 and stage 2 → 3 boundary intervals have a larger change per
  step than the last within-stage interval before them: 32B 1.76e-4 vs 6.16e-5 and 1.53e-4 vs 4.96e-6; 7B 1.39e-4 vs
  3.27e-5 and 1.14e-4 vs 3.98e-6 (7B's last stage-2 interval is 84 steps).
- **RL at its released spacing:** 8.12 × LR per step for 32B (50-step intervals) and 8.94 × for 7B (25-step). The run
  doc reads this as "**RL breaks this pattern.** At a constant LR, its change per step grows to about 8–9 × the LR."
  The review: "numbers reproduce; the interpretation is not supported", and it puts "RL's large per-step change
  relative to its LR as a property of RL" under "Not supported". Per step ÷ LR falls with checkpoint spacing inside
  the same run. The chords come exactly from the stored statistics, which the review verified on raw weights:

  | run | native spacing | 2× | 3× | 4× | first 1,000 steps (from the stage start) |
  |---|---|---|---|---|---|
  | 32B RL (Olmo 3.1 Think) | 7.98 (50 steps) | 4.96 | 3.75 | 3.15 (200) | 0.97 (1.08 if the lineage after the restart is 900 steps) |
  | 7B RL (Olmo 3 7B Think) | 8.92 (25) | 5.67 | 4.29 | 3.51 (100) | 0.68 |
  | 7B SFT | 0.658 (1,000) | 0.452 | 0.364 | 0.314 (4,000) | 0.86 |
  | 32B SFT 5e-5 / 1e-4 | 0.83 / 0.80 (1,000) | – | – | – | 0.90 / 0.86 |
  | 32B mid-training | ingredient 1: 0.87 (1,000) | – | – | ingredient 2 (same schedule): 0.41 (4,000) | – |

  The native-spacing RL values (7.98, 8.92) include the first interval from DPO (46 / 55 intervals); the per-stage
  table above is within-stage (45 / 54).

  - Over ~1,000 steps RL moves 0.7–1.1 × LR per step, like every other stage. That is ≈ 11–25× less per step than
    SFT (at matched 1,000-step spacing: 10.8–13.6× for 32B, 24.6× for 7B), about the ratio of their LRs.
  - The review names two causes. The first is spacing with anti-correlated changes: consecutive RL changes have cosine
    ≈ −0.2, so the displacement grows sublinearly with steps (here ∝ L^0.33), and short intervals inflate the per-step
    rate.
  - The second is bf16 rounding, which "plausibly adds to this" at RL's and DPO's tiny LRs. At 25–50 steps, ≥ 60 % of
    the moved weights move by ≤ 1 ulp. 7B DPO, one 1,172-step interval at a peak LR of 8e-8 (*[librarian]* the ratio
    uses the mean, 4e-8), also gives 8.1 × LR. In the 32B RL run the per-step rate drops back to the first-interval
    level after each optimizer restart and then climbs again,
    which fits a rounding effect. The pre-registration listed both mechanisms under G2.
  - RL's "first 1,000 steps" values are chords from DPO, not mid-run intervals. Mid-run RL at 1,000-step spacing is
    the review's first next measurement.
  - As pre-registered (native spacing), E1 is a miss and G2 a hit.

### RL's total move, straightness and unchanged weights

- **Totals (E2, hit).** RL's last checkpoint is 0.364 % of ‖θ_DPO‖ from DPO for 32B (`rl/2300`) and 0.0806 % for 7B
  (`rlmain`, beyond `rl/1375`'s 0.0798 %). SFT moved 11.31 % for 32B (SFT `main`, the soup, from base `main`) and
  14.95 % for 7B (`sftmain` from `s3/11921`). So RL moved 31× and 185× less.
- **Within one bf16 step of DPO at RL's end (E3, hit):** 65.20 % of the 32B weights (34.30 % bitwise equal; the
  review's raw-weight value is 0.6519792 exactly) and 87.61 % of the 7B weights (67.89 %).
- **Between RL checkpoints,** a median 80.1 % (32B, per 50 steps) and 92.4 % (7B, per 25 steps) of weights are bitwise
  unchanged; all 46 32B RL intervals exceed 50 % (minimum 0.580). Of the weights that move, **about three quarters**
  move by ≤ 1 rounding step: 32B median 76 % (range 60–79 %), 7B 75–83 %. The run doc's "at least three quarters" is
  wrong for 32B.
- **Embedding against MLP (G6, hit):** in all 46 32B RL intervals the embedding is more often bitwise unchanged than
  every MLP matrix type (medians: embedding 0.985, each MLP type ≤ 0.803).
- **Consecutive changes:** the median cosine is −0.221 for 32B RL and −0.198 for 7B RL, against −0.017 / −0.037 for
  32B SFT (5e-5 / 1e-4) and −0.057 for 7B SFT. At 2× spacing the RL values are −0.219 / −0.236. *[librarian]* The
  review prints −0.038 and −0.058 for the last two; the underlying values are −0.03746 and −0.05750 (`stages.csv`,
  the review's `rv_chains.csv`), as in the table above.
- **Straightness (E4, miss).** S_RL is 0.1006 against S_SFT 0.393 for 32B (the executor's 5e-5 chain without the soup
  name; 0.263 on the pre-registered chain), and 0.0707 against 0.2086 for 7B. Subsampled to SFT's interval count, RL
  gives 0.240 at 12 intervals for 32B (the exact value nearest to 11 from the stored lags) and 0.084 at 43 for 7B.
  These still miss, although the spacing still differs (≈ 200 against 1,000 steps for 32B; *[librarian]* ≈ 25–50
  for 7B, from the review's subsample gaps). The review on the subsample:
  "Coarser subsampling can only raise S, so this bound is informative but not decisive at 11."
- **Distance from each stage's start** (`tables/chains.csv`, the executor's chains). *[librarian]* The review recomputed
  its own 16 chains on the pre-registered definitions; the mid-training ingredient, 32B stage-3 and RL-Zero rows here
  use other end points or exclusions, so it did not check those rows as printed, and it checked the 32B SFT 5e-5 row
  (bridged) only for S.

  | model | stretch | start → end | intervals | end distance / ‖θ_start‖ | path / ‖θ_start‖ | S | mean cos | excluded names |
  |---|---|---|---|---|---|---|---|---|
  | 32B | stage 1 (sampled) | `s1/0` → `s1/656000` | 48 | 1.67 | 35.2 | 0.0475 | −0.110 | |
  | 32B | stage 2, ingredient 1 | `s1/656000` → `s2i1/23842` | 24 | 0.378 | 2.21 | 0.171 | −0.057 | |
  | 32B | stage 2, ingredient 2 (every 4th) | `s1/656000` → `s2i2/23842` | 6 | 0.378 | 1.02 | 0.371 | −0.103 | |
  | 32B | stage 3 | `s2merge` → `s3/11921` | 12 | 0.273 | 1.06 | 0.259 | −0.063 | |
  | 32B | SFT 5e-5 | `s3merge` → `sft5e-5/10790` | 10 | 0.0868 | 0.221 | 0.393 | −0.034 | `sft5e-5/8000` |
  | 32B | SFT 1e-4 | `s3merge` → `sft1e-4/10790` | 11 | 0.159 | 0.441 | 0.361 | −0.035 | |
  | 32B | RL | `dpo` → `rl/2300` | 46 | 0.00364 | 0.0362 | 0.101 | −0.206 | |
  | 7B | stage 1 (sampled) | `s1/0` → `s1/1413814` | 58 | 1.81 | 21.9 | 0.0827 | −0.079 | |
  | 7B | stage 2 | `s1/1413814` → `s2/47684` | 49 | 0.37 | 3.53 | 0.105 | −0.090 | |
  | 7B | stage 3 | `s2/47684` → `s3/11921` | 13 | 0.192 | 0.802 | 0.239 | −0.088 | |
  | 7B | SFT | `s3/11921` → `sft/43000` | 43 | 0.15 | 0.717 | 0.209 | −0.059 | |
  | 7B | RL | `dpo` → `rl/1375` | 55 | 0.000798 | 0.0113 | 0.0707 | −0.191 | |
  | 7B | RL-Zero IF | `s3/11921` → `zif/1900` | 17 | 0.00135 | 0.00879 | 0.154 | −0.261 | `zif/100`, `zif/400` |
  | 7B | RL-Zero Math | `s3/11921` → `zmath/1900` | 19 | 0.00123 | 0.0083 | 0.148 | −0.230 | |
  | 7B | RL-Zero Code | `s3/11921` → `zcode/2900` | 26 | 0.00147 | 0.0124 | 0.119 | −0.243 | `zcode/100`, `200`, `300` |

  The 32B RL chain runs through steps 150 and 200, which are off the lineage (the review's new finding below).
- **RL from the base against RL from DPO (7B; G7, miss).** At step 1,000, RL-Zero IF (1.071e-3) and Math (8.35e-4)
  are farther from their start than RL from DPO (6.80e-4), but Code (6.61e-4) is not.

### Where the change happens

- Attention and MLP matrices move **≈ 1–2×** the model average in every stage. The run doc's "1.1–1.9×" is reworded:
  the executor's table pooled by stage gives 1.05–1.93, and per family 0.96–1.99 (in 32B stage 1, v moves 0.96× and o
  1.01×). Norm vectors move 0.00–0.30×.
- DPO's jump concentrates on attention. For 32B, q, k, v and o move 2.16, 2.59, 2.65 and 2.56 × the model average,
  against 0.96–1.42 for the MLP. For 7B, attention moves 1.84–2.15× and the MLP 1.51–1.84×.
- The review recounted per-module and per-layer r. Figure 3's heatmaps were not among the figures it read.

### LoRA against RL (32B; E5, hit)

| RL step | cos `lora_main` | cos `lora_ctrl` | random mean ± sd (8 draws) | ‖ΔW_main‖ / ‖θ_t − θ_DPO‖ |
|---|---|---|---|---|
| 50 | 0.00078 | 0.00000 | −2.0e-6 ± 4.5e-6 | 5.125 |
| 100 | 0.00107 | 0.00002 | −5.1e-8 ± 5.6e-6 | 3.330 |
| 200 (*[librarian]* off the lineage) | 0.00133 | 0.00008 | −2.2e-6 ± 7.2e-6 | 2.343 |
| 500 | 0.00280 | 0.00007 | 1.6e-6 ± 6.5e-6 | 1.482 |
| 750 | 0.00369 | 0.00016 | −5.7e-7 ± 3.8e-6 | 1.083 |
| 1,000 | 0.00423 | 0.00014 | −1.3e-6 ± 7.0e-6 | 0.880 |
| 1,100 | 0.00449 | 0.00015 | −1.9e-6 ± 6.3e-6 | 0.807 |
| 1,500 | 0.00459 | 0.00021 | −2.6e-6 ± 5.0e-6 | 0.536 |
| 1,600 | 0.00472 | 0.00022 | −2.4e-6 ± 5.2e-6 | 0.527 |
| 2,000 | 0.00504 | 0.00019 | −9.6e-7 ± 3.7e-6 | 0.495 |
| 2,300 | 0.00525 | 0.00019 | −5.6e-7 ± 5.0e-6 | 0.468 |

(The executor's `numbers.md` selection of steps; `tables/lora.csv` has all 46.)

- At step 2,300, over the 448 targeted matrices, the cosine of `lora_main` with RL's change is 0.00525. The run doc's
  0.0053 should read 0.0052 (0.0052497). That is ≈ 1,000 random sd above chance: z = 1,044 on the executor's 8
  random draws, 1,088 on the review's own 8. It is 28× the control's 0.000186, and both are below 0.1. The review
  recomputed both cosines from the raw weights.
- The alignment grows during RL, near monotone: 0.00078 at step 50, 0.00525 at 2,300. Over the targeted matrices the
  adapter is 0.468× the size of RL's final change.
- By module and depth (the review): `lora_main` beats `lora_ctrl` in all 7 module types and all 8 depth bins. The
  cosine is largest in `down` (0.0129), `v` (0.0127) and `o` (0.0107).
- The inherited claim, labelled: `lora_main` elicits 40–41 of 48 IFBench / IFEval reach prompts on
  `Olmo-3-32B-Think-DPO`, and `lora_ctrl` 0 (olmo-ifladder, reviewed).

### Checkpoint findings

- **The 32B SFT `main` is the soup of the two learning-rate runs' finals, filed under the branch `5e-5-step8000`.**
  The review: "reproduces, and more strongly than claimed".
  - The Gram fit is 0.5000 / 0.5000 on `sft5e-5/10790` + `sft1e-4/10790`, with residual 10.09, the bf16-rounding
    level. On the tested tensors it is bit for bit equal to round_bf16 of their mean.
  - In the 5e-5 chain the distance from base goes 424.6 → 583.3 → 443.7 at 7,000 → "8,000" → 9,000. The change
    into "8,000" is 6.40e-2 against 1.70e-2 for the interval before it, and the next change reverses it (cosine
    −0.969). The branch uses SFT `main`'s shard layout, not that of the other 21 branches (the executor's header
    check).
  - The report says so: §4.2.2, "a linearly weighted merge of two checkpoints trained with different learning rates",
    and Table 47, "1.0e-4 souped with 5.0e-5". The pre-registration's "the released SFT is the 5e-5 run at step
    8,000" and `steps_lr.md`'s "LR there ≈ 1.33e-5" are wrong. The 5e-5 run's own step 8,000 is not released
    (`numbers.md`).
  - DPO started from it. θ_DPO is 1.955 from it (3.8e-4 relative, 93.9 % within 1 ulp) and ≥ 310.2 from every other
    SFT checkpoint. G5 misses for that reason.
  - The executor excluded the name from its 5e-5 chain, bridging 7,000 → 9,000, and calls this applying "the
    pre-registered duplicate rule". The review scores E1 and E4 on the pre-registered chain, which keeps the soup:
    E1's 32B ratio is 1.62 there (1.29 bridged) and S_SFT 0.263 (0.393), as `log.md` also gives them. With the two
    intervals around the soup dropped, the review gets 1.39. Every variant is still a miss.
- **Stage 3 started from the stage-2 merge.** `s3/1000` is 776.1 from the merge and 1,712.9 from `s1/656000`, so it did
  not start from the script's `load_path` `step656000`.
- **Merges.** Stage 2 is the equal mean of the two ingredients' finals (weight 0.5000, residual 0.46 % of their
  distance; bit for bit on the tested tensors). Stage 3 is "the equal-thirds mean up to rounding-level differences (88 %
  bit-identical to the rounded mean of the released branches)". The best weights on 9,000 / 10,000 / 11,000 / 11,921
  are 0.0004 / 0.3327 / 0.3344 / 0.3325. The executor reads this as three merged checkpoints, as the report and the
  branch name say, where the model card says four (`numbers.md`).
- **RL-Zero branch names.** Five hold misfiled weights and are excluded:
  - IF `step_100` (= `step_1000`) and `step_400` (17.94e-4 of ‖θ_base‖ from base `main`, beyond IF's `main` at 13.79e-4,
    fitting no position from 100 to 2,000);
  - Code `step_100` / `step_200` (= `step_1000` / `step_2000`) and `step_300` (= Code `main`; 15.11e-4, beyond
    `step_2900`'s 14.73e-4, so ≈ step 3,000).

  Math is clean. Byte identity alone cannot catch IF `step_400` or Code `step_300`. In label order the IF and Code
  chains each have 4 corrupted intervals.
- **Starts.** The 7B SFT and all three RL-Zero runs start from base `main`: their first checkpoints are nearer to it
  than to `s3/11000` (SFT 52.07 against 56.44; IF 3.35, Math 0.51 and Code 2.07 against ≈ 22). The IF and Code "first"
  checkpoints are the misfiled step-1,000 weights, but IF `step_200` (1.00 from base) and Code `step_400` (1.14) give
  the same answer. 7B DPO started from `sftmain` (1.202 away, against 1.837 from `sft/43000`).
- **Mains beyond the last branch:** 7B `rlmain` ≈ step 1,400 (the report's 1,400); RL-Zero IF and Math `main` ≈ step
  2,000.
- `allenai/Olmo-3-32B-Think`'s 15 checkpoints are 3.1's `step_0050`–`step_0750` (its `main` = `step_0750`), and 3.1's
  `main` = `step_2300`.

### The 32B RL checkpoints are not one line (the review's new finding; `Olmo-3.1-32B-Think`, identical to `Olmo-3-32B-Think` for steps 50–750)

The review computed chords between checkpoints from the stored statistics (`rv_spacing.py`; units of 1e-4 of
‖θ_DPO‖):
- `step_0150` and `step_0250` are both "`step_100` + one fresh 50-step move" (the review's words). Each is 3.32
  from step 100 (the size of DPO → 50, 3.31), 5.42 from step 50 and 6.09 from DPO. They are 4.67 apart, which is
  √2 × 3.32, so the two moves are orthogonal (cosine 0.008).
- Step 200 descends from 150 (4.32), not from 250 (6.10). Steps 300 onward descend from 250 (d(250, 300) = 4.40,
  d(200, 300) = 7.29).
- So 150 → 200 is a branch the run abandoned, and the released line restarts from step 100. The 200 → 250 "interval" is
  a jump between branches (cosine −0.645 with the interval before it).
- Three intervals show a fresh-optimizer signature (first-interval size, cosine ≈ 0 with the previous interval): DPO →
  50, 100 → 150, and 1,850 → 1,900. The last is a resume without rollback. The 7B RL chain shows none of this.
- Consequences, in the review's words: labels from 250 up "probably overstate the optimizer steps along the released
  lineage by ≈ 100", which is small for per-step rates and ≈ 4 % for totals. "olmo-ifcurve's step-200 point (Figure
  5's pass@1 0.347 and the rug of 7 first-genuine prompts at step 200; same commit `15d5e3cd`) is on the abandoned
  branch, not on the lineage of steps 250–2,300." The executor did not detect this; its duplicate rule covered only
  byte-identical names. The review asks for confirmation with direct differences and AI2's RL logs.
- *[librarian]* This is marked in the olmo-ifcurve summary (a dated note under its models, and pointers at its step-200
  results) and in its index line. Its STATE line and plain-summary copy do not mention step 200 and are unchanged. The
  review did not re-derive olmo-ifcurve's readings without step 200. Figure 5's rug is olmo-ifcurve's executor F1
  (`emergence.csv`: 8 prompts at 1,100 and 2 at 2,300), which differs from the olmo-ifcurve review's on two prompts.

### Expectations (pre-registered 04:48:59, before the first pod), scored by the review

E1–E5 come from the proposal; G1–G7 are the executor's own guesses.

| id | expectation | 32B | 7B | the review's score |
|---|---|---|---|---|
| E1 | median r / step: RL ≥ 10× smaller than SFT | RL 1.624e-5 (45 intervals) vs SFT 5e-5 run 2.633e-5 (10): 1.62× (bridged chain: 1.29×) | RL 8.94e-6 (54) vs SFT 1.669e-5 (42): 1.87× | **miss**, as pre-registered (native spacing). At matched 1,000-step spacing: 10.8–13.6× and 24.6× |
| E2 | RL's total from DPO < SFT's total from base | 0.003642 vs 0.1131 (SFT `main`, the soup): 31× | 0.000806 vs 0.1495: 185× | **hit** both |
| E3 | > 50 % of weights within 1 ulp of DPO at RL's end | 65.20 % (bitwise 34.30 %) | 87.61 % (67.89 %) | **hit** both |
| E4 | S_RL > S_SFT (native spacing) | 0.1006 vs 0.263 (pre-registered chain; 0.393 bridged); subsampled 0.240 at 12 intervals | 0.0707 vs 0.2086; 0.084 at 43 | **miss** both |
| E5 | LoRA at 2,300: cos_main > cos_ctrl, both < 0.1, main > random + 3 sd | 0.005250 vs 0.000186; random −5.6e-7 ± 5.0e-6, z = 1,044 | – | **hit** |
| G1 | 32B RL intervals > 50 % bitwise unchanged; SFT 1,000-step intervals < 10 % | RL 46 / 46 (minimum 0.580); SFT 16 / 18 below 10 % (median 0.048), both 9,000 → 10,000 intervals above it (1e-4: 0.130; 5e-5: 0.227) | – | RL half hit; the SFT half holds at the median and fails in 2 of 18 intervals. The executor scored it at the median (hit); the pre-registration does not say |
| G2 | per step ÷ LR ≥ 2× larger for RL than SFT | 8.12 vs 0.830: 9.8× | 8.94 vs 0.658: 13.6× | **hit** both (native spacing) |
| G3 | boundary intervals > the last within-stage interval | 1.76e-4 vs 6.16e-5; 1.53e-4 vs 4.96e-6 | 1.39e-4 vs 3.27e-5; 1.14e-4 vs 3.98e-6 | **hit** |
| G4 | merges are averages (< 1 % residuals) | stage 2: 0.46 % (weight 0.5000). Stage 3: equal thirds (0.3327 / 0.3344 / 0.3325; 9,000 gets 0.0004), but the residual is 16.7 % (8.56 vs 51.20) | – | **miss** on the stage-3 residual. The threshold is below the bf16 rounding floor: a perfectly rounded equal-thirds mean leaves ≈ 14 %, so no recipe could pass |
| G5 | 32B DPO did not start from SFT `main` | 1.955 from SFT `main`, ≥ 310.2 from all others | – | **miss**: DPO started from the released SFT `main`, which is itself the soup |
| G6 | 32B RL: embedding unchanged more often than every MLP type | 46 / 46 (median 0.985 vs 0.803) | – | **hit** |
| G7 | 7B at step 1,000: each RL-Zero run farther from its start than RL from DPO | – | IF 1.071e-3, Math 8.35e-4 > 6.80e-4; Code 6.61e-4 < 6.80e-4 | **miss** (2 of 3) |

So 6 or 7 of 12 hit, depending on the G1 reading. The run says "7 of 12" and must say that G1 was scored at the
median. The review found the expectations committed before the first pod request and unchanged since, and misses
reported as misses, except for G1's scoring rule.

### Compute (the review's §R8)

| job | b32_s1 | b32_s2 | b32_s3 | b7_s1 | b7_s2 | b7_s3 | p32_rl | p32_sft | p7_rl | p7_sft | z7_code | z7_if | z7_math |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A40 seconds | 3,613 | 2,461 | 1,157 | 992 | 822 | 226 | 3,405 | 2,175 | 884 | 750 | 491 | 372 | 349 |
| TB from the Hub | 3.159 | 2.063 | 0.967 | 0.861 | 0.730 | 0.204 | 3.030 | 1.483 | 0.832 | 0.686 | 0.452 | 0.321 | 0.321 |

- 13 engine runs, 17,698 A40-seconds (4.92 h, one A40 each), 15.110 TB downloaded. Not in the table: the p32_sft
  restage (450 s), one failed p32_sft attempt on `ow-d` (06:01, rc 1), and stage 0 and 0b (≈ 30 min on `ow-s0`).
- Generated tokens, training steps, training tokens and Lean checks are all 0: nothing was sampled or trained. All
  3,244 expected member × shard files are logged with status ok, the manifest sha256, size and commit; the 14
  pre-staged files match `restage.json`, whose 14 originals match the manifest (R4).
- The registry holds 91 rows, 13 jobs × 7 metrics (`gpu_seconds`, `gen_tokens`, `train_steps`, `train_tokens`,
  `lean_checks`, `bytes_downloaded`, `files_verified`). `gpu_seconds` equals each job's `summary.json`. The review:
  "Present and correct … No arm comparison is at stake". *[librarian]* Each job measures different checkpoints, so
  no arm has a comparator for the 1.25× flag.
- Pods: `~/podhours.log` lists 7 A40 pods (SECURE, $0.59/h; 6.194 h) and the RTX 4000 Ada request that never got a
  machine (0.36 h, logged at $0.10): **6.557 h and $3.77** of $50. *[librarian]* The review gives the A40 pods as
  $3.65, which is 6.194 h × $0.59; the log's seven A40 rows add to $3.67, and $3.67 + $0.10 is the $3.77. Stage 0
  cost $0.32 + $0.10, under the ≤ $3 gate. The reviewer pod: 0.476 A40-hours, $0.28.

### Other policy checks (the review)

- Every number in the write-up and `numbers.md` names its repo, branch or commit. The exception is the plain summary's
  inherited olmo-ifladder claim, which is labelled in the copy here.
- The language-model standards (hand gate, templates, budgets) and the seed and noise-floor standard do not apply: no
  text is generated, and the statistics are deterministic. The pre-registration says so, and the review agrees.
- No secret is in the committed or uploaded files (the scan included the gzipped CSVs). Weights stayed on pods: the
  bucket holds no `.safetensors`.

### The review's rewordings (§Verdict), all applied above

1. "RL breaks this pattern … 8–9 × the LR" (run doc), and the plain summary's "about ten times larger than the
   learning rate predicts" (the review abbreviates it to "…LR predicts") and "an RL step is nearly as large as an SFT
   step". These hold at the released 25–50-step spacing. Over
   ~1,000 steps RL moves 0.7–1.1 × LR per step, like every other stage, and ≈ 11–25× less per step than SFT. The causes
   to name are spacing (with anti-correlated changes) and bf16 rounding.
2. E1's sentence needs the same spacing caveat. The miss stands as pre-registered.
3. "At least three quarters" → "about three quarters (32B median 76 %, range 60–79 %; 7B 75–83 %)".
4. "7 of 12" → say that G1 was scored at the median. Two of the 18 32B SFT 1,000-step intervals have more than 10 %
   unchanged.
5. G4's "bf16 rounding of an exact average" → "the equal-thirds mean up to rounding-level differences (88 %
   bit-identical to the rounded mean of the released branches)".
6. Minor:
   - "1.1–1.9×" → "≈ 1–2×";
   - 0.0053 → 0.0052;
   - "15.2 TB" is the download volume (the 453 checkpoints are 14.7 TB);
   - give the pre-registered-chain values next to the soup-excluded ones (E1 32B 1.62, S_SFT 0.263), as `log.md` does;
   - label the plain summary's olmo-ifladder claim;
   - `steps_lr.md`'s SFT `main` row is superseded.

**The fork's executor files are not corrected.** On branch `dan_olmo-weights`, `run_olmo_weights.md`, `numbers.md`,
`olmo_weights/plain_summary.md` and `steps_lr.md` still carry the wording the review asks to change, and the
pre-registration keeps its SFT line, which the review calls wrong (R7.1). Only the plain-summary copy here is marked.

## For Proposal 48 (`olmo-kl`) and other reuse of these checkpoints

Proposal 48 (not approved or launched) says "It can reuse Proposal 47's checkpoint manifest and hash checks", and it
plans to "Plot behavioural distance on Proposal 47's x-axis". What to take (the review's findings, with file and
commit facts from the fork):
- **The manifest**, pre-registered at `a92b8cb7`: `olmo_weights/manifest/` with `checkpoints.csv` (commits and shard
  sha256), `refs.json`, `trees/`, `jobs.json` and `hash_checks.json`; the real safetensors `headers/` were added at
  `7ef655dd`. The review's own Hub fetch matched it (0 mismatches on 3,141 shard files).
- **The corrected readings:**
  - the 32B SFT `main` / `5e-5-step8000` is the soup of both runs' finals, not a training step. Under
    `5e-5-step8000` the files use `main`'s shard layout, not the one that branch's own index file describes (549 of
    707 tensors in another file; the executor's header check, `log.md`);
  - five RL-Zero names are misfiled: IF `step_100` and `step_400`; Code `step_100`, `step_200` and `step_300`;
  - 32B RL `step_0150` and `step_0200` are off the lineage of steps 250–2,300, whose labels probably overstate the
    lineage's optimizer steps by ≈ 100. 1,850 → 1,900 shows a fresh-optimizer signature (a resume without rollback).
- **Spacing.** Per-step rates depend on checkpoint spacing, so compare stages at matched spacing, not per step at
  native spacing.
- **Steps and LRs.** `steps_lr.md`'s SFT `main` row is superseded, and the 7B long-context schedule rests on Table 35
  and the last step, not on the script's lines.
- **Statistics.** The per-tensor statistics are in the bucket (`…/olmo-weights/artifacts/olmo_weights/stats/<job>/`).
  Chords between non-adjacent checkpoints of one chain follow exactly from them only for gaps of up to 4 intervals
  (the review's note to E4). *[librarian]* Most of Proposal 48's planned intervals span 2–4 of this run's (SFT every
  2nd, stage 2 every 4th, RL every 100 steps), so their chords follow from the stored statistics; the direct
  computation it plans would then be a cross-check, needed only for wider gaps such as between stage-1 locations.

## Limitations and next step

- **Descriptive; weight distance is not behaviour** (the pre-registration). One model family, and one released run per
  stage (32B SFT: two learning-rate runs). The optimizer state and the fp32 training copies are not released, so a
  weight unchanged in bf16 may still have moved in training (Proposal 47's caveat).
- **Per-step rates depend on spacing.** RL's released checkpoints are 25–50 steps apart (RL-Zero 100), the other
  stages' 1,000 (32B ingredient 2 sampled every 4th, so 4,000; DPO one jump). Comparisons per step across stages
  therefore need matched spacing. RL was measured at 1,000 steps only from its start, not mid-run.
- Stage 1 is sampled: 25 / 30 locations as adjacent pairs, and its chain statistics use the sampled sequence.
- The executor excluded the soup name (bridging 7,000 → 9,000) and the five misfiled RL-Zero names, calling this the
  pre-registered duplicate rule (IF `step_400`, with no twin, was "excluded as anomalous"). The review scores E1 and
  E4 on the pre-registered chain, which keeps the soup, and notes that byte identity cannot catch IF `step_400` or
  Code `step_300`. The 32B RL branch was not detected, so every 32B RL figure and chain statistic runs in label order
  through steps 150 and 200.
- **Next measurements** (the review):
  1. **RL at matched spacing, mid-run.** On a pod (< $1), difference RL checkpoints 1,000 steps apart: 32B 300 →
     1,300 and 1,300 → 2,300; 7B 375 → 1,375. "This settles whether mid-run RL also moves ≈ 1 × LR per step at
     1,000-step spacing, as the from-start chords say."
  2. **Confirm the 32B RL branch point** with direct differences (θ_250 − θ_100 against θ_150 − θ_100), and look for a
     rollback at step 200 in AI2's RL logs. Mark olmo-ifcurve's step-200 point as off-lineage. *[librarian]* Marked
     in its summary, pending this confirmation.
  3. **Measure the bf16 rounding share of the RL and DPO changes.** Compare the per-step change against the time since
     each optimizer restart (the 1,850 → 1,900 reset is a natural experiment), or model the flips against the fraction
     of weights near rounding boundaries.
- Paused or separate: the low-rank analysis (Dan paused it until he has these results) and behavioural KL (Proposal
  48).
