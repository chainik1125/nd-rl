The final implementation audit passed. Completed configurations, input/code/checkpoint identities, raw proof rewards, declared compute, and final aggregation reconcile within the scopes below. This is the implementation agent's audit of the sprint it helped build, not a fresh-agent or external replication. No GPU execution or training was launched by these audits.

The [provenance receipt](final_provenance_independent_v1.json) checks 102 distinct completed post-baseline model operations and 612 unique files. Nine jobs acquired GPU4 and completed their workers successfully. One additional development attempt timed out in the queue without acquiring a GPU or doing model work. Prepared second-seed and RL-specific calibration plans are not counted as executions. The initial base-SFT supervisor had a documented final-log error after successful worker completion; its original launcher is preserved and matches the recorded hash in `tools/archive/launch_v1.py`.

The final compute ledger reconciles to **35,868,523,813,466,880 post-baseline dense-compute proxy units**, **9,760.824638 GPU lease seconds (2.711340177 hours)**, and **$0 new cloud spending**. Each physical output is charged once. Aggregate calibration costs include their nested arms. Previously acquired candidate samples are not charged again in the fit or projected-gradient subtotals. The fresh fit, fresh audit, and projected-gradient work add 201,132,229,620,480 proxy units in total.

The shared base completed 8,000 SFT updates, processing 153,702,418 response tokens in 398.905066 training seconds. It did not record the same dense proxy, so its proxy is explicitly unavailable rather than silently treated as zero; its lease time is included above. This proxy approximates model arithmetic rather than measuring hardware FLOPs. CPU data collection, proof replay, and interactive analysis are outside the proxy.

At the [dated GPU observation](final_gpu_observation_v1.json), **02:42:00 UTC on September 11**, GPU4 had 1 MiB memory usage and 0% utilization. All 19 recorded sprint launcher/worker PIDs were absent. Three existing GPU4 lock files were free: the sprint lock, the prior ND sprint lock, and `/tmp/olmo_single_gpu_4.lock`; the optional CRRA lock did not exist. Other users' GPUs are outside this claim. The sandboxed CPU subprocess could not initialize NVML, so the audit explicitly binds the separate, previously approved direct `nvidia-smi` observation. No process-inspection approval wait was needed.

**Training runtime replay.** All four main endpoints have complete semantic replay, including every saved source and prefix state, independently checked reward, group success count, leave-one-out advantage, action/context/padding mask, seeded source/reset/retention schedule, controller counter and transition, and per-step generation/audit/backward/retention proxy. First budget crossings and final checkpoint hashes reconcile.

| Main arm | Completed updates | Strictly replayed attempts | Response groups | Retained examples |
|---|---:|---:|---:|---:|
| Selected, full starts | 9,056 | 289,792 | 36,224 | 144,896 |
| Selected, reset starts | 9,054 | 289,728 | 36,216 | 144,864 |
| Uniform, full starts | 9,096 | 291,072 | 36,384 | 145,536 |
| Uniform, reset starts | 8,827 | 282,464 | 35,308 | 141,232 |
| Total | 36,033 | 1,153,056 | 144,132 | 576,528 |

Receipts are `main_runtime_selected_full_endpoint_complete_v1.json`, `main_runtime_selected_reset_endpoint.json`, `main_runtime_uniform_full_endpoint.json`, and `main_runtime_uniform_reset_endpoint.json`. The earlier selected/full receipt covered complete scalar accounting but only the first 300 updates semantically; it remains preserved, and the new complete receipt closes that gap. Full-start arms retained an unused controller state, while every actual start had cut zero.

The [retention endpoint audit](retention_seed1_endpoint_independent_v1.json) reconstructs all 32,034 updates and 512,544 retained examples from the exact ordered stream of 512 sources, checks all 15 saved exposure checkpoints, and reconciles the first crossing of the 4e15 proxy ceiling. Its final proxy is 4,000,052,734,894,080. The exposure snapshots match update counts and retained examples, not total compute; the separate final checkpoint is the equal-total-compute control.

**Evaluation replay and aggregation.** The [final receipt](final_seed1_independent_v2/receipt.json) checks every one of **196,608 raw attempts from 32 models on 768 final sources**, including 402 canonical contexts. Model/input/code hashes, completed producer results, empty-prefix sampling, eight attempts per source, temperature 0.8, full 512-token response budget, and recorded likelihood-audit bounds reconcile. The independently rerun aggregate agrees **exactly** with the root report for all model summaries, 62 paired comparisons, and 12 factorial contrasts.

Final payload access occurred only after root's explicit notification and the checked global closure/release. The closure at 02:09:45 and release at 02:09:47 bind 56 frozen checkpoints: 32 sampled-evaluation models plus 24 score-only family/control models. No training or selector change follows closure in the executed finish or retention-development pipelines.

The [retention-development receipt](retention_development_independent_receipt_v1.json) adds complete replay of **107,520 attempts from 35 models on 384 development sources**. This includes 52,224 already checked main/adapted attempts and 55,296 newly evaluated control attempts; the former are not new acquisitions. Its independent aggregate exactly matches all 66 paired comparisons and 36 contrasts. The [plan review](retention_development_plan_independent_v1.json) verifies all 16 main-to-retention exposure mappings and the 12 identities `main contrast = retention reference + residual`, including cancellation when both full-start arms share the step-1,107 control. This identity is arithmetic, not an additive causal decomposition of the optimizer trajectory.

The first final-audit copy sorted coefficient keys, producing 78 differences no larger than 2.78e-17 in derived bits/token contrasts. The corrected copy preserves original coefficient order and agrees exactly, without relaxing any comparison tolerance. Earlier exports are preserved. Four gate and addition-order regression tests pass.

**Diagnostic scope.** The [completed diagnostics audit](completed_diagnostics_independent_v1.json) strictly replays all 18,432 fresh chosen/full/latest attempts and both 6,144-attempt projected-input draw sets, with full state, source, seed, action, and producer hash binding. The fresh full component is included in both counts, so these are 24,576 distinct saved attempts, not 30,720 distinct acquisitions. It checks all 768 initial-cut decisions, source-bank reuse, equal-reward algebraic zeros, the projection sign and fixed-512 scaling, saved family contrasts, and incremental compute. Projected diagnostics launch no new rollouts and perform no updates. The completed executable verifies exact equality of its parameters before and after projection; this CPU audit checks that executable and its receipts rather than repeating its gradients. The 23,968 all-cut fitting attempts are not separately replayed in full by this final audit, although their immutable producer, admissible-state map, selection rule, and cost are checked.

All line caps and bins count **total written ND proof lines, including PR premises and AS assumptions**. The prompt and PRF/FINAL/QED markers are excluded. A reference length is a supplied witness length, not a minimum proof length.

**Remaining limits.** These checks do not retrain models or reconstruct every historical parameter/Adam update. Per-update online policies were not saved, so changing-policy teacher probabilities rely on the saved runtime likelihood audits; final saved NLL and behavior values are hash-bound and aggregated without new model forwards. Base/adaptation/calibration/oracle/prefix SFT have input/source/completion/compute validation and the existing meaningful CPU tests, rather than full independent numerical replay of every optimization step. RL budget pilots and every original all-cut fit attempt are not covered by the full main-runtime replay count. One initialization, one online seed, one decoding seed, and synthetic source support limit scientific generalization. Context bootstrap intervals do not estimate variability across training seeds or account for all multiple comparisons.

The following commands repeat the principal read-only CPU checks from the sprint directory, writing fresh outputs under `/tmp`. They load saved weights on CPU only when required to inspect configuration/metadata; they do not train or sample. Resource-state checks are point-in-time checks and may differ on a later rerun. The saved GPU observation preserves the original audit time.

```bash
cd /data/users/dmitry/nd-rl/proof-fragments-20260910-1800
SPRINT_PY=/data/users/dmitry/simplex-research/.venv/bin/python
CPU_AUDIT_OUT=$(mktemp -d /tmp/proof-fragment-cpu-audit.XXXXXX)

"$SPRINT_PY" -B code/audit_main_runtime.py --run results/rl_selected_full_seed1_v1 --steps 9056 --output "$CPU_AUDIT_OUT/selected_full.json"
"$SPRINT_PY" -B code/audit_main_runtime.py --run results/rl_selected_reset_seed1_v1 --steps 9054 --output "$CPU_AUDIT_OUT/selected_reset.json"
"$SPRINT_PY" -B code/audit_main_runtime.py --run results/rl_uniform_full_seed1_v1 --steps 9096 --output "$CPU_AUDIT_OUT/uniform_full.json"
"$SPRINT_PY" -B code/audit_main_runtime.py --run results/rl_uniform_reset_seed1_v1 --steps 8827 --output "$CPU_AUDIT_OUT/uniform_reset.json"

"$SPRINT_PY" -B code/audit_retention_endpoint.py --run results/retention_control_seed1_v1 --output "$CPU_AUDIT_OUT/retention_endpoint.json"

"$SPRINT_PY" -B code/audit_final_evaluations.py --plan configs/analysis_final_seed1_v1.json --closure reports/training_closed_v1.json --release configs/release_final_seed1_v1.json --authorized-by-root-notification --root-analysis reports/final_seed1_v1.json --output "$CPU_AUDIT_OUT/final"

"$SPRINT_PY" -B tools/reverify_attempts.py configs/analysis_retention_development_v1.json --output "$CPU_AUDIT_OUT/retention_development_replay.json"

"$SPRINT_PY" -B code/audit_final_provenance.py --gpu-observation code/tests/final_gpu_observation_v1.json --output "$CPU_AUDIT_OUT/provenance.json"

cd code
"$SPRINT_PY" -B -m unittest test_final_audit_gates test_retention_only test_projected_loo
```

The three test modules above cover final-release identity and ordering, actual zero-advantage versus pure-retention trajectory equivalence on a small CPU model, and analytic/finite-distribution projection sign, group normalization, fixed-token scaling, and equal-reward behavior. The retained tests and frozen experiment programs are distinct from the prose and plot review; no frozen training snapshots or root reports were edited during validation.
