# Initial proof-cap comparison

Six configurations compare supervised proof caps **6, 10 and 12**, each with a direct-proof model and a symbolic-CoT model. All share an 8.87M-parameter architecture, initialization, nominal seed, RL prompt distribution and training-compute targets. Raising the cap also expands synthetic proof-family support; this experiment does not isolate length from training distribution.

All six configurations are complete. Read the [results and graphs](reports/summary.md). The best sampled coverage is **10/58**, but the longest solved textbook reference remains **7 lines for Direct and 6 for CoT**; the seven-line result already existed after SFT. Scores use the **58-problem adaptive textbook development set**, released separately after each full RL run. The historical final-test population is excluded. Only the already permitted 14 textbook training classes enter training, as prompts without proof labels.

## Inspect progress

Run from this directory on simplex1:

```bash
/data/users/dmitry/simplex-research/.venv/bin/python code/status.py
```

This reads training telemetry and coordinator state. Its remaining-time estimate covers the current training stage only and assumes the observed average throughput continues. It does not include later stages, evaluations or queue time.

The [Trackio dashboard](http://127.0.0.1:7862/?project=cap-comparison-v2&smoothing=0&metric_filter=%5Eaccuracy%2F&x_axis=compute%2Frl_budget_fraction) shows individual runs, with smoothing disabled. From another machine, forward the port:

```bash
ssh -N -L 7862:127.0.0.1:7862 dmitry@simplex1
```

Pretraining accuracy uses the same 512 common short SFT examples; synthetic validation uses the same 256 separate problems. Both probes have reference proofs of 2–6 lines and measure short-proof competence. Each point is a checkpoint evaluation. The intermediate checkpoint evaluations run after RL finishes, so an active run initially has only its SFT baseline. Textbook accuracy has only SFT and full-RL endpoints, both evaluated after the run. No averages combine different training configurations.

## Reproduce completed-run analysis

```bash
/data/users/dmitry/simplex-research/.venv/bin/python code/analyze.py --root .
/data/users/dmitry/simplex-research/.venv/bin/python code/report.py --root .
```

Analysis requires all six configurations, checks completed training budgets, immutable identities, checkpoint hashes and saved evaluation artifacts, then strictly replays every textbook attempt. The finalizer completed these checks and exited. The authored interpretation is bound to the resulting analysis hash.

The [completion audit](reports/completion_audit.json) records frozen-input and artifact checks. The [dashboard accuracy audit](reports/dashboard_accuracy_audit.json) checks all 72 imported points and native SQLite records against the verified results; the [browser check](reports/dashboard_browser_check.json) and [snapshot](reports/dashboard.png) verify labels, grouping, smoothing and the compute axis. Reproduce these checks with:

```bash
/data/users/dmitry/simplex-research/.venv/bin/python code/audit_dashboard.py --root . --imports /data/users/dmitry/nd-rl/observability/.observatory/imports/cap-comparison-v2
/data/users/dmitry/nd-rl/observability/.venv/bin/python code/check_dashboard_browser.py --root .
```

Relevant records:

- [Frozen training protocol](plans/protocol.json) and [prespecified analysis](reports/analysis_protocol.json).
- [Compute accounting](reports/compute_implementation.md) and [prelaunch review](reports/prelaunch_review.md).
- [Infrastructure interruption receipt](reports/recovery_protocol.json), [same-seed restart variability](reports/restart_reproducibility.json) and [GPU placement adjustment](reports/placement_adjustment.json).
- [First completed model's proof examples](reports/cap10_cot_s20260910_examples.md) and [cap-6 CoT examples](reports/cap6_cot_s20260910_examples.md). These are posthoc illustrations of completed results.

Training uses the bound `snapshots/main_v1` code and data. Operational recovery, monitoring and reporting code is separate from those frozen inputs. Infrastructure retries start from the same initialization with complete budgets; archived interrupted attempts are excluded from comparisons and are not extra seeds.
