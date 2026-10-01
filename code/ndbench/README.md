# ndbench: capability benchmarks for ND provers

`ndbench` samples proofs from a model on fixed theorem suites, judges every attempt, stores the
results, and draws one chart per benchmark. It holds 8 benchmarks (glue, families, skills, laws,
padding, prefix, EDL, Yue pass@k) and 4 interp measurements (likelihood, steering, weight
interpolation, probes).

- Results and their caveats: [`experiment-summaries/2026-09-30-benchmarking/`](../../experiment-summaries/2026-09-30-benchmarking/README.md).
- Plan, configs, schedules and GPU rules: [`code/experiments/current/benchmarking_20260930/`](../experiments/current/benchmarking_20260930/README.md)
  (`PLAN.md`, `INTERP_SPEC.md`, `runbook.md`).
- Outputs (store, suites, logs, dashboards): `/scratch/cowifornia/nd-rl/bench/out/` on serrano.
- Models are labels in `configs/models.yaml` (adapter, checkpoint, sha256 prefix).

## Cost tiers: what needs a GPU

Most things an agent wants to do with these benchmarks need no GPU. Every stored sample is already
judged, so reading results, re-analysing them and redrawing charts are CPU work. A GPU is needed
only to sample a model (or fine-tune it, for EDL) on a suite again.

The panel file [`panel.yaml`](panel.yaml) lists every job with a tier. GPU minutes are the
largest measured per model on one RTX 3090 (`out/store/ledger.jsonl`, 2026-09-30 to 10-01) for
the base / rl / armC models (small absgpt transformers). Other models can be much slower or faster.

| Tier | Job | GPU-min per model | Benchmark it feeds |
| --- | --- | --- | --- |
| cpu | `summary_page`, `detail_pages` (`ndbench dash`) | 0 | every chart, from stored results |
| cpu | `probe_position_baselines` | 0 | probes (position-only baselines) |
| cpu | suite builds (`ndbench build`), analysis views in Python | 0 | all |
| cheap | `prefix_<m>` | 0.9 | prefix completion |
| cheap | `parts_<m>`, `glue_parts_renamed_<m>`, `glue_<m>` | 1.4 + 1.1 + 2.1 | glue |
| cheap | `padding_<m>`, `gen_ctrl_<m>` | 2.4 + 1.3 | free lines (padding) |
| cheap, pair | `interp_vectors`, `interp_probes`, `interp_likelihood_crux_control` | 0.6, 2.1, 0.3 | steering, probes, likelihood control |
| moderate | `yue_textbook_dev_<m>` (k to 4096), `yue_textbook_law_<m>` (k to 1024) | 5.4, 6.2 | Yue pass@k |
| moderate | `families_<m>` | 11.8 | restricted families |
| moderate | `laws_<m>` | 11.4 | law x substitution size |
| moderate, pair | `interp_interpolate` | 14.8 | weight interpolation |
| expensive | `skills_<m>` | about 40, not measured | skills (base/rl results are legacy imports) |
| expensive | `edl_<m>` (6 learning rates) | 43-49 | EDL |
| expensive, pair | `interp_steer_sweep` | 26.3 | steering |
| expensive, pair | `interp_likelihood` | about 50 | likelihood |

The whole cheap tier costs about 14 GPU-min per model, and the cpu tier costs nothing. You can run
either one alone; no cheap job needs a moderate or expensive one first. Within the cheap tier, glue
needs `parts_<m>` and `glue_parts_renamed_<m>`, and padding's natural reference curve needs
`gen_ctrl_<m>`. The panel lists these in order.

"pair" jobs measure a (base, rl) pair, and their configs name base and rl. For another pair, copy
the config and edit it.

## Running a tier

From the worktree root, always dry-run first:

```bash
NDBENCH_ALLOWED_GPUS=4 code/.venv/bin/python \
    code/experiments/current/benchmarking_20260930/run_schedule.py \
    --schedule code/ndbench/panel.yaml --model <label> --max-tier cheap --budget-h 0.5 --dry-run
```

- `--max-tier {cpu,cheap,moderate,expensive}` selects every job at or below that tier. `--jobs`
  narrows the selection further (ids or job names, e.g. `--jobs prefix_<label>`).
- `--model` fills `{model}`. `--include-pair` adds the interp pair jobs.
- `--budget-h` caps the GPU hours this invocation may spend. Without it, the cap is `cap_h` on
  the whole ledger (6.9 of 9.5 h already spent on 2026-10-01).
- `cpu` jobs run with no visible GPU and no GPU lease. GPU jobs run one at a time on GPU 4 under
  `runner.gpu.lease`, which writes the ledger.
- `--scale` multiplies every estimate. Use it after timing one cheap job on a new kind of model.
  A job is stopped at 3x its estimate; a stopped sampling run resumes from its stored rows.

Before any GPU tier, ask whoever owns the GPU budget; in this workspace that is Leon. The cpu tier
needs no approval. On serrano, GPUs 0-3 belong to another user: only GPU 4 may be used
(`runbook.md`).

## Pitfalls

- gen_ctrl and dev L8-11 were the RL hill-climb's selection set, so rl's numbers there are
  optimistic. Use the sealed set for clean comparisons.
- Theorems seen in training: the laws view excludes them by default. The families view reports
  the horizon with and without them (`n_star_unseen`; the summary uses unseen only, since most
  of the smallest family sizes are training theorems). The skills view does not exclude them.
- EDL run ids carry a generation (`edl`, `edl2`, `edl3`, ...), and the view keeps the newest per
  (model, n, m). Re-running a model that already has EDL rows needs a new `--run-id`.
- `dash --summary` currently draws only base, rl and armC. A new model's results land in the store
  and in the analysis views, but its charts need a summary-page change.
- A new adapter (a model that is not an absgpt whole-proof model) has its own throughput. Time one
  cheap job before trusting any estimate above.
