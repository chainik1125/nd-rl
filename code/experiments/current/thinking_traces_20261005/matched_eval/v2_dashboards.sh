#!/usr/bin/env bash
# Render the RL monitoring dashboard for every GRPO recipe-v2 line (stages L2 then L3, end to end).
#   bash code/experiments/current/thinking_traces_20261005/matched_eval/v2_dashboards.sh [OUTDIR]
# rl_dashboard.py lives on branch leon/grpo-v2; it is run read-only from that worktree (no edits, no
# bytecode written), and only needs numpy and the figure style. EI lines have no GRPO metrics and
# are skipped; a line whose L3 has not started is drawn with L2 only.
set -u
R=/scratch/cowifornia/nd-rl/think/runs/grpo_v2
WT=/scratch/cowifornia/nd-rl/think/nd-rl-v2
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
OUT=${1:-/scratch/cowifornia/nd-rl/think/nd-rl/experiment-summaries/2026-10-05-thinking-traces/figures/rl_dash}
cd "$WT" || exit 1
for l2 in "$R"/grpo_*_s[0-9]_L2; do
  [ -f "$l2/metrics.jsonl" ] || continue
  line=$(basename "$l2" _L2); l3=$R/${line}_L3
  runs=$l2; [ -f "$l3/metrics.jsonl" ] && runs=$l2,$l3
  ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice "$PY" -m \
    code.experiments.current.thinking_traces_20261005.matched_eval.rl_dashboard \
    --runs "$runs" --label "$line" --out "$OUT"
done
