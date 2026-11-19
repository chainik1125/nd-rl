#!/usr/bin/env bash
# g0_collect.sh: ON SERRANO. Interim copy (no checkpoints) of all six lines + their cross-read folders, then the summary.
G0=$(cd "$(dirname "$0")" && pwd); NN=${1:-120}
for p in "rp-stab4 tA_p10_s0" "rp-stab6 tB_p10_s0" "rp-stab5 tA_p10_s1" "rp-tag1 tB_p10_s1" "rp-stab7 tA_p10_s2" "rp-stab2 tB_p10_s2"; do
  set -- $p; timeout 300 bash "$G0/g0_pod.sh" peek "$1" "$2" &
done; wait
/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python "$G0/g0_summary.py" --N "$NN" --json /scratch/cowifornia/nd-rl/author_tags/runs_g0/summary_interim.json
