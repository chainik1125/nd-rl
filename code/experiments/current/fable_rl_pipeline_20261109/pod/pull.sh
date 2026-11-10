#!/usr/bin/env bash
# pull.sh <alias> [all]: copy logs, round records and checkpoints from a pod's /workspace/runs to
# /scratch/cowifornia/nd-rl/fable_rl/runs/<alias>/ on serrano. One way, no --delete. Without
# "all" the large per-round state (state.pkl, opt.pt, x cache) is skipped.
set -euo pipefail
h=${1:?usage: pull.sh <alias> [all]}
F=/scratch/cowifornia/nd-rl/fable_rl
mkdir -p $F/runs/$h
# cur.pt is the workers' copy of the policy, rewritten in place every round: reading it while it
# is written fails on the pods' network volume (errno 61) and nothing needs it
EX=(--exclude state.pkl --exclude state.pkl.tmp --exclude opt.pt --exclude cur.pt --exclude 'smoke*')
[[ ${2:-} == all ]] && EX=(--exclude cur.pt --exclude 'smoke*')
timeout 1200 rsync -rlpt "${EX[@]}" -e "ssh -F $F/ssh_config" "$h:/workspace/runs/" "$F/runs/$h/"
echo "PULL_OK $h $(date -u +%FT%TZ)"
