#!/usr/bin/env bash
# Pull this run's artifacts and checkpoints from every state-env pod into the run worktree.
# Usage: bash pod/se/pullall.sh [pod ...]
set -e
cd /home/dan/work/state-env
for P in "${@:-se-1 se-2 se-3 se-4}"; do
  [ -f ~/.config/nd-rl/pods/$P ] || { echo "skip $P (no pod)"; continue; }
  echo "=== $P"
  ./pod/se/pull.sh $P artifacts/se/ || echo "pull artifacts failed $P"
  ./pod/se/pull.sh $P ckpts/se/ || echo "pull ckpts failed $P"
done
du -sh artifacts/se ckpts/se 2>/dev/null
