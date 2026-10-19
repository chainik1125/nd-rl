#!/usr/bin/env bash
# Continue a worker's previous conversation (same cwd, same home) with a new message.
# Usage: resume_worker.sh <arm> <worker> <container-gpu-index> "<message>" [model]
set -euo pipefail
ARM=$1; W=$2; GPU=$3; MSG=$4; MODEL=${5:-opus}
NAME=ab-$ARM
docker exec "$NAME" bash -c "cd /work/$W && [ -f stderr.log ] && mv stderr.log stderr.prev.\$(date +%s).log || true"
docker exec "$NAME" tmux kill-session -t "$W" 2>/dev/null || true
docker exec "$NAME" tmux new-session -d -s "$W" -c "/work/$W" \
  "export CUDA_VISIBLE_DEVICES=$GPU AB_ARM=$ARM AB_WORKER=$W; \
   claude -p --continue \"$MSG\" --model $MODEL --dangerously-skip-permissions --verbose --output-format stream-json \
     >> /work/$W/transcript.jsonl 2> /work/$W/stderr.log; echo EXIT \$? >> /work/$W/stderr.log"
echo "resumed $ARM/$W on container GPU $GPU"
