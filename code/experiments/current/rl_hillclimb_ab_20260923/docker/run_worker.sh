#!/usr/bin/env bash
# Start one autonomous Claude Code worker inside an arm's container, in a detached tmux session.
# Usage: run_worker.sh <arm> <worker> <container-gpu-index> <brief path inside container> [model]
#   e.g. run_worker.sh gear w1 0 /work/w1/BRIEF.md opus
set -euo pipefail
ARM=$1; W=$2; GPU=$3; BRIEF=$4; MODEL=${5:-opus}
NAME=ab-$ARM
docker exec "$NAME" mkdir -p "/work/$W"
docker exec "$NAME" tmux new-session -d -s "$W" -c "/work/$W" \
  "export CUDA_VISIBLE_DEVICES=$GPU AB_ARM=$ARM AB_WORKER=$W; \
   claude -p \"\$(cat $BRIEF)\" --model $MODEL --dangerously-skip-permissions --verbose --output-format stream-json \
     > /work/$W/transcript.jsonl 2> /work/$W/stderr.log; echo EXIT \$? >> /work/$W/stderr.log"
echo "started $ARM/$W on container GPU $GPU with model $MODEL"
