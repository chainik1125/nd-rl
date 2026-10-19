#!/usr/bin/env bash
# Usage: stop_worker.sh <arm> <worker>
set -euo pipefail
docker exec "ab-$1" tmux kill-session -t "$2" 2>/dev/null && echo "stopped $1/$2" || echo "no session $1/$2"
