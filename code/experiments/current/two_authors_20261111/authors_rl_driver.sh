#!/usr/bin/env bash
# authors_rl_driver.sh: one RL line on one pod, run ON SERRANO by the orchestrator: [setup,] copy, check-env, start.
# It never creates or deletes a pod and does not wait for the line (use `authors_pod.sh check` / `fetch`).
#   [ND_GUARD=off] authors_rl_driver.sh <alias> <line> [--after <pid>]
#     --after <pid>   a setup is already running on serrano as <pid>: wait for it instead of running setup
#     ND_GUARD=off    the line runs without rollback or lr cut (tag_line.grpo_argv)
set -uo pipefail
h=${1:?alias}; line=${2:?line}
W=$(cd "$(dirname "$0")" && pwd); P="$W/authors_pod.sh"
say() { echo "[$(date -u +%FT%TZ)] $h: $*"; }
die() { say "STOP: $*"; exit 1; }
if [[ ${3:-} == --after ]]; then
  while kill -0 "$4" 2>/dev/null; do sleep 5; done
  say "setup process $4 ended"
else
  bash "$P" setup "$h" || die setup
fi
bash "$P" copy "$h" rl "$line" || die copy
bash "$P" check-env "$h" || die check-env
bash "$P" start "$h" rl --line "$line" || die start
say "RL LINE STARTED: $line (ND_GUARD=${ND_GUARD:-on})"
