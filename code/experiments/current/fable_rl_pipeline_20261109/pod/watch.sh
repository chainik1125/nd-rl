#!/usr/bin/env bash
# watch.sh: supervisor for the RL runs listed in $F/watch.tsv (alias<TAB>name<TAB>timeout s<TAB>args).
# Every 5 minutes, per run: if its process is gone and the log has no RL_DONE, restart it with
# --resume (at most 6 restarts per run). Every 20 minutes: pull logs and checkpoints to serrano.
# Events go to $F/logs/watch.log. Bounded by the caller's timeout.
set -uo pipefail
F=/scratch/cowifornia/nd-rl/fable_rl
HERE=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -F $F/ssh_config)
log() { echo "$(date -u +%FT%TZ) $*" >> $F/logs/watch.log; }
declare -A restarts
i=0
while true; do
  while IFS=$'\t' read -r h name to args; do
    [[ -z ${h:-} || $h == \#* ]] && continue
    st=$("${SSH[@]}" -n "$h" "if pgrep -f '[f]r\.rl.*--out /workspace/runs/$name( |\$)' >/dev/null; then echo RUNNING; elif grep -q RL_DONE /workspace/runs/$name.log 2>/dev/null; then echo DONE; else echo DEAD; fi" 2>/dev/null || echo UNREACHABLE)
    if [[ $st == DEAD ]]; then
      n=${restarts[$name]:-0}
      if (( n < 6 )); then
        restarts[$name]=$((n + 1))
        log "RESTART $h $name (restart $((n + 1))): $("${SSH[@]}" -n "$h" "tail -c 300 /workspace/runs/$name.log | tr '\n' ' '" 2>/dev/null)"
        "${SSH[@]}" -n "$h" "cp /workspace/runs/$name.log /workspace/runs/$name.log.$n 2>/dev/null"
        bash "$HERE/run.sh" "$h" "$name" "$to" $args --resume >> $F/logs/watch.log 2>&1
      else
        log "GIVING UP on $h $name after 6 restarts"
      fi
    elif [[ $st == UNREACHABLE ]]; then
      log "UNREACHABLE $h ($name)"
    fi
  done < $F/watch.tsv
  if (( i % 4 == 0 )); then
    for h in $(cut -f1 $F/watch.tsv | grep -v '^#' | sort -u); do bash "$HERE/pull.sh" "$h" >> $F/logs/watch.log 2>&1; done
  fi
  i=$((i + 1))
  sleep 300
done
