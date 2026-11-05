#!/usr/bin/env bash
# poll.sh [SECS] [ALIASES...]: print new event lines from base_saturation pods every SECS (120),
# and UNREACHABLE after 3 failed polls in a row. One line per event (for a Monitor). Read only.
iv=${1:-120}
shift || true
hs=("$@")
[ ${#hs[@]} -eq 0 ] && hs=(rp-sat-a rp-sat-b025 rp-sat-b4 rp-sat-c rp-sat-d1)
S=/scratch/cowifornia/nd-rl/base_saturation/poll
mkdir -p "$S"
declare -A fails
while true; do
  for h in "${hs[@]}"; do
    n=$(cat "$S/$h.n" 2>/dev/null || echo 0)
    if out=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "tail -n +$((n + 1)) /workspace/sat/events.log 2>/dev/null"); then
      fails[$h]=0
      if [ -n "$out" ]; then
        printf '%s\n' "$out"
        echo $(( n + $(printf '%s\n' "$out" | wc -l) )) > "$S/$h.n"
      fi
    else
      fails[$h]=$(( ${fails[$h]:-0} + 1 ))
      [ "${fails[$h]}" -eq 3 ] && echo "$(date -u +%FT%TZ) $h UNREACHABLE (3 polls failed)"
    fi
  done
  sleep "$iv"
done
