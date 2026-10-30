#!/usr/bin/env bash
# hirl_sync.sh (2026-10-30, agent:claude for Leon): on serrano, pull the finished think_hirl reads
# from ynez into serrano's runs/eval/v2 under their names, never over an existing dir.
#   bash hirl_sync.sh [INTERVAL_S]     loop (default 900 s); one pass with INTERVAL_S = 0
# A read is finished when its ynez dir eval/final/NAME has rows_plain_B64.jsonl and either
# chunks.json (a merged read: hirl_queue.py writes it to a temporary dir and renames it) or a done
# state file queue/state/NAME.json (a single read). It is copied to runs/eval/v2/.NAME.partial and
# renamed to NAME with mv -n (no clobber); a NAME already there is left alone and logged once.
set -u
V2=/scratch/cowifornia/nd-rl/think/runs/eval/v2
H=/scratch/cowifornia/nd-rl/think_hirl
E="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes"
IV=${1:-900}
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
SEEN=" "
once() {
  local names n
  names=$($E ynez "cd $H/eval/final 2> /dev/null || exit 0
    for d in *; do case \$d in '*'|*.tmp|*.crashed_*) continue ;; esac
      [ -f \$d/rows_plain_B64.jsonl ] || continue
      if [ -f \$d/chunks.json ] || grep -q '\"status\": \"done\"' $H/queue/state/\$d.json 2> /dev/null; then echo \$d; fi
    done") || { say "ssh ynez failed: next pass"; return; }
  for n in $names; do
    [[ $n =~ ^[A-Za-z0-9_.]+$ ]] || { say "skipping odd name '$n'"; continue; }
    if [ -e "$V2/$n" ]; then
      case $SEEN in *" $n "*) ;; *) say "$n: already in $V2 (left alone)"; SEEN="$SEEN$n " ;; esac
      continue
    fi
    rm -rf "$V2/.$n.partial"  # only ever our own temporary copy
    if rsync -a -e "$E" "ynez:$H/eval/final/$n/" "$V2/.$n.partial/"; then
      mv -n -T "$V2/.$n.partial" "$V2/$n"
      if [ -e "$V2/.$n.partial" ]; then say "$n: appeared in $V2 meanwhile: copy left at .$n.partial"
      else say "$n: copied to $V2/$n"; SEEN="$SEEN$n "; fi
    else say "$n: rsync failed: next pass"; fi
  done
}
say "start (interval $IV s)"
while true; do
  once
  [ "$IV" = 0 ] && break
  sleep "$IV"
done
