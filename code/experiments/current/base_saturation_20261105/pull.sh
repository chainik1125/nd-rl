#!/usr/bin/env bash
# pull.sh <alias> [--verify]: bring a base_saturation pod's outputs back to serrano
# (/scratch/cowifornia/nd-rl/base_saturation/pods/<alias>/: out/, logs/, status/, events.log).
# Pulling only (pod -> serrano), no --delete. With --verify: md5 of every file under out/ on both
# sides (files of a step still running are skipped), so the pod can be torn down.
set -euo pipefail
h=${1:?usage: pull.sh <alias> [--verify]}
[[ $h =~ ^rp-sat-[a-z0-9-]+$ ]] || { echo "alias must look like rp-sat-<name>"; exit 2; }
D=/scratch/cowifornia/nd-rl/base_saturation/pods/$h
mkdir -p "$D"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
timeout 3600 rsync -rlpt -e "ssh -o BatchMode=yes" "$h:/workspace/sat/out" "$h:/workspace/sat/logs" \
  "$h:/workspace/sat/status" "$h:/workspace/sat/events.log" "$D/"
echo "$h: pulled to $D ($(du -sh "$D" | cut -f1)) $(date -u +%FT%TZ)"
if [[ ${2:-} == --verify ]]; then
  if "${SSH[@]}" "$h" 'test -f /workspace/sat/status/current'; then
    echo "a step is still running on $h: verify after it ends"; exit 1
  fi
  "${SSH[@]}" "$h" 'cd /workspace/sat && find out -type f | sort | xargs -d "\n" md5sum' > "$D/md5.pod"
  (cd "$D" && cut -c35- "$D/md5.pod" | xargs -d '\n' md5sum) > "$D/md5.lab"
  if cmp -s "$D/md5.pod" "$D/md5.lab"; then
    echo "VERIFIED: all $(wc -l < "$D/md5.pod") output files of $h equal on serrano"
  else
    diff "$D/md5.pod" "$D/md5.lab" | head; echo "MD5 MISMATCH: pull again"; exit 1
  fi
fi
