#!/usr/bin/env bash
# fetch.sh <alias> <test>: pod /workspace/at/stability/<test>/ -> serrano /scratch/cowifornia/nd-rl/author_tags/stability/<test>/
# (metrics, logs, cmd/args, canary base, exit status, the Lean gate log; no checkpoints, no lean_dump), md5 of every
# fetched file on both sides. Run ON SERRANO. Fetching while the run is live gives md5 differences for files still
# being written; fetch again after it ends.
set -euo pipefail
h=$1 test=$2
DST=/scratch/cowifornia/nd-rl/author_tags/stability/$test
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=15)
mkdir -p "$DST"
"${SSH[@]}" "$h" "cd /workspace/at/stability/$test && find . -type f ! -name '*.pt' ! -name 'lean_dump.jsonl' | sed 's|^\./||' | sort" > "$DST/.fetch.list"
[[ -s $DST/.fetch.list ]] || { echo "nothing to fetch for $test on $h"; exit 1; }
rsync -t -e "ssh -o BatchMode=yes -o ConnectTimeout=15" --files-from="$DST/.fetch.list" "$h:/workspace/at/stability/$test/" "$DST/"
ssh -o BatchMode=yes "$h" "cd /workspace/at/stability/$test && xargs -d '\n' md5sum" < "$DST/.fetch.list" > "$DST/.fetch.pod.md5"
(cd "$DST" && xargs -d '\n' md5sum < .fetch.list) > "$DST/.fetch.lab.md5"
if cmp -s "$DST/.fetch.pod.md5" "$DST/.fetch.lab.md5"; then
  echo "$test: md5 all $(wc -l < "$DST/.fetch.list") files equal on $h and serrano ($(date -u +%FT%TZ))"
else
  diff "$DST/.fetch.pod.md5" "$DST/.fetch.lab.md5" | grep '^[<>]' | head
  echo "$test: MD5 DIFFERENCES (a live run's files); fetch again after it ends"; exit 1
fi
