#!/usr/bin/env bash
# g0_stopper.sh <line> <epoch>: ON THE POD. Replaces a running driver's 3.3 h STOP timer (coordinator, 2026-11-18: the
# stop rule decides, hard end 7:50 pm PT): at <epoch>, STOP in the line's newest segment (tag_line ends GRPO cleanly).
line=$1; t=$2
sleep $(( t - $(date +%s) ))
d=$(ls -d /workspace/atg0/runs/$line/seg[0-9]* | sort -V | tail -1)
[[ -e /workspace/atg0/DONE_$line ]] && { echo "$(date -u +%FT%TZ) $line already done"; exit 0; }
touch "$d/STOP"; echo "$(date -u +%FT%TZ) STOP written in $d"
