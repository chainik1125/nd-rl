#!/usr/bin/env bash
# g0_move.sh <from> <to> <line> <lean_workers> <read_frac>: ON SERRANO. Move a running guard-off line to another pod
# by tag_line's documented continuation (seg k+1 --resume <its newest rolling checkpoint>):
#   1. on <from>: end the line's driver and stopper (by PID; tag_line keeps running), STOP_AT_CKPT in its segment:
#      tag_line ends GRPO after the next rolling checkpoint (end.json how=moved)
#   2. copy AT/runs/<line> (no alarm checkpoints) and its logs <from> -> serrano -> <to>, md5-checked; old folder on
#      <from> moved aside to /workspace/atg0/moved_<line>
#   3. on <to>: g0_driver2.sh with G0_SEG=k+1 G0_RESUME=<rolling> and the 7:50 pm PT stop
set -euo pipefail
from=$1; to=$2; line=$3; lw=${4:-8}; rf=${5:-0.3}
[[ $line =~ ^t[AB]_p10_s([0-9])$ ]] || { echo "bad line"; exit 2; }
seed=${BASH_REMATCH[1]}
bn=0; [[ $line == tA_* ]] && bn=1
G0=$(cd "$(dirname "$0")" && pwd)
AT=/scratch/cowifornia/nd-rl/author_tags
TEND=1795060200  # 2026-11-19T03:50:00Z = 7:50 pm PT
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
utc() { date -u +%FT%TZ; }
echo "$(utc) move $line $from -> $to"
# stage the target first (idempotent): RL kit from the md5-checked rp-ta5 stage, then code + base (g0_pod.sh prep)
"${SSH[@]}" "$to" 'grep -q "^setup done" /workspace/setup.log && test -x /workspace/venv/bin/python' || { echo "$to: setup not done"; exit 1; }
KS=/scratch/cowifornia/nd-rl/two_authors/pod_stage/rp-ta5; KREF=$KS.manifest.txt.lab.md5; KT=$(mktemp -d)
grep -E "  (kd|kg)/" "$KREF" > "$KT/kit.md5"; cut -c35- "$KT/kit.md5" > "$KT/kit.list"
(cd "$KS" && md5sum -c --quiet "$KT/kit.md5")
"${RSYNC[@]}" --files-from="$KT/kit.list" "$KS/" "$to:/workspace/"
"${SSH[@]}" "$to" 'cd /workspace && md5sum -c --quiet' < "$KT/kit.md5" && echo "$to: RL kit md5 ok"
bash "$G0/g0_pod.sh" prep "$to" "$line"
seg=$("${SSH[@]}" "$from" "ls -d /workspace/atg0/runs/$line/seg[0-9]* | sort -V | tail -1")
k=${seg##*seg}
"${SSH[@]}" "$from" "
drv=\$(ps -eo pid,ppid,args | awk '\$2==1 && /g0_driver.*\.sh $line/ {print \$1}')
sub=\$(ps -eo pid,ppid,args | awk -v d=\"\$drv\" 'd!=\"\" && \$2==d && /g0_driver/ {print \$1}')
stp=\$(ps -eo pid,args | awk '/g0_stopper.sh $line/ && !/awk/ {print \$1}')
stps=\$(for p in \$stp \$sub; do ps -eo pid,ppid,args | awk -v p=\$p '\$2==p && /sleep/ {print \$1}'; done)
echo \"driver \$drv subshell \$sub stopper \$stp sleeps \$stps\"
for p in \$drv \$sub \$stp \$stps; do kill \$p; done
touch $seg/STOP_AT_CKPT; echo STOP_AT_CKPT written in $seg"
echo "$(utc) waiting for seg$k to end at its next rolling checkpoint"
for _ in $(seq 120); do
  how=$("${SSH[@]}" "$from" "grep -o '\"how\": \"[a-z]*\"' $seg/end.json 2>/dev/null; true")
  [[ -n $how ]] && break; sleep 15
done
echo "$(utc) seg$k end: $how"
[[ $how == *moved* ]] || { echo "seg$k did not end as moved: stop here (nothing copied)"; exit 1; }
"${SSH[@]}" "$from" "ps -eo pid,args | grep -E 'tag_line.py --line $line|tag_grpo.py.*$line|tag_danread.*$line' | grep -v grep; true"
roll=$("${SSH[@]}" "$from" "ls $seg/ckpts/${line}_it*_rolling.pt" | sed -E 's/.*_it([0-9]+)_rolling\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
echo "resume from $roll"
L=$AT/runs_g0/_move_$line; rm -rf "$L"; mkdir -p "$L/runs/$line" "$L/logs"
"${RSYNC[@]}" --exclude '*_alarm.pt' "$from:/workspace/atg0/runs/$line/" "$L/runs/$line/"
"${RSYNC[@]}" "$from:/workspace/atg0/logs/$line*" "$L/logs/"
(cd "$L" && find . -type f | sed 's|^\./||' | sort) > "$L.man"
"${SSH[@]}" "$from" "cd /workspace/atg0 && xargs -d '\n' md5sum" < <(grep '^runs/' "$L.man") > "$L.from.md5"
(cd "$L" && grep '^runs/' "$L.man" | xargs -d '\n' md5sum) > "$L.lab.md5"
cmp -s "$L.from.md5" "$L.lab.md5" || { echo "MD5 MISMATCH from $from"; exit 1; }
"${SSH[@]}" "$to" "test -e /workspace/atg0/runs/$line && echo EXISTS; true" | grep -q EXISTS && { echo "$to already has runs/$line"; exit 1; }
"${RSYNC[@]}" "$L/" "$to:/workspace/atg0/"
"${SSH[@]}" "$to" "cd /workspace/atg0 && xargs -d '\n' md5sum" < <(grep '^runs/' "$L.man") > "$L.to.md5"
cmp -s "$L.lab.md5" "$L.to.md5" || { echo "MD5 MISMATCH to $to"; exit 1; }
"${RSYNC[@]}" "$G0/g0_driver.sh" "$to:/workspace/g0exp/g0_driver2.sh"
"${SSH[@]}" "$to" "test -s /workspace/atg0/bases/stage1_tagAB_p10_s$seed.pt && md5sum /workspace/atg0/bases/stage1_tagAB_p10_s$seed.pt"
md5sum "$AT/bases/stage1_tagAB_p10_s$seed.pt"
"${SSH[@]}" "$from" "mv /workspace/atg0/runs/$line /workspace/atg0/moved_$line"
n=$((k + 1))
"${SSH[@]}" "$to" "cd /workspace/g0exp; G0_LEAN_WORKERS=$lw G0_READ_FRAC=$rf G0_T_END=$TEND G0_SEG=$n G0_RESUME=$roll setsid nohup bash /workspace/g0exp/g0_driver2.sh $line $seed $bn >> /workspace/atg0/logs/$line.driver.log 2>&1 < /dev/null & sleep 2; tail -3 /workspace/atg0/logs/$line.driver.log"
echo "$(utc) $line resumed on $to as seg$n from $roll"
