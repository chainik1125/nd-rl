#!/usr/bin/env bash
# g0_pod.sh: guard-off reruns of the six author-tag p10 lines on rented Runpod pods (2026-11-18), run ON SERRANO.
# Never creates/stops/deletes a pod, never reads an API key, never copies a sealed set.
#   g0_pod.sh prep   <alias> <line>             own code copy (/workspace/g0exp) + the line's base (/workspace/atg0/bases),
#                                               md5-checked; md5-checks the RL kit (kd/, kg/) already on the pod
#   g0_pod.sh launch <alias> <line> <base_none> detached driver: tag_line (ND_GUARD=off, 3.3 h timeout), then cross reads
#   g0_pod.sh fetch  <alias> <line>             pod atg0/runs/<line> (no alarm/rolling .pt), logs, cross reads -> serrano, md5
set -euo pipefail
cmd=$1; h=$2; line=$3
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "bad alias"; exit 2; }
[[ $line =~ ^t[AB]_p10_s([0-9])$ ]] || { echo "bad line"; exit 2; }
seed=${BASH_REMATCH[1]}
G0=$(cd "$(dirname "$0")" && pwd)
EXP=/scratch/cowifornia/nd-rl/worktrees/author-tags/code/experiments/current
AT=/scratch/cowifornia/nd-rl/author_tags
REF=/scratch/cowifornia/nd-rl/two_authors/pod_stage/rp-ta5.manifest.txt.lab.md5
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
SEALED='textbook72|tb72|test_13_20|test_21_30|transfer_track832'
utc() { date -u +%FT%TZ; }

case $cmd in
  prep)
    ST=$(mktemp -d /tmp/g0stage.XXXX)
    mkdir -p "$ST"/g0exp/{author_tags_20261110,knockout_danbase_20261104} "$ST/atg0/bases"
    for f in habits tag_core tag_danread tag_grpo tag_launch tag_line tag_measure tag_pretrain tagcorpus; do
      cp -p "$EXP/author_tags_20261110/$f.py" "$ST/g0exp/author_tags_20261110/"; done
    for f in common gpu merge danread; do cp -p "$EXP/knockout_danbase_20261104/$f.py" "$ST/g0exp/knockout_danbase_20261104/"; done
    cp -p "$G0/g0_driver.sh" "$ST/g0exp/"
    # pod copy only: LEAN_GATE_WORKERS and the reads' DANREAD_MEM_FRAC (a memory cap, no effect on samples) settable per pod
    python3 - "$ST/g0exp/author_tags_20261110/tag_line.py" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'LEAN_GATE_WORKERS="8",'; b = '{"DANREAD_MEM_FRAC": "0.3"}'
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, 'LEAN_GATE_WORKERS=os.environ.get("G0_LEAN_WORKERS", "8"),')
s = s.replace(b, '{"DANREAD_MEM_FRAC": os.environ.get("G0_READ_FRAC", "0.3")}')
open(p, "w").write(s)
PYEOF
    S5=/scratch/cowifornia/nd-rl/two_authors/pod_stage/rp-ta5
    (cd "$S5" && md5sum -c --quiet <(grep -E "  (kd/dan_code_13e41bbd/data/coe/|kg/dan_axis)" "$REF")) || { echo "rp-ta5 stage changed"; exit 1; }
    (cd "$S5" && "${RSYNC[@]}" -R kd/dan_code_13e41bbd/data/coe/sigma_heldout_strict.txt kd/dan_code_13e41bbd/data/coe/transfer_track832.jsonl kg/dan_axis.json "$h:/workspace/")
    for x in .pt .tag.json .pt.args.json; do cp -p "$AT/bases/stage1_tagAB_p10_s$seed$x" "$ST/atg0/bases/"; done
    (cd "$ST" && find . -type f | sed 's|^\./||' | sort) > "$ST.man"
    if grep -E "$SEALED" "$ST.man"; then echo "SEALED NAME: nothing copied"; exit 1; fi
    "${RSYNC[@]}" "$ST/" "$h:/workspace/"
    (cd "$ST" && xargs -d '\n' md5sum < "$ST.man") > "$ST.lab"
    "${SSH[@]}" "$h" 'cd /workspace && xargs -d "\n" md5sum' < "$ST.man" > "$ST.pod"
    cmp -s "$ST.lab" "$ST.pod" || { diff "$ST.lab" "$ST.pod"; echo "MD5 MISMATCH (code/base)"; exit 1; }
    echo "$h: code+base md5 ok ($(wc -l < "$ST.man") files)"
    grep -E "  (kd|kg)/" "$REF" > "$ST.kit"
    "${SSH[@]}" "$h" 'cd /workspace && md5sum -c --quiet' < "$ST.kit" && echo "$h: RL kit md5 ok ($(wc -l < "$ST.kit") files)" \
      || { echo "$h: RL KIT MISSING/DIFFERENT"; exit 1; }
    "${SSH[@]}" "$h" 'wc -l < /workspace/kd/dan_code_13e41bbd/data/coe/transfer_track832.jsonl'
    rm -rf "$ST" "$ST".* ;;
  launch)
    bn=${4:?base_none 0|1}
    lw=${5:-8}; rf=${6:-0.3}; tend=${7:-}; gseg=${8:-1}; gres=${9:-}
    "${SSH[@]}" "$h" "mkdir -p /workspace/atg0/logs; cd /workspace/g0exp; G0_LEAN_WORKERS=$lw G0_READ_FRAC=$rf G0_T_END=$tend G0_SEG=$gseg G0_RESUME=$gres setsid nohup bash /workspace/g0exp/g0_driver.sh $line $seed $bn >> /workspace/atg0/logs/$line.driver.log 2>&1 < /dev/null & sleep 1; echo launched $(utc)" ;;
  fetch)
    D=$AT/runs_g0/$line; R=$AT/reads/cross_g0_p10_s$seed
    mkdir -p "$D/logs" "$R"
    "${SSH[@]}" "$h" "cd /workspace/atg0/runs/$line && find . -type f ! -name '*_alarm.pt' ! -name '*_rolling.pt' | sed 's|^\./||' | sort" > "$D.fetch.list"
    "${RSYNC[@]}" --files-from="$D.fetch.list" "$h:/workspace/atg0/runs/$line/" "$D/"
    "${SSH[@]}" "$h" "cd /workspace/atg0/runs/$line && xargs -d '\n' md5sum" < "$D.fetch.list" > "$D.fetch.pod.md5"
    (cd "$D" && xargs -d '\n' md5sum < "$D.fetch.list") > "$D.fetch.lab.md5"
    "${RSYNC[@]}" "$h:/workspace/atg0/logs/$line*" "$D/logs/"
    "${SSH[@]}" "$h" "test -d /workspace/atg0/reads/cross_g0_p10_s$seed" && \
      "${RSYNC[@]}" "$h:/workspace/atg0/reads/cross_g0_p10_s$seed/" "$R/"
    if cmp -s "$D.fetch.pod.md5" "$D.fetch.lab.md5"; then echo "$line: md5 ok, $(wc -l < "$D.fetch.list") files ($(utc))"
    else diff "$D.fetch.pod.md5" "$D.fetch.lab.md5" | head; echo "$line: MD5 DIFF (files still being written?)"; exit 1; fi ;;
  peek)  # interim copy without checkpoints (the final fetch replaces it)
    D=$AT/runs_g0/$line; mkdir -p "$D/logs"
    "${RSYNC[@]}" --exclude '*.pt' "$h:/workspace/atg0/runs/$line/" "$D/"
    "${RSYNC[@]}" "$h:/workspace/atg0/logs/$line*" "$D/logs/"
    R=$AT/reads/cross_g0_p10_s$seed; mkdir -p "$R"
    "${SSH[@]}" "$h" "test -d /workspace/atg0/reads/cross_g0_p10_s$seed" && "${RSYNC[@]}" "$h:/workspace/atg0/reads/cross_g0_p10_s$seed/" "$R/"
    echo "$line: peeked $(utc)" ;;
  *) echo "unknown $cmd"; exit 2 ;;
esac
