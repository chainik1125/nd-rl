#!/usr/bin/env bash
# g0_resume_tag1.sh [alias=rp-tag1]: ON SERRANO. Resume tB_p10_s1 (seg1 ended `moved` at it 25 on rp-stab1, fetched to
# AT/runs_g0/tB_p10_s1) on a fresh, set-up pod as seg 2 from tB_p10_s1_it25_rolling.pt (tag_line's documented resume).
#   1. RL kit (the 100 kd/ kg/ files of the md5-checked rp-ta5 stage: GRPO code subset, pools, dan_axis, Dan's modules,
#      sigma_heldout_strict and the 106-row transfer file), md5-checked on the pod
#   2. g0_pod.sh prep (code copy, seed-1 base, kit check)
#   3. the run folder and logs, md5-checked; dry run of seg 2 (must refuse nothing); launch with the 7:50 pm PT stop
set -euo pipefail
h=${1:-rp-tag1}; line=tB_p10_s1
G0=$(cd "$(dirname "$0")" && pwd)
AT=/scratch/cowifornia/nd-rl/author_tags
S=/scratch/cowifornia/nd-rl/two_authors/pod_stage/rp-ta5
REF=/scratch/cowifornia/nd-rl/two_authors/pod_stage/rp-ta5.manifest.txt.lab.md5
D=$AT/runs_g0/$line
ROLL=/workspace/atg0/runs/$line/seg1/ckpts/${line}_it25_rolling.pt
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
RS=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
T=$(mktemp -d)
"${SSH[@]}" "$h" 'grep -q "^setup done" /workspace/setup.log && test -x /workspace/venv/bin/python && test -L /scratch/cowifornia/nd-rl/knockout_danbase' \
  || { echo "$h: setup not done"; exit 1; }
grep -E "  (kd|kg)/" "$REF" > "$T/kit.md5"
cut -c35- "$T/kit.md5" > "$T/kit.list"
grep -iE "textbook72|tb72|test_13_20|test_21_30" "$T/kit.list" && { echo "sealed name"; exit 1; }
(cd "$S" && md5sum -c --quiet "$T/kit.md5")
"${RS[@]}" --files-from="$T/kit.list" "$S/" "$h:/workspace/"
"${SSH[@]}" "$h" 'cd /workspace && md5sum -c --quiet' < "$T/kit.md5" && echo "$h: RL kit md5 ok ($(wc -l < "$T/kit.list") files)"
bash "$G0/g0_pod.sh" prep "$h" "$line"
"${SSH[@]}" "$h" "test -e /workspace/atg0/runs/$line && echo EXISTS; true" | grep -q EXISTS && { echo "$h already has runs/$line"; exit 1; }
(cd "$D" && find . -type f ! -path './logs/*' | sed 's|^\./||' | sort) > "$T/run.list"
"${SSH[@]}" "$h" "mkdir -p /workspace/atg0/runs/$line /workspace/atg0/logs"
"${RS[@]}" --files-from="$T/run.list" "$D/" "$h:/workspace/atg0/runs/$line/"
"${RS[@]}" "$D/logs/${line}_seg1.log" "$D/logs/$line.events.log" "$D/logs/$line.driver.log" "$h:/workspace/atg0/logs/"
(cd "$D" && xargs -d '\n' md5sum < "$T/run.list") > "$T/run.lab.md5"
"${SSH[@]}" "$h" "cd /workspace/atg0/runs/$line && xargs -d '\n' md5sum" < "$T/run.list" > "$T/run.pod.md5"
cmp "$T/run.lab.md5" "$T/run.pod.md5" && echo "$h: run folder md5 ok ($(wc -l < "$T/run.list") files)"
"${SSH[@]}" "$h" "cd /workspace/g0exp/author_tags_20261110 && AT=/workspace/atg0 ND_GUARD=off /workspace/venv/bin/python tag_line.py --line $line --gpu 0 --seg 2 --resume $ROLL --dry-run" > "$T/dry.json"
python3 - "$T/dry.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); c = d["cmd"]
print("dry run: refuses", d["refuses"], "| resume in cmd", "--resume" in c, "| guard flags", "--rollback_on_alarm" in c, "--no_alarm_ckpt" in c, "--stop_nonfinite" in c)
assert d["refuses"] == [] and "--rollback_on_alarm" not in c
EOF
bash "$G0/g0_pod.sh" launch "$h" "$line" 0 8 0.3 1795060200 2 "$ROLL"
sleep 5; "${SSH[@]}" "$h" "tail -4 /workspace/atg0/logs/$line.driver.log"
rm -rf "$T"
