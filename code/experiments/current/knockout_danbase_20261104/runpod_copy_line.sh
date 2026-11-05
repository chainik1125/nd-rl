#!/usr/bin/env bash
# runpod_copy_line.sh <line> <alias> [<checkpoint on serrano>]: the one-way copy that lets one
# knockout_danbase line resume on a Runpod pod (runpod_copy.sh of 05:13Z, for any line).
# Written by agent:claude, 2026-11-05, for the coordinator to run (copies to pods are the
# coordinator's, on Leon's approval; the supervisor never copies to a pod). Run it after a
# "NEEDS POD:" event; the event names the line and its checkpoint on serrano.
#
# WHAT IT COPIES (from serrano; rsync -rlpt into the pod's /workspace, NO --delete anywhere,
# nothing touches the Runpod key). About 190 to 200 MB.
#   1. The four segment scripts (common.py, gpu.py, dline.py, danread.py) -> /workspace/exp/.
#   2. The GRPO code subset of the patched f35d3daf tree: code/__init__.py, code/thinking/ (with
#      vendor/danstate), code/ndbench/{__init__,core/__init__,core/keys,core/types}.py, COMMIT*.
#   3. Dan's 22 modules the reads import, his strict list (data/coe/sigma_heldout_strict.txt,
#      the 106 strict σ held-out theorems, read only, never trained on) and transfer_track832.jsonl
#      TRIMMED to those 106 rows.
#   4. The line's base: stage1_ko_s<seed>.pt for dko_s* and dkoour_s* (stage1_full_s3.pt for
#      dfull_s3), Dan's public create-or-elicit checkpoints.
#   5. The line's arm's training pool: dan_rl_targets.jsonl (Dan's 4,495 rl_targets) for the
#      primary dko_* lines, L2_pool.jsonl (5,000) for the secondary dkoour_* lines; and
#      canary_long.jsonl (the canary's text; its builder excluded every sealed set).
#   6. The line's resume checkpoint: by default the newest rolling or alarm checkpoint in
#      KD/backup/<line>/ on serrano (the "NEEDS POD:" event puts it there), to
#      /workspace/kd/runs/<line>/pod_resume/resume_it<i>.pt, where the supervisor looks for it.
# NOT copied: textbook72 (sealed), Charles149 and its holdout / long-reach sets, heldout1000, the
# 726 other transfer theorems, the other arm's pool, other bases, any other eval or benchmark set.
# DRY_RUN=1 stages and lists the files without copying anything. Every copied file is md5-checked (serrano against the pod). When the pod's env is installed
# (podsetup.sh), podcheck.sh runs; last, it prints the register_pod.py line that pins the line to
# the pod (the supervisor then resumes it there).
set -euo pipefail
line=${1:?usage: runpod_copy_line.sh <line> <alias> [<checkpoint>]}
h=${2:?usage: runpod_copy_line.sh <line> <alias> [<checkpoint>]}
ck=${3:-}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name>"; exit 2; }
KD=/scratch/cowifornia/nd-rl/knockout_danbase
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
G=$KD/snap/f35d3daf_danbase
D=$KD/dan_code_13e41bbd
read -r base pool < <(cd "$W" && python3 -c "
import common as C, sys
L = C.LINES[sys.argv[1]]
print(L['base'], L['pool'])" "$line")
if [[ -z $ck ]]; then  # the newest checkpoint with a training state in serrano's backup
  ck=$({ ls "$KD/backup/$line"/*_rolling.pt "$KD/backup/$line"/*_alarm.pt 2>/dev/null || true; } |
       sed -E 's/.*_it([0-9]+)(_rolling|_alarm)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
fi
[[ -s $ck ]] || { echo "no checkpoint for $line (looked in $KD/backup/$line/)"; exit 2; }
it=$(sed -E 's/.*_it([0-9]+)(_rolling|_alarm)?\.pt$/\1/' <<< "$ck")
[[ $it =~ ^[0-9]+$ ]] || { echo "cannot read the iteration from $ck"; exit 2; }
echo "line $line (base $base, pool $pool) -> $h; resume checkpoint $ck (it $it)"

# ---- stage on serrano, in the pod's /workspace layout, exactly what goes over
ST=$KD/runpod_stage_$line
rm -rf "$ST"
mkdir -p "$ST/exp" "$ST/kd/snap/f35d3daf_danbase/code/ndbench/core" "$ST/kd/dan_code_13e41bbd/coe" \
  "$ST/kd/dan_code_13e41bbd/nd_verify" "$ST/kd/dan_code_13e41bbd/data/coe" "$ST/kd/bases" \
  "$ST/kd/pools" "$ST/kd/runs/$line/pod_resume" "$ST/kd/logs"
cp -p "$W"/{common.py,gpu.py,dline.py,danread.py} "$ST/exp/"
S=$ST/kd/snap/f35d3daf_danbase
cp -p "$G/code/__init__.py" "$S/code/"
cp -rp "$G/code/thinking" "$S/code/"
cp -p "$G/code/ndbench/__init__.py" "$S/code/ndbench/"
cp -p "$G"/code/ndbench/core/{__init__.py,keys.py,types.py} "$S/code/ndbench/core/"
cp -p "$G"/COMMIT "$G"/COMMIT_PATCHED "$S/"
T=$ST/kd/dan_code_13e41bbd
for f in best_model eval_set expert_iter gen lean_gate lean_judge lean_prefilter lean_tok model \
         nd2lean normalize patterns prune record sample state_env state_ladder_ei state_sample \
         tokenizer; do cp -p "$D/$f.py" "$T/"; done
cp -p "$D/coe/coe_sigma.py" "$T/coe/"
cp -p "$D"/nd_verify/{__init__.py,verify.py} "$T/nd_verify/"
cp -p "$D/data/coe/sigma_heldout_strict.txt" "$T/data/coe/"
python3 - "$D" "$T" <<'EOF'
import json, os, sys
d, t = sys.argv[1], sys.argv[2]
strict = set(open(f"{d}/data/coe/sigma_heldout_strict.txt").read().split())
src = f"{d}/data/coe/transfer_track832.jsonl"
rows = [l for l in open(src) if json.loads(l)["name"] in strict]
assert len(rows) == 106, len(rows)
out = f"{t}/data/coe/transfer_track832.jsonl"
open(out, "w").writelines(rows)
m = os.stat(src).st_mtime
os.utime(out, (m, m))
EOF
# the large files as hard links (same filesystem): no extra space, same bytes
ln "$KD/bases/$base" "$ST/kd/bases/$base"
ln "$KD/pools/$pool" "$KD/pools/canary_long.jsonl" "$ST/kd/pools/"
ln "$ck" "$ST/kd/runs/$line/pod_resume/resume_it$it.pt"
find "$ST" -name __pycache__ -prune -exec rm -rf {} +
(cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$KD/runpod_manifest_$line.txt")
echo "staged $(wc -l < "$KD/runpod_manifest_$line.txt") files, $(du -shL "$ST" | cut -f1)"
if [[ ${DRY_RUN:-} == 1 ]]; then  # preview: what would go over; nothing leaves serrano
  echo "DRY_RUN=1: nothing copied; manifest $KD/runpod_manifest_$line.txt"
  exit 0
fi

# ---- copy (no --delete), then md5 of every file on both sides
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
"${SSH[@]}" "$h" 'mkdir -p /workspace'
timeout 3600 rsync -rlpt -e "ssh -o BatchMode=yes" "$ST/" "$h:/workspace/"
(cd "$ST" && xargs -d '\n' md5sum < "$KD/runpod_manifest_$line.txt") > "$KD/runpod_md5_$line.lab"
"${SSH[@]}" "$h" 'cd /workspace && xargs -d "\n" md5sum' < "$KD/runpod_manifest_$line.txt" > "$KD/runpod_md5_$line.pod"
if ! cmp -s "$KD/runpod_md5_$line.lab" "$KD/runpod_md5_$line.pod"; then
  diff "$KD/runpod_md5_$line.lab" "$KD/runpod_md5_$line.pod" | head -20
  echo "MD5 MISMATCH: do not register; rerun this script"; exit 1
fi
echo "md5: all $(wc -l < "$KD/runpod_md5_$line.lab") files equal on serrano and $h ($(date -u +%FT%TZ))"
grep -E "bases/|pools/|pod_resume/" "$KD/runpod_md5_$line.lab" | sed -E 's/^(.{8})[0-9a-f]+/\1…/'

# ---- check the pod when its env is installed
if "${SSH[@]}" "$h" 'test -x /workspace/venv/bin/python && test -x /root/.elan/bin/lean'; then
  bash "$W/podcheck.sh" "$h" "$base"
else
  echo "env not installed yet: run podsetup.sh $h $line (it runs podcheck.sh)"
fi
echo
echo "then pin the line to the pod (the supervisor resumes it there within a pass):"
echo "  python3 $W/register_pod.py add $h <pod id> $line <pod createdAt> --gpuh <remaining GPU-h from the NEEDS POD event>"
