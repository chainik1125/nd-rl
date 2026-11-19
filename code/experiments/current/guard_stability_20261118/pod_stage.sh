#!/usr/bin/env bash
# pod_stage.sh <alias>: put the GRPO kit for the guard-stability tests on a Runpod pod. Run ON SERRANO.
# Reuses authors_pod.sh's `stage <alias> rl rl_authors_half_s0` (GRPO code subset, Dan's modules, the two pools,
# sealed-name check), then drops what these tests do not need (the authors base, the strict-106 transfer rows and
# their name list, dan_axis.json) and adds the tag base stage1_tagAB_p10_s2 (.pt, .tag.json, .pt.args.json) under
# at/bases (pod: /workspace/at, linked at /scratch/cowifornia/nd-rl/author_tags). rsync (no --delete), md5 of every
# file on both sides. Never creates/stops/deletes a pod, never reads an API key, never copies a sealed set.
set -euo pipefail
h=${1:?usage: pod_stage.sh <rp-alias>}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name>"; exit 2; }
AP=/scratch/cowifornia/nd-rl/worktrees/two-authors/code/experiments/current/two_authors_20261111/authors_pod.sh
TA=/scratch/cowifornia/nd-rl/two_authors
AT=/scratch/cowifornia/nd-rl/author_tags
BASE=stage1_tagAB_p10_s2
SEALED='textbook72|tb72|test_13_20|test_21_30|transfer_track832'
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")

bash "$AP" stage "$h" rl rl_authors_half_s0
ST=$TA/pod_stage/$h
MAN=$ST.guard_stability.manifest.txt
rm -rf "$ST/ta/bases" "$ST/kd/dan_code_13e41bbd/data" "$ST/kg"
mkdir -p "$ST/at/bases"
for x in "$BASE.pt" "$BASE.tag.json" "$BASE.pt.args.json"; do cp -p "$AT/bases/$x" "$ST/at/bases/"; done
(cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$MAN")
if grep -i -E "$SEALED" "$MAN"; then echo "SEALED OR TRANSFER NAME IN THE STAGE: nothing copied"; exit 1; fi
echo "staged $(wc -l < "$MAN") files, $(du -shL "$ST" | cut -f1) ($MAN)"
"${SSH[@]}" "$h" 'mkdir -p /workspace/at /scratch/cowifornia/nd-rl && ln -sfn /workspace/at /scratch/cowifornia/nd-rl/author_tags && ls -la /scratch/cowifornia/nd-rl/'
timeout 1800 "${RSYNC[@]}" "$ST/" "$h:/workspace/"
(cd "$ST" && xargs -d '\n' md5sum < "$MAN") > "$MAN.lab.md5"
"${SSH[@]}" "$h" 'cd /workspace && xargs -d "\n" md5sum' < "$MAN" > "$MAN.pod.md5"
if ! cmp -s "$MAN.lab.md5" "$MAN.pod.md5"; then
  diff "$MAN.lab.md5" "$MAN.pod.md5" | head -20
  echo "MD5 MISMATCH: do not start; rerun"; exit 1
fi
echo "md5: all $(wc -l < "$MAN.lab.md5") files equal on serrano and $h ($(date -u +%FT%TZ))"
grep -E "at/bases/|kd/pools/" "$MAN.lab.md5"
