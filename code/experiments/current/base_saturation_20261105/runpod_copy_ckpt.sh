#!/usr/bin/env bash
# runpod_copy_ckpt.sh <alias> <checkpoint on serrano> <name>: copy ONE checkpoint that a pod made
# (pulled to serrano under /scratch/cowifornia/nd-rl/base_saturation/pods/) to another pod's
# /workspace/sat/ckpts/<name>, for that pod to read it. Written by agent:claude, 2026-11-05, for
# the coordinator to run (copies to pods are the coordinator's). One .pt file; no --delete; md5
# checked on both sides. Planned uses: C_s0 (out/C_s0/model.pt, 38 MB) to rp-sat-a and
# rp-sat-b025 as C_s0.pt; D2's L3 it39 (115 MB) to rp-sat-d1 as D2_s0.pt.
set -euo pipefail
h=${1:?usage: runpod_copy_ckpt.sh <alias> <ckpt> <name>}
ck=${2:?usage: runpod_copy_ckpt.sh <alias> <ckpt> <name>}
name=${3:?usage: runpod_copy_ckpt.sh <alias> <ckpt> <name>}
[[ $h =~ ^rp-sat-[a-z0-9-]+$ ]] || { echo "alias must look like rp-sat-<name>"; exit 2; }
[[ $name =~ ^[A-Za-z0-9_.-]+\.pt$ ]] || { echo "name must be a bare *.pt file name"; exit 2; }
ck=$(readlink -f "$ck")
[[ $ck == /scratch/cowifornia/nd-rl/base_saturation/pods/*.pt ]] || { echo "only checkpoints pulled from this experiment's pods"; exit 2; }
[ -s "$ck" ] || { echo "no file $ck"; exit 2; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
"${SSH[@]}" "$h" 'mkdir -p /workspace/sat/ckpts'
timeout 1800 rsync -pt -e "ssh -o BatchMode=yes" "$ck" "$h:/workspace/sat/ckpts/$name"
a=$(md5sum < "$ck" | cut -d' ' -f1)
b=$("${SSH[@]}" "$h" "md5sum < /workspace/sat/ckpts/$name" | cut -d' ' -f1)
[ "$a" = "$b" ] || { echo "MD5 MISMATCH ($a on serrano, $b on $h): rerun"; exit 1; }
echo "md5 ${a:0:8}…: $ck -> $h:/workspace/sat/ckpts/$name ($(date -u +%FT%TZ))"
