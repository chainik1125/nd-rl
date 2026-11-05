#!/usr/bin/env bash
# runpod_copy.sh <alias>: the one-way copy that gives one base_saturation pod exactly what its jobs
# need. Written by agent:claude, 2026-11-05, for the coordinator to run (copies to pods are the
# coordinator's, on Leon's approval; the experiment agent never copies to a pod).
#
# Stages on serrano (/scratch/cowifornia/nd-rl/base_saturation/stage/<alias>/sat, the pod's
# /workspace/sat layout; large files hard-linked), then rsync -rlpt to the pod's /workspace (NO
# --delete), then md5 of every file on both sides, then podcheck.sh.
#
# WHAT EACH POD GETS (all pods: pod/*.sh -> sat/exp/, the read snapshot code/rd (cce4dfc, the
# CPT reads' code), and Charles149 runs/pools/charles_1002/charles_clean.jsonl, an eval set, read only):
#   rp-sat-a      code/s1 (cd089700 + anneal.py), S1g3_s0 model.pt, S1g3.pkl, heldout.jsonl   ~195 MB
#   rp-sat-b025   code/cpt (32bc1da: circuit_diffing cpt.py), S1g3_s0 model.pt, S1g3.pkl         ~190 MB
#   rp-sat-b4     as rp-sat-b025                                                                 ~190 MB
#   rp-sat-c      code/s1 (+ stage1_to0.py), code/l2 (fa28f6eb), code/l3 (f35d3daf), S1g3.pkl,
#                 heldout.jsonl, the GRPO pools (L2, L3, L3_3p, canary_long)                     ~230 MB
#   rp-sat-d1     code/l2, code/l3, CPT_s0 (circuits/ckpts/cpt/cpt_lr0.00015.pt -> ckpts/cpt_s0.pt),
#                 the GRPO pools                                                                 ~115 MB
#   rp-sat-r*     (extra read pods) code/rd and Charles149 only                                   ~2 MB
# Code snapshots are `git archive` of code/__init__.py, code/thinking, code/ndbench/{__init__.py,
# core} (plus code/experiments/{,current/}__init__.py and the arm's scripts), with a COMMIT file.
# NEVER staged: textbook72 (refused by md5 against every textbook72 file and by name), any
# credentials (refused by name), any other eval set. DRY_RUN=1 stages and lists, copies nothing.
set -euo pipefail
h=${1:?usage: runpod_copy.sh <alias>}
[[ $h =~ ^rp-sat-[a-z0-9-]+$ ]] || { echo "alias must look like rp-sat-<name>"; exit 2; }
E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=/scratch/cowifornia/nd-rl/base_saturation
R=/scratch/cowifornia/nd-rl/think/runs
REPO=/home/cowifornia/code/research/nd-rl
CPT0=/scratch/cowifornia/nd-rl/circuits/ckpts/cpt/cpt_lr0.00015.pt
TB=/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72
ST=$S/stage/$h
rm -rf "$ST"
mkdir -p "$ST/sat/code" "$ST/sat/exp" "$ST/sat/runs/pools/charles_1002" "$ST/sat/ckpts"
cp -p "$E"/pod/*.sh "$ST/sat/exp/"
echo "$h" > "$ST/sat/alias"
snap() {  # TAG COMMIT [more paths]
  local tag=$1 c=$2
  shift 2
  mkdir -p "$ST/sat/code/$tag"
  git -C "$REPO" archive "$c" -- code/__init__.py code/thinking code/ndbench/__init__.py \
    code/ndbench/core "$@" | tar -x -C "$ST/sat/code/$tag"
  git -C "$REPO" rev-parse "$c" > "$ST/sat/code/$tag/COMMIT"
}
mine() {  # TAG FILE...: this experiment's scripts into a snapshot, as importable modules
  local d=$ST/sat/code/$1/code/experiments/current/base_saturation_20261105
  shift
  mkdir -p "$d"
  for f in "$@"; do cp -p "$E/$f" "$d/"; done
}
INITS=(code/experiments/__init__.py code/experiments/current/__init__.py)
CD=code/experiments/current/circuit_diffing_20261101
data_s1() {
  mkdir -p "$ST/sat/runs/stage1/S1g3_s0" "$ST/sat/runs/data"
  ln "$R/data/S1g3.pkl" "$ST/sat/runs/data/S1g3.pkl"
}
pools() {
  mkdir -p "$ST/sat/runs/pools/curriculum/L2" "$ST/sat/runs/pools/curriculum/L3" "$ST/sat/runs/pools/curriculum_3p"
  ln "$R/pools/curriculum/L2/pool.jsonl" "$ST/sat/runs/pools/curriculum/L2/pool.jsonl"
  ln "$R/pools/curriculum/L3/pool.jsonl" "$ST/sat/runs/pools/curriculum/L3/pool.jsonl"
  ln "$R/pools/curriculum_3p/L3_3p.jsonl" "$ST/sat/runs/pools/curriculum_3p/L3_3p.jsonl"
  ln "$R/pools/canary_long.jsonl" "$ST/sat/runs/pools/canary_long.jsonl"
}
snap rd cce4dfc
ln "$R/pools/charles_1002/charles_clean.jsonl" "$ST/sat/runs/pools/charles_1002/charles_clean.jsonl"
case $h in
  rp-sat-a)
    snap s1 cd089700 "${INITS[@]}"; mine s1 anneal.py
    data_s1; ln "$R/stage1/S1g3_s0/model.pt" "$ST/sat/runs/stage1/S1g3_s0/model.pt"
    ln /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl "$ST/sat/runs/heldout.jsonl" ;;
  rp-sat-b025|rp-sat-b4)
    snap cpt 32bc1da "${INITS[@]}" $CD/common.py $CD/cpt.py $CD/stage0_weights.py
    data_s1; ln "$R/stage1/S1g3_s0/model.pt" "$ST/sat/runs/stage1/S1g3_s0/model.pt" ;;
  rp-sat-c)
    snap s1 cd089700 "${INITS[@]}"; mine s1 stage1_to0.py
    snap l2 fa28f6eb; snap l3 f35d3daf
    data_s1; rmdir "$ST/sat/runs/stage1/S1g3_s0" "$ST/sat/runs/stage1"
    ln /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl "$ST/sat/runs/heldout.jsonl"
    pools ;;
  rp-sat-d1)
    snap l2 fa28f6eb; snap l3 f35d3daf
    ln "$CPT0" "$ST/sat/ckpts/cpt_s0.pt"; pools ;;
  rp-sat-r*) ;;
  *) echo "unknown alias $h"; exit 2 ;;
esac
find "$ST" -name __pycache__ -prune -exec rm -rf {} +
(cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$S/stage/manifest_$h.txt")
# guards: no sealed set, no credentials
(cd "$ST" && xargs -d '\n' md5sum < "$S/stage/manifest_$h.txt") > "$S/stage/md5_$h.lab"
if find "$TB" -type f -exec md5sum {} + | cut -d' ' -f1 | grep -qxFf - <(cut -d' ' -f1 "$S/stage/md5_$h.lab"); then
  echo "REFUSED: a staged file has the md5 of a textbook72 file"; exit 1
fi
if grep -iqE 'textbook|tb72|runpod_env|id_rsa|id_ecdsa|id_runpod|id_ynez|\.ssh/|credential|secret' "$S/stage/manifest_$h.txt"; then
  echo "REFUSED: a staged path names a sealed set or a credential"; exit 1
fi
echo "$h: staged $(wc -l < "$S/stage/manifest_$h.txt") files, $(du -shL "$ST" | cut -f1) (manifest $S/stage/manifest_$h.txt)"
grep -vE '  sat/code/' "$S/stage/md5_$h.lab" | sed -E 's/^(.{8})[0-9a-f]+/\1…/'
if [[ ${DRY_RUN:-} == 1 ]]; then echo "DRY_RUN=1: nothing copied"; exit 0; fi

# ---- copy (no --delete), then md5 of every file on both sides
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
"${SSH[@]}" "$h" 'mkdir -p /workspace'
timeout 3600 rsync -rlpt -e "ssh -o BatchMode=yes" "$ST/" "$h:/workspace/"
"${SSH[@]}" "$h" 'cd /workspace && xargs -d "\n" md5sum' < "$S/stage/manifest_$h.txt" > "$S/stage/md5_$h.pod"
if ! cmp -s "$S/stage/md5_$h.lab" "$S/stage/md5_$h.pod"; then
  diff "$S/stage/md5_$h.lab" "$S/stage/md5_$h.pod" | head -20
  echo "MD5 MISMATCH on $h: rerun this script"; exit 1
fi
echo "md5: all $(wc -l < "$S/stage/md5_$h.lab") files equal on serrano and $h ($(date -u +%FT%TZ))"
"${SSH[@]}" "$h" 'mkdir -p /workspace/sat/status && cp /workspace/sat/alias /workspace/sat/status/alias'
bash "$E/podcheck.sh" "$h"
