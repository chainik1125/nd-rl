#!/usr/bin/env bash
# Runs on the pod: ../remote.sh's setup (fork at the pinned commit, Lean), then passk.py on each checkpoint in ckpt/,
# or, given a jobs file (passk/backfill.py), on each checkpoint it lists.
set -euo pipefail
FORK_COMMIT=51604b3feaa5e66821d1faba22238fe6d96d46d0   # same as ../remote.sh
export ND_FORK=/workspace/nd-takehome ND_ARCH=untied ND_RL=ei
cd /workspace/long_rl
[ -d "$ND_FORK/.git" ] || git clone -q https://github.com/dan-pandori/nd-takehome.git "$ND_FORK"
git -C "$ND_FORK" cat-file -e "$FORK_COMMIT" 2>/dev/null || git -C "$ND_FORK" fetch -q origin
git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
[ -x ~/.elan/bin/lean ] || curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0 > /dev/null
export LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG=/workspace/long_rl/lean_gate.jsonl ND_DATA=/workspace/long_rl/data
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
if [ $# -eq 0 ]; then
  for c in ckpt/*.pt; do
    t=$(basename "$c" .pt)
    $PY passk/passk.py "$c" --tag "$t" --out "results/$t.json"
  done
  exit
fi
# passk/backfill.py: <jobs> lines are "<HF url> <tag> <arch>"; textbook72 only; a failure doesn't stop the rest.
mkdir -p ckpt results
while read -r url tag arch; do
  [ -f "results/$tag.json" ] && continue
  curl -sSfL -o "ckpt/$tag.pt" "$url" \
    && ND_ARCH=$arch $PY passk/passk.py "ckpt/$tag.pt" --tag "$tag" --out "results/$tag.json" --sets textbook72 < /dev/null \
    || echo "FAILED $tag"
done < "$1"
