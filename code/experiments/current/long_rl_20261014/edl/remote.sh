#!/usr/bin/env bash
# Runs on the pod: the fork at ../remote.sh's pinned commit (no Lean: EDL only scores likelihoods), then edl.py on each
# checkpoint in <jobs> (lines "<HF url> <tag> <arch>", from edl/launch.py). A failure doesn't stop the rest.
set -euo pipefail
FORK_COMMIT=51604b3feaa5e66821d1faba22238fe6d96d46d0   # same as ../remote.sh
export ND_FORK=/workspace/nd-takehome ND_RL=ei ND_DATA=/workspace/long_rl/data
cd /workspace/long_rl
[ -d "$ND_FORK/.git" ] || git clone -q https://github.com/dan-pandori/nd-takehome.git "$ND_FORK"
git -C "$ND_FORK" cat-file -e "$FORK_COMMIT" 2>/dev/null || git -C "$ND_FORK" fetch -q origin
git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
mkdir -p ckpt edl/results
while read -r url tag arch; do
  [ -f "edl/results/$tag.json" ] && continue
  curl -sSfL -o "ckpt/$tag.pt" "$url" \
    && ND_ARCH=$arch $PY edl/edl.py "ckpt/$tag.pt" --tag "$tag" --out "edl/results/$tag.json" < /dev/null \
    || echo "FAILED $tag"
  rm -f "ckpt/$tag.pt"
done < "$1"
