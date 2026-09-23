#!/usr/bin/env bash
# Fetch the data files and the checkpoint listed in MANIFEST.json and verify their sha256.
# Code files (model.py, tokenizer.py, nd_verify/) are committed; this script also re-verifies them.
#
#   bash code/ws3/fork_pin/fetch.sh            # fetch what is missing, verify everything
#   bash code/ws3/fork_pin/fetch.sh --verify   # verify only, no network
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=$HERE/MANIFEST.json
VERIFY_ONLY=${1:-}
SCRATCH=${FORK_PIN_SCRATCH:-${TMPDIR:-/tmp}/nd-rl-fork-pin}

jq_() { python3 -c "import json,sys; d=json.load(open('$M')); print(eval('d'+sys.argv[1]))" "$1"; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
check() { # check <file> <expected>
  local got; got=$(sha "$1")
  if [ "$got" = "$2" ]; then echo "ok    $(basename "$1")"; else echo "FAIL  $1: sha256 $got != $2"; return 1; fi
}

COMMIT=$(jq_ "['fork']['commit']")
REPO=$(jq_ "['fork']['repo']")
fail=0

# 1. committed code
for f in model.py tokenizer.py nd_verify/__init__.py nd_verify/verify.py; do
  check "$HERE/$f" "$(jq_ "['code_committed_here']['$f']")" || fail=1
done

# 2. data from the fork at the pinned commit
mkdir -p "$HERE/data"
need_clone=0
for name in $(python3 -c "import json; print(' '.join(json.load(open('$M'))['data_fetched_to_data_dir']))"); do
  [ -f "$HERE/data/$name" ] || need_clone=1
done
if [ $need_clone = 1 ] && [ "$VERIFY_ONLY" != "--verify" ]; then
  if [ ! -d "$SCRATCH/.git" ]; then git clone -q "$REPO" "$SCRATCH"; fi
  git -C "$SCRATCH" fetch -q origin "$COMMIT" 2>/dev/null || true
  git -C "$SCRATCH" checkout -q "$COMMIT"
  for name in $(python3 -c "import json; print(' '.join(json.load(open('$M'))['data_fetched_to_data_dir']))"); do
    src=$(jq_ "['data_fetched_to_data_dir']['$name']['fork_path']")
    [ -f "$HERE/data/$name" ] || cp "$SCRATCH/$src" "$HERE/data/$name"
  done
fi
for name in $(python3 -c "import json; print(' '.join(json.load(open('$M'))['data_fetched_to_data_dir']))"); do
  if [ -f "$HERE/data/$name" ]; then
    check "$HERE/data/$name" "$(jq_ "['data_fetched_to_data_dir']['$name']['sha256']")" || fail=1
  else echo "MISSING data/$name"; fail=1; fi
done
[ -f "$HERE/data/train.jsonl" ] || { [ -f "$HERE/data/train.jsonl.gz" ] && gunzip -k "$HERE/data/train.jsonl.gz"; }

# 3. the checkpoint
mkdir -p "$HERE/ckpt"
if [ ! -f "$HERE/ckpt/pretrain.pt" ] && [ "$VERIFY_ONLY" != "--verify" ]; then
  curl -sL -o "$HERE/ckpt/pretrain.pt" "$(jq_ "['checkpoint']['url']")"
fi
if [ -f "$HERE/ckpt/pretrain.pt" ]; then
  check "$HERE/ckpt/pretrain.pt" "$(jq_ "['checkpoint']['sha256']")" || fail=1
else echo "MISSING ckpt/pretrain.pt"; fail=1; fi

[ $fail = 0 ] && echo "fork_pin: all hashes match" || { echo "fork_pin: VERIFICATION FAILED"; exit 1; }
