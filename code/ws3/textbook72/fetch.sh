#!/usr/bin/env bash
# Restore the sealed textbook set (gitignored) from Dmitry's branch and verify MANIFEST.json.
#   bash code/ws3/textbook72/fetch.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BRANCH=origin/dmitry/cap-comparison-textbook-eval-data
SRC=code/experiments/archived/cap_comparison_20260910/evaluation
git -C "$HERE" fetch -q origin dmitry/cap-comparison-textbook-eval-data
for f in textbook_dev.jsonl textbook_train.jsonl; do
  git -C "$HERE" show "$BRANCH:$SRC/$f" > "$HERE/$f"
  want=$(awk -v f="\"$f\"" 'index($0, f) {hit=1} hit && /sha256/ {gsub(/[^0-9a-f]/, "", $2); print $2; exit}' "$HERE/MANIFEST.json")
  got=$(shasum -a 256 "$HERE/$f" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "sha256 mismatch for $f" >&2; exit 1; }
  echo "ok $f"
done
