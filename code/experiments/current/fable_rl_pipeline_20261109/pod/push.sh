#!/usr/bin/env bash
# push.sh <alias> [data]: copy the pipeline code (and, with "data", x and the stripped y splits)
# to a pod. Only these go over: fr/, vendor_dan/, tests/, *.py of this folder; and
# train_k12.jsonl.gz, y_train.jsonl, y_dev.jsonl. No test set is ever copied.
set -euo pipefail
h=${1:?usage: push.sh <alias> [data]}
F=/scratch/cowifornia/nd-rl/fable_rl
HERE=$(cd "$(dirname "$0")/.." && pwd)
R=(rsync -rlpt --exclude __pycache__ -e "ssh -F $F/ssh_config")
"${R[@]}" "$HERE/fr" "$HERE/vendor_dan" "$HERE/tests" "$h:/workspace/fr/"
ls "$HERE"/*.py >/dev/null 2>&1 && "${R[@]}" "$HERE"/*.py "$h:/workspace/fr/"
if [[ ${2:-} == data ]]; then
  "${R[@]}" "$F/data/train_k12.jsonl.gz" "$F/data/y_train.jsonl" "$F/data/y_dev.jsonl" "$h:/workspace/data/"
  ssh -F $F/ssh_config "$h" 'cd /workspace/data && md5sum train_k12.jsonl.gz y_train.jsonl y_dev.jsonl' > /tmp/fr_md5_$h.txt
  (cd $F/data && md5sum train_k12.jsonl.gz y_train.jsonl y_dev.jsonl) | diff - /tmp/fr_md5_$h.txt && echo "md5 equal on $h"
fi
echo "PUSH_OK $h"
