#!/usr/bin/env bash
# after_pretrain.sh <alias> <seed> <hours>: when the pod's pretraining ends, score the base on the
# dev split (plain mode, 5 draws) and start RL.
set -euo pipefail
h=$1; s=$2; hours=$3
F=/scratch/cowifornia/nd-rl/fable_rl
HERE=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -F $F/ssh_config)
until "${SSH[@]}" "$h" "grep -q 'PRETRAIN_DONE\|Traceback' /workspace/runs/pre_s*.log" 2>/dev/null; do sleep 30; done
"${SSH[@]}" "$h" "grep -q PRETRAIN_DONE /workspace/runs/pre_s*.log" || { echo "PRETRAIN FAILED on $h"; exit 1; }
bash "$HERE/push.sh" "$h" | tail -1
"${SSH[@]}" "$h" "cd /workspace/fr && LEAN_GATE_WORKERS=7 timeout 900 /workspace/venv/bin/python solve.py --ckpt /workspace/runs/pre_s$s/pre.pt --in /workspace/data/y_dev.jsonl --out /workspace/runs/pre_s$s/dev_plain_k5.jsonl --k 5 --seed 0 --mode plain --judge 2>&1 | tail -13"
bash "$HERE/run.sh" "$h" rl_s$s $(( ${hours%.*} * 3600 + 5400 )) -m fr.rl --init /workspace/runs/pre_s$s/pre.pt --out /workspace/runs/rl_s$s --seed $s --hours $hours
echo "AFTER_PRETRAIN_OK $h $(date -u +%FT%TZ)"
