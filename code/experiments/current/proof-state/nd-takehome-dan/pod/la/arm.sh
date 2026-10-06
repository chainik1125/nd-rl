#!/usr/bin/env bash
# One ladder arm, then base-reachability scoring of its found proofs. Usage: bash pod/la/arm.sh <name> [ladder_ei args...]
# Log: artifacts/ladder/<name>.log ; DONE marker artifacts/ladder/<name>.done
cd /workspace/nd-takehome; export OMP_NUM_THREADS=4
NAME=$1; shift
[ -f data/train.jsonl ] || gunzip -k data/train.jsonl.gz
python3 ladder_ei.py --init ckpts/stage1_abs.pt --name $NAME --batch 512 "$@" > artifacts/ladder/$NAME.log 2>&1 || { echo FAILED >> artifacts/ladder/$NAME.log; exit 1; }
bash pod/la/novelty.sh $NAME >> artifacts/ladder/$NAME.log 2>&1
touch artifacts/ladder/$NAME.done
