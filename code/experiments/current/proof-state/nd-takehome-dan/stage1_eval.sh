#!/usr/bin/env bash
# Stage-1 evaluation of one checkpoint: held-out greedy by length, transfer greedy and pass@16 (T=0.8), validation-36 greedy.
# usage: bash stage1_eval.sh ckpts/stage1_rel.pt rel
set -e
CK=$1; TAG=$2
mkdir -p artifacts
python3 eval_set.py --ckpt $CK --in data/heldout.jsonl --out artifacts/${TAG}_heldout_greedy.jsonl --temperature 0 --summary artifacts/${TAG}_heldout_greedy.json
python3 eval_set.py --ckpt $CK --in data/transfer.jsonl --out artifacts/${TAG}_transfer_greedy.jsonl --temperature 0 --summary artifacts/${TAG}_transfer_greedy.json
python3 eval_set.py --ckpt $CK --in data/transfer.jsonl --out artifacts/${TAG}_transfer_k16.jsonl --k 16 --temperature 0.8 --seed 0 --summary artifacts/${TAG}_transfer_k16.json
python3 prove.py --ckpt $CK --in targets/validation_36.jsonl --out artifacts/${TAG}_val36_greedy.jsonl --greedy
python3 eval_targets.py --proofs artifacts/${TAG}_val36_greedy.jsonl --judged artifacts/${TAG}_val36_judged.jsonl | tee artifacts/${TAG}_val36_greedy.txt
