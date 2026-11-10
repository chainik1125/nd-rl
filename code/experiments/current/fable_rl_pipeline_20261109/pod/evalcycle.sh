#!/usr/bin/env bash
# evalcycle.sh <rl pod alias> <run name> <seed tag> [<round>]: on the evaluation pod (fr5), train a
# verifier from the run's saved step data so far and score a policy checkpoint of that run on dev
# in plain and full mode (5 draws). Files go pod -> serrano (pull.sh) -> fr5. Default round: latest.
set -euo pipefail
h=$1; run=$2; tag=$3; rnd=${4:-}
F=/scratch/cowifornia/nd-rl/fable_rl
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/pull.sh" "$h" | tail -1
D=$F/runs/$h/$run
[[ -z $rnd ]] && rnd=$(ls $D/r*.pt | sed -E 's/.*\/r([0-9]+)\.pt/\1/' | sort -n | tail -1)
nver=$(ls $D/ver_r*.pkl | wc -l)
echo "run $run: policy r$rnd.pt, verifier data from $nver rounds"
ssh -n -F $F/ssh_config fr5 "mkdir -p /workspace/ck/$tag /workspace/ev"
rsync -t -e "ssh -F $F/ssh_config" $D/r$rnd.pt $D/ver_r*.pkl fr5:/workspace/ck/$tag/
bash "$HERE/push.sh" fr5 | tail -1
ssh -n -F $F/ssh_config fr5 "cd /workspace/fr && PYTHONUNBUFFERED=1 timeout 2400 /workspace/venv/bin/python -m fr.verifier --run /workspace/ck/$tag --out /workspace/ck/$tag/verifier_n$nver.pt --steps 6000 --max_per_round 150000 --x /workspace/data/train_k12.jsonl.gz --x_pos 400000 2>&1 | grep 'holdout\|examples' | cut -c1-600
for m in plain full; do SOLVER_MODE=\$m LEAN_GATE_WORKERS=7 timeout 1500 /workspace/venv/bin/python solve.py --ckpt /workspace/ck/$tag/r$rnd.pt --verifier /workspace/ck/$tag/verifier_n$nver.pt --in /workspace/data/y_dev.jsonl --out /workspace/ev/${tag}_r${rnd}_\$m.jsonl --k 5 --seed 0 --judge 2>&1 | grep -v Warn | cut -c1-200 > /workspace/ev/${tag}_r${rnd}_\$m.log; echo \"== $tag r$rnd \$m (verifier n$nver)\"; grep -v '^{\|^---' /workspace/ev/${tag}_r${rnd}_\$m.log; done"
echo "EVALCYCLE_OK $tag r$rnd $(date -u +%FT%TZ)"
