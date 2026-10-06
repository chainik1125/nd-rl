#!/usr/bin/env bash
# Pod side: generate the arm's raw pool with the knobbed generator, merge (class dedup, validation-36 dropped), assemble the
# 155,000-record flat set (depth-3 excluded, evaluation classes excluded), shape table, overlap table, render check.
# Usage: bash pod/dsg/gen.sh <g1|g2>      (called as a job by pod/dsg/jobs.py; log artifacts/dsg/logs/gen_<arm>.log)
set -e; cd /workspace/nd-takehome; export PATH=$HOME/.elan/bin:$PATH; mkdir -p data/dsg artifacts/dsg
ARM=$1
W=$(python3 -c "
import os
try:
    q = open('/sys/fs/cgroup/cpu.max').read().split(); n = os.cpu_count() if q[0] == 'max' else max(1, int(int(q[0]) / int(q[1])))
except Exception: n = os.cpu_count()
print(max(2, min(n, 32)))")
echo "workers $W  $(date -u +%FT%TZ)"
EXCL="data/p2/heldout.jsonl data/p2/targets_depth3.jsonl data/p2/transfer_depth3.jsonl data/p2/targets_reductio_req.jsonl data/p2/transfer_reductio_req.jsonl data/r3_1/depth3_req.jsonl data/r3_1/depth3_req_transfer.jsonl data/ladder/rl_targets.jsonl data/ladder/transfer.jsonl"
NOCAP="--cap_np 1000000000 --cap_pat 1000000000"
if [ "$ARM" = g1 ]; then
  T=$((6000000 / W))
  python3 make_coverage_sets.py gen --out data/dsg/raw_g1 --workers $W --tries $T --seed 20000 $NOCAP --ore_steps 3 --ore_boxes > artifacts/dsg/gen_g1_stats.json
  python3 make_coverage_sets.py merge --glob "data/dsg/raw_g1.w*.jsonl" --out data/dsg/pool_g1.jsonl --prefix g1 > artifacts/dsg/merge_g1.txt
  python3 dsg_assemble.py --pool data/dsg/pool_g1.jsonl --out data/dsg/train_g1.jsonl --report data/dsg/assemble_g1.json --exclude $EXCL --prefix train_g1
else
  T=$((30000000 / W))
  python3 make_coverage_sets.py gen --out data/dsg/raw_g2 --workers $W --tries $T --seed 30000 $NOCAP --ladder --drop_contra --ore_boxes > artifacts/dsg/gen_g2_stats.json
  python3 make_coverage_sets.py merge --glob "data/dsg/raw_g2.w*.jsonl" --out data/dsg/pool_g2.jsonl --prefix g2 > artifacts/dsg/merge_g2.txt
  # fill pool: G1's generator, a different seed (used only for length bins G2 cannot fill; fraction disclosed per bin)
  TF=$((6000000 / W))
  python3 make_coverage_sets.py gen --out data/dsg/raw_g1fill --workers $W --tries $TF --seed 40000 $NOCAP --ore_steps 3 --ore_boxes > artifacts/dsg/gen_g1fill_stats.json
  python3 make_coverage_sets.py merge --glob "data/dsg/raw_g1fill.w*.jsonl" --out data/dsg/pool_g1fill.jsonl --prefix g1fill > artifacts/dsg/merge_g1fill.txt
  python3 dsg_assemble.py --pool data/dsg/pool_g2.jsonl --fill data/dsg/pool_g1fill.jsonl --out data/dsg/train_g2.jsonl --report data/dsg/assemble_g2.json --exclude $EXCL --prefix train_g2
fi
echo "assembled $(date -u +%FT%TZ)"
python3 dsg_shape.py data/dsg/train_$ARM.jsonl --lean --out artifacts/dsg/shape_$ARM.json
python3 dsg_overlap.py --sets data/dsg/train_$ARM.jsonl --pools $EXCL data/train.jsonl data/transfer.jsonl data/rl_targets.jsonl data/p2/train_depth3_f0_a1.jsonl --out artifacts/dsg/overlap_$ARM.json
python3 dsg_render_check.py data/dsg/train_$ARM.jsonl --out artifacts/dsg/render_$ARM.json
cp data/dsg/assemble_$ARM.json artifacts/dsg/
echo "GEN_DONE $(date -u +%FT%TZ)"
