#!/usr/bin/env bash
# Pod side: generate one null pool with the UNMODIFIED generator (no knob flags), merge (class dedup, validation-36
# dropped), assemble the 155,000-record flat set (depth-3 excluded, evaluation classes excluded), shape / overlap /
# render tables.  Usage: bash pod/nf/gen.sh <p1|p2|p3|p4> <generator-seed>
set -e; cd /workspace/nd-takehome; export PATH=$HOME/.elan/bin:$PATH; mkdir -p data/nf artifacts/nf
P=$1; SEED=$2
W=$(python3 -c "
import os
try:
    q = open('/sys/fs/cgroup/cpu.max').read().split(); n = os.cpu_count() if q[0] == 'max' else max(1, int(int(q[0]) / int(q[1])))
except Exception: n = os.cpu_count()
print(max(2, min(n, 32)))")
echo "pool $P seed $SEED workers $W  $(date -u +%FT%TZ)"
EXCL="data/p2/heldout.jsonl data/p2/targets_depth3.jsonl data/p2/transfer_depth3.jsonl data/p2/targets_reductio_req.jsonl data/p2/transfer_reductio_req.jsonl data/r3_1/depth3_req.jsonl data/r3_1/depth3_req_transfer.jsonl data/ladder/rl_targets.jsonl data/ladder/transfer.jsonl"
NOCAP="--cap_np 1000000000 --cap_pat 1000000000"
T=$((6000000 / W))
# no generator knob flags: this is gen.py's control path (ore_steps 1, no ore_boxes, 45/55 goal/forward, strict off, bot_p 0.02)
python3 make_coverage_sets.py gen --out data/nf/raw_$P --workers $W --tries $T --seed $SEED $NOCAP > artifacts/nf/gen_${P}_stats.json
python3 make_coverage_sets.py merge --glob "data/nf/raw_$P.w*.jsonl" --out data/nf/pool_$P.jsonl --prefix $P > artifacts/nf/merge_$P.txt
python3 dsg_assemble.py --pool data/nf/pool_$P.jsonl --out data/nf/train_$P.jsonl --report data/nf/assemble_$P.json --exclude $EXCL --prefix train_$P
echo "assembled $(date -u +%FT%TZ)"
python3 dsg_shape.py data/nf/train_$P.jsonl --lean --out artifacts/nf/shape_$P.json
python3 dsg_overlap.py --sets data/nf/train_$P.jsonl --pools $EXCL data/transfer.jsonl data/rl_targets.jsonl data/heldout.jsonl --out artifacts/nf/overlap_$P.json
python3 dsg_render_check.py data/nf/train_$P.jsonl --out artifacts/nf/render_$P.json
cp data/nf/assemble_$P.json artifacts/nf/
rm -f data/nf/raw_$P.w*.jsonl
echo "GEN_DONE $(date -u +%FT%TZ)"
