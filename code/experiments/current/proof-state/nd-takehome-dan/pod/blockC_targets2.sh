#!/usr/bin/env bash
cd /workspace/nd-takehome
set -e
python3 make_coverage_sets.py merge --glob "data/p2/raw_long_c8b.w*.jsonl" --out data/p2/pool_long_c8b.jsonl --prefix c8longb
nice -n 10 python3 minlen.py --in data/p2/pool_long_c8b.jsonl --out data/p2/pool_long_c8b_minlen.jsonl --bound 8 --time 10 --procs 4
echo BLOCKC_TARGETS2_MINLEN_DONE
