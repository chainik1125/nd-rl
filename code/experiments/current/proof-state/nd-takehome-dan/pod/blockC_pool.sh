#!/usr/bin/env bash
# Block C cap-8 natural pool (p3): merge the uncapped cap-8 shards (dedupe by class, drop val-36), report per-length pattern rates.
cd /workspace/nd-takehome
set -e
python3 make_coverage_sets.py merge --glob "data/p2/raw_cap8nat.w*.jsonl" --out data/p2/pool_cap8.jsonl --prefix cap8
echo BLOCKC_POOL_MERGED
