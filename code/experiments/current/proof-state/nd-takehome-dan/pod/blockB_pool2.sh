#!/usr/bin/env bash
# Block B pool v2: schema-instantiated classical-only theorems without double negation + generator-native ones.
cd /workspace/nd-takehome
set -e
python3 reductio_pool.py gen --out data/p2/reductio2_cands.jsonl --per_schema 45 --exclude data/p2/pool_cap6_recon.jsonl data/p2/targets_reductio.jsonl data/p2/transfer_reductio.jsonl data/p2/heldout.jsonl
python3 minlen.py --in data/p2/reductio2_cands.jsonl --out data/p2/reductio2_cands_minlen.jsonl --bound 8 --time 10 --procs 4
python3 reductio_pool.py finalize --cands data/p2/reductio2_cands.jsonl --minlen data/p2/reductio2_cands_minlen.jsonl --native data/p2/pool_rnd_co_minlen.jsonl --native_pool data/p2/pool_rnd_co.jsonl --outdir data/p2 --per_schema_t 30 --per_schema_x 15
echo BLOCKB_POOL2_DONE
