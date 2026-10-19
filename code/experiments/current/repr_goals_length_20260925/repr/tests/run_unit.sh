#!/usr/bin/env bash
# All unit tests on one GPU (run through tests/with_gpu.py). Writes tests/out/unit.out.
R=/scratch/cowifornia/nd-rl/ab3/repr; PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
cd $R
export OMP_NUM_THREADS=8
$PY -m pytest -q -p no:cacheprovider tests/test_variants.py tests/test_meter_variants.py tests/test_equivalence.py 2>&1 | tail -n 30
echo "== frozen test_meter.py, unchanged, inside a variant frozen copy (posline)"
cd $R/tests/out/frozen_posline && $PY -m pytest -q -p no:cacheprovider test_meter.py 2>&1 | tail -n 5
