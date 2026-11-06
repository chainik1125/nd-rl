#!/bin/zsh
# Tightness measurements for approach E, one certify process at --mem 512 (the production run uses 3).
cd ${0:A:h}/..
export FASTMIN_BUILD=$PWD/../certifier_exact/build_v8_frozen
P=~/repos/nd-takehome/.venv/bin/python
$P measure.py measure/small_rec.jsonl measure/small_rec_m.jsonl --floor written --secs 15 --procs 1 --mem 512 > measure/small_rec.log 2>&1
$P measure.py measure/small_def.jsonl measure/small_def_m.jsonl --floor written --secs 15 --procs 1 --mem 512 > measure/small_def.log 2>&1
$P probe_cost.py measure/long.jsonl --floor 40 --secs 60 --procs 1 --mem 512 --limit 100 -v > measure/long_levels.log 2>&1
$P measure.py measure/mid.jsonl measure/mid_m.jsonl --floor written --secs 60 --procs 1 --mem 512 > measure/mid.log 2>&1
