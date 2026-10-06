#!/usr/bin/env bash
# la-1: after T1 s0 -> T4 s0 + T5 s0
cd /workspace/nd-takehome
until [ -f artifacts/ladder/la_T1_s0.done ]; do sleep 60; done
bash pod/la/arm2.sh la_T4_s0 --seed 0 --alloc window &
bash pod/la/arm2.sh la_T5_s0 --seed 0 --inject_pool data/ladder/pool_inject_cap6.jsonl --inject_round 4 &
wait; echo Q1C_DONE $(date -u +%FT%TZ)
