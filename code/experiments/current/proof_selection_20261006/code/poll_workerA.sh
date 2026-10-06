#!/bin/bash
# exits when "POOLS READY" appears in logs/worker_A.md or new files appear in pools_v2 / scores (verifier_*), or after 40 min
S=/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006
for i in $(seq 1 16); do
  if grep -q "POOLS READY" $S/logs/worker_A.md 2>/dev/null; then echo "POOLS READY seen $(date -u)"; ls -la ~/runs/proof-selection/pools_v2; exit 0; fi
  if ls ~/runs/proof-selection/scores/*verifier* >/dev/null 2>&1; then echo "verifier scores seen"; exit 0; fi
  sleep 150
done
echo "timeout $(date -u)"; ls -la ~/runs/proof-selection/pools_v2
