#!/bin/bash
# Pull results from all pods into the local repo.
# Usage: bash code/exp/sync_results.sh   (from repo root)
set -u
KEY=~/.ssh/id_ed25519
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY"
mkdir -p results/pods

pull() { # tag port host remote_paths...
  tag=$1; port=$2; host=$3; shift 3
  mkdir -p "results/pods/$tag"
  $SSH -p "$port" "root@$host" "cd /root/nd && tar czf - $* 2>/dev/null" \
    | tar xzf - -C "results/pods/$tag" 2>/dev/null
  echo "pulled $tag: $(find results/pods/$tag -type f | wc -l | tr -d ' ') files"
}

pull a 22074 63.141.33.5    results runs/L12/metrics.json runs/L16/metrics.json runs/L8_2xtrue/metrics.json
pull b 22163 194.68.245.144 results runs/ei_rl/history.json "runs/ei_rl/round*/discoveries.jsonl" "runs/ei_rl/round*/eval_*.jsonl" runs/L8/metrics.json runs/ei_oracle/history.json
pull c 22107 69.30.85.12    results runs/ei_rl_off/history.json "runs/ei_rl_off/round*/discoveries.jsonl" "runs/ei_rl_off/round*/eval_*.jsonl" runs/L8off/metrics.json
pull d 32032 157.157.221.29 results runs/ei_prim/history.json
