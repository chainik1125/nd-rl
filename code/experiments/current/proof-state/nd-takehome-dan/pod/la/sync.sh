#!/usr/bin/env bash
# ladder-A helpers (worktree ~/work/ladder-A, not ~/nd-takehome). Usage:
#   bash pod/la/sync.sh <pod>                  push code (no .git/data/artifacts/ckpts) + targets + stage1_abs.pt
#   bash pod/la/sync.sh <pod> pull <relpath>   pull a path from the pod into the worktree (into its parent dir)
#   bash pod/la/sync.sh <pod> push <relpath>   push a path from the worktree to the pod
. ~/.config/nd-rl/env; N=$1; shift; F=~/.config/nd-rl/pods/$N; [ -f "$F" ] || { echo "no pod $N"; exit 1; }; . "$F"
W=/home/dan/work/ladder-A
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
R="root@$POD_IP:/workspace/nd-takehome"
case "${1:-code}" in
  pull) mkdir -p "$W/$(dirname "$2")"; exec rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$R/$2" "$W/$(dirname "$2")/" ;;
  push) ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "mkdir -p /workspace/nd-takehome/$(dirname "$2")"; exec rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/$2" "$R/$2" ;;
  code)
    rsync -rlptz --no-o --no-g --exclude .git --exclude data/ --exclude artifacts/ --exclude ckpts/ --exclude __pycache__ --exclude .venv -e "ssh $SSHO -p $POD_PORT" "$W/" "$R/"
    ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "mkdir -p /workspace/nd-takehome/data/ladder /workspace/nd-takehome/ckpts/ladder /workspace/nd-takehome/artifacts/ladder /workspace/nd-takehome/targets"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/targets/" "$R/targets/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/ckpts/stage1_abs.pt" "$R/ckpts/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/data/heldout.jsonl" "$W/data/train.jsonl.gz" "$W/data/rl_targets.jsonl" "$W/data/transfer.jsonl" "$R/data/"
    ;;
esac
