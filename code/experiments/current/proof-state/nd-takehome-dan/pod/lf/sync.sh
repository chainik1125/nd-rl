#!/usr/bin/env bash
# lean-format helpers (worktree ~/work/lean-format). Usage:
#   bash pod/lf/sync.sh <pod>                  push code + data + token Stage-1 ckpts
#   bash pod/lf/sync.sh <pod> pull <relpath>   pull a path from the pod into the worktree (into its parent dir)
#   bash pod/lf/sync.sh <pod> push <relpath>
. ~/.config/nd-rl/env; N=$1; shift; F=~/.config/nd-rl/pods/$N; [ -f "$F" ] || { echo "no pod $N"; exit 1; }; . "$F"
W=/home/dan/work/lean-format
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
R="root@$POD_IP:/workspace/nd-takehome"
RS="rsync -rlptz --no-o --no-g -e \"ssh $SSHO -p $POD_PORT\""
case "${1:-code}" in
  pull) mkdir -p "$W/$(dirname "$2")"; exec rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$R/$2" "$W/$(dirname "$2")/" ;;
  push) ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "mkdir -p /workspace/nd-takehome/$(dirname "$2")"; exec rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/$2" "$R/$2" ;;
  pullall) exec rsync -rlptz --no-o --no-g --exclude 'mix_*.jsonl' -e "ssh $SSHO -p $POD_PORT" "$R/artifacts/lf/" "$W/artifacts/lf/" ;;
  pullckpt) exec rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$R/ckpts/lf" "$R/ckpts/ladder" "$W/ckpts/" ;;
  code)
    rsync -rlptz --no-o --no-g --exclude .git --exclude data/ --exclude artifacts/ --exclude ckpts/ --exclude __pycache__ --exclude .venv -e "ssh $SSHO -p $POD_PORT" "$W/" "$R/"
    ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "mkdir -p /workspace/nd-takehome/{data/ladder,data/p2,ckpts/ladder,ckpts/lf,artifacts/lf/logs,targets}"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/targets/" "$R/targets/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/ckpts/stage1_abs.pt" "$R/ckpts/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/ckpts/lf/token_stage1_a1.pt" "$R/ckpts/lf/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/data/heldout.jsonl" "$W/data/train.jsonl.gz" "$W/data/transfer.jsonl" "$R/data/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/data/ladder/rl_targets.jsonl" "$W/data/ladder/transfer.jsonl" "$R/data/ladder/"
    rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$W/data/p2/heldout.jsonl" "$W/data/p2/targets_depth3.jsonl" "$W/data/p2/transfer_depth3.jsonl" "$W/data/p2/train_depth3_f0_a1.jsonl" "$R/data/p2/"
    ;;
esac
