#!/usr/bin/env bash
# ds-generator helpers (worktree ~/work/ds-generator; the ~/bin helpers address ~/nd-takehome, so never use podsync/podpush here).
#   bash pod/dsg/sync.sh <pod>                  push code + pools + control checkpoints
#   bash pod/dsg/sync.sh <pod> pull <relpath>   pull a path from the pod into the worktree (into its parent dir)
#   bash pod/dsg/sync.sh <pod> push <relpath>   push one path
#   bash pod/dsg/sync.sh <pod> pullall          pull artifacts/dsg/ (found files, logs, coverage) and ckpts/dsg/
. ~/.config/nd-rl/env; N=$1; shift; F=~/.config/nd-rl/pods/$N; [ -f "$F" ] || { echo "no pod $N"; exit 1; }; . "$F"
W=/home/dan/work/ds-generator
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
R="root@$POD_IP:/workspace/nd-takehome"
RS() { rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "$@"; }
case "${1:-code}" in
  pull) mkdir -p "$W/$(dirname "$2")"; RS "$R/$2" "$W/$(dirname "$2")/" ;;
  push) ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "mkdir -p /workspace/nd-takehome/$(dirname "$2")"; RS "$W/$2" "$R/$2" ;;
  pullall) mkdir -p "$W/artifacts/dsg" "$W/ckpts/dsg"; RS --exclude 'mix_*.jsonl' "$R/artifacts/dsg/" "$W/artifacts/dsg/"; RS "$R/ckpts/dsg/" "$W/ckpts/dsg/" ;;
  sh) shift; ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "cd /workspace/nd-takehome && $*" ;;
  code)
    RS --exclude .git --exclude data/ --exclude artifacts/ --exclude ckpts/ --exclude __pycache__ --exclude .venv "$W/" "$R/"
    ssh $SSHO -p "$POD_PORT" "root@$POD_IP" "mkdir -p /workspace/nd-takehome/{data/ladder,data/p2,data/r3_1,data/dsg,ckpts/dsg,ckpts/lf,artifacts/dsg/logs,targets}"
    RS "$W/targets/" "$R/targets/"
    RS "$W/ckpts/lf/stage1_a1_seq_s0.pt" "$W/ckpts/lf/stage1_a1_seq_s1.pt" "$R/ckpts/lf/"
    RS "$W/data/heldout.jsonl" "$W/data/transfer.jsonl" "$W/data/rl_targets.jsonl" "$W/data/train.jsonl" "$R/data/"
    RS "$W/data/ladder/rl_targets.jsonl" "$W/data/ladder/transfer.jsonl" "$R/data/ladder/"
    RS "$W/data/r3_1/depth3_req.jsonl" "$W/data/r3_1/depth3_req_transfer.jsonl" "$R/data/r3_1/"
    RS "$W/data/p2/heldout.jsonl" "$W/data/p2/targets_depth3.jsonl" "$W/data/p2/transfer_depth3.jsonl" "$W/data/p2/targets_reductio_req.jsonl" "$W/data/p2/transfer_reductio_req.jsonl" "$W/data/p2/train_depth3_f0_a1.jsonl" "$R/data/p2/"
    ;;
esac
