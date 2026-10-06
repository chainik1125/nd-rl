#!/usr/bin/env bash
# Push THIS run worktree's code to a pod (podsync is hardwired to ~/nd-takehome). Usage: pod/se/push.sh <pod>
. ~/.config/nd-rl/env; N=$1; F=~/.config/nd-rl/pods/$N; [ -f "$F" ] || { echo "no pod $N"; exit 1; }; . "$F"
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
exec rsync -rlptz --no-o --no-g --exclude .git --exclude data/ --exclude artifacts/ --exclude ckpts/ --exclude __pycache__ \
  -e "ssh $SSHO -p $POD_PORT" /home/dan/work/state-env/ "root@$POD_IP:/workspace/nd-takehome/"
