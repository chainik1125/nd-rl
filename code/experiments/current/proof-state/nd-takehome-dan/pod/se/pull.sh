#!/usr/bin/env bash
# Pull one path from a pod back into THIS run worktree. Usage: pod/se/pull.sh <pod> <relpath>
. ~/.config/nd-rl/env; N=$1; shift; F=~/.config/nd-rl/pods/$N; [ -f "$F" ] || { echo "no pod $N"; exit 1; }; . "$F"
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
D=/home/dan/work/state-env/$1; mkdir -p "$(dirname "$D")"
exec rsync -rlptz --no-o --no-g -e "ssh $SSHO -p $POD_PORT" "root@$POD_IP:/workspace/nd-takehome/$1" "$D"
