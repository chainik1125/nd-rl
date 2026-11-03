#!/usr/bin/env bash
# Pull evals, run metrics (no checkpoints), bases and logs from ynez and temescal into serrano's KO.
K=/scratch/cowifornia/nd-rl/knockout_grpo
SSHC="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
for h in ynez temescal; do
  SSH_AUTH_SOCK= timeout 600 rsync -a -e "$SSHC" --exclude 'ckpts/' --exclude 'registry/' --exclude 'lean_gate*.jsonl' \
    $h:$K/evals/ $K/evals/ 2>&1 | tail -1
  SSH_AUTH_SOCK= timeout 600 rsync -a -e "$SSHC" --exclude 'ckpts/' --exclude 'registry/' --exclude 'lean_gate*.jsonl' \
    $h:$K/runs/ $K/runs/ 2>&1 | tail -1
  SSH_AUTH_SOCK= timeout 600 rsync -a -e "$SSHC" --exclude 'full_*' $h:$K/bases/ $K/bases/ 2>&1 | tail -1
  mkdir -p $K/logs/$h
  SSH_AUTH_SOCK= timeout 600 rsync -a -e "$SSHC" $h:$K/logs/ $K/logs/$h/ 2>&1 | tail -1
  SSH_AUTH_SOCK= timeout 60 rsync -a -e "$SSHC" $h:$K/STATUS_$h.md $K/ 2>/dev/null
done
