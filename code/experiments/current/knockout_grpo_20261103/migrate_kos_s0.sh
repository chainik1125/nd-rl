#!/usr/bin/env bash
# Copy kos_s0_gr and kos_s0_dp_gr (with their source L2 runs and reads) from temescal to ynez via
# serrano, read-only on temescal except the queue STOP file. Usage: migrate_kos_s0.sh
set -u
K=/scratch/cowifornia/nd-rl/knockout_grpo
S=$K/tmp/migrate_temescal
SSHT="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
SSHY="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
mkdir -p $S/runs $S/evals $S/bases $S/logs
r() { SSH_AUTH_SOCK= timeout 3600 rsync -a "$@"; }
for src in kos_s0 kos_s0_dp; do
  mkdir -p $S/runs/${src}_L2/ckpts
  r -e "$SSHT" temescal:$K/runs/${src}_L2/metrics.jsonl temescal:$K/runs/${src}_L2/args.json $S/runs/${src}_L2/
  r -e "$SSHT" temescal:$K/runs/${src}_L2/ckpts/${src}_L2_it39.pt $S/runs/${src}_L2/ckpts/
done
r -e "$SSHT" --exclude '*_snap.pt' --include 'kos_s0_gr_L3_it283_alarm.pt' --exclude '*_alarm.pt' \
  temescal:$K/runs/kos_s0_gr_L3 $S/runs/
r -e "$SSHT" --exclude '*_snap.pt' --include 'kos_s0_dp_gr_L3_it328_alarm.pt' --exclude '*_alarm.pt' \
  temescal:$K/runs/kos_s0_dp_gr_L3 $S/runs/
r -e "$SSHT" temescal:$K/runs/kos_s0_gr_L3_lean_dump.jsonl temescal:$K/runs/kos_s0_dp_gr_L3_lean_dump.jsonl $S/runs/
r -L -e "$SSHT" temescal:$K/evals/kos_s0_gr temescal:$K/evals/kos_s0_dp_gr $S/evals/
mkdir -p $S/bases/kos_s0
r -e "$SSHT" temescal:$K/bases/kos_s0/stage1_tokens.json $S/bases/kos_s0/
r -e "$SSHT" "temescal:$K/logs/kos_s0_gr_*" "temescal:$K/logs/kos_s0_dp_gr_*" "temescal:$K/logs/gr_kos_s0*" $S/logs/
du -sh $S; ls $S/runs/*/ckpts
r -e "$SSHY" $S/runs/ ynez:$K/runs/
r -e "$SSHY" $S/evals/ ynez:$K/evals/
r -e "$SSHY" $S/bases/ ynez:$K/bases/
r -e "$SSHY" $S/logs/ ynez:$K/logs/
echo "copied $(date -u +%FT%TZ)"
