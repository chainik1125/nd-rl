#!/usr/bin/env bash
# go.sh ROLE [ARGS]: one pod's job list for base_saturation_20261105 (started by launch.sh, detached).
#   A               train arm A, then read it (16 chunks)
#   B STEPS         train arm B at STEPS (78 or 1248), then read it
#   C [SEEDS]       train arm C; then D2 (the RL chain from C); then read D2's chunks SEEDS (0-15)
#   D1              the RL chain from CPT_s0 (ckpts/cpt_s0.pt); then read D1 (16 chunks)
#   READ CKPT NAME SEEDS   read chunks SEEDS (e.g. 0-7 or 3,5) of CKPT as read NAME
# One go.sh holds the pod's GPU at a time (flock; a second one waits up to 8 h). Events go to
# /workspace/sat/events.log (poll.sh on serrano shows them); watchdog.sh watches every step.
set -uo pipefail
source /workspace/sat/exp/steps.sh
role=${1:?usage: go.sh ROLE [ARGS]}
shift
exec 9> "$W/status/gpu.lock"
ev "go $role $*: waiting for the GPU lock"
flock -w 28800 9 || { ev "FAIL go $role: GPU lock wait timed out"; exit 1; }
echo $$ > "$W/status/go.pid"
rm -f "$W/status/go.end"
if ! { [ -f "$W/status/wd.pid" ] && kill -0 "$(cat "$W/status/wd.pid")" 2>/dev/null; }; then
  setsid nohup bash "$W/exp/watchdog.sh" > "$W/logs/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$W/status/wd.pid"
fi
ev "go $role $*: start"
rc=0
case $role in
  A) train_A && read_chunks "$W/out/A_s0/model.pt" charles149_sat_A_s0_plain_k1024 0-15 || rc=1 ;;
  B) s=${1:?B needs STEPS}
     train_B "$s" && read_chunks "$W/out/B$s/cpt_lr0.00015.pt" "charles149_sat_B${s}_s0_plain_k1024" 0-15 || rc=1 ;;
  C) sd=${1:-0-15}
     train_C && rl_chain "$W/out/C_s0/model.pt" fromC2x \
       && read_chunks "$(rl_final fromC2x)" charles149_sat_D2_s0_plain_k1024 "$sd" || rc=1 ;;
  D1) rl_chain "$W/ckpts/cpt_s0.pt" fromcpt \
       && read_chunks "$(rl_final fromcpt)" charles149_sat_D1_s0_plain_k1024 0-15 || rc=1 ;;
  READ) read_chunks "${1:?}" "${2:?}" "${3:?}" || rc=1 ;;
  *) ev "FAIL go: unknown role $role"; rc=2 ;;
esac
ev "go $role $*: END rc $rc"
touch "$W/status/go.end"
exit $rc
