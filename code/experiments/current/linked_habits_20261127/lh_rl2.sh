#!/usr/bin/env bash
# lh_rl2.sh <corpus> [--dry-run]: PREREG Amendment 10's replication, run ON THE POD (started detached by
# `lh_pod.sh rl2`): c24's two GRPO lines again with RL seed 1 (rl_c24_rlhfreit_s1, rl_c24_anc_s1), from the FIRST base
# (stage1_c24_s0.pt), the reward line with the existing reward model rm_c24_reit_s0 (no refit, no new probe), 40
# iterations, snapshots 10/20/30/40 measured untagged on marker_plain; one stage-2 job (`stage2 --rl_seed 1`), the two
# lines one after the other (each takes the whole card, as in stage 2), finished lines skipped on a rerun.
# Starts only after <c>.BASE2_DONE (RL2_ANYTIME=1 overrides) and while no driver runs (it holds <c>.driver.lock, so a
# driver cannot start beside it either). Ends with LH/pod/<c>.RL2_DONE; a line that did not finish is appended to
# <c>.FAILED ("rl2 <line> did not finish"). Bounded: each line under its own timeout (3 h), the job under its cap, this
# wait at most MAX_H_RL2 (8) h, and `lh_pod.sh rl2` runs it under `timeout` (TOTAL_H, 9 h).
# --dry-run: the job's plan and each line's GRPO command, nothing launched or written.
# Environment: LH, PY, GPU (0), POLL (60 s), MAX_H_RL2 (8), RL2_ANYTIME; LH_LAUNCH and LH_RL2_TASKS replace the launcher
# and the task list (tests).
set -uo pipefail
c=${1:?usage: lh_rl2.sh <corpus> [--dry-run]}
[[ $c =~ ^[a-z][a-z0-9]*$ ]] || { echo "bad corpus name"; exit 2; }
W=$(cd "$(dirname "$0")" && pwd)
LH=${LH:-/scratch/cowifornia/nd-rl/linked_habits}
PY=${PY:-/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python}
read -r -a LAUNCH <<< "${LH_LAUNCH:-$PY $W/lh_launch.py}"
export LH
POD=$LH/pod
utc() { date -u +%FT%TZ; }
say() { echo "[$(utc)] $c rl2: $*"; }
if [[ -n ${LH_RL2_TASKS:-} ]]; then tasks=$LH_RL2_TASKS; else
  tasks=$(cd "$W" && "$PY" -c "import lh_core as C; print(' '.join(C.rl2_tasks('$c')))") || { say "task list"; exit 2; }
fi
[[ -n $tasks ]] || { say "no RL seed-1 lines for $c (PREREG Amendment 10: c24 only)"; exit 2; }
if [[ ${2:-} == --dry-run ]]; then
  ND_GUARD=off "${LAUNCH[@]}" stage2 --corpus "$c" --rl_seed 1 --parallel 2 --gpu "${GPU:-0}" --dry-run
  for t in $tasks; do ND_GUARD=off "${LAUNCH[@]}" rl --line "$t" --gpu "${GPU:-0}" --dry-run; done
  exit 0
fi
mkdir -p "$POD"
[[ -e $POD/$c.BASE2_DONE || ${RL2_ANYTIME:-0} == 1 ]] || { say "no $POD/$c.BASE2_DONE: the second base pass has not ended; not started"; exit 3; }
if ! mkdir "$POD/$c.driver.lock" 2>/dev/null; then
  say "a driver holds $POD/$c.driver.lock ($(cat "$POD/$c.driver.lock/pid" 2>/dev/null)): not started"; exit 3
fi
echo "$(hostname) $$ $(utc) rl2" > "$POD/$c.driver.lock/pid"
trap 'rm -f "$POD/$c.driver.lock/pid"; rmdir "$POD/$c.driver.lock" 2>/dev/null' EXIT
finished() {  # a GRPO line whose seg1/end.json says done (as lh_driver.sh)
  local how; how=$(sed -n 's/.*"how": "\([a-z]*\)".*/\1/p' "$LH/runs/$1/seg1/end.json" 2>/dev/null | head -1)
  [[ $how == done || $how == rule ]]
}
pending=()
for t in $tasks; do
  if finished "$t"; then say "$t finished before, skipped"; else pending+=("$t"); fi
done
if (( ${#pending[@]} > 0 )); then
  joined=$(IFS=,; echo "${pending[*]}")
  say "stage2 --corpus $c --rl_seed 1 --tasks $joined"
  if ND_GUARD=off "${LAUNCH[@]}" stage2 --corpus "$c" --rl_seed 1 --tasks "$joined" --parallel 2 --gpu "${GPU:-0}"; then
    "${LAUNCH[@]}" wait --job "stage2_${c}_rl1" --max_h "${MAX_H_RL2:-8}" --every "${POLL:-60}" \
      --heartbeat "$POD/$c.heartbeat.json" --stage rl2 || say "the job did not end rc 0"
  else
    say "the launch refused or failed"
  fi
fi
for t in $tasks; do
  if finished "$t"; then say "$t finished"
  else say "$t FAILED"; echo "$(utc) rl2 $t did not finish" >> "$POD/$c.FAILED"; fi
done
utc > "$POD/$c.RL2_DONE"
say "done"
