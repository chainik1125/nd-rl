#!/usr/bin/env bash
# serrano PSU guard (coordinator, 23:1xZ): if more than 3 GPUs carry any user's process, stop
# kos_s0_dp_gr right after its next checkpoint (an availability stop); kos_s0_gr has priority.
# Touches only our own processes. Ends at DEADLINE, or when kos_s0_dp_gr is no longer running.
K=/scratch/cowifornia/nd-rl/knockout_grpo; DEADLINE=${1:-"2026-11-04 04:00:00"}
CK=$K/runs/kos_s0_dp_gr_L3/ckpts
gpus_busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u | grep -c .; }
grpo_pid() { pgrep -u "$(id -u)" -f -- "--name kos_s0_dp_gr_L[3]" | head -1; }
while [ $(date -u +%s) -lt $(date -u -d "$DEADLINE" +%s) ]; do
  p=$(grpo_pid)
  if [ -z "$p" ]; then echo "kos_s0_dp_gr not running $(date -u +%TZ)"; exit 0; fi
  n=$(gpus_busy)
  if [ "$n" -gt 3 ]; then
    echo "PSU: $n GPUs busy at $(date -u +%TZ); waiting for kos_s0_dp_gr's next checkpoint"
    cur=$(ls -t $CK/*_alarm.pt $CK/*_rolling.pt 2>/dev/null | head -1)
    for i in $(seq 1 120); do
      new=$(ls -t $CK/*_alarm.pt $CK/*_rolling.pt 2>/dev/null | head -1)
      if [ "$new" != "$cur" ] && [ -f "$new.upload.json" ]; then s0=$(stat -c %s "$new"); sleep 10; [ "$s0" = "$(stat -c %s "$new")" ] && break; fi
      sleep 5
    done
    job=$(python3 -c "import json;print(json.load(open('$K/queue/state/resume_kos_s0_dp_gr.json'))['pid'])" 2>/dev/null)
    echo "stopped for machine availability at $(date -u +%FT%TZ): $n GPUs busy on serrano (PSU limit 3); resume from $(basename "$new")" > $K/evals/kos_s0_dp_gr/AVAILABILITY_STOP
    kill -TERM -"$p" 2>/dev/null; sleep 20; [ -n "$job" ] && kill -TERM -"$job" 2>/dev/null
    echo "STOPPED kos_s0_dp_gr at $(date -u +%TZ) after $(basename "$new")"; exit 0
  fi
  sleep 30
done
echo "DEADLINE $(date -u +%TZ)"
