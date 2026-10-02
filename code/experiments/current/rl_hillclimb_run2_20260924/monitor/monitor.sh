#!/usr/bin/env bash
# Event stream for run 2: scores, promotions, worker DONE/LIMIT/VIOLATION lines, digest updates, 20-min heartbeat.
AB=/scratch/cowifornia/nd-rl/ab2
declare -A seen_status
SEEN=$AB/logs/monitor_seen.txt; touch $SEEN
while IFS= read -r k; do seen_status["$k"]=1; done < $SEEN
scores_n=$(wc -l < $AB/logs/scores.jsonl 2>/dev/null || echo 0)
daemon_n=$(wc -l < $AB/logs/score_daemon.log 2>/dev/null || echo 0)
digest_h=$(md5sum $AB/digest.md 2>/dev/null | cut -c1-8)
inbox_h=$(md5sum $AB/INBOX_MONITOR.md 2>/dev/null | cut -c1-8)
last_hb=$(date +%s)
while true; do
  scores_n=$(wc -l < $AB/logs/scores.jsonl 2>/dev/null || echo 0)
  dn=$(wc -l < $AB/logs/score_daemon.log 2>/dev/null || echo 0)
  if [ "$dn" -gt "$daemon_n" ]; then
    tail -n +$((daemon_n + 1)) $AB/logs/score_daemon.log | grep -E "status=error|Traceback" | grep -v smoke | head -3 | sed 's/^/DAEMON /'
    daemon_n=$dn
  fi
  for w in w1 w2 w3 w4 w5 w6 w7 audit; do
    f=$AB/$w/STATUS.md
    [ -f $f ] || continue
    while IFS= read -r line; do
      key="$w|$line"
      if [ -z "${seen_status[$key]}" ]; then seen_status[$key]=1; echo "$key" >> $SEEN; echo "STATUS $w: $line"; fi
    done < <(grep -E "DONE|LIMIT|VIOLATION|QUARANTINE|BEST|crash|timeout|champion" $f 2>/dev/null)
  done
  h=$(md5sum $AB/digest.md 2>/dev/null | cut -c1-8)
  if [ "$h" != "$digest_h" ]; then digest_h=$h; echo "DIGEST updated: $(sed -n '1,12p' $AB/digest.md | tr '\n' ' ' | cut -c1-600)"; fi
  h=$(md5sum $AB/INBOX_MONITOR.md 2>/dev/null | cut -c1-8)
  if [ "$h" != "$inbox_h" ]; then inbox_h=$h; echo "INBOX_MONITOR: $(tail -c 500 $AB/INBOX_MONITOR.md | tr '\n' ' ')"; fi
  now=$(date +%s)
  if [ $((now - last_hb)) -ge 1800 ]; then
    last_hb=$now
    gpus=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ';' | tr -d ' ')
    locks=$(python3 $AB/frozen/gpulock.py status 2>/dev/null | tr '\n' ';' | cut -c1-200)
    runs=$(ls -d $AB/w*/runs/*/ $AB/audit/runs/*/ 2>/dev/null | wc -l)
    fr=$(cd $AB && python3 controller.py --root $AB frontier 2>/dev/null | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin); print(" ".join("%s:%s/%s" % (e.get("id"), e.get("metric"), e.get("frontier")) for e in d.get("frontier", [])))
except Exception: print("?")')
    echo "HB $(date -u +%H:%M) runs=$runs scores=$scores_n frontier=[$fr] gpus=$gpus locks=$locks"
  fi
  sleep 30
done
