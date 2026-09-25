#!/usr/bin/env bash
# Event stream for the monitor session: one line per thing worth acting on. Polls every 30 s.
AB=/scratch/cowifornia/nd-rl/ab
declare -A seen_exit seen_status seen_q
scores_n=$(wc -l < $AB/logs/scores.jsonl 2>/dev/null || echo 0)
daemon_n=$(wc -l < $AB/logs/score_daemon.log 2>/dev/null || echo 0)
last_hb=$(date +%s)
while true; do
  # new score lines
  n=$(wc -l < $AB/logs/scores.jsonl 2>/dev/null || echo 0)
  if [ "$n" -gt "$scores_n" ]; then
    tail -n +$((scores_n + 1)) $AB/logs/scores.jsonl | python3 -c '
import sys, json
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    if d.get("smoke"): continue
    bl = d.get("by_len") or {}
    bl = {k: (v[0] if isinstance(v, list) else v) for k, v in bl.items()}
    ck = str(d.get("ckpt", "")).replace("/scratch/cowifornia/nd-rl/ab/", "")
    print("SCORE", d.get("arm"), "k=%s s=%s solved=%s by_len=%s %s %s" % (d.get("k"), d.get("seed"), d.get("solved"), bl, ck, d.get("status", "")))'
    scores_n=$n
  fi
  # daemon errors
  dn=$(wc -l < $AB/logs/score_daemon.log 2>/dev/null || echo 0)
  if [ "$dn" -gt "$daemon_n" ]; then
    tail -n +$((daemon_n + 1)) $AB/logs/score_daemon.log | grep -E "status=error|Traceback|ERROR" | head -3 | sed 's/^/DAEMON /'
    daemon_n=$dn
  fi
  for w in gear/w1 gear/w2 orch/exec orch/review; do
    d=$AB/$w
    # worker exit
    if [ -z "${seen_exit[$w]}" ] && grep -q '^EXIT' $d/stderr.log 2>/dev/null; then
      seen_exit[$w]=1; echo "EXIT $w: $(grep -m1 '^EXIT' $d/stderr.log) | stderr tail: $(tail -c 300 $d/stderr.log | tr '\n' ' ')"
    fi
    # status lines of interest
    if [ -f $d/STATUS.md ]; then
      while IFS= read -r line; do
        key="$w|$line"
        if [ -z "${seen_status[$key]}" ]; then seen_status[$key]=1; echo "STATUS $w: $line"; fi
      done < <(grep -E "QUARANTINE|LIMIT|DONE|CRASH|crash|timeout" $d/STATUS.md 2>/dev/null)
    fi
    # questions
    if [ -f $d/QUESTIONS.md ]; then
      h=$(md5sum $d/QUESTIONS.md | cut -c1-8)
      if [ "${seen_q[$w]}" != "$h" ]; then seen_q[$w]=$h; echo "QUESTIONS $w updated: $(tail -c 400 $d/QUESTIONS.md | tr '\n' ' ')"; fi
    fi
  done
  # heartbeat every 15 min
  now=$(date +%s)
  if [ $((now - last_hb)) -ge 900 ]; then
    last_hb=$now
    gpus=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')
    alive=""
    for w in w1 w2; do docker exec ab-gear tmux has-session -t $w 2>/dev/null && alive="$alive gear/$w" ; done
    for w in exec review; do docker exec ab-orch tmux has-session -t $w 2>/dev/null && alive="$alive orch/$w" ; done
    runs=$(ls -d $AB/gear/w*/runs/*/ $AB/orch/exec/runs/*/ 2>/dev/null | wc -l)
    echo "HB $(date -u +%H:%M) alive:[$alive] runs_dirs=$runs scores=$scores_n gpus=$gpus"
  fi
  sleep 30
done
