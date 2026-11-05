#!/usr/bin/env bash
# watchdog.sh: on a base_saturation pod, every 60 s: STALL (the running step's log quiet past its
# stall limit), OVERRUN (the step past 1.5x its expected minutes; its own timeout still bounds it),
# EXIT (go.sh gone without its END record). Events to /workspace/sat/events.log, once per kind
# and step. Exits 10 min after the last go.sh ends.
source /workspace/sat/exp/steps.sh
idle=0
while true; do
  if read -r n pid log t0 exp stall < "$W/status/current" 2>/dev/null; then
    now=$(date +%s)
    if kill -0 "$pid" 2>/dev/null; then
      age=$(( now - $(stat -c %Y "$log" 2>/dev/null || echo "$t0") ))
      if [ "$age" -gt $(( stall * 60 )) ] && [ ! -f "$W/status/$n.stall" ]; then
        ev "STALL $n: log quiet for $(( age / 60 )) min"; touch "$W/status/$n.stall"
      fi
      if [ $(( now - t0 )) -gt $(( exp * 90 )) ] && [ ! -f "$W/status/$n.overrun" ]; then
        ev "OVERRUN $n: $(( (now - t0) / 60 )) min against ${exp} expected"; touch "$W/status/$n.overrun"
      fi
    fi
  fi
  gp=$(cat "$W/status/go.pid" 2>/dev/null)
  if [ -n "$gp" ] && ! kill -0 "$gp" 2>/dev/null && [ ! -f "$W/status/go.end" ]; then
    ev "EXIT go.sh (pid $gp) is gone without an END record"; touch "$W/status/go.end"
  fi
  if [ -f "$W/status/go.end" ] && ! { [ -n "$gp" ] && kill -0 "$gp" 2>/dev/null; }; then
    idle=$(( idle + 1 ))
  else idle=0; fi
  [ $idle -ge 10 ] && { rm -f "$W/status/wd.pid"; exit 0; }
  sleep 60
done
