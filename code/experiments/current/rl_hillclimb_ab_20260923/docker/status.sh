#!/usr/bin/env bash
# Per arm and worker: tmux alive?, last transcript event, EXIT line, STATUS.md tail.
AB=/scratch/cowifornia/nd-rl/ab
for ARM in gear orch; do
  NAME=ab-$ARM
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "$ARM: container not running"; continue; }
  for W in $(ls "$AB/$ARM" | grep -E '^(w[0-9]+|exec|review)$'); do
    alive=$(docker exec "$NAME" tmux has-session -t "$W" 2>/dev/null && echo yes || echo no)
    last=$(tail -c 4000 "$AB/$ARM/$W/transcript.jsonl" 2>/dev/null | python3 -c '
import sys, json
lines=[l for l in sys.stdin.read().splitlines() if l.strip()]
for l in reversed(lines):
    try: d=json.loads(l)
    except Exception: continue
    t=d.get("type");
    if t=="assistant":
        c=d.get("message",{}).get("content",[])
        txt=" ".join(x.get("text","") for x in c if isinstance(x,dict) and x.get("type")=="text")
        if txt: print(txt[:160].replace("\n"," ")); break
        names=[x.get("name") for x in c if isinstance(x,dict) and x.get("type")=="tool_use"]
        if names: print("tool:", ",".join(names)); break
    elif t=="result": print("RESULT:", str(d.get("result",""))[:160].replace("\n"," ")); break
' 2>/dev/null)
    ex=$(grep -m1 '^EXIT' "$AB/$ARM/$W/stderr.log" 2>/dev/null || true)
    echo "[$ARM/$W] tmux=$alive $ex | $last"
    [ -f "$AB/$ARM/$W/STATUS.md" ] && tail -2 "$AB/$ARM/$W/STATUS.md" | sed 's/^/    /'
  done
done
echo "--- scores (last 12) ---"
tail -12 "$AB/logs/scores.jsonl" 2>/dev/null | python3 -c '
import sys, json
for l in sys.stdin:
    try: d=json.loads(l)
    except Exception: continue
    print(d.get("arm"), d.get("k"), d.get("seed"), d.get("solved"), d.get("by_len"), str(d.get("ckpt",""))[-60:])'
