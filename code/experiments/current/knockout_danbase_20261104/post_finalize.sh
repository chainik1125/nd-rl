#!/usr/bin/env bash
# post_finalize.sh: after the agent's handback, wait (bounded) until every line has ended, then
# rerun finalize.sh once and log one event in KD/sup/events.md for the coordinator. Commits
# nothing; the summary's hand-written text (Answer, predictions) is left for the coordinator.
#   (setsid nohup timeout 8h bash post_finalize.sh > KD/sup/post_finalize.log 2>&1 &)
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
KD=/scratch/cowifornia/nd-rl/knockout_danbase
pending() {
  python3 -c "
import json
s = json.load(open('$KD/sup/state.json'))
print(' '.join(k for k, v in s.items() if not k.startswith('_') and v['status'] != 'done'))"
}
while true; do
  p=$(pending)
  [[ -z $p ]] && break
  echo "$(date -u +%FT%TZ) waiting for: $p"
  sleep 120
done
sleep 180  # the supervisor's last collect and the pods' teardown
if timeout 1800 bash "$W/finalize.sh"; then
  echo "- $(date -u +%FT%TZ) every line has ended: finalize.sh rerun (numbers.md, figures and the summary's AUTO sections now hold the final secondary numbers; its Status / Answer / P3-P4 text still says pending; nothing committed)" >> "$KD/sup/events.md"
else
  echo "- $(date -u +%FT%TZ) every line has ended, but the post-handback finalize.sh failed (see KD/sup/post_finalize.log)" >> "$KD/sup/events.md"
fi
