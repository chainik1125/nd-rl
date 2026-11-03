#!/usr/bin/env bash
# Like wait_event.sh, and also exits on any new EXIT / STALL / OVERRUN / FAIL line in the kq or
# watchdog logs of serrano, ynez or temescal. Prints the new lines and the kos digest.
K=/scratch/cowifornia/nd-rl/knockout_grpo; MAXMIN=${1:-120}
SSHC="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
snap() { ls $K/evals/*/line_done.json $K/evals/*/STOP_LINE 2>/dev/null | sort; }
qlog() { cat $K/logs/kq_serrano.log $K/logs/watchdog.log 2>/dev/null | sed 's/^/[serrano] /'
  for h in ynez; do SSH_AUTH_SOCK= timeout 40 $SSHC $h "cat $K/logs/kq_$h.log $K/logs/watchdog.log 2>/dev/null" 2>/dev/null | sed "s/^/[$h] /"; done; }
bash $K/tools/collect_ynez.sh >/dev/null 2>&1; before=$(snap); q0=$(qlog | grep -E "EXIT|STALL|OVERRUN|FAIL" | sort)
end=$(( $(date +%s) + 60 * MAXMIN ))
while [ $(date +%s) -lt $end ]; do
  sleep 300; bash $K/tools/collect_ynez.sh >/dev/null 2>&1
  al=$(python3 $K/tools/health.py 2>&1 | grep ALERT)
  now=$(snap); q1=$(qlog | grep -E "EXIT|STALL|OVERRUN|FAIL" | sort)
  newq=$(comm -13 <(echo "$q0") <(echo "$q1"))
  if [ -n "$al" ] || [ "$now" != "$before" ] || [ -n "$newq" ]; then echo "$al"; comm -13 <(echo "$before") <(echo "$now"); echo "$newq"; break; fi
done
for t in kos_s0_gr kos_s0_dp_gr kos_s2_gr; do [ -f $K/evals/$t/curve.jsonl ] && python3 -c "
import json
rows=[json.loads(l) for l in open('$K/evals/$t/curve.jsonl')]
print('$t', ' '.join('c%d:%.1f'%(r['c'],r['e32']) for r in rows[-6:]), '| R %.3f'%rows[-1]['R'], 'best %.1f'%max(r['e32'] for r in rows))"; done
date -u
