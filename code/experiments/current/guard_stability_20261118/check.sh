#!/usr/bin/env bash
# check.sh [--events]: one line per run: pod, test, last iteration, its pass@1 / P(:) / s, alarms and rollbacks so
# far, exit status, seconds since the log last changed. --events prints only EXIT / STALL (log idle > 300 s) / ERROR
# lines, once each (state in logs/.events.seen), for a watchdog. Run ON SERRANO.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="rp-stab4:relaxed_s2 rp-stab5:noguard_s2 rp-stab6:noguard_wsd_s2 rp-stab7:noguard_kl02_s2"  # rp-stab1 noguard_s0, rp-stab2 noguard_s1: stopped 00:17Z on request
ev=${1:-}
touch logs/.events.seen
for r in $RUNS; do
  h=${r%%:*} t=${r#*:}
  line=$(timeout 30 ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" "
    L=/workspace/at/stability/$t/seg1/train.log
    [ -e \$L ] || { echo 'no log'; exit 0; }
    it=\$(grep -E '^it [0-9]+:' \$L | tail -1 | sed -E 's/^it ([0-9]+): pass@1 ([0-9.]+%).* ([0-9]+)s \(.*/it \1 pass@1 \2 \3s/')
    can=\$(grep -E '^canary it' \$L | tail -1 | sed -E 's/.*P\(:\) ([0-9.]+%) \(log ([-0-9.]+)\).*/P(:) \1 log \2/')
    al=\$(grep -c '^ALARM' \$L); rb=\$(grep -c '^ROLLBACK' \$L)
    err=\$(grep -c -E 'Traceback|Error|OutOfMemory|NONFINITE|nonfinite_stop' \$L)
    ex=\$(cat /workspace/at/stability/$t/seg1/exit.json 2>/dev/null | tr -d '\n')
    age=\$(( \$(date +%s) - \$(stat -c %Y \$L) ))
    echo \"\$it | \$can | alarms \$al rollbacks \$rb | err \$err | idle \${age}s | \${ex:-running}\"
  " 2>&1 | tail -1)
  if [[ $ev == --events ]]; then
    key=""
    [[ $line == *'"rc"'* ]] && key="EXIT $t"
    [[ $line =~ err\ ([1-9][0-9]*) ]] && key="ERROR $t"
    [[ $line =~ idle\ ([0-9]+)s && ${BASH_REMATCH[1]} -gt 300 && $line != *'"rc"'* ]] && key="STALL $t"
    [[ $line == *"Connection"* || $line == *"timed out"* ]] && key="SSHFAIL $t"
    if [[ -n $key ]] && ! grep -qxF "$key" logs/.events.seen; then
      echo "$key" >> logs/.events.seen
      echo "$key ($h): $line"
    fi
  else
    printf '%-9s %-16s %s\n' "$h" "$t" "$line"
  fi
done
