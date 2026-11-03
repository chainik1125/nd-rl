#!/usr/bin/env bash
# One ssh to temescal every 15 min (ConnectTimeout 10) until it answers or until DEADLINE (UTC).
DEADLINE=${1:-"2026-11-04 01:00:00"}
SSHC="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
while [ $(date -u +%s) -lt $(date -u -d "$DEADLINE" +%s) ]; do
  if SSH_AUTH_SOCK= timeout 20 $SSHC temescal "uptime" 2>/dev/null; then echo "UP $(date -u +%FT%TZ)"; exit 0; fi
  echo "down $(date -u +%FT%TZ)"
  sleep 900
done
echo "DEADLINE $(date -u +%FT%TZ)"
