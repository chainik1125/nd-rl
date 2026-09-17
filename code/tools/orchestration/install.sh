#!/usr/bin/env bash
# Install the orchestration tools into ~/bin on the agent host and add the podnew log + cron tick.
set -e; H=$(cd "$(dirname "$0")" && pwd); mkdir -p ~/bin ~/runs
for t in runrole gate0 gate1 gate2 review_workspace orchestrator notify; do install -m755 "$H/$t" ~/bin/$t; done
grep -q "pods.log" ~/bin/podnew 2>/dev/null || sed -i 's|^printf "POD_ID=%s\\nPOD_IP=%s\\nPOD_PORT=%s\\n" "$ID" "$IP" "$PORT" > ~/.config/nd-rl/pods/$N|&; echo "$(date -u +%FT%TZ) $N $ID" >> ~/pods.log|' ~/bin/podnew
( crontab -l 2>/dev/null | grep -v orchestrator; echo "*/10 * * * * /home/dan/bin/orchestrator" ) | crontab -
echo "installed: $(ls ~/bin | tr '\n' ' ')"; crontab -l
