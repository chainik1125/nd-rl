#!/usr/bin/env bash
# Install (or update) every orchestration tool in this directory into ~/bin, and set the host's cron
# guards. Safe to re-run at any time:
#   - `install` writes each tool as a new file, so loops already running keep the copy they started with;
#   - the managed cron lines below are rewritten, other lines are kept (balance_alert stays only where
#     it already runs, so there is one daily balance message, not one per host), and one-off dated lines
#     are dropped — e.g. the take-home night's `0 12 25 9 * killswitch`, which would re-fire every year.
set -e; H=$(cd "$(dirname "$0")" && pwd); mkdir -p ~/bin ~/runs
for f in "$H"/*; do t=$(basename "$f"); case "$t" in install.sh|README.md) continue;; esac; install -m755 "$f" ~/bin/"$t"; done
CRON="*/10 * * * * $HOME/bin/orchestrator
*/15 * * * * $HOME/bin/pod_watchdog
*/10 * * * * $HOME/bin/pod_budget_watch
17 * * * * $HOME/bin/balance_floor
37 * * * * $HOME/bin/disk_watch"
( crontab -l 2>/dev/null | grep -v -E "bin/(orchestrator|pod_watchdog|pod_budget_watch|balance_floor|disk_watch)$" \
    | grep -v -E '^[0-9]+ [0-9]+ [0-9]+ [0-9]+ \* .*killswitch' || true; echo "$CRON" ) | crontab -
echo "installed into ~/bin: $(ls "$H" | grep -v -x -e install.sh -e README.md | tr '\n' ' ')"; echo "crontab:"; crontab -l
