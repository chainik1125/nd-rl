#!/usr/bin/env bash
# One worker per GPU for GRPO recipe v2 on ynez: take the next ARM:SEED line from a shared job file and run
# line.sh for it, until no job is left. Unlike queue.sh, the jobs are not bound to a GPU when the worker starts:
# reorder or add pending jobs by editing the job file under its lock (flock JOBFILE.lock), and start another
# worker when a GPU frees up. A job is removed from the file when a worker takes it.
#   [BACKUP=1] bash pull_queue.sh GPU JOBFILE
# BACKUP=1: a backup.py pass after each job. Off by default since 2026-10-15: its destination on /home was removed
# on purpose (Leon: minimal data duplication on /home), and a pass would recreate it.
source /scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh
g=$1; jf=$2
take() {  # the first ARM:SEED line of $jf, removed from the file under the lock; empty when none is left
  flock "$jf.lock" python3 - "$jf" <<'EOF'
import re, sys
p = sys.argv[1]
lines = open(p).read().splitlines()
for i, x in enumerate(lines):
    if re.fullmatch(r"[a-z0-9_]+:[0-9]+", x.strip()):
        open(p, "w").write("".join(y + "\n" for j, y in enumerate(lines) if j != i))
        print(x.strip())
        break
EOF
}
say "worker GPU $g: start ($jf)"
while j=$(take) && [ -n "$j" ]; do
  IFS=: read -r arm s <<< "$j"
  say "worker GPU $g: start $arm seed $s"
  bash $R/grpo_v2/line.sh $g $s $arm > $L/v2_${arm}_s$s.log 2>&1
  say "worker GPU $g: $arm seed $s exit $?: $(tail -1 $L/v2_${arm}_s$s.log | cut -c1-160)"
  if [ "${BACKUP:-0}" = 1 ]; then python3 $R/backup.py >> $L/backup.log 2>&1 || true; fi  # opt-in (2026-10-15)
done
say "worker GPU $g done"
