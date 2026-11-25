#!/usr/bin/env bash
# stage0_regrade.sh: Stage 0a on serrano (CPU, at most 8 cores). Rebuilds Dan's olmo-ifcurve outputs from his stored
# samples and grades, and re-grades every stored sample with his grader.
#   1. a work tree that mirrors his repository layout (his code copied or linked read-only, his fetched samples and
#      grades linked, his hand-gate files copied), so that his scripts run unchanged without writing into his clone;
#   2. `olmo_ifcurve/cells.py rows <job>` for the 31 jobs his analysis reads -> work/artifacts/olmo_ifcurve/rows/;
#   3. `olmo_ifcurve/analyze.py --reps 10000` -> work/olmo_ifcurve/{passk,results,emergence,...}.csv;
#   4. `olmo_ifomega/grade.py` (venv_grade: the checker versions of his pod/olmo_ifcurve/setup.sh) on every samples
#      file -> stage0/regrade/<job>.jsonl;
#   5. a second work tree whose graded/ files are our re-grades, then cells.py rows + analyze.py again (work_regrade/).
# stage0_compare.py then writes stage0/repro.md.
# Usage: bash stage0_regrade.sh [--skip-grade]
set -euo pipefail
ROOT=/scratch/cowifornia/nd-rl/olmo_elicit
D=$ROOT/ext/nd-takehome
S=$ROOT/stage0
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
GPY=$ROOT/venv_grade/bin/python
export NLTK_DATA=$ROOT/venv_grade/nltk_data
NP=8
JOBS=$(ls $D/artifacts/olmo_ifcurve/rows | sed 's/\.csv$//')

mk_tree() {  # mk_tree <dir> <graded source: dan|ours>
  local W=$1 G=$2
  rm -rf "$W"; mkdir -p "$W"/artifacts/{olmo_ifcurve,olmo_ifomega,olmo_iffu}
  cp -r $D/olmo_ifcurve "$W"/olmo_ifcurve
  for x in olmo_ifomega olmo32 olmo_ifladder olmo_iffu; do ln -sfn $D/$x "$W"/$x; done
  for r in olmo_ifcurve olmo_ifomega olmo_iffu; do ln -sfn $D/artifacts/$r/samples "$W"/artifacts/$r/samples; done
  if [ "$G" = dan ]; then
    for r in olmo_ifcurve olmo_ifomega olmo_iffu; do ln -sfn $D/artifacts/$r/graded "$W"/artifacts/$r/graded; done
  else
    for r in olmo_ifcurve olmo_ifomega olmo_iffu; do mkdir -p "$W"/artifacts/$r/graded; done
    for j in $JOBS; do
      r=olmo_ifcurve; case $j in F_*_ifb) r=olmo_ifomega;; T4_s0750|T4b_s0750) r=olmo_iffu;; esac
      ln -sfn $S/regrade/$j.jsonl "$W"/artifacts/$r/graded/$j.jsonl
    done
  fi
  cp -r $D/artifacts/olmo_ifcurve/hand "$W"/artifacts/olmo_ifcurve/hand
  mkdir -p "$W"/artifacts/olmo_ifcurve/rows
}

run_tree() {  # rows + analyze in a tree
  local W=$1
  (cd "$W" && echo $JOBS | tr ' ' '\n' | xargs -P $NP -I{} $PY olmo_ifcurve/cells.py rows {} > "$W"/rows.log 2>&1
   $PY olmo_ifcurve/analyze.py --reps 10000 > "$W"/analyze.log 2>&1)
  echo "tree $W: $(grep -c rows "$W"/rows.log) rows files; analyze rc ok"
}

mkdir -p $S/regrade
mk_tree $S/work dan
run_tree $S/work

if [ "${1:-}" != "--skip-grade" ]; then
  for j in $JOBS; do
    r=olmo_ifcurve; case $j in F_*_ifb) r=olmo_ifomega;; T4_s0750|T4b_s0750) r=olmo_iffu;; esac
    echo "$j $D/artifacts/$r/samples/$j.jsonl.gz"
  done > $S/regrade/jobs.txt
  # grade.py inserts its own vendor/ and olmo32/ paths; -I keeps the working directory off sys.path
  (cd $S/work && cat $S/regrade/jobs.txt | xargs -P $NP -L 1 sh -c \
    "$GPY -I olmo_ifomega/grade.py \"\$1\" olmo_ifomega/problems.jsonl $S/regrade/\$0.jsonl > $S/regrade/\$0.log 2>&1")
  echo "re-graded $(ls $S/regrade/*.jsonl | wc -l) files"
fi
mk_tree $S/work_regrade ours
run_tree $S/work_regrade
echo "STAGE0_REGRADE_DONE $(date -u +%FT%TZ)"
