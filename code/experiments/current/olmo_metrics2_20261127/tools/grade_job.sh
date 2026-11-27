#!/usr/bin/env bash
# grade_job.sh <samples.jsonl.gz> <graded.jsonl> [<problems.jsonl>]
# Grades a samples file with Dan's grader (olmo_ifomega/grade.py <samples> <problems> <graded>) without writing into his
# tree (PYTHONDONTWRITEBYTECODE, python -I). Problems default: <dan>/olmo_ifomega/problems.jsonl.
#   on the pod (inside a job, after sampling):  bash tools/grade_job.sh runs/<job>/samples.jsonl.gz runs/<job>/graded.jsonl
#   on serrano (re-grade):  same call; picks venv_grade (the checker versions of Dan's pod setup) when /workspace/nd-takehome
#                           is absent. env overrides: DAN_ROOT, GRADE_PY, NLTK_DATA.
set -euo pipefail
S=${1:?usage: grade_job.sh <samples.jsonl.gz> <graded.jsonl> [<problems.jsonl>]}
G=${2:?graded output path}
if [[ -d /workspace/nd-takehome/olmo_ifomega ]]; then
  DAN=${DAN_ROOT:-/workspace/nd-takehome}; PY=${GRADE_PY:-python3}
else
  OE=/scratch/cowifornia/nd-rl/olmo_elicit
  DAN=${DAN_ROOT:-$OE/ext/nd-takehome}; PY=${GRADE_PY:-$OE/venv_grade/bin/python}
  export NLTK_DATA=${NLTK_DATA:-$OE/venv_grade/nltk_data}
fi
Pb=${3:-$DAN/olmo_ifomega/problems.jsonl}
[[ -s $S ]] || { echo "no samples file $S"; exit 1; }
[[ -s $Pb ]] || { echo "no problems file $Pb"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$(dirname "$G")"
tmp="$G.tmp.$$"
# A job killed by its timeout leaves the gzip without its end marker; grade the complete rows of a repaired copy
# (tools/gz_repair.py; the original file is never modified). Orchestrator addition, 2026-11-27.
if ! gzip -t "$S" 2>/dev/null; then
  R="${S%.jsonl.gz}_complete.jsonl.gz"
  python3 "$(dirname "$0")/gz_repair.py" "$S" "$R" || { echo "gz repair failed for $S"; exit 1; }
  echo "GRADING the repaired copy $R (the original stream is incomplete)"
  S=$R
fi
"$PY" -I "$DAN/olmo_ifomega/grade.py" "$S" "$Pb" "$tmp"
mv "$tmp" "$G"
echo "GRADED $(wc -l < "$G") rows -> $G"
