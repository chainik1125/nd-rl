#!/bin/zsh
# per-level sem probe over a prompt file: sem_probe.sh PROMPTS.txt MAXBODY MAXMEMO > log
while IFS= read -r p; do
  [[ -z "$p" || "$p" == \#* ]] && continue
  ~/repos/nd-takehome/.venv/bin/python minlen_sem.py "$p" $2 $3
done < $1
