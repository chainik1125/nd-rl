#!/bin/sh
# Launch command of record for eval_dan_best_state (proposal.md). Run from the repo root.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SCR=/private/tmp/claude-501/-Users-cyu-Desktop-github-nd-rl/0e8fdc82-572d-4fd1-ba25-ff48dbf29acf/scratchpad
E=$ROOT/experiments/textbook-problem/eval_dan_best_state
RUN=$E/run_best12_r8
T72=$ROOT/docs/textbook-problem/textbook72/dan_eval
PY=$ROOT/code/.venv/bin/python
cd $SCR/fork_dan || exit 1
for s in 0 1 2; do
  mkdir -p $RUN/s$s $T72/s$s
  for set in v0 v1 hand14 batch3 lean_v2 derived_plain textbook72; do
    if [ "$set" = textbook72 ]; then IN=$T72/textbook72.jsonl; OUT=$T72/s$s; else IN=$E/inputs/$set.jsonl; OUT=$RUN/s$s; fi
    echo "$(date -u +%FT%TZ) == s$s $set"
    LEAN_GATE_LOG=$RUN/s$s/lean_gate.jsonl $PY state_eval.py --ckpt $SCR/dan_ckpts/la_T1_best12_s${s}_r8.pt \
      --in $IN --out $OUT/$set.jsonl --summary $OUT/$set.summary.json --k 32 --temperature 1.0 --batch 256 --seed 0 2>&1 | grep -v Warning | grep -E "SOLVED|^env |Traceback|Error"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE"
