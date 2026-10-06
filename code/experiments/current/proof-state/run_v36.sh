#!/usr/bin/env bash
# proof-state: validation-36 (textbook theorems) for the state-env arms (S, SH, SN) and the whole-proof control C0.
# Stage-1 and T1 round-8 checkpoints, seeds 0/1. Greedy (k=1, T=0) and k=256 at T=0.8. Lean 4.34.0 alone decides.
#
#   bash run_v36.sh [ckpt-name ...]      # default: all 16 in $RUN/ckpts; runs in parallel over GPUs 1-7
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/nd-takehome-dan                 # dan-pandori/nd-takehome @ add0a682 (dan_state-env)
RUN=${RUN:-$HOME/runs/proof-state}
PY=${PY:-$HOME/venvs/proof-state/bin/python}
IN=$CODE/targets/validation_36.jsonl
K=${K:-256}
GPUS=(${GPUS:-1 2 3 4 5 6 7})

export PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 ND_SAMPLE_PATH=fast LEAN_GATE_WORKERS=${LEAN_GATE_WORKERS:-16}
mkdir -p "$RUN/out" "$RUN/logs"

names=("$@")
[[ ${#names[@]} -gt 0 ]] || mapfile -t names < <(cd "$RUN/ckpts" && ls *.pt | sed 's/\.pt$//')

one() {  # $1 = ckpt name, $2 = gpu
  local n=$1 g=$2 ev=state_eval.py
  [[ $n == *C0* ]] && ev=eval_set.py
  cd "$CODE"
  for mode in greedy k$K; do
    local args=(--k 1 --temperature 0)
    [[ $mode == greedy ]] || args=(--k "$K" --temperature 0.8 --seed 0)
    [[ -s $RUN/out/${n}_$mode.json ]] && continue
    CUDA_VISIBLE_DEVICES=$g LEAN_GATE_LOG=$RUN/logs/${n}_$mode.gate.jsonl LEAN_GATE_DUMP=$RUN/out/${n}_$mode.dump.jsonl \
      "$PY" $ev --ckpt "$RUN/ckpts/$n.pt" --in "$IN" --lenfield min_lines_ub --batch 2048 "${args[@]}" \
      --out "$RUN/out/${n}_$mode.jsonl" --summary "$RUN/out/${n}_$mode.json" > "$RUN/logs/${n}_$mode.log" 2>&1
  done
}

i=0
for n in "${names[@]}"; do
  one "$n" "${GPUS[i % ${#GPUS[@]}]}" &
  i=$((i + 1))
  (( i % ${#GPUS[@]} == 0 )) && wait
done
wait
echo "done: $(ls "$RUN"/out/*.json | wc -l) summaries in $RUN/out"
