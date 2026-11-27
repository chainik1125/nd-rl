#!/bin/bash
# Amendment 6(a) step 2 analysis (SERRANO, CPU, <= 3 cores, nice): builds the union runs tree v2/runs_a6s2 (symlinks
# only) from batch_old and batch_r (rp-om2) and batch_a6s2 (the sampling pod), with every involved cell.json, then runs
# analyze_cells.py with the contrast list fixed in PREREG Amendment 8 and a6_reading.py. Writes v2/a6/step2_*.
#   bash a6s2_analyze.sh <sampling pod alias>
set -euo pipefail
A=${1:?alias of the sampling pod}
X=$(cd "$(dirname "$0")" && pwd)
V2=${V2:-/scratch/cowifornia/nd-rl/olmo_elicit/v2}
PY=${PY:-/scratch/cowifornia/venvs/olmo-edl/bin/python}
U=$V2/runs_a6s2/dpo
S=$V2/runs_$A/dpo
[ -f "$S/sample/batch_a6s2/dpo__final25__A16__n64__r1p/graded.jsonl" ] || { echo "no fetched batch_a6s2 under $S"; exit 2; }
mkdir -p "$U/sample" "$U/gate" "$U/train/cells"
ln -sfn "$V2/runs_rp-om2/dpo/sample/batch_old" "$U/sample/batch_old"
ln -sfn "$V2/runs_rp-om2/dpo/sample/batch_r" "$U/sample/batch_r"
ln -sfn "$S/sample/batch_a6s2" "$U/sample/batch_a6s2"
for b in batch_old batch_r; do ln -sfn "$V2/runs_rp-om2/dpo/gate/$b.json" "$U/gate/$b.json"; done
ln -sfn "$S/gate/batch_a6s2.json" "$U/gate/batch_a6s2.json"
for c in "$V2"/runs_union/dpo/train/cells/*; do ln -sfn "$(readlink -f "$c")" "$U/train/cells/$(basename "$c")"; done
for c in "$V2"/runs_rp-om4/dpo/train/cells/*; do ln -sfn "$c" "$U/train/cells/$(basename "$c")"; done
ln -sfn "$V2/runs_rp-om3/dpo/train/cells/dpo__final25__A16__n64__r1p_s523373504" "$U/train/cells/dpo__final25__A16__n64__r1p_s523373504"
cd "$X"
CON=$($PY a6_reading.py --print-contrasts-arg)
OMP_NUM_THREADS=3 nice -n 10 timeout 3600 "$PY" analyze_cells.py --runs "$V2/runs_a6s2" --ckpt dpo \
  --out "$V2/a6/step2_summary" --contrasts-out "$V2/a6/step2_contrasts" --contrasts "$CON"
OMP_NUM_THREADS=1 nice -n 10 "$PY" a6_reading.py --contrasts "$V2/a6/step2_contrasts.json" --out "$V2/a6/step2_reading.json"
