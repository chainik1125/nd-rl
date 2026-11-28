#!/usr/bin/env bash
# wi_rm.sh <a050|a075|a025>   (POD side; its own job after the cell, in the cells chain and under the GPU lock,
# so it never meets a write job on the CPU lock; it uses no GPU)
# Deletes /workspace/wi/models/interp_<tag> only when that cell's job is DONE and its count.json says complete with 192 graded
# rows. Anything else leaves the folder in place and exits 11 (the chain goes on). It can only ever remove a folder named
# interp_<tag> that carries interp_write.py's marker; dpo, s2300 and the tokenizer folder cannot be named.
# The writer's MANIFEST.json and the check report already live in runs/write_<tag>/out/ and stay.
# Exit: 0 removed (or already gone)   11 kept: the cell is not complete   2 usage or a folder that is not ours
set -uo pipefail
TAG=${1:?usage: wi_rm.sh <a050|a075|a025>}
[[ $TAG =~ ^a(025|050|075)$ ]] || { echo "REFUSED: $TAG is not a merged cell"; exit 2; }
WI=${M2_ROOT:-/workspace/wi}; MERGED=${WI_MERGED_ROOT:-$WI/models}
D=$MERGED/interp_$TAG; C=$WI/runs/cell_$TAG; OUT=$WI/runs/rm_$TAG/out
mkdir -p "$OUT"
utc() { date -u +%FT%TZ; }
if [[ ! -f $C/state/DONE ]]; then echo "KEPT $D: cell_$TAG has no DONE marker"; exit 11; fi
python3 - "$C/out/count.json" <<'PY' || { echo "KEPT $D: count.json does not say complete with 192 graded rows"; exit 11; }
import json, sys
c = json.load(open(sys.argv[1]))
ok = c.get("complete") is True and c.get("graded_rows") == 192 == c.get("planned") and c.get("n_missing") == 0
sys.exit(0 if ok else 1)
PY
if [[ ! -d $D ]]; then echo "$(utc) already gone: $D" | tee "$OUT/removed.txt"; exit 0; fi
[[ -f $D/WI_WRITE_OK && ! -L $D && $(basename "$D") == interp_$TAG ]] || { echo "REFUSED: $D does not carry the write marker"; exit 2; }
[[ -f $WI/runs/write_$TAG/out/MANIFEST.json && -f $WI/runs/write_$TAG/out/check.json ]] || { echo "REFUSED: write report of $TAG is not in runs/write_$TAG/out"; exit 2; }
B=$(du -sb "$D" | cut -f1)
rm -rf -- "$D" || { echo "rm failed"; exit 1; }
echo "$(utc) removed $D ($B bytes) after cell_$TAG was counted complete" | tee "$OUT/removed.txt"
exit 0
