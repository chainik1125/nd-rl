#!/usr/bin/env bash
# wi_release.sh --fetched   (POD side; run once by `wi_pod.sh release`, which first requires an md5-checked fetch)
# Ends this run's use of the shared pod. It touches only what this run made, and stops or restarts nothing:
#   refuses (exit 3) while any job or chain of this run is running;
#   removes the merged checkpoints still under this run's own folder (interp_a050, interp_a075, interp_a025);
#   removes /workspace/models/s2300 only if this run downloaded it (runs/download_s2300/out/origin.txt says ABSENT)
#     and it is the pinned commit; otherwise it is left and the reason is printed;
#   never touches dpo, the tokenizer folder, Dan's checkouts or anything else outside /workspace/wi;
#   writes /workspace/wi/GPU_FREE (one line) when no GPU memory is in use. If memory is in use (exit 4) it writes
#     no marker and prints the processes; with none of this run's jobs running they belong to someone else.
# Report: /workspace/wi/meta/release.txt (also printed). Exit: 0 released, 2 usage, 3 still running, 4 GPU in use.
set -uo pipefail
[[ ${1:-} == --fetched ]] || { echo "usage: wi_release.sh --fetched   (only after an md5-checked fetch; it deletes weights)"; exit 2; }
WI=${M2_ROOT:-/workspace/wi}; MODELS=${MODELS_ROOT:-/workspace/models}; MERGED=${WI_MERGED_ROOT:-$WI/models}
S2300_COMMIT=29884259e8c87406883e817e0628fa5ce9b529c3
OUT=$WI/meta/release.txt
utc() { date -u +%FT%TZ; }
say() { echo "$*" | tee -a "$OUT"; }
cd "$WI" || exit 2
st=$(python3 pod/m2_status.py --root "$WI" 2>/dev/null) || { echo "RELEASE_REFUSED: cannot read this run's status"; exit 3; }
if grep -q " state=RUNNING " <<< "$st"; then
  echo "RELEASE_REFUSED: a job or chain of this run is still running"; grep " state=RUNNING " <<< "$st" | cut -c1-120; exit 3
fi
mkdir -p "$WI/meta"; : > "$OUT"; rm -f "$WI/GPU_FREE"
say "release of the weight-interpolation run, $(utc)"
for t in a050 a075 a025; do
  D=$MERGED/interp_$t
  if [[ -d $D && ! -L $D ]]; then B=$(du -sb "$D" | cut -f1); rm -rf -- "$D" && say "removed $D ($B bytes)"; fi
done
rmdir "$MERGED" 2> /dev/null || true
D=$MODELS/s2300; O=$WI/runs/download_s2300/out/origin.txt
if [[ ! -e $D ]]; then say "s2300: not present"
elif [[ -L $D || ! -d $D ]]; then say "kept $D: not a plain folder"
elif ! grep -q "^ABSENT " "$O" 2> /dev/null; then say "kept $D: it existed before this run (or there is no origin record)"
elif [[ $(cat "$D/.sha" 2> /dev/null) != "$S2300_COMMIT" ]]; then say "kept $D: it is not the pinned commit"
else B=$(du -sb "$D" | cut -f1); rm -rf -- "$D" && rm -f -- "$D.lock" && say "removed $D ($B bytes)"; fi
for k in dpo tok; do
  [[ -d $MODELS/$k ]] && say "left in place: $MODELS/$k ($(du -sb "$MODELS/$k" | cut -f1) bytes, commit $(cat "$MODELS/$k/.sha" 2> /dev/null || echo unknown))"
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | tr '\n' ' ')
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2> /dev/null | tr '\n' ';')
say "GPU memory in use (MiB): ${used:-unknown}; compute processes: ${apps:-none}"
busy=0
for u in $used; do [[ $u =~ ^[0-9]+$ && $u -ge ${GPU_FREE_MIB:-2000} ]] && busy=1; done
[[ -n $used ]] || busy=1
if [[ $busy == 1 ]]; then
  say "GPU_NOT_FREE $(utc): memory is in use or could not be read; no job of this run is running and nothing was stopped"
  exit 4
fi
line="GPU_FREE $(utc) weight-interpolation run finished: none of its jobs is running; dpo is at $MODELS/dpo"
echo "$line" > "$WI/GPU_FREE"
say "$line"
exit 0
