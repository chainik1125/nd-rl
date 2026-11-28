#!/usr/bin/env bash
# wi_download.sh <dpo|s2300>   (POD side; the command of the job download_<ckpt>)
# Records, once, whether the checkpoint folder already existed before this run touched it, then runs metrics 2's
# unedited pod/m2_download.sh (Dan's fetch at the pinned commit into /workspace/models/<ckpt>, then commit, shard count,
# bytes and every shard's sha256 against the hub). The pod is shared: pod/wi_release.sh deletes s2300 at the end only if
# this record says ABSENT, that is, only if this run downloaded it. dpo is never deleted by this run.
set -uo pipefail
CK=${1:?usage: wi_download.sh <dpo|s2300>}
[[ $CK == dpo || $CK == s2300 ]] || { echo "this run downloads dpo and s2300 only"; exit 2; }
WI=${M2_ROOT:-/workspace/wi}; MODELS=${MODELS_ROOT:-/workspace/models}; O=$WI/runs/download_$CK/out
mkdir -p "$O"
if [[ ! -f $O/origin.txt ]]; then
  if [[ -e $MODELS/$CK ]]; then echo "PREEXISTING $(date -u +%FT%TZ) $MODELS/$CK"; else echo "ABSENT $(date -u +%FT%TZ) $MODELS/$CK"; fi > "$O/origin.txt"
fi
echo "origin: $(cat "$O/origin.txt")"
cd "$WI" || exit 2
exec env M2_ROOT="$WI" VERIFY_SHA=1 bash pod/m2_download.sh "$CK"
