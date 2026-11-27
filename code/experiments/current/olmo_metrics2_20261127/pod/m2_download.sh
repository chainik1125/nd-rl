#!/usr/bin/env bash
# m2_download.sh <ckpt> (POD side; run as a job by `m2_pod.sh download`). Dan's pod/olmo_ifomega/fetch.sh at the pinned
# commit of olmo_ifcurve/checkpoints.csv into $MODELS/<ckpt>; skipped when .complete exists; then ALWAYS verified
# (commit, shard count, bytes; VERIFY_SHA=1 adds the Hub sha256). env: DAN_ROOT, MODELS_ROOT, VERIFY_SHA, TO_FETCH.
set -uo pipefail
CK=${1:?usage: m2_download.sh <ckpt>}
DAN=${DAN_ROOT:-/workspace/nd-takehome}; MODELS=${MODELS_ROOT:-/workspace/models}
P=$(cd "$(dirname "$0")" && pwd)
export HF_HOME=/workspace/.hf HF_HUB_DISABLE_TELEMETRY=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
row=$(awk -F, -v c="$CK" '$1==c' "$DAN/olmo_ifcurve/checkpoints.csv")
[ -n "$row" ] || { echo "UNKNOWN_CKPT $CK"; exit 2; }
IFS=, read -r _ REPO _ _ COMMIT _ <<< "$row"
if [ -f "$MODELS/$CK/.complete" ]; then echo "already complete: $MODELS/$CK (verifying)"; else
  echo "fetch $REPO@$COMMIT -> $MODELS/$CK"
  (cd "$DAN" && timeout --signal=TERM --kill-after=60 "${TO_FETCH:-2400}" \
     bash pod/olmo_ifomega/fetch.sh "$REPO" "$COMMIT" "$CK") || { echo "FETCH_FAILED rc=$?"; exit 1; }
fi
ARGS=(); [ "${VERIFY_SHA:-0}" = 1 ] && ARGS=(--sha256)
python3 "$P/m2_verify_weights.py" "$CK" --models "$MODELS" --dan "$DAN" "${ARGS[@]}" | tee "${M2_ROOT:-/workspace/m2}/runs/download_$CK/download_verify.json"
exit "${PIPESTATUS[0]}"
