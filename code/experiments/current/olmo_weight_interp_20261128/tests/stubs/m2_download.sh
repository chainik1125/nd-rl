#!/usr/bin/env bash
# Stands in for metrics 2's pod/m2_download.sh in the chain tests. The fixture weights are either already in
# $MODELS_ROOT/<ckpt> (a folder that existed before the run) or are copied there from $STUB_HUB/<ckpt> (a download).
CK=$1; M=${MODELS_ROOT:?}; R=${M2_ROOT:?}/runs/download_$CK
[[ ${STUB_DOWNLOAD_FAIL:-} != "$CK" ]] || { echo "FETCH_FAILED"; exit 1; }
if [[ ! -d $M/$CK ]]; then
  [[ -d ${STUB_HUB:-/nonexistent}/$CK ]] || { echo "FETCH_FAILED"; exit 1; }
  mkdir -p "$M" && cp -r "$STUB_HUB/$CK" "$M/$CK" && : > "$M/$CK.lock"
fi
case $CK in s2300) echo 29884259e8c87406883e817e0628fa5ce9b529c3 ;; *) echo 97604023f2602a7dfe6d31b1fc229985e0002022 ;; esac > "$M/$CK/.sha"
touch "$M/$CK/.complete"
echo "{\"ckpt\": \"$CK\", \"ok\": true, \"sha256_checked\": true}" | tee "$R/download_verify.json"
