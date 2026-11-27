#!/usr/bin/env bash
# m2_setup.sh (POD side; started by m2_pod.sh setup): runs the 2026-11-25 pod/oe_setup.sh UNEDITED in content, with its
# two path prefixes redirected: the copy of oe_setup.sh at pod/oe_setup.sh is copied to meta/oe_setup.m2.sh with
# `sed s#/workspace/oe#/workspace/m2#g` (the only places are its meta and runs directories; the diff is kept in
# meta/oe_setup.sed.diff) and that copy runs. Everything else (Dan's pinned clones at /workspace/nd-takehome and
# /workspace/nd-answeronly, vLLM 0.11.0 etc., models under /workspace/models) is as before and idempotent, so a pod that
# already holds the other run's set-up and weights is reused. Writes /workspace/m2/meta/setup.done.
set -uo pipefail
M2=${M2_ROOT:-/workspace/m2}
mkdir -p "$M2/meta" "$M2/runs" /workspace/models
sed "s#/workspace/oe#$M2#g" "$M2/pod/oe_setup.sh" > "$M2/meta/oe_setup.m2.sh"
diff "$M2/pod/oe_setup.sh" "$M2/meta/oe_setup.m2.sh" > "$M2/meta/oe_setup.sed.diff"
[ -s "$M2/meta/oe_setup.sed.diff" ] || { echo "SED_NOOP: oe_setup.sh has no /workspace/oe path"; exit 1; }
bash "$M2/meta/oe_setup.m2.sh"
