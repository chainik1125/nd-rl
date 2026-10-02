#!/usr/bin/env bash
# Usage: fill_briefs.sh <DEADLINE_UTC e.g. "2026-09-23 04:15"> <HOURS>
set -euo pipefail
AB=/scratch/cowifornia/nd-rl/ab; D="$1"; H="$2"
BASE='7 of 1,610 (pretrain checkpoint, seed 0; 5 with seed 1). One 600 s run of the unmodified seed rl.py scored 238 (L7 47, L8 89, L9 102); that is the number to beat'
fill() { sed -e "s/{{DEADLINE_UTC}}/$D/g" -e "s/{{HOURS}}/$H/g" -e "s/{{BASELINE}}/$BASE/g" -e "s/{{WORKER}}/$3/g" "$1" > "$2"; }
fill $AB/common/briefs/gear_worker.md   $AB/gear/w1/BRIEF.md w1
fill $AB/common/briefs/gear_worker.md   $AB/gear/w2/BRIEF.md w2
fill $AB/common/briefs/orch_executor.md $AB/orch/exec/BRIEF.md exec
fill $AB/common/briefs/orch_reviewer.md $AB/orch/review/BRIEF.md review
grep -c "{{" $AB/gear/w1/BRIEF.md $AB/orch/exec/BRIEF.md $AB/orch/review/BRIEF.md || true
echo filled
