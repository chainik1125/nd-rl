#!/usr/bin/env bash
# pod_run.sh <test> [extra grpo flags...]: ONE guard-stability GRPO run, ON THE POD (copied to /workspace/gs/).
# Mirrors the guarded baseline's command (author_tags/runs/tA_p10_s2/seg1/cmd.json: tag A, base tagAB_p10_s2, GRPO
# seed 2; env BASE_SEED / SEED pick tA_p10_s0 / s1's base and seed, which differ only there) with pod paths, except: --iters 60 (the schedule's horizon), no --snap_iters (model-only snapshots for the
# reads, which these tests skip), and no --rollback_on_alarm / --max_rollbacks (the caller adds them back for the
# guarded variants). The environment is tag_line.Seg.env()'s. Bounded: timeout 80 min (75 min of GRPO + start-up),
# then SIGTERM, KILL 60 s later. Writes <out>/cmd.json, <out>/train.log, <out>/exit.json.
set -uo pipefail
test=${1:?usage: pod_run.sh <test> [extra grpo flags...]}
shift
AT=/scratch/cowifornia/nd-rl/author_tags
KD=/scratch/cowifornia/nd-rl/knockout_danbase
CODE=$KD/snap/f35d3daf_danbase
EXP=/scratch/cowifornia/nd-rl/worktrees/two-authors/code/experiments/current/author_tags_20261110
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
BASE_SEED=${BASE_SEED:-2}  # the base stage1_tagAB_p10_s<BASE_SEED>; GRPO --seed SEED (default: the same)
SEED=${SEED:-$BASE_SEED}
BASE=$AT/bases/stage1_tagAB_p10_s$BASE_SEED.pt
WORKERS=${LEAN_GATE_WORKERS:-8}
OUT=$AT/stability/$test/seg1
LIMIT=${LIMIT_MIN:-80}
[[ -e $OUT/metrics.jsonl ]] && { echo "refusing: $OUT/metrics.jsonl exists"; exit 2; }
mkdir -p "$OUT"
args=(--init "$BASE" --name "$test" --out "$OUT"
  --targets "$KD/pools/dan_rl_targets.jsonl"
  --iters 60 --prompts 256 --group 16 --batch 1024 --lr 1.5e-4 --unlikely_beta 0.25 --overlong fail
  --save_every 400 --save_rolling 5 --keep_rolling 4 --max_steps 48 --max_action 256 --seed "$SEED" --budgets 0
  --canary "$KD/pools/canary_long.jsonl" --canary_n 1000 --canary_every 1
  --kl_anchor_coef 0 --kl_anchor_ref "$BASE" --monitor --arm logical --max_rej 0 --rej_penalty 0.1
  "$@")
cmd=("$PY" "$EXP/tag_grpo.py" --tag A --variant p10 -- "${args[@]}")
"$PY" - "$OUT/cmd.json" "$CODE" "$LIMIT" "${cmd[@]}" <<'EOF'
import json, sys, time
out, cwd, limit, cmd = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump({"cmd": cmd, "cwd": cwd, "limit_min": limit, "utc": utc}, open(out, "w"), indent=1)
EOF
export ND_OFFLINE=1 ND_REGISTRY_SYNC=0 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=$WORKERS OMP_NUM_THREADS=8 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LEAN_GATE_DUMP=$OUT/lean_dump.jsonl \
  ND_REGISTRY_DIR=$OUT/registry PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
  AUTHOR_TAG=A
cd "$CODE"
t0=$(date +%s)
echo "start $(date -u +%FT%TZ) $test base s$BASE_SEED seed $SEED lean workers $WORKERS limit ${LIMIT} min" > "$OUT/train.log"
timeout --kill-after=60 "${LIMIT}m" "${cmd[@]}" >> "$OUT/train.log" 2>&1
rc=$?
echo "end $(date -u +%FT%TZ) rc $rc ($(( $(date +%s) - t0 )) s)" >> "$OUT/train.log"
printf '{"rc": %d, "secs": %d, "utc": "%s", "timed_out": %s}\n' "$rc" "$(( $(date +%s) - t0 ))" \
  "$(date -u +%FT%TZ)" "$([[ $rc == 124 || $rc == 137 ]] && echo true || echo false)" > "$OUT/exit.json"
