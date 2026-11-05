# steps.sh: sourced by go.sh on a base_saturation pod. Every step is bounded (timeout) and logged;
# watchdog.sh watches the running step. Layout under /workspace/sat (runpod_copy.sh puts it there):
#   code/{s1,cpt,l2,l3,rd}  git-archive snapshots (COMMIT inside): Stage 1 cd089700, CPT 32bc1da,
#                           GRPO L2 fa28f6eb, GRPO L3 f35d3daf, reads cce4dfc
#   runs/                   stage1/S1g3_s0/model.pt, data/S1g3.pkl, heldout.jsonl, pools/...
#   ckpts/                  other inits (CPT_s0; C and D2 when the coordinator copies them)
#   out/, logs/, status/, events.log
W=/workspace/sat
PY=/workspace/venv/bin/python
RUNS=$W/runs
CODE=$W/code
export ND_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ND_LEAN=/root/.elan/bin/lean TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=0
mkdir -p "$W/tmp" "$W/logs" "$W/out" "$W/status"
SET=$RUNS/pools/charles_1002/charles_clean.jsonl  # Charles149 (md5 b104d60f...), read only
READ_FLAGS=(--budgets 64 --arms plain --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10)

ev() { echo "$(date -u +%FT%TZ) $(cat $W/status/alias 2>/dev/null || hostname) $*" >> "$W/events.log"; }

cpu_quota() {  # the container's CPU quota (cgroup v2 cpu.max), else nproc
  local q p
  if read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then
    echo $(( (q + p - 1) / p ))
  else nproc; fi
}

# step NAME DIR TIMEOUT_S EXPECTED_MIN STALL_MIN CMD...: run CMD in DIR, bounded, logged to
# logs/NAME.log; status/NAME.done on rc 0 (a done step is skipped), status/NAME.fail otherwise.
step() {
  local n=$1 dir=$2 to=$3 exp=$4 stall=$5
  shift 5
  if [ -f "$W/status/$n.done" ]; then ev "$n: done before, skipped"; return 0; fi
  local log=$W/logs/$n.log
  ev "$n: start (timeout ${to}s, expected ${exp} min)"
  echo "=== $(date -u +%FT%TZ) $*" >> "$log"
  (cd "$dir" && exec timeout --kill-after=120 "$to" "$@") >> "$log" 2>&1 &
  local pid=$! t0
  t0=$(date +%s)
  echo "$n $pid $log $t0 $exp $stall" > "$W/status/current"
  wait $pid
  local rc=$?
  rm -f "$W/status/current"
  local took=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    touch "$W/status/$n.done"; ev "$n: ok in ${took} min"
  else
    touch "$W/status/$n.fail"; ev "FAIL $n: rc $rc after ${took} min (124 = timeout); tail: $(tail -3 "$log" | tr '\n' '|' | cut -c1-300)"
  fi
  return $rc
}

seeds() { case $1 in *-*) seq "${1%-*}" "${1#*-}" ;; *) echo "$1" | tr ',' ' ' ;; esac; }

# read CKPT READ SEEDS: Charles149 chunks of k64 with --seed i (the paper's read; cpt_control's reads.py)
read_chunks() {
  local ck=$1 rd=$2 i d lgw
  [ -s "$ck" ] || { ev "FAIL read $rd: no checkpoint $ck"; return 1; }
  lgw=$(cpu_quota); [ "$lgw" -gt 16 ] && lgw=16
  for i in $(seeds "$3"); do
    d=$W/out/reads/$rd/seed$i
    if [ -f "$d/rows_plain_B64.jsonl" ] && [ -f "$d/summary.json" ]; then continue; fi
    [ -d "$d" ] && mv "$d" "$d.crashed_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$(dirname "$d")"
    step "rd_${rd}_c$i" "$CODE/rd" 2700 6 15 env ND_GIT_SHA="$(cat $CODE/rd/COMMIT)" \
      LEAN_GATE_WORKERS="$lgw" OMP_NUM_THREADS=3 \
      "$PY" -m code.thinking.eval --ckpt "$ck" --sets "$SET" --out "$d" --k 64 --batch 1024 \
      "${READ_FLAGS[@]}" --seed "$i" || return 1
  done
  ev "read $rd seeds $3: all chunks ok"
}

train_A() {  # arm A: S1g3_s0, Stage 1's recipe from its floor to lr 0 over 1,200 steps
  step train_A "$CODE/s1" 3600 10 10 env ND_GIT_SHA="$(cat $CODE/s1/COMMIT)" \
    "$PY" -m code.experiments.current.base_saturation_20261105.anneal \
    --base "$RUNS/stage1/S1g3_s0/model.pt" --data "$RUNS/data/S1g3.pkl" \
    --heldout "$RUNS/heldout.jsonl" --out "$W/out/A_s0"
}

train_B() {  # arm B: cpt.py exactly as CPT_s0 (lr 1.5e-4, cosine to 30%, record order seed 0), STEPS steps
  local s=$1 to
  to=$(( 1800 + s * 9 ))
  step "train_B$s" "$CODE/cpt" "$to" $(( s * 5 / 60 + 5 )) 10 env ND_GIT_SHA="$(cat $CODE/cpt/COMMIT)" \
    CIRCUITS_RUNS="$RUNS" CIRCUITS_ALLOWED_GPUS=0 \
    "$PY" -m code.experiments.current.circuit_diffing_20261101.cpt --gpu 0 --lr 1.5e-4 \
    --steps "$s" --base_seed 0 --out_dir "$W/out/B$s"
}

train_C() {  # arm C: Stage 1 from scratch, seed 0, 48,000 steps, cosine to lr 0
  step train_C "$CODE/s1" 10800 50 10 env ND_GIT_SHA="$(cat $CODE/s1/COMMIT)" \
    "$PY" -m code.experiments.current.base_saturation_20261105.stage1_to0 \
    --data "$RUNS/data/S1g3.pkl" --heldout "$RUNS/heldout.jsonl" --out "$W/out/C_s0" \
    --seed 0 --best_steps 48000
}

# rl_chain INIT TAG: the paper's GRPO recipe, grpo_think_vs_s0_L2 (fa28f6eb) then
# grpo_think_vs_mb_kkl0_s0_L3 (f35d3daf), each stage's argv exactly as its args.json records it,
# with only --init / --name / --out / --kl_anchor_ref (each stage's own init, as the recipe) and
# the pool paths changed. Seed 0. Each stage: timeout 6 h (line.sh's 21600 s).
rl_chain() {
  local init=$1 tag=$2 O=$W/out/grpo
  local n2=grpo_think_vs_${tag}_s0_L2 n3=grpo_think_vs_mb_kkl0_${tag}_s0_L3
  local P=$RUNS/pools
  [ -s "$init" ] || { ev "FAIL rl $tag: no init $init"; return 1; }
  mkdir -p "$O"
  local common=(--iters 39 --prompts 256 --group 16 --batch 1024 --lr 1.5e-4)
  local tail2=(--overlong fail --save_every 39 --max_steps 96 --seed 0 --keep_rolling 100 --effort
    --budgets 0,64,256 --canary "$P/canary_long.jsonl" --canary_n 1000 --canary_every 1
    --rollback_on_alarm --max_rollbacks 3 --ban_restore_check --ban_stop 0.005)
  step "$n2" "$CODE/l2" 21600 85 25 env ND_GIT_SHA="$(cat $CODE/l2/COMMIT)" \
    "$PY" -m code.thinking.grpo --init "$init" --name "$n2" --out "$O/$n2" \
    --targets "$P/curriculum/L2/pool.jsonl" "${common[@]}" "${tail2[@]}" \
    --kl_anchor_coef 0.02 --kl_anchor_ref "$init" --monitor --arm logical --max_rej 0 \
    --rej_penalty 0.1 || return 1
  local ck2=$O/$n2/ckpts/${n2}_it39.pt
  [ -s "$ck2" ] || { ev "FAIL $n2: no it39 checkpoint"; return 1; }
  ev "$n2 guard: $(grep -E '^guard it' "$W/logs/$n2.log" | tail -1 | cut -c1-120)"
  step "$n3" "$CODE/l3" 21600 125 25 env ND_GIT_SHA="$(cat $CODE/l3/COMMIT)" \
    "$PY" -m code.thinking.grpo --init "$ck2" --name "$n3" --out "$O/$n3" \
    --targets "$P/curriculum/L3/pool.jsonl" "${common[@]}" \
    --lr_schedule wsd --decay_frac 0.2 --min_lr_frac 0.3 --decay_shape sqrt --unlikely_beta 0.25 \
    "${tail2[@]}" --kl_anchor_coef 0 --kl_anchor_ref "$ck2" --monitor --arm logical --max_rej 0 \
    --rej_penalty 0.1 --targets_extra "$P/curriculum_3p/L3_3p.jsonl" --targets_extra_frac 0.33 \
    || return 1
  [ -s "$O/$n3/ckpts/${n3}_it39.pt" ] || { ev "FAIL $n3: no it39 checkpoint"; return 1; }
  ev "$n3 guard: $(grep -E '^guard it' "$W/logs/$n3.log" | tail -1 | cut -c1-120)"
}
rl_final() { echo "$W/out/grpo/grpo_think_vs_mb_kkl0_$1_s0_L3/ckpts/grpo_think_vs_mb_kkl0_$1_s0_L3_it39.pt"; }
