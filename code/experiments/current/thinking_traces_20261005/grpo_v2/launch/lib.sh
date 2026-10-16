# helpers for the 2026-10-07 follow-up queue (sourced)
R=/scratch/cowifornia/nd-rl/think/runs; L=$R/logs; F=$R/followup_20261007; G=$R/restart3_grpofix
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp
say() { echo "$(date -u +%H:%M)Z $*"; }
waitpid() { while kill -0 $1 2>/dev/null; do sleep 60; done; }
waitline() { until grep -q "$2" $1 2>/dev/null; do sleep 120; done; }
waitfree() {  # GPU: no compute process and < 500 MiB, for up to 3 h
  local i
  for i in $(seq 1 180); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 500 ] && \
      [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1)" ] && return 0; sleep 60; done
  say "GPU $1 not free after 3 h: stopping this pipeline"; exit 1; }
train() {  # GPU NAME INIT EXPECTED_H [grpo args]: fixed-code GRPO to iteration 78, with a watchdog
  local g=$1 n=$2 init=$3 h=$4; shift 4; waitfree $g
  ( cd $G && bash launch.sh $g $n $init --iters 78 --keep_rolling 100 "$@" > $L/f_$n.log 2>&1 ) & local p=$!
  sleep 5; say "launched $n on GPU $g (pid $p): $*"
  python3 $R/watchdog.py $n:$p:$g:$L/f_$n.log:$h >> $L/watchdog_followup.log 2>&1 &
  wait $p; say "$n exit $?: $(grep -E '^it [0-9]+:' $L/f_$n.log | tail -1 | cut -c1-160)"; }
evdone() {  # OUT BUDGETS: every budget's rows file exists (summary.json is rewritten after each budget,
  local b    # so its presence alone does not mean the eval finished)
  for b in ${2//,/ }; do [ -f $1/rows_plain_B$b.jsonl ] || return 1; done; }
ev() {  # GPU NAME CKPT SETS BUDGETS K MIN_THINK: one eval call
  local out=$R/eval/followup/$2; evdone $out $5 && { say "$2 already done"; return; }; waitfree $1; local t=$(date +%s)
  CUDA_VISIBLE_DEVICES=$1 timeout 5400 code/.venv/bin/python -m code.thinking.eval --ckpt $3 --sets $4 --out $out --k $6 \
    --budgets $5 --min_think $7 --arms plain --batch 1024 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10 \
    > $L/eval_f_$2.log 2>&1
  say "$2 exit $? $(( $(date +%s) - t ))s: $(grep -E '^\[plain' $L/eval_f_$2.log | tr '\n' ' ')"; }
