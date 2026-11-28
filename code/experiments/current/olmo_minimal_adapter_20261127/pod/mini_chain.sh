#!/usr/bin/env bash
# mini_chain.sh <ckpt> <stages> (POD side; started by mini_pod.sh under setsid + nohup + timeout).
# One checkpoint of the minimal-adapter experiment. <stages> is a comma list, run in the order given, from
#   download         Dan's pod/olmo_ifomega/fetch.sh at the commit pinned in his checkpoints.csv, verified (commit, shard
#                    count, bytes; VERIFY_SHA=1 also the Hub's LFS sha256). Always runs first. Weights are KEPT unless
#                    DELETE_WEIGHTS=1.
#   train            mini_train.py run --model <weights> --ckpt <ck> --plan <plan> --root <pools> --out runs/<ck>/train
#                    ${TRAIN_ARGS}   (PLAN = plan path relative to $MINI_ROOT, e.g. plans/stage1.json)
#   gate:<batch>     lora_gate.py on the staged cells of <batch> ($MINI_ROOT/adapters/<batch>/<cell>/...): vLLM applies each
#                    adapter as PEFT does? -> runs/<ck>/gate/<batch>.json. Per-cell FAIL verdicts do NOT fail the stage (the
#                    stage is DONE with a note; the cells are simply not sampled); a missing JSON or an engine error does.
#   sample:<batch>   multi_sample.py (Dan's sampler, several adapters in one engine) on the cells that did not FAIL the gate,
#                    then Dan's grade.py per cell -> runs/<ck>/sample/<batch>/<cell>/{samples.jsonl.gz,graded.jsonl,meta.json}
#   plain:<name>     Dan's olmo_ifladder/sample.py UNCHANGED, no LoRA, on the problems file pushed from serrano
#                    ($MINI_ROOT/stage/problems_<name>.jsonl, pids_<name>.txt, seed_<name>.txt; --n 0 --chunk 4 --order given --tp 1),
#                    then his grade.py on the same problems file (it scores `prompt` + kwargs, not `prompt_sample`)
#                    -> runs/<ck>/plain_<name>/{samples.jsonl.gz,meta.json,graded.jsonl}
# State/log files use the stage name with ':' -> '_' (gate:b1 -> gate_b1): state/<stage>.{RUNNING,DONE,FAILED},
# logs/<stage>.log, state/<stage>.expected (draws expected, sample stages). Every stage is resumable: a stage with
# state/<stage>.DONE is skipped, the sampler resumes its own files. Heartbeat (state/heartbeat: "<epoch> <stage>") and a GPU
# line (logs/gpu.log) every 60 s. One chain per pod (flock on $MINI_ROOT/gpu.lock). Every command runs under `timeout`.
# env: MINI_ROOT / DAN_ROOT / MODELS_ROOT override the /workspace paths; PY (python, default python); PLAN, TRAIN_ARGS,
#      DELETE_WEIGHTS=1, VERIFY_SHA=1, TO_DOWNLOAD / TO_TRAIN / TO_GATE / TO_SAMPLE / TO_GRADE (seconds), DRY_RUN=1 (print only).
#      On a pod shared with another job (PREREG Amendment 9; both unset = the behaviour of every earlier run):
#        ALSO_LOCK=<file>  the other job's GPU lock file. If it exists it is locked too for the whole chain (opened for
#                          reading; never created or written), so that job refuses to start while this chain runs.
#        GPU_FREE_MIB=<n>  do not start if a GPU has n MiB or more in use (another job is on it).
set -uo pipefail
CK=${1:?usage: mini_chain.sh <ckpt> <stages>}
STAGES=${2:-download}
MINI=${MINI_ROOT:-/workspace/mini}
PY=${PY:-python}
R=$MINI/runs/$CK
ST=$R/state
LG=$R/logs
DAN=${DAN_ROOT:-/workspace/nd-takehome}
MODELS=${MODELS_ROOT:-/workspace/models}
MD=$MODELS/$CK
export HF_HOME=/workspace/.hf HF_HUB_DISABLE_TELEMETRY=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 OMP_NUM_THREADS=8
TO_DOWNLOAD=${TO_DOWNLOAD:-2400}
TO_TRAIN=${TO_TRAIN:-21600}
TO_GATE=${TO_GATE:-3600}
TO_SAMPLE=${TO_SAMPLE:-21600}
TO_GRADE=${TO_GRADE:-1800}
utc() { date -u +%FT%TZ; }
mkdir -p "$ST" "$LG"
log() { echo "$(utc) $*" | tee -a "$LG/chain.log"; }
run() {  # run <timeout> <cmd...>: under timeout; DRY_RUN prints instead
  local to=$1; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: timeout $to $*"; return 0; fi
  # not --foreground: on expiry timeout signals its whole process group (vLLM's engine-core child included). It runs in its
  # own process group, so `mini_pod.sh stop` signals every process group below the chain (see stop), not only the chain's.
  timeout --signal=TERM --kill-after=60 "$to" "$@"
}
sname() { echo "${1//:/_}"; }

# ---- validate the stage list
for s in ${STAGES//,/ }; do
  [[ $s =~ ^(download|train|train:[A-Za-z0-9._-]+|gate:[A-Za-z0-9._-]+|sample:[A-Za-z0-9._-]+|plain:[A-Za-z0-9.-]+)$ ]] || { log "BAD_STAGE $s"; exit 2; }
done

# ---- checkpoint row (his olmo_ifcurve/checkpoints.csv) and the L seed (stage2/seeds_L.csv)
row=$(awk -F, -v c="$CK" '$1==c' "$DAN/olmo_ifcurve/checkpoints.csv")
[ -n "$row" ] || { log "UNKNOWN_CKPT $CK"; exit 2; }
IFS=, read -r _ REPO REV STEP COMMIT NSH WBYTES _ <<< "$row"
SEED_L=$(awk -F, -v c="$CK" '$1==c {print $3}' "$MINI/stage2/seeds_L.csv" 2>/dev/null)

# ---- one chain per pod
if [ "${DRY_RUN:-0}" != 1 ]; then
  exec 8> "$MINI/gpu.lock"
  flock -n 8 || { log "LOCKED: another chain holds $MINI/gpu.lock; not starting $CK"; exit 3; }
  if [ -n "${ALSO_LOCK:-}" ] && [ -f "$ALSO_LOCK" ]; then
    exec 7< "$ALSO_LOCK"
    flock -n 7 || { log "LOCKED: another job holds $ALSO_LOCK; not starting $CK"; exit 3; }
  fi
  if [ -n "${GPU_FREE_MIB:-}" ] && command -v nvidia-smi > /dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -n 1)
    [ "${used:-x}" -lt "$GPU_FREE_MIB" ] 2>/dev/null \
      || { log "GPU_BUSY: ${used:-no answer} MiB in use on a GPU (limit $GPU_FREE_MIB); another job may be on it; not starting $CK"; exit 3; }
  fi
  grep -q "^GPU_OK" "$MINI/meta/gpu_status" 2>/dev/null || { log "GPU_BAD or no setup: not starting"; exit 3; }
  echo "$$ $(ps -o pgid= $$ | tr -d ' ') $(utc) $STAGES" > "$ST/chain.pid"
  rm -f "$ST/chain.DONE"
  echo download > "$ST/current"
  ( exec 8>&- 7<&-  # the heartbeat must not hold the GPU lock (nor the other job's)
    while sleep "${HB_SECONDS:-60}"; do
      echo "$(date +%s) $(cat "$ST/current" 2>/dev/null)" > "$ST/heartbeat.tmp" && mv "$ST/heartbeat.tmp" "$ST/heartbeat"
      echo "$(utc) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')" >> "$LG/gpu.log"
    done ) > /dev/null 2>&1 &  # its own output to /dev/null: never holds the caller's stdout open
  HB=$!
  echo "$(date +%s) download" > "$ST/heartbeat"
  trap 'kill $HB 2>/dev/null' EXIT
fi
log "CHAIN_START $CK repo=$REPO rev=$REV commit=$COMMIT stages=$STAGES seed_L=${SEED_L:-none}"

begin() { echo "$1" > "$ST/current"; rm -f "$ST/$(sname "$1").FAILED"; echo "$(utc)" > "$ST/$(sname "$1").RUNNING"; log "STAGE_START $1"; }
finish() {  # finish <stage> <rc> [note]
  local n; n=$(sname "$1")
  rm -f "$ST/$n.RUNNING"
  if [ "$2" = 0 ]; then echo "$(utc) rc=0 ${3:-}" > "$ST/$n.DONE"; log "STAGE_DONE $1 ${3:-}";
  else echo "$(utc) rc=$2 ${3:-}" > "$ST/$n.FAILED"; log "STAGE_FAILED $1 rc=$2 ${3:-}"; fi
}
todo() { [ ! -f "$ST/$(sname "$1").DONE" ]; }

# ---- download + verify (his pod/olmo_ifomega/fetch.sh; revision = the pinned commit of checkpoints.csv)
WEIGHTS_OK=0
if [ -f "$ST/download.DONE" ] && [ -f "$MD/.complete" ]; then WEIGHTS_OK=1; log "download already done"; else
  begin download
  ( cd "$DAN" && run "$TO_DOWNLOAD" bash pod/olmo_ifomega/fetch.sh "$REPO" "$COMMIT" "$CK" ) >> "$LG/download.log" 2>&1
  rc=$?
  if [ $rc = 0 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    VERIFY_SHA=${VERIFY_SHA:-0} $PY - "$MD" "$REPO" "$COMMIT" "$NSH" "$WBYTES" > "$R/download_verify.json" 2>> "$LG/download.log" <<'EOF'
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
md, repo, commit, nsh, wbytes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sha = open(os.path.join(md, ".sha")).read().strip()
sh = sorted(f for f in os.listdir(md) if f.endswith(".safetensors"))
size = sum(os.path.getsize(os.path.join(md, f)) for f in sh)
out = {"commit_expected": commit, "commit_got": sha, "shards": len(sh), "shards_expected": nsh, "bytes": size,
       "bytes_expected": wbytes, "sha256_checked": False}
ok = sha == commit and len(sh) == nsh and size == wbytes
if ok and os.environ.get("VERIFY_SHA") == "1":
    from huggingface_hub import HfApi
    lfs = {s.rfilename: s.lfs.sha256 for s in HfApi().model_info(repo, revision=commit, files_metadata=True).siblings if s.lfs}
    def h(f):
        d = hashlib.sha256()
        with open(os.path.join(md, f), "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                d.update(b)
        return f, d.hexdigest()
    with ThreadPoolExecutor(len(sh)) as ex:
        got = dict(ex.map(h, sh))
    bad = [f for f in sh if got[f] != lfs.get(f)]
    out.update(sha256_checked=True, sha256_bad=bad)
    ok = ok and not bad
out["ok"] = ok
print(json.dumps(out))
sys.exit(0 if ok else 1)
EOF
    rc=$?
  fi
  [ $rc = 0 ] && WEIGHTS_OK=1
  finish download $rc "$(cat "$R/download_verify.json" 2>/dev/null)"
fi
if [ $WEIGHTS_OK != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  for s in ${STAGES//,/ }; do [ "$s" = download ] || { todo "$s" && finish "$s" 9 "no verified weights"; }; done
  echo "$(utc) download failed" > "$ST/chain.DONE"; log "CHAIN_END $CK (download failed)"; exit 1
fi

# ---- the stages, in the order asked
for s in ${STAGES//,/ }; do
  [ "$s" = download ] && continue
  todo "$s" || { log "skip $s (DONE)"; continue; }
  kind=${s%%:*}; B=${s#*:}; n=$(sname "$s")
  case $kind in
    train)
      begin "$s"
      if [ -z "${PLAN:-}" ] || [ ! -f "$MINI/$PLAN" ]; then
        finish "$s" 7 "no plan file (PLAN=${PLAN:-unset})"; continue
      fi
      mkdir -p "$R/train"
      # shellcheck disable=SC2086
      ( cd "$MINI" && run "$TO_TRAIN" $PY -u mini_train.py run --model "$MD" --ckpt "$CK" --plan "$MINI/$PLAN" \
          --root "$MINI/pools" --out "$R/train" ${TRAIN_ARGS:-} ) >> "$LG/$n.log" 2>&1
      finish "$s" $? ;;
    gate)
      begin "$s"
      AD=$MINI/adapters/$B; GJ=$R/gate/$B.json; mkdir -p "$R/gate"
      if [ ! -d "$AD" ]; then finish "$s" 7 "no staged batch $AD"; continue; fi
      if ! $PY "$MINI/pod/mini_status.py" --verify-batch "$AD" >> "$LG/$n.log" 2>&1; then
        finish "$s" 8 "staged batch differs from its manifest (md5)"; continue
      fi
      run "$TO_GATE" $PY -u "$MINI/lora_gate.py" --model "$MD" --tokenizer "$MODELS/tok" --cells-dir "$AD" --out "$GJ" >> "$LG/$n.log" 2>&1
      rc=$?
      if { [ $rc = 0 ] || [ $rc = 1 ]; } && [ -s "$GJ" ]; then
        finish "$s" 0 "$($PY "$MINI/pod/mini_status.py" --gate-summary "$GJ" 2>/dev/null)"
      else finish "$s" "$rc" "gate engine or json failed (see logs/$n.log)"; fi ;;
    sample)
      begin "$s"
      AD=$MINI/adapters/$B; GJ=$R/gate/$B.json; OUT=$R/sample/$B
      if [ ! -s "$GJ" ]; then finish "$s" 7 "no gate result $GJ (run gate:$B first)"; continue; fi
      if [ -z "${SEED_L:-}" ]; then finish "$s" 7 "no seed for $CK in stage2/seeds_L.csv"; continue; fi
      mkdir -p "$OUT"
      PIDS_S=${PIDS_FILE:-$MINI/stage2/pids_L.txt}; SEED_S=${SEED_SAMPLE:-$SEED_L}
      log "SAMPLE_INPUTS batch=$B pids=$PIDS_S seed=$SEED_S"
      $PY "$MINI/pod/mini_status.py" --expected-draws "$GJ" "$PIDS_S" > "$ST/$n.expected" 2>/dev/null
      ( cd "$DAN" && VLLM_ATTENTION_BACKEND=FLASH_ATTN run "$TO_SAMPLE" $PY -u "$MINI/multi_sample.py" --dan "$DAN" \
          --model "$MD" --tokenizer "$MODELS/tok" --name "$CK" --rev "$REV" --problems olmo_ifomega/problems.jsonl \
          --pids "@$PIDS_S" --n 0 --chunk 4 --seed "$SEED_S" --job "L_${CK}_$B" --cells-dir "$AD" \
          --gate "$GJ" --out "$OUT" ) >> "$LG/$n.log" 2>&1
      rc=$?
      if [ $rc = 0 ]; then
        for d in "$OUT"/*/; do
          [ -f "${d}samples.jsonl.gz" ] || continue
          ( cd "$DAN" && run "$TO_GRADE" $PY olmo_ifomega/grade.py "${d}samples.jsonl.gz" olmo_ifomega/problems.jsonl "${d}graded.jsonl" ) >> "$LG/$n.log" 2>&1 \
            || { rc=$?; echo "grade failed for $d" >> "$LG/$n.log"; }
        done
      fi
      finish "$s" $rc ;;
    plain)
      begin "$s"
      PR=$MINI/stage/problems_$B.jsonl; PF=$MINI/stage/pids_$B.txt; SEEDP=$(cat "$MINI/stage/seed_$B.txt" 2>/dev/null); O=$R/plain_$B
      if [ ! -s "$PR" ] || [ ! -s "$PF" ] || [ -z "$SEEDP" ]; then finish "$s" 7 "missing stage/problems_$B.jsonl, pids_$B.txt or seed_$B.txt"; continue; fi
      mkdir -p "$O"
      awk -F: '{t+=$2} END {print t}' "$PF" > "$ST/$n.expected"
      ( cd "$DAN" && VLLM_ATTENTION_BACKEND=FLASH_ATTN run "$TO_SAMPLE" $PY -u olmo_ifladder/sample.py --model "$MD" \
          --tokenizer "$MODELS/tok" --name "$CK" --rev "$REV" --problems "$PR" --pids "@$PF" --n 0 --chunk 4 \
          --seed "$SEEDP" --job "plain_${B}_$CK" --tp 1 --order given --out "$O/samples.jsonl.gz" --meta "$O/meta.json" && \
        run "$TO_GRADE" $PY olmo_ifomega/grade.py "$O/samples.jsonl.gz" "$PR" "$O/graded.jsonl" ) >> "$LG/$n.log" 2>&1
      finish "$s" $? ;;
  esac
done

# ---- weights (kept by default)
echo done > "$ST/current"
if [ "${DELETE_WEIGHTS:-0}" = 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  rm -rf "$MD" "$MD.lock" && echo "$(utc)" > "$ST/weights_deleted" && log "WEIGHTS_DELETED $MD"
fi
nf=$(ls "$ST"/*.FAILED 2>/dev/null | wc -l)
echo "$(utc) failed_stages=$nf" > "$ST/chain.DONE"
log "CHAIN_END $CK failed_stages=$nf"
