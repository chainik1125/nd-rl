#!/usr/bin/env bash
# oe_chain.sh <ckpt> <stages> (POD side; started by olmo_pod.sh run under setsid + nohup + timeout).
# One checkpoint, in SPEC order: download -> edl -> T -> H -> L -> rep -> weights deleted. <stages> is a comma list
# from edl,T,H,L,rep (download always runs first), or `download` alone (weights kept). Every stage is resumable: a stage with state/<stage>.DONE is skipped, the
# samplers resume their own output files. Per stage: logs/<stage>.log, state/<stage>.{RUNNING,DONE,FAILED},
# state/<stage>.expected (draws expected, sampling stages). A heartbeat (state/heartbeat: "<epoch> <stage>") and a GPU
# line (logs/gpu.log) are written every 60 s. One chain per pod at a time (flock on /workspace/oe/gpu.lock), so never
# two GPU processes at once. EDL's exit code is its gate; a failed EDL is recorded and T / H / rep still run.
# env: OE_ROOT / DAN_ROOT / DAO_ROOT / MODELS_ROOT override the /workspace paths (local dry runs),
#      VERIFY_SHA=1 (sha256 every shard against the Hub's LFS sha256; default 0: commit + shard count + bytes only),
#      EDL_ARGS (extra olmo_edl.py flags, e.g. "--lrs 3e-5,1e-4,3e-4 --orders 0,1"), KEEP_WEIGHTS=1,
#      TO_DOWNLOAD / TO_EDL / TO_T / TO_H / TO_L / TO_REP (seconds), N_H (H draws per prompt, 8 or 4), DRY_RUN=1 (print only),
#      L_CHECK=1 (stage L: run lora_check.py also at a checkpoint other than dpo; at dpo it always runs).
set -uo pipefail
CK=${1:?usage: oe_chain.sh <ckpt> <stages>}
STAGES=${2:-edl,T,H}
[ "$STAGES" = download ] && KEEP_WEIGHTS=1  # download only: keep the weights for the next run
OE=${OE_ROOT:-/workspace/oe}
R=$OE/runs/$CK
ST=$R/state
LG=$R/logs
DAN=${DAN_ROOT:-/workspace/nd-takehome}
DAO=${DAO_ROOT:-/workspace/nd-answeronly}
MODELS=${MODELS_ROOT:-/workspace/models}
MD=$MODELS/$CK
export HF_HOME=/workspace/.hf HF_HUB_DISABLE_TELEMETRY=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 OMP_NUM_THREADS=8
TO_DOWNLOAD=${TO_DOWNLOAD:-2400}
TO_EDL=${TO_EDL:-14400}
TO_T=${TO_T:-3600}
TO_H=${TO_H:-7200}
TO_REP=${TO_REP:-7200}
TO_L=${TO_L:-7200}
N_H=${N_H:-8}
utc() { date -u +%FT%TZ; }
mkdir -p "$ST" "$LG"
log() { echo "$(utc) $*" | tee -a "$LG/chain.log"; }
run() {  # run <timeout> <cmd...>: under timeout; DRY_RUN prints instead
  local to=$1; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: timeout $to $*"; return 0; fi
  timeout --signal=TERM --kill-after=60 "$to" "$@"
}

# ---- checkpoint row (his olmo_ifcurve/checkpoints.csv) and seeds (ours, new; stage2/seeds.csv)
row=$(awk -F, -v c="$CK" '$1==c' $DAN/olmo_ifcurve/checkpoints.csv)
[ -n "$row" ] || { log "UNKNOWN_CKPT $CK"; exit 2; }
IFS=, read -r _ REPO REV STEP COMMIT NSH WBYTES _ <<< "$row"
SEEDS=$(awk -F, -v c="$CK" '$1==c' $OE/stage2/seeds.csv)
IFS=, read -r _ _ SEED_T SEED_H SEED_REP _ <<< "$SEEDS"
SEED_L=$(awk -F, -v c="$CK" '$1==c {print $3}' $OE/stage2/seeds_L.csv 2>/dev/null)

# ---- one chain per pod
if [ "${DRY_RUN:-0}" != 1 ]; then
  exec 8> $OE/gpu.lock
  flock -n 8 || { log "LOCKED: another chain holds /workspace/oe/gpu.lock; not starting $CK"; exit 3; }
  grep -q "^GPU_OK" $OE/meta/gpu_status 2>/dev/null || { log "GPU_BAD or no setup: not starting"; exit 3; }
  echo "$$ $(ps -o pgid= $$ | tr -d ' ') $(utc) $STAGES" > "$ST/chain.pid"
  rm -f "$ST/chain.DONE"
  echo download > "$ST/current"
  ( exec 8>&-  # the heartbeat must not hold the GPU lock
    while sleep 60; do
      echo "$(date +%s) $(cat "$ST/current" 2>/dev/null)" > "$ST/heartbeat.tmp" && mv "$ST/heartbeat.tmp" "$ST/heartbeat"
      echo "$(utc) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')" >> "$LG/gpu.log"
    done ) > /dev/null 2>&1 &  # its own output to /dev/null: never holds the caller's stdout open
  HB=$!
  echo "$(date +%s) download" > "$ST/heartbeat"
  trap 'kill $HB 2>/dev/null' EXIT
fi
log "CHAIN_START $CK repo=$REPO rev=$REV commit=$COMMIT stages=$STAGES seeds T=$SEED_T H=$SEED_H rep=$SEED_REP L=${SEED_L:-none}"

begin() { echo "$1" > "$ST/current"; rm -f "$ST/$1.FAILED"; echo "$(utc)" > "$ST/$1.RUNNING"; log "STAGE_START $1"; }
finish() {  # finish <stage> <rc> [note]
  rm -f "$ST/$1.RUNNING"
  if [ "$2" = 0 ]; then echo "$(utc) rc=0 ${3:-}" > "$ST/$1.DONE"; log "STAGE_DONE $1";
  else echo "$(utc) rc=$2 ${3:-}" > "$ST/$1.FAILED"; log "STAGE_FAILED $1 rc=$2 ${3:-}"; fi
}
want() { [[ ",$STAGES," == *",$1,"* ]] && [ ! -f "$ST/$1.DONE" ]; }

# ---- download + verify (his pod/olmo_ifomega/fetch.sh; revision = the pinned commit of checkpoints.csv)
WEIGHTS_OK=0
if [ -f "$ST/download.DONE" ] && [ -f "$MD/.complete" ]; then WEIGHTS_OK=1; log "download already done"; else
  begin download
  ( cd $DAN && run "$TO_DOWNLOAD" bash pod/olmo_ifomega/fetch.sh "$REPO" "$COMMIT" "$CK" ) >> "$LG/download.log" 2>&1
  rc=$?
  if [ $rc = 0 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    VERIFY_SHA=${VERIFY_SHA:-0} python - "$MD" "$REPO" "$COMMIT" "$NSH" "$WBYTES" > "$R/download_verify.json" 2>> "$LG/download.log" <<'EOF'
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
  for s in edl T H rep; do want $s && finish $s 9 "no verified weights"; done
  echo "$(utc) download failed" > "$ST/chain.DONE"; log "CHAIN_END $CK (download failed)"; exit 1
fi

# ---- stage 1: EDL (the other agent's runner; exit code is the gate, failures recorded, chain continues)
if want edl; then
  begin edl
  # shellcheck disable=SC2086
  run "$TO_EDL" python -u $OE/olmo_edl.py run --model "$MD" --ckpt "$CK" --pool $OE/pool --out "$R/edl" ${EDL_ARGS:-} >> "$LG/edl.log" 2>&1
  finish edl $?
fi

# ---- stage 2 T: his answer-only sampler on his transplant inputs (writer label = this checkpoint), his grader
if want T; then
  begin T; mkdir -p "$R/T"
  RQ=$OE/stage2/T_$CK.jsonl.gz
  python -c "import gzip,json; print(sum(json.loads(l)['n'] for l in gzip.open('$RQ','rt')))" > "$ST/T.expected" 2>/dev/null
  ( cd $DAO && VLLM_ATTENTION_BACKEND=FLASH_ATTN run "$TO_T" python -u olmo_answeronly/sample_prefix.py --model "$MD" \
      --tokenizer $MODELS/tok --name "$CK" --rev "$REV" --requests "$RQ" --seed "$SEED_T" --job "T_$CK" \
      --out "$R/T/samples.jsonl.gz" --meta "$R/T/meta.json" && \
    run 1800 python olmo_transplant/grade_rows.py "$R/T/samples.jsonl.gz" "$R/T/graded.jsonl" ) >> "$LG/T.log" 2>&1
  finish T $?
fi

# ---- stage 2 H: his sampler (olmo_ifladder/sample.py) on the hinted reach prompts, his grader
if want H; then
  begin H; mkdir -p "$R/H"
  PF=$OE/stage2/pids_H_n$N_H.txt
  awk -F: '{s+=$2} END {print s}' "$PF" > "$ST/H.expected"
  ( cd $DAN && VLLM_ATTENTION_BACKEND=FLASH_ATTN run "$TO_H" python -u olmo_ifladder/sample.py --model "$MD" \
      --tokenizer $MODELS/tok --name "$CK" --rev "$REV" --problems $OE/stage2/problems_H.jsonl --pids "@$PF" \
      --n 0 --chunk 4 --seed "$SEED_H" --job "H_$CK" --tp 1 --order given --out "$R/H/samples.jsonl.gz" \
      --meta "$R/H/meta.json" && \
    run 1800 python olmo_ifomega/grade.py "$R/H/samples.jsonl.gz" $OE/stage2/problems_H.jsonl "$R/H/graded.jsonl" ) >> "$LG/H.log" 2>&1
  finish H $?
fi

# ---- stage L: the 48 reach prompts, plain (as Dan's arm B sent them), through this checkpoint's stage-1 adapter on
# vLLM's LoRA path (his olmo_ifladder/sample.py --lora, as his LoRA evaluations ran; never a merge). Adapter staged
# from serrano (stage_adapter.py: DONE lists the grid, BEST.json = the shortest order-0 row) and md5-checked here;
# at dpo (or with L_CHECK=1) lora_check.py first confirms the adapter is applied (teacher-forced nats vs PEFT's).
if want L; then
  begin L; mkdir -p "$R/L"
  AD=$OE/adapters/$CK/best_order0
  SRCJ=$OE/adapters/$CK/source.json
  awk -F: '{s+=$2} END {print s}' $OE/stage2/pids_L.txt > "$ST/L.expected"
  rc=0; note=""
  if [ ! -f "$AD/adapter_model.safetensors" ] || [ ! -f "$SRCJ" ] || [ -z "${SEED_L:-}" ]; then
    rc=7; note="no staged adapter, source.json or seed"
  else
    ( cd "$AD" && md5sum adapter_config.json adapter_model.safetensors ) > "$R/L/adapter_md5.txt"
    cp "$SRCJ" "$R/L/adapter_source.json"
    python3 - "$SRCJ" "$R/L/adapter_md5.txt" >> "$LG/L.log" 2>&1 <<'PYEOF'
import json, sys
src = json.load(open(sys.argv[1]))
got = {f: m for m, f in (ln.split() for ln in open(sys.argv[2]) if ln.strip())}
bad = [f for f, v in src["files"].items() if got.get(f) != v["md5"]]
print("ADAPTER_MD5", "ok" if not bad else f"MISMATCH {bad}", src["row"])
sys.exit(1 if bad else 0)
PYEOF
    [ $? = 0 ] || { rc=8; note="adapter md5 differs from serrano's"; }
  fi
  if [ $rc = 0 ] && { [ "$CK" = dpo ] || [ "${L_CHECK:-0}" = 1 ]; } && ! grep -q '"verdict": "PASS"' "$R/L/lora_check.json" 2>/dev/null; then
    run 1800 python -u $OE/lora_check.py --model "$MD" --tokenizer $MODELS/tok --adapter "$AD" --source "$SRCJ" \
      --pool $OE/pool --out "$R/L/lora_check.json" >> "$LG/L.log" 2>&1 || { rc=9; note="lora_check FAILED: adapter not applied as in PEFT"; }
  fi
  if [ $rc = 0 ]; then
    ( cd $DAN && VLLM_ATTENTION_BACKEND=FLASH_ATTN run "$TO_L" python -u olmo_ifladder/sample.py --model "$MD" \
        --tokenizer $MODELS/tok --name "$CK" --rev "$REV" --problems olmo_ifomega/problems.jsonl \
        --pids "@$OE/stage2/pids_L.txt" --n 0 --chunk 4 --seed "$SEED_L" --job "L_$CK" --tp 1 --order given \
        --lora "$AD" --lora_name "${CK}_edl64" --lora_rank 16 --out "$R/L/samples.jsonl.gz" --meta "$R/L/meta.json" && \
      run 1800 python olmo_ifomega/grade.py "$R/L/samples.jsonl.gz" olmo_ifomega/problems.jsonl "$R/L/graded.jsonl" ) >> "$LG/L.log" 2>&1
    rc=$?
    [ $rc = 0 ] || note="sampler or grader failed (see logs/L.log; a rejected adapter fails here)"
  fi
  finish L $rc "$note"
fi

# ---- Stage 0c replication draw: arm-A prompts x 2, plain prompts, new seed (only when asked)
if want rep; then
  begin rep; mkdir -p "$R/rep"
  awk -F: '{s+=$2} END {print s}' $OE/stage2/pids_rep.txt > "$ST/rep.expected"
  ( cd $DAN && VLLM_ATTENTION_BACKEND=FLASH_ATTN run "$TO_REP" python -u olmo_ifladder/sample.py --model "$MD" \
      --tokenizer $MODELS/tok --name "$CK" --rev "$REV" --problems olmo_ifomega/problems.jsonl \
      --pids "@$OE/stage2/pids_rep.txt" --n 0 --chunk 2 --seed "$SEED_REP" --job "rep_$CK" --tp 1 --order given \
      --out "$R/rep/samples.jsonl.gz" --meta "$R/rep/meta.json" && \
    run 1800 python olmo_ifomega/grade.py "$R/rep/samples.jsonl.gz" olmo_ifomega/problems.jsonl "$R/rep/graded.jsonl" ) >> "$LG/rep.log" 2>&1
  finish rep $?
fi

# ---- weights
echo done > "$ST/current"
if [ "${KEEP_WEIGHTS:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  rm -rf "$MD" "$MD.lock" && echo "$(utc)" > "$ST/weights_deleted" && log "WEIGHTS_DELETED $MD"
fi
nf=$(ls "$ST"/*.FAILED 2>/dev/null | wc -l)
echo "$(utc) failed_stages=$nf" > "$ST/chain.DONE"
log "CHAIN_END $CK failed_stages=$nf"
