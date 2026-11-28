#!/usr/bin/env bash
# wi_cell.sh <a100|a050|a075|a025>   (POD side; one GPU job of the chain, started through m2_job.sh, cwd = $M2_ROOT)
# One cell of PREREG.md: 48 reach prompts x 4 draws through Dan's unedited sampler (metrics 2's sample_wrap.py), every
# setting equal across cells; only the model folder and its labels differ.
#   a100           serves the stored s2300 checkpoint as downloaded (the same-engine yardstick)
#   a050/a075/a025 serve /workspace/wi/models/interp_<tag> (this run's own folder), and only if interp_write.py left
#                  its WI_WRITE_OK marker
# Steps: wait for the job it depends on (preflight for a100, write_<tag> otherwise; they run in the other chain) -> precheck
# (weights verified) -> ledger gate -> sampler under its own timeout (WI_CELL_TIMEOUT, 5400 s) -> grade whatever rows
# exist -> count (out/count.json). Nothing is deleted here (pod/wi_rm.sh does that as its own job).
# An earlier attempt's rows are never resumed (Dan's sampler would fill only the missing draws: a retry bias).
# Exit: 0 complete (192 rows, settings as registered)   9 skipped by the ledger gate   21 weights not verified
#       124 the sampler was cut by its timeout (rows kept and graded; count.json lists the missing draws)
#       31 rows missing / extra / duplicated   32 a setting differs   2 usage   other = the sampler's own exit code
# env: M2_ROOT (/workspace/wi), MODELS_ROOT, WI_MERGED_ROOT, DAN_ROOT, WI_POD_START_UTC (required), WI_CAP_USD (21), WI_RATE (5.29),
#      WI_PRIOR_USD (0), WI_CELL_USD (7.94 = 90 min), WI_CELL_TIMEOUT (5400), WI_SEED (710012), WI_PREP_CHAIN (prep),
#      WI_WAIT_S (3600); tests override the two commands with WI_SAMPLER and WI_GRADER.
set -uo pipefail
TAG=${1:?usage: wi_cell.sh <a100|a050|a075|a025>}
WI=${M2_ROOT:-/workspace/wi}; MODELS=${MODELS_ROOT:-/workspace/models}; DAN=${DAN_ROOT:-/workspace/nd-takehome}
MERGED=${WI_MERGED_ROOT:-$WI/models}
case $TAG in
  # the label is not "s2300", so these rows cannot be mistaken for Dan's stored s2300 rows
  a100) MODEL=$MODELS/s2300; NAME=s2300_same_engine; REV=29884259e8c87406883e817e0628fa5ce9b529c3 ;;
  a025|a050|a075) MODEL=$MERGED/interp_$TAG; NAME=interp_$TAG; REV=interp_$TAG ;;
  *) echo "unknown cell $TAG"; exit 2 ;;
esac
JOB=cell_$TAG; R=$WI/runs/$JOB; O=$R/out
TO=${WI_CELL_TIMEOUT:-5400}; SEED=${WI_SEED:-710012}
read -r -a SAMPLER <<< "${WI_SAMPLER:-python3 sample_wrap.py}"
read -r -a GRADER <<< "${WI_GRADER:-bash tools/grade_job.sh}"
PROBLEMS=$DAN/olmo_ifomega/problems.jsonl
utc() { date -u +%FT%TZ; }
cd "$WI" || exit 2
mkdir -p "$O" "$R/state"; echo 192 > "$R/state/expected"
status() { echo "$(utc) $*" | tee -a "$O/cell_status.txt"; }

python3 cell_count.py wait --tag "$TAG" --wi "$WI" || { status "NOT_STARTED the job this cell depends on did not finish"; exit 21; }
python3 cell_count.py precheck --tag "$TAG" --wi "$WI" --models "$MODELS" --merged "$MERGED" || { status "NOT_STARTED weights not verified"; exit 21; }
[[ -n ${WI_POD_START_UTC:-} ]] || { status "NOT_STARTED WI_POD_START_UTC is not set (ledger gate)"; exit 2; }
python3 tools/ledger_ok.py --start-utc "$WI_POD_START_UTC" --rate "${WI_RATE:-5.29}" --prior "${WI_PRIOR_USD:-0}" \
  --cap "${WI_CAP_USD:-21}" --cell-usd "${WI_CELL_USD:-7.94}" || { status "SKIPPED by the ledger gate"; exit 9; }
if [[ -s $O/samples.jsonl.gz ]]; then
  # Dan's sampler would resume and fill only the missing draws (a retry bias, PREREG "Cut cells"): never silently.
  [[ ${WI_ALLOW_RESUME:-0} == 1 ]] || { status "NOT_STARTED rows of an earlier attempt exist in $O (move them away, or WI_ALLOW_RESUME=1)"; exit 2; }
  status "RESUME allowed by WI_ALLOW_RESUME=1"
fi

status "SAMPLING $TAG model=$MODEL seed=$SEED timeout=${TO}s"
timeout --signal=TERM --kill-after=60 "$TO" "${SAMPLER[@]}" --dan-root "$DAN" --sidecar "$O/wrap.json" -- \
  --model "$MODEL" --tokenizer "$MODELS/tok" --name "$NAME" --rev "$REV" --problems "$PROBLEMS" \
  --pids @inputs/spec_reach4.txt --n 1 --chunk 1 --seed "$SEED" --job "$JOB" \
  --out "$O/samples.jsonl.gz" --meta "$O/meta.jsonl" &
SP=$!
# `timeout` leads its own process group; a stop of this job (SIGTERM to this script's group) is passed on to it, and it
# passes it to the sampler's whole group (SIGKILL after 60 s).
trap 'kill -TERM "$SP" 2>/dev/null' TERM INT
wait "$SP"; src=$?
while kill -0 "$SP" 2>/dev/null; do wait "$SP"; src=$?; done
trap - TERM INT
status "SAMPLER_EXIT rc=$src"

if [[ -s $O/samples.jsonl.gz ]]; then
  "${GRADER[@]}" "$O/samples.jsonl.gz" "$O/graded.jsonl" "$PROBLEMS" || status "GRADE_FAILED rc=$?"
fi
python3 cell_count.py count --dir "$O" --spec inputs/spec_reach4.txt --tag "$TAG" --seed "$SEED" --model-dir "$MODEL" \
  --prompt-sha inputs/prompt_sha256_reach.json
crc=$?
if [[ $src == 124 || $src == 137 ]]; then status "CELL_CUT $TAG (sampler timeout ${TO}s): rows kept and graded, see count.json"; exit 124; fi
if [[ $src != 0 ]]; then status "CELL_FAILED $TAG sampler rc=$src"; exit "$src"; fi
if [[ $crc != 0 ]]; then status "CELL_INCOMPLETE $TAG count rc=$crc"; exit "$crc"; fi
status "CELL_COMPLETE $TAG"
exit 0
