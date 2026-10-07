#!/usr/bin/env bash
# Rerun the hard12r pilot (2026-10-07): exclusion index, candidate sets, the RL'd filter model at
# k 8 on GPU 0 (shared-lock protocol in run_eval.sh), and the yield report. CPU steps take minutes;
# the two GPU jobs took 777 s + 285 s. Sampling at T 0.8 on a GPU is not bit-reproducible, so a
# rerun gives statistically equivalent counts, not the same ones. The project stopped after the
# pilot (see README.md); nothing here builds a pool or a test split.
set -eu
R=/scratch/cowifornia/nd-rl/think/runs
H12=$R/pools/hard12_work
W=$R/pools/hard12r/work
H=code/experiments/current/thinking_traces_20261005/pools/hard12r
M=code.experiments.current.thinking_traces_20261005.pools
F=$R/restart3_grpofix/grpo_T0_crashed_it38/ckpts/grpo_T0_it35_rolling.pt
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp
PY=code/.venv/bin/python
mkdir -p $W/raw $W/cands $W/sets $W/eval $W/logs

# 1. Exclusion index: hard12's files + rl_hard12, hard12 test (+ 1k subset), test_13_20 / test_21_30
$PY -m $M.hard12r.exclude_r index --out $W/excl_index.pkl

# 2a. Prior candidates: hard12 candidates T0_s0 measured at k 32 that are in neither rl_hard12 nor
#     the hard12 test set: spare in-band (2-16 of 32) and T0 <= 1 of 32
$PY -m $M.hard12r.cands prior --index $W/excl_index.pkl --out $W/cands/prior.jsonl
$PY -m $M.hard12r.cands evalset --in $W/cands/prior.jsonl --out $W/sets/pilot_prior.jsonl

# 2b. Fresh candidates: hard12's steered streams (unmeasured rows) + a new 300 s stream of the same
#     configs (band score >= 0.3), excluded against the index and every measured hard12 candidate;
#     300 per config at random among band score >= 0.3 (hard12 select.py, <= 30% per length)
$PY -m $M.hard12.gen_steer --configs strict_long,takehome,fwd_strict_d2,fwd_strict_d1,fwd_takehome_d2,goal_takehome \
  --model $H12/band_model.json --secs 300 --workers 3 --seed 1300000 --min_score 0.3 --out $W/raw/fresh1.jsonl
cat $H12/raw/steer1_x.jsonl $H12/raw/pilot_x.jsonl $H12/cc_scored.jsonl > $W/raw/h12streams.jsonl
SEEN="$H12/b1_s0.jsonl $H12/b1_s1.jsonl $H12/b1_s2.jsonl $H12/pilot_s0.jsonl $H12/doing/*.jsonl"
$PY -m $M.hard12r.exclude_r filter --index $W/excl_index.pkl --in $W/raw/h12streams.jsonl \
  --out $W/raw/h12streams_x.jsonl --seen $SEEN
$PY -m $M.hard12r.exclude_r filter --index $W/excl_index.pkl --in $W/raw/fresh1.jsonl \
  --out $W/raw/fresh1_x.jsonl --seen $W/raw/h12streams_x.jsonl $SEEN
$PY -m $M.hard12.select --in $W/raw/h12streams_x.jsonl $W/raw/fresh1_x.jsonl \
  --cfg strict_long:300,takehome:300,fwd_strict_d2:300,fwd_strict_d1:300,fwd_takehome_d2:300,goal_takehome:300 \
  --min_score 0.3 --per_len_frac 0.3 --tag fp --shards 1 --seed 31 --out $W/pilot_fresh
$PY - <<'EOF'
import json
W = "/scratch/cowifornia/nd-rl/think/runs/pools/hard12r/work"
rs = [json.loads(l) for l in open(f"{W}/pilot_fresh_s0.jsonl")]
with open(f"{W}/cands/pilot_fresh.jsonl", "w") as f:
    f.writelines(json.dumps({**r, "src": "fresh"}) + "\n" for r in rs)
EOF
rm $W/pilot_fresh_s0.jsonl
$PY -m $M.hard12r.cands evalset --in $W/cands/pilot_fresh.jsonl --out $W/sets/pilot_fresh.jsonl

# 3. Filter model (fixed no-think GRPO, it 35), k 8, GPU 0 under the shared lock
$H/run_eval.sh 0 pilot_prior_k8 $F $W/sets/pilot_prior.jsonl $W/eval/pilot_prior_k8 8 0 1
$H/run_eval.sh 0 pilot_fresh_k8 $F $W/sets/pilot_fresh.jsonl $W/eval/pilot_fresh_k8 8 0 1

# 4. Yield report (json + markdown tables); copies into meta/ and runs/pools/hard12r/meta.json
$PY -m $M.hard12r.pilot --work $W --out $W/pilot_report.json > $W/pilot_report.md
cp $W/pilot_report.json $H/meta/pilot_report.json
cp $W/pilot_report.json $R/pools/hard12r/meta.json
