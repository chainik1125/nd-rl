#!/usr/bin/env bash
# Rebuild runs/pools/rl_hard12.jsonl and runs/pools/hard12_test/hard12_test.jsonl end to end (as run 2026-10-06/07).
# CPU steps take minutes; the T0 pass-rate batches take ~2.5 GPU-hours (k 32), T1v1 another ~1-2 GPU-hours.
# GPU steps refuse to start on a GPU with >= 500 MiB in use and never use GPU 0 (run_passrate.sh / pull2.sh: < 500 MiB and no compute process).
# Sampling at T 0.8 on a GPU is not bit-reproducible, so a rerun gives a statistically equivalent, not identical, set.
set -eu
R=/scratch/cowifornia/nd-rl/think/runs/pools
W=$R/hard12_work
H=code/experiments/current/thinking_traces_20261005/pools/hard12
M=code.experiments.current.thinking_traces_20261005.pools.hard12
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_LEAN=$HOME/.elan/bin/lean TMPDIR=/scratch/cowifornia/nd-rl/think/runs/tmp
PY=code/.venv/bin/python
mkdir -p $W/raw $W/todo $W/doing

# 1. Task 1: features vs T0 pass rate on the previous agent's uniform k 16 measurements (7-12 lines)
SETS="$R/short_pilot.jsonl $R/short_cands.jsonl"
ROWS="$R/passrate/short_pilot/rows.jsonl $R/passrate/short/rows.jsonl"
K12=/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl
$PY -m $M.analyze --sets $SETS --rows $ROWS --k12 $K12 --out $W/feature_analysis_k16.json
$PY -m $M.analyze --sets $SETS --rows $ROWS --k12 $K12 --max_fdepth 5 --out $W/feature_analysis_k16_d5.json
$PY -m $M.score fit --sets $SETS --rows $ROWS --out $W/band_model.json

# 2. Exclusion index (K12 train/held-out/cap-6, all fork data, OOD sets, rl_hard*, test_21_30 if present)
$PY -m $M.exclude index --out $W/excl_index.pkl

# 3. Candidate streams
#  (a) the previous agent's strict-long stream, scored; batch 1 = top 9,000 unmeasured by band score
#      (<= 1,700 per length; states <= 386 tokens; not in rl_hard*), 3 shards b1_s{0,1,2}
$PY -m $M.score score --model $W/band_model.json --in $R/cands_clean.jsonl --out $W/cc_scored.jsonl --workers 12
$PY - <<'EOF'
import json, collections, random
R = "/scratch/cowifornia/nd-rl/think/runs/pools"
meas = {json.loads(l)["name"] for p in ("short_pilot", "short") for l in open(f"{R}/passrate/{p}/rows.jsonl")}
hk = {json.loads(l)["key"] for p in ("rl_hard.jsonl", "rl_hard_filtered.jsonl") for l in open(f"{R}/{p}")}
rs = [r for r in map(json.loads, open(f"{R}/hard12_work/cc_scored.jsonl"))
      if r["name"] not in meas and r["key"] not in hk and r["max_state"] <= 386]
rs.sort(key=lambda r: -r["band_score"])
per, out = collections.Counter(), []
for r in rs:
    if per[r["n_lines"]] >= 1700:
        continue
    per[r["n_lines"]] += 1
    out.append(r)
    if len(out) >= 9000:
        break
random.Random(1).shuffle(out)
for i in range(3):
    with open(f"{R}/hard12_work/b1_s{i}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out[i::3])
EOF
#  (b) steered generator configs (pilot: 40 s x 2 workers per config; main stream: 180 s x 3 workers, score >= 0.2)
$PY -m $M.gen_steer --configs strict_long,takehome,fwd_strict_d2,fwd_strict_d1,fwd_takehome_d2,goal_strict,goal_takehome \
  --model $W/band_model.json --secs 40 --workers 2 --seed 900000 --out $W/raw/pilot.jsonl
$PY -m $M.exclude filter --index $W/excl_index.pkl --in $W/raw/pilot.jsonl --out $W/raw/pilot_x.jsonl \
  --seen $W/b1_s0.jsonl $W/b1_s1.jsonl $W/b1_s2.jsonl
$PY -m $M.select --in $W/raw/pilot_x.jsonl --min_score 0.3 --per_len_frac 0.35 --tag p --shards 1 --seed 1 \
  --cfg takehome:200,fwd_strict_d1:200,fwd_takehome_d2:200,goal_takehome:200,fwd_strict_d2:150 --out $W/pilot
$PY -m $M.select --in $W/raw/pilot_x.jsonl --cfg takehome:100 --tag pr --shards 1 --seed 2 \
  --skip $W/pilot_s0.jsonl --out $W/pilotrand
cat $W/pilotrand_s0.jsonl >> $W/pilot_s0.jsonl
$PY -m $M.gen_steer --configs takehome,fwd_strict_d1,fwd_takehome_d2,goal_takehome,fwd_strict_d2 \
  --model $W/band_model.json --secs 180 --workers 3 --seed 930000 --min_score 0.2 --out $W/raw/steer1.jsonl
$PY -m $M.exclude filter --index $W/excl_index.pkl --in $W/raw/steer1.jsonl --out $W/raw/steer1_x.jsonl \
  --seen $W/b1_s0.jsonl $W/b1_s1.jsonl $W/b1_s2.jsonl $W/pilot_s0.jsonl
SKIP="--skip_measured $W/passrate --skip $W/b1_s0.jsonl $W/b1_s1.jsonl $W/b1_s2.jsonl $W/pilot_s0.jsonl"
$PY -m $M.select --in $W/raw/steer1_x.jsonl --cfg fwd_takehome_d2:3500,fwd_strict_d2:2500,goal_takehome:1000 \
  --min_score 0.4 --per_len_frac 0.3 $SKIP --tag b2a --shards 7 --seed 21 --out $W/b2a
$PY -m $M.select --in $W/raw/steer1_x.jsonl --cfg takehome:3000,fwd_strict_d1:2000 \
  --min_score 0.35 --per_len_frac 0.3 $SKIP --tag b2b --shards 5 --seed 22 --out $W/b2b
i=1
for f in b2a_s0 b2b_s0 b2a_s1 b2b_s1 b2a_s2 b2b_s2 b2a_s3 b2b_s3 b2a_s4 b2b_s4 b2a_s5 b2a_s6; do
  mv $W/$f.jsonl $W/todo/b2_q$(printf %02d $i).jsonl; i=$((i + 1))
done
#  (c) batch 3 (added once batch 2's first shards showed fwd_takehome_d2 at the highest band yield): 2,000 more
cat $W/todo/b2_q*.jsonl > $W/b2_all_tmp.jsonl
$PY -m $M.select --in $W/raw/steer1_x.jsonl --cfg fwd_takehome_d2:1500,fwd_strict_d2:500 --min_score 0.4 \
  --per_len_frac 0.3 $SKIP $W/b2_all_tmp.jsonl --tag b3 --shards 2 --seed 23 --out $W/b3
mv $W/b3_s0.jsonl $W/todo/b3_q13.jsonl; mv $W/b3_s1.jsonl $W/todo/b3_q14.jsonl; rm $W/b2_all_tmp.jsonl

# 4. T0 k 32 pass rates (GPUs 1-4; seeds: b1 11/12/13, pilot 21, queue shard q<i> 100 + i)
$H/run_passrate.sh 1 b1_s0 $W/b1_s0.jsonl T0 32 11 &
$H/run_passrate.sh 2 b1_s1 $W/b1_s1.jsonl T0 32 12 &
$H/run_passrate.sh 3 b1_s2 $W/b1_s2.jsonl T0 32 13 &
$H/run_passrate.sh 4 pilot $W/pilot_s0.jsonl T0 32 21 &
wait
for G in 1 2 3 4; do $H/pull2.sh $G 90 T0 32 & done
wait

# 5. Band filter, exclusion + dedupe, Lean on the references, random split, keys sidecar, T1v1 samples
$PY -m $M.exclude index --out $W/excl_index.pkl   # rebuilt: picks up test_21_30.jsonl if it exists by now
LEAN_GATE_WORKERS=16 LEAN_GATE_LOG=$W/lean_refs.jsonl $PY -m $M.assemble12 build --work $W \
  --index $W/excl_index.pkl --n_pool 5000 --n_test 2500

# 6. T1v1 (B 256, plain, T 0.8, k 32) on 1,000 pool + 1,000 test targets, for reference only
#    (split into 4 shards per set: t1_pool_q*, t1_test_q*, merged into passrate/t1_{pool,test}/rows.jsonl)
$PY - <<'EOF'
import json
W = "/scratch/cowifornia/nd-rl/think/runs/pools/hard12_work"
for nm in ("pool", "test"):
    rows = [json.loads(l) for l in open(f"{W}/t1_{nm}.jsonl")]
    for i in range(4):
        with open(f"{W}/todo/t1_{nm}_q{i}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows[i::4])
EOF
for G in 1 2 3 4; do $H/pull2.sh $G 90 T1v1 32 & done
wait
# 6b. T0_s0 again on the same 2 x 1,000 targets with fresh seeds (120-127): the regression-to-the-mean baseline
$PY - <<'EOF2'
import json
W = "/scratch/cowifornia/nd-rl/think/runs/pools/hard12_work"
j = 20
for nm in ("pool", "test"):
    rows = [json.loads(l) for l in open(f"{W}/t1_{nm}.jsonl")]
    for i in range(4):
        with open(f"{W}/todo/t0re_{nm}_q{j}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows[i::4])
        j += 1
EOF2
for G in 1 2 3 4; do $H/pull2.sh $G 90 T0 32 & done
wait
for nm in pool test; do
  for run in t1 t0re; do
    mkdir -p $W/passrate/${run}_$nm
    cat $W/passrate/${run}_${nm}_q*/rows.jsonl > $W/passrate/${run}_$nm/rows.jsonl
  done
done

# 7. Meta files (runs/pools/rl_hard12.meta.json, hard12_test/hard12_test.meta.json, copies in $H/meta/)
$PY -m $M.assemble12 meta --work $W
