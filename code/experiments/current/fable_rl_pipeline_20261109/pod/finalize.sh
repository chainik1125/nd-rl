#!/usr/bin/env bash
# finalize.sh <alias> <run> <tag>: after the run has logged RL_DONE, on the run's own pod:
#   1. pull the run to serrano and pick the policy checkpoint by the rule (report.py --best:
#      highest dev plain pass@1 among the scored rounds, ties to the later round);
#   2. train the run's two verifier networks on the step data of all its rounds;
#   3. pack policy + verifier + the frozen solver settings ($F/final/solver.json) into
#      /workspace/final/final_<tag>.pt;
#   4. score it on dev in plain and full mode (5 draws, seed 0, Lean-judged);
#   5. copy the packed checkpoint, verifier, dev rows and logs to $F/final/<tag>/ and compare md5.
# The ablation run has no step data: with VERIFIER=none it is packed without a verifier and scored
# in plain mode only.
set -euo pipefail
h=$1; run=$2; tag=$3
F=/scratch/cowifornia/nd-rl/fable_rl
HERE=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -n -F $F/ssh_config)
mkdir -p $F/final/$tag
"${SSH[@]}" "$h" "grep -q RL_DONE /workspace/runs/$run.log" || { echo "no RL_DONE in $run.log on $h"; exit 1; }
bash "$HERE/pull.sh" "$h" | tail -1
D=$F/runs/$h/$run
python3 "$HERE/../report.py" --best "$D" > $F/final/$tag/checkpoint_rule.txt
rnd=$(grep -o 'picked round [0-9]*' $F/final/$tag/checkpoint_rule.txt | grep -o '[0-9]*')
echo "$tag: rule picks r$rnd.pt"
bash "$HERE/push.sh" "$h" | tail -1
rsync -t -e "ssh -F $F/ssh_config" $F/final/solver.json "$h:/workspace/solver.json"
R=/workspace/runs/$run
PY=/workspace/venv/bin/python
modes="plain full"
if [[ ${VERIFIER:-} == none ]]; then
  modes="plain"
  "${SSH[@]}" "$h" "mkdir -p /workspace/final && cd /workspace/fr && $PY pack.py --policy $R/r$rnd.pt --out /workspace/final/final_$tag.pt --solver \"\$(cat /workspace/solver.json)\" --note '$tag: policy r$rnd of $run, no verifier'"
else
  # the two verifier networks of PREREG amendment 5, each trained on this run's own step data
  "${SSH[@]}" "$h" "mkdir -p /workspace/final && cd /workspace/fr && export PYTHONUNBUFFERED=1 && timeout 2400 $PY -m fr.verifier --kind disc --run $R --out $R/verifier_disc.pt --x /workspace/data/train_k12.jsonl.gz --steps 6000 --max_per_round 150000 --x_pos 400000 > /workspace/final/verifier_disc_$tag.log 2>&1; timeout 4800 $PY -m fr.verifier --kind mslot --run $R --out $R/verifier_mslot.pt --x /workspace/data/train_k12.jsonl.gz --steps 8000 --n_layer 6 --d 384 --x_pos 1200000 > /workspace/final/verifier_mslot_$tag.log 2>&1; grep -h 'examples\|^holdout' /workspace/final/verifier_*_$tag.log | cut -c1-500; grep -q VERIFIER_DONE /workspace/final/verifier_disc_$tag.log && grep -q VERIFIER_DONE /workspace/final/verifier_mslot_$tag.log && $PY pack.py --policy $R/r$rnd.pt --verifier $R/verifier_mslot.pt:$(cat $F/final/thr_mslot),$R/verifier_disc.pt:$(cat $F/final/thr_disc) --out /workspace/final/final_$tag.pt --solver \"\$(cat /workspace/solver.json)\" --note '$tag: policy r$rnd of $run, verifiers from all its rounds'"
fi
"${SSH[@]}" "$h" "cd /workspace/fr && for m in $modes; do SOLVER_MODE=\$m LEAN_GATE_WORKERS=7 timeout 1800 $PY solve.py --ckpt /workspace/final/final_$tag.pt --in /workspace/data/y_dev.jsonl --out /workspace/final/dev_${tag}_\$m.jsonl --k 5 --seed 0 --judge 2>&1 | grep -v Warn > /workspace/final/dev_${tag}_\$m.log; echo \"== $tag r$rnd \$m\"; grep -v '^{\|^---' /workspace/final/dev_${tag}_\$m.log | head -3; done; cd /workspace/final && md5sum final_$tag.pt dev_${tag}_*.jsonl > md5_$tag.txt"
rsync -rlpt -e "ssh -F $F/ssh_config" "$h:/workspace/final/" $F/final/$tag/
(cd $F/final/$tag && md5sum -c md5_$tag.txt | tr '\n' ' ')
echo "FINALIZE_OK $tag r$rnd $(date -u +%FT%TZ)"
