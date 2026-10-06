#!/bin/bash
# Verifier stage. dev: bash run_verifier_stage.sh dev ; (log a freeze) ; test: bash run_verifier_stage.sh test
set -e
S=/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006
cd ~/runs/proof-selection; PY=~/venvs/proof-state/bin/python
V=$(ls ~/runs/proof-selection/scores/ | grep -E "^dev1108__s0_r16__k16__(verifier|shallow|completion_verifier)" | sed "s/^dev1108__s0_r16__k16__//; s/.jsonl$//" | paste -sd,)
echo "verifier scorers: $V"
if [ "$1" = dev ]; then
  [ -f stopping/tau_grid_dev_v.json ] || cp stopping/tau_grid_dev.json stopping/tau_grid_dev_v.json
  $PY $S/code/stopping.py --pool pools_v2/dev1108__s0_r16__k16.jsonl --split dev --scorers $V --grid stopping/tau_grid_dev_v.json --write_grid stopping/tau_grid_dev_v.json --B 10 --out_dir stopping/dev/gridpass_v >/dev/null
  $PY $S/code/stopping.py --pool pools_v2/rr600__s0_r16__k32.jsonl --split dev --scorers $V --grid stopping/tau_grid_dev_v.json --write_grid stopping/tau_grid_dev_v.json --B 10 --out_dir stopping/dev/gridpass_v >/dev/null
  for p in dev1108__s0_r16__k16 rr600__s0_r16__k32; do
    $PY $S/code/stopping.py --pool pools_v2/$p.jsonl --split dev --scorers $V --grid stopping/tau_grid_dev_v.json
    $PY $S/code/auroc.py --pool pools_v2/$p.jsonl --split dev --out $S/results/auroc_${p%%__*}__dev.json
  done
  $PY $S/code/stopping_headline.py --dev stopping/dev1108__s0_r16__k16__dev stopping/rr600__s0_r16__k32__dev --test stopping/dev1108__s0_r16__k16__dev stopping/rr600__s0_r16__k32__dev --scorers completion_loglik,$V --out $S/results/stopping_headline_devonly_v.json
else
  for p in holdout250__s0_r16__k64 textbook72__s0_r16__k64; do
    $PY $S/code/stopping.py --pool pools_v2/$p.jsonl --scorers $V --grid stopping/tau_grid_dev_v.json
    $PY $S/code/auroc.py --pool pools_v2/$p.jsonl --out $S/results/auroc_$p.json
    $PY $S/code/curve_table.py --prefix stopping/$p --scorers completion_loglik,completion_loglik_sum,completion_minstep,loglik_mean,$V --md $S/results/curve_table_$p.md > /dev/null
  done
  $PY $S/code/stopping_headline.py --dev stopping/dev1108__s0_r16__k16__dev stopping/rr600__s0_r16__k32__dev --test stopping/holdout250__s0_r16__k64 stopping/textbook72__s0_r16__k64 --scorers completion_loglik,completion_minstep,completion_loglik_sum,completion_entropy,shortest,loglik_mean,loglik_sum,loglik_min_step,entropy_mean,loglik_mean_full,$V --out $S/results/stopping_headline.json
  for p in holdout250__s0_r16__k64 textbook72__s0_r16__k64; do set=${p%%__*}
    $PY $S/code/stopping_figs.py --prefix stopping/$p --title "$set (test), r16 s0 generator, T 0.8, k=64" --loglik completion_loglik --verifier verifier_ens --headline $S/results/stopping_headline.json --out $S/figures/stopping_$set
  done
  $PY $S/code/report_tables.py completion,completion_loglik,completion_loglik_sum,completion_minstep,loglik_mean,$V > $S/results/worker_B_tables.md
fi
