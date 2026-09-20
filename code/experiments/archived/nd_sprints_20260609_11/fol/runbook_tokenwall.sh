#!/bin/bash
# FOL constant token-wall experiment.
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
RES=/root/nd/results
export PYTHONUNBUFFERED=1
mkdir -p $D $R $RES

python3 -m fol.gen_data --n 250000 --consts abcde --min-lines 2 --max-lines 16 --seed 0 --out $D/fol_pool.jsonl 2>$D/folgen.log
echo "pool done"
python3 -m fol.make_splits_fol $D/fol_pool.jsonl $D
echo "splits done"

python3 -m fol.run_fol train --train $D/fol_train_bounded.jsonl --out $R/fol_bounded --iters 6000 --bs 128 --layers 4 > $R/fol_bounded.log 2>&1
echo "bounded trained"
python3 -m fol.run_fol train --train $D/fol_train_full.jsonl --out $R/fol_full --iters 6000 --bs 128 --layers 4 > $R/fol_full.log 2>&1
echo "full trained"

for model in bounded full; do
  for ev in iv oov; do
    python3 -m fol.run_fol eval --ckpt $R/fol_$model/ckpt.pt --targets $D/fol_eval_$ev.jsonl --k 8 --temp 0.8 --bs 1024 --limit 600 --out $RES/fol_${model}_${ev}.jsonl > $RES/log_fol_${model}_${ev}.txt 2>&1
    echo "eval $model $ev: $(grep SOLVED $RES/log_fol_${model}_${ev}.txt)"
  done
done
echo "FOL TOKENWALL DONE"
