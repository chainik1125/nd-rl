#!/bin/bash
cd /root/nd/code
D=/root/nd/data; R=/root/nd/runs; RES=/root/nd/results
export PYTHONUNBUFFERED=1; mkdir -p $D $R $RES
python3 -m fol.gen_data --n 250000 --consts abcde --min-lines 2 --max-lines 18 --seed 3 --out $D/fol_pool2.jsonl 2>$D/g2.log
echo pool done
python3 -m fol.make_splits_struct $D/fol_pool2.jsonl $D
echo splits done
python3 -m fol.run_fol train --train $D/fol_struct_train.jsonl --out $R/fol_struct --iters 8000 --bs 128 --layers 4 > $R/fol_struct.log 2>&1
echo trained
python3 -m fol.forensic --ckpt $R/fol_struct/ckpt.pt --targets $D/fol_eval_eigen.jsonl --k 8 --temp 0.8 --bs 1024 --limit 600 --out $RES/forensic_eigen.jsonl > $RES/forensic_eigen.txt 2>&1
python3 -m fol.forensic --ckpt $R/fol_struct/ckpt.pt --targets $D/fol_eval_noeigen.jsonl --k 8 --temp 0.8 --bs 1024 --limit 600 --out $RES/forensic_noeigen.jsonl > $RES/forensic_noeigen.txt 2>&1
echo "=== EIGEN ==="; cat $RES/forensic_eigen.txt
echo "=== NOEIGEN ==="; cat $RES/forensic_noeigen.txt
echo STRUCT DONE
