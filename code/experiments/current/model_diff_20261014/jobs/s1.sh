export PYTORCH_ALLOC_CONF=expandable_segments:True
python cc.py --models g0_pre,g0_fin --layer 3 --eval_only --out runs/g0_L3
for L in 1 2 4; do python cc.py --models g0_pre,g0_fin --layer $L --tokens 40e6 --oversample lad3:3 --out runs/g0_L$L; done
