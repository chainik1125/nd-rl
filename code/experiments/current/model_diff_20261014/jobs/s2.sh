export PYTORCH_ALLOC_CONF=expandable_segments:True
python cc.py --models e1_pre,e1_fin --layer 5 --eval_only --out runs/e1_L5
python cc.py --models g1_pre,g1_fin --layer 3 --tokens 40e6 --oversample lad3:3 --out runs/g1_L3
python cc.py --models e1_pre,e1_fin --layer 3 --tokens 40e6 --oversample lad3:3 --out runs/e1_L3
python cc.py --models g1_pre,g1_fin --layer 5 --tokens 40e6 --oversample lad3:3 --out runs/g1_L5
