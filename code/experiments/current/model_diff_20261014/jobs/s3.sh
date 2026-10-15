export PYTORCH_ALLOC_CONF=expandable_segments:True
python cc.py --models g0_pre,g1_pre --layer 3 --tokens 40e6 --oversample lad3:3 --out runs/x01pre_L3
