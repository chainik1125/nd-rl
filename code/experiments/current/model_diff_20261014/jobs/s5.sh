export PYTORCH_ALLOC_CONF=expandable_segments:True
while pgrep -f "x01pre_L3" > /dev/null; do sleep 30; done
python cc.py --models g1_pre,g1_fin --layer 3 --eval_only --out runs/g1_L3
