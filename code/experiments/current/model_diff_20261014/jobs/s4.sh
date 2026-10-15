export PYTORCH_ALLOC_CONF=expandable_segments:True
for P in g0 g1 e1; do python cc.py --models ${P}_pre,${P}_fin --diff --layer 3 --m 4096 --k 16 --tokens 30e6 --oversample lad3:3 --out runs/diff_${P}_L3; done
