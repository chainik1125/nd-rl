export PYTORCH_ALLOC_CONF=expandable_segments:True
while pgrep -f "jobs/s[1245].sh" > /dev/null; do sleep 30; done
for r in g0_L2 e1_L3 g1_L3 diff_g1_L3 g0_L4 g1_L5 diff_e1_L3; do
  if [ -f runs/$r/weights.pt ] && [ ! -f runs/$r/cc.pt ]; then
    P=$(python -c "import torch;print(','.join(torch.load('runs/$r/weights.pt',map_location='cpu')['names']))")
    L=$(python -c "import torch;print(torch.load('runs/$r/weights.pt',map_location='cpu')['args']['layer'])")
    EX=$(python -c "import torch;a=torch.load('runs/$r/weights.pt',map_location='cpu')['args'];print(('--diff --m %d --k %d' % (a['m'],a['k'])) if a.get('diff') else '')")
    python cc.py --models $P --layer $L $EX --eval_only --out runs/$r
  fi
done
echo ALLDONE
