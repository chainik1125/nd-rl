"""ICL capability ladder figure with our weak specialized model anchored on it.

Frontier models: solve rate from icl_w4.jsonl (40-shot, 2 samples, n=135).
Weak 3.3M model: greedy (pass@1) and pass@8 on the same 135 targets.
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

icl = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
weak_base = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]      # greedy
weak_base8 = [json.loads(l) for l in open(sys.argv[3]) if l.strip()]     # pass@8
out = sys.argv[4]

bm = defaultdict(lambda: [0, 0])
for r in icl:
    bm[r['model']][0] += r['solved']
    bm[r['model']][1] += 1
rates = {m: s / n for m, (s, n) in bm.items()}

wb = sum(r['solved'] for r in weak_base) / len(weak_base)
wb8 = sum(r['solved'] for r in weak_base8) / len(weak_base8)

# assemble bars sorted by rate
bars = [(m, rates[m], 'frontier') for m in rates]
bars.append(('weak-3.3M (greedy)', wb, 'weak'))
bars.append(('weak-3.3M (pass@8)', wb8, 'weak'))
bars.sort(key=lambda x: x[1])

plt.figure(figsize=(9, 4.6))
colors = ['tab:purple' if t == 'weak' else 'tab:gray' for _, _, t in bars]
plt.barh([b[0] for b in bars], [b[1] for b in bars], color=colors)
for i, (_, v, _) in enumerate(bars):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
plt.xlabel('ND-completion solve rate on 135 held-out targets (verifier-judged)')
plt.title('Capability ladder: a 3.3M specialized prover vs frontier LLMs (40-shot)\n'
          '(purple = our weak model; frontier = 2 samples; GPT is format-penalized)',
          fontsize=10)
plt.xlim(0, 0.9)
plt.tight_layout()
plt.savefig(out, dpi=160)
print('wrote', out)
print('weak base greedy', round(wb, 3), 'pass@8', round(wb8, 3))
