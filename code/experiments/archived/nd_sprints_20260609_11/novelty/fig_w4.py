"""W4 figure: two orthogonal axes — difficulty predicts ICL ladder, interest does not."""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out = sys.argv[2]

byh = defaultdict(list)
for r in rows:
    byh[r['ladder_height']].append(r)
heights = sorted(byh)


def mean(h, k):
    vs = [x[k] for x in byh[h] if x.get(k) is not None]
    return sum(vs) / len(vs) if vs else None


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
# left: difficulty measures rise with ladder height (twin axis for bits)
axes[0].plot(heights, [mean(h, 'min_lines') for h in heights], 'o-', color='tab:blue', label='min proof length (lines)')
axes[0].plot(heights, [mean(h, 'genus') for h in heights], '^-', color='tab:green', label='genus')
axes[0].set_ylim(0, 13)
axes[0].set_xlabel('ICL ladder height (0=weakest model solves .. 6=none do)')
axes[0].set_ylabel('min lines / genus', color='tab:blue')
axes[0].set_title('Difficulty measures track the ICL ladder\n'
                  r'($\rho$: bits +0.52, min-lines +0.50, genus +0.39)', fontsize=11)
ax0b = axes[0].twinx()
ax0b.plot(heights, [mean(h, 'mdl_bits') for h in heights], 's-', color='tab:red', label='MDL bits')
ax0b.set_ylabel('MDL bits', color='tab:red')
ax0b.set_ylim(60, 170)
lines = axes[0].get_lines() + ax0b.get_lines()
axes[0].legend(lines, [l.get_label() for l in lines], fontsize=8, loc='upper left')
axes[0].grid(alpha=0.3)

# right: lemma-value flat
axes[1].plot(heights, [mean(h, 'lemma_value') for h in heights], 'D-', color='tab:purple', label='lemma-value (interest)')
axes[1].set_xlabel('ICL ladder height')
axes[1].set_ylabel('mean lemma-value')
axes[1].set_title('Interest (lemma-value) is ORTHOGONAL to the ladder\n'
                  r'($\rho = -0.04$; vs min-lines $\rho=0.008$)', fontsize=11)
axes[1].set_ylim(0, max(mean(h, 'lemma_value') or 0 for h in heights) * 1.3)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

fig.suptitle('W4: two independent axes — proof difficulty vs proof interest', fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=160)
print('wrote', out)
