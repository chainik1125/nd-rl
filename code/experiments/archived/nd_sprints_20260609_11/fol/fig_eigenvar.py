import json
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(p):
    return [json.loads(l) for l in open(p)]


eig = load(sys.argv[1])
noe = load(sys.argv[2])
out = sys.argv[3]


def by_len(rows):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        g[r['n_lines']][0] += r['solved']
        g[r['n_lines']][1] += 1
    return g


ge, gn = by_len(eig), by_len(noe)
xs = [b for b in sorted(set(ge) | set(gn))
      if 3 <= b <= 8 and ge.get(b, [0, 0])[1] >= 8 and gn.get(b, [0, 0])[1] >= 8]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot(xs, [ge[b][0] / ge[b][1] for b in xs], 'o-', color='tab:red',
           label='needs eigenvariable rule (forall-I / exists-E)')
ax[0].plot(xs, [gn[b][0] / gn[b][1] for b in xs], 's-', color='tab:blue',
           label='no eigenvariable rule')
ax[0].set_xlabel('proof length (lines)')
ax[0].set_ylabel('solve rate (pass@8, verifier-judged)')
ax[0].set_title('Solve rate: shared length cliff,\nonly a small eigenvariable gap', fontsize=11)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_ylim(0.6, 1.02)


def evrate(rows):
    ev = sum(r['cats'].get('EIGENVAR-VIOLATION', 0) for r in rows)
    tot = sum(sum(r['cats'].values()) for r in rows)
    return ev / tot if tot else 0


vals = [evrate(eig), evrate(noe)]
ax[1].bar(['eigenvariable\ntargets', 'non-eigenvariable\ntargets'], vals,
          color=['tab:red', 'tab:blue'])
for i, vv in enumerate(vals):
    ax[1].text(i, vv + .005, f'{vv:.2f}', ha='center')
ax[1].set_ylabel('share of failed attempts that are\neigenvariable-freshness violations')
ax[1].set_title('The eigenvariable error signature\n(verifier-caught freshness violations)', fontsize=11)
ax[1].grid(alpha=0.3, axis='y')
ax[1].set_ylim(0, 0.28)
fig.suptitle('Eigenvariable structural wall: not a solve cliff, but a distinctive freshness-violation error mode', fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=150)
print('wrote', out)
