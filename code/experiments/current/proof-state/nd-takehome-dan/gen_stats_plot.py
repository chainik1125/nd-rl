#!/usr/bin/env python3
"""Generator/data histograms for the writeup (figures/data_stats.png) from data/splits_stats.json + data/train.jsonl."""
import json, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st = json.load(open('data/splits_stats.json'))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
# 1. length histograms of every pool
for name, col in (('train', '#d1782f'), ('heldout', '#f2b27a'), ('rl_targets', '#1f5fbf'), ('transfer', '#8fb3e6')):
    h = st[name + '_by_len']
    ks = sorted(int(k) for k in h)
    axes[0].bar([k + {'train': -0.3, 'heldout': -0.1, 'rl_targets': 0.1, 'transfer': 0.3}[name] for k in ks], [h[str(k)] for k in ks], width=0.2, color=col, label=f'{name} (n={st[name+"_n"]:,})')
axes[0].set_yscale('log'); axes[0].set_xlabel('proof length of the generating proof (lines)'); axes[0].set_ylabel('theorems'); axes[0].legend(fontsize=7); axes[0].axvline(6.5, color='k', ls=':', lw=.8)
axes[0].set_title('Pools by length (cap 6 = dotted line)', fontsize=9)
# 2. rule usage: fraction of proofs containing the rule
for name, col, off in (('train', '#d1782f', -0.2), ('rl_targets', '#1f5fbf', 0.2)):
    rc = st[name + '_rules']; n = st[name + '_n']
    rules = ['AS', 'IMPI', 'IMPE', 'ANDI', 'ANDE1', 'ANDE2', 'ORI1', 'ORI2', 'ORE', 'NEGI', 'NEGE', 'DN', 'BOTE', 'R']
    axes[1].bar([i + off for i in range(len(rules))], [rc.get(r, 0) / n for r in rules], width=0.4, color=col, label=name)
axes[1].set_xticks(range(len(rules))); axes[1].set_xticklabels(rules, rotation=60, fontsize=7); axes[1].set_ylabel('fraction of proofs using the rule'); axes[1].legend(fontsize=7)
axes[1].set_title('Rule usage', fontsize=9)
# 3. premises
for name, col, off in (('train', '#d1782f', -0.2), ('rl_targets', '#1f5fbf', 0.2)):
    h = st[name + '_n_prem']; n = st[name + '_n']
    axes[2].bar([int(k) + off for k in h], [h[k] / n for k in h], width=0.4, color=col, label=name)
axes[2].set_xlabel('number of premises'); axes[2].set_ylabel('fraction of theorems'); axes[2].legend(fontsize=7); axes[2].set_title('Premise count', fontsize=9)
fig.tight_layout(); fig.savefig('figures/data_stats.png', dpi=150)
print('figures/data_stats.png')
