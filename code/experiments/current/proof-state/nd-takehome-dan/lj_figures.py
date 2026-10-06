#!/usr/bin/env python3
"""Two figures for run lean-judge: what the Lean-only class is made of, and what the judge costs.
Reads artifacts/lj/t3_classify.json and artifacts/lj/t6_throughput.json; writes figures/lj_*.png."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
C = json.load(open('artifacts/lj/t3_classify.json'))
T = json.load(open('artifacts/lj/t6_throughput.json'))

SHORT = {'omitted premise re-statement': 'omitted premise\nre-statement',
         'BOTE on a non-F line (Not.elim)': 'BOTE on a non-F line\n(Not.elim)',
         'no ND denotation (outside the lean_seq grammar; excluded)': 'no ND denotation\n(excluded)',
         'other': 'other',
         'NEGE on A and A>F (Lean: ~A is A→False)': '`NEGE` on A and A>F\n(¬A is A→False)'}
items = list(C['classes'].items())[::-1]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
cols = ['#b0b0b0' if 'excluded' in k else '#3b6ea5' for k, _ in items]
ax.barh([SHORT.get(k, k) for k, _ in items], [v for _, v in items], color=cols)
for i, (k, v) in enumerate(items):
    ax.text(v + 60, i, f'{v:,}  ({100*v/C["n"]:.1f} %)', va='center', fontsize=8)
ax.set_xlim(0, max(v for _, v in items) * 1.42)
ax.set_xlabel('distinct stored samples')
ax.set_title(f'Proofs Lean accepts and nd_verify rejects, n = {C["n"]:,} stored samples\n'
             'five runs, all lean_seq from-scratch; grey = outside the grammar, counted by neither judge',
             fontsize=8.5)
ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig('figures/lj_leanonly_classes.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.0))
lab = ['old: nd_verify pass\n(dropped)', 'new: registry hit', 'new: marker', 'new: fallback\n(nd2lean + Lean)']
val = [T['old_nd_verify_per_1000_s'], T['new_registry_hit_per_1000_s'], T['new_marker_per_1000_s'], T['new_fallback_per_1000_s']]
ax.barh(lab[::-1], val[::-1], color=['#3b6ea5', '#3b6ea5', '#7aa8d0', '#c26a3a'][::-1])
for i, v in enumerate(val[::-1]):
    ax.text(v * 1.25, i, f'{v:.4g} s', va='center', fontsize=8)
ax.set_xscale('log'); ax.set_xlim(1e-3, 30)
ax.set_xlabel('seconds per 1,000 strings judged  (log scale)')
ax.set_title(f'Judging cost, n = {T["n"]:,} stored proofs, A40 pod, 6 Lean workers\n'
             'the gate’s Lean run on the literal texts is unchanged and not shown', fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig('figures/lj_throughput.png', dpi=160); plt.close(fig)
print('figures/lj_leanonly_classes.png figures/lj_throughput.png')
