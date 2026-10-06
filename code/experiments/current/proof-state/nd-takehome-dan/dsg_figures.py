#!/usr/bin/env python3
"""Figures for run ds-generator from artifacts/dsg/summary.json (dsg_analysis.py) -> figures/dsg_shape.png, figures/dsg_readiness.png.
Categorical colours = the dataviz reference palette slots 1-3 in fixed order (C0 blue, G1 orange, G2 aqua; validated in the
reference palette file; no node on the VPS to re-run the validator); identity is never colour-alone: every bar group is labelled,
seeds are two markers, frozen is hollow."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = json.load(open('artifacts/dsg/summary.json'))
ARMS = ('c0', 'g1', 'g2'); COL = {'c0': '#2a78d6', 'g1': '#eb6834', 'g2': '#1baf7a'}
NAME = {'c0': 'C0 control (a1)', 'g1': 'G1 ORE-shape knobs (null at cap 6)', 'g2': 'G2 ladder generator at cap 6'}
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})

# ---------- figure 1: shape tables as grouped bars ----------
sh = {a: S['sets'][a]['shape'] for a in ARMS if S['sets'][a]['shape']}
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6), gridspec_kw={'width_ratios': [1.6, 0.9, 0.9]})
rules = ('AS', 'IMPI', 'IMPE', 'ANDI', 'ORI1', 'NEGI', 'NEGE', 'DN', 'ANDE1', 'BOTE', 'R', 'ORE')
x = np.arange(len(rules)); w = 0.26
for i, a in enumerate(sh):
    ax[0].bar(x + (i - 1) * w, [100 * sh[a]['rule_share'][r] for r in rules], w - 0.03, color=COL[a], label=NAME[a], lw=0)
ax[0].set_xticks(x); ax[0].set_xticklabels(rules, fontsize=7.5); ax[0].set_ylabel('% of proofs containing the rule'); ax[0].set_title('(a) per-proof rule shares', loc='left', fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5); ax[0].grid(axis='y', color='#e6e5e0', lw=0.6)
for i, a in enumerate(sh):
    ax[1].bar(np.arange(3) + (i - 1) * w, [100 * sh[a]['box_depth_hist'].get(str(b), 0) / sh[a]['n'] for b in range(3)], w - 0.03, color=COL[a], lw=0)
    ax[2].bar(np.arange(4) + (i - 1) * w, [100 * sh[a]['n_prem_hist'].get(str(b), 0) / sh[a]['n'] for b in range(4)], w - 0.03, color=COL[a], lw=0)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['0', '1', '2']); ax[1].set_xlabel('box depth'); ax[1].set_title('(b) box-depth histogram (%)', loc='left', fontsize=9); ax[1].grid(axis='y', color='#e6e5e0', lw=0.6)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(['0', '1', '2', '3']); ax[2].set_xlabel('premises'); ax[2].set_title('(c) premise-count histogram (%)', loc='left', fontsize=9); ax[2].grid(axis='y', color='#e6e5e0', lw=0.6)
fig.suptitle('ds-generator: shape of the three 155,000-proof sets (flat 31,000 per length 2–6, depth-3 excluded)', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('figures/dsg_shape.png', dpi=160); plt.close(fig)

# ---------- figure 2: readiness panel ----------
rows = {(r['arm'], r['seed']): r for r in S['rows']}
panels = [('held-out greedy\n(pattern-free / all)', lambda r: (r['heldout']['by_pattern']['none']['rate'], r['heldout']['rate']) if r['heldout'] else None, (0, 1)),
          ('depth-3 pass@2,000\n(1,000 pool / req8)', lambda r: (r['coverage']['d3']['targets_hit'] / r['coverage']['d3']['n_targets'], r['coverage']['req8']['targets_hit'] / r['coverage']['req8']['n_targets']) if r['coverage']['d3'] and r['coverage']['req8'] else None, (0, 1)),
          ('reductio-req targets hit / 300\n(pass@2,000)', lambda r: (r['coverage']['red']['targets_hit'], None) if r['coverage']['red'] else None, None),
          ('dial round 4: depth-3 acquisition\nEI (filled) / frozen (hollow)', lambda r: (r['dial']['ei']['acq_round4'], r['dial']['frozen']['acq_round4']) if r['dial'] else None, (0, 0.7)),
          ('ladder transfer solved / 2,285\nT1 (filled) / frozen (hollow)', lambda r: (r['ladder']['T1']['transfer_solved'], r['ladder']['frozen']['transfer_solved']) if r['ladder'] and r['ladder']['T1'] and r['ladder']['frozen'] else None, None),
          ('ladder transfer L*\nT1 (filled) / frozen (hollow)', lambda r: (r['ladder']['T1']['lstar_transfer'], r['ladder']['frozen']['lstar_transfer']) if r['ladder'] and r['ladder']['T1'] and r['ladder']['frozen'] else None, (6, 13))]
fig, ax = plt.subplots(1, len(panels), figsize=(15, 3.4))
for a_, (title, fn, ylim) in zip(ax, panels):
    for i, arm in enumerate(ARMS):
        for s, mk in ((0, 'o'), (1, 's')):
            r = rows.get((arm, s))
            v = fn(r) if r else None
            if not v: continue
            a_.plot([i - 0.12 + 0.24 * s], [v[0]], marker=mk, ms=7, color=COL[arm], mec=COL[arm], mfc=COL[arm], lw=0)
            if v[1] is not None:
                a_.plot([i - 0.12 + 0.24 * s], [v[1]], marker=mk, ms=7, color=COL[arm], mec=COL[arm], mfc=SURF, mew=1.6, lw=0)
                a_.plot([i - 0.12 + 0.24 * s] * 2, [v[1], v[0]], color=COL[arm], lw=1, alpha=0.6)
    a_.set_xticks(range(3)); a_.set_xticklabels(['C0', 'G1', 'G2']); a_.set_title(title, fontsize=8.5, loc='left'); a_.grid(axis='y', color='#e6e5e0', lw=0.6); a_.set_xlim(-0.6, 2.6)
    if ylim: a_.set_ylim(*ylim)
ax[0].plot([], [], 'o', color=MUTED, label='seed 0'); ax[0].plot([], [], 's', color=MUTED, label='seed 1'); ax[0].legend(frameon=False, fontsize=7.5, loc='lower left')
fig.suptitle('ds-generator: readiness panel per arm (two Stage-1 seeds; filled = EI / T1 / all, hollow = frozen / pattern-free)', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('figures/dsg_readiness.png', dpi=160); plt.close(fig)
print('wrote figures/dsg_shape.png figures/dsg_readiness.png')
