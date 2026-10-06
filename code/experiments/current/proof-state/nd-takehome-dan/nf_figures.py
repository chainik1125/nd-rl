#!/usr/bin/env python3
"""Figures for run noise-floor from artifacts/nf/summary.json (nf_analysis.py) ->
figures/nf_cells.png (the eight null cells per quantity, with the control's published value marked)
and figures/nf_resolvable.png (the resolvable-difference table as a chart, with every standing
finding this run retests plotted against its own floor).
Categorical colours = the dataviz reference palette slots used by dsg_figures.py, in the same fixed order
(blue = null cells, orange = a standing finding below its floor, aqua = above it); identity is never
colour-alone: Stage-1 seed 0 is a circle and seed 1 a triangle, gap-closers are hollow diamonds, and
findings below the floor are additionally drawn hollow."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = json.load(open('artifacts/nf/summary.json'))
F = S['floors']
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})

# quantity -> (label, control's published value and the model it was measured on, extra hollow points)
# every published value is the C0 control of ds-generator / ds-composition: ckpts/lf/stage1_a1_seq_s{0,1}.pt,
# 3.3 M-parameter lean_seq GPT trained on data/p2/train_depth3_f0_a1.jsonl.
ROWS = [
    ('ladder_frozen_transfer_solved', 'frozen ladder, transfer solved / 2,285', [158, 114]),
    ('ladder_frozen_transfer_lstar', 'frozen ladder, transfer $L^*$', [9, 9]),
    ('cov_red_solved', '`targets_reductio_req` solved / 300', [31, 27]),
    ('cov_req8_solved', '`depth3_req` required@8 solved / 300', [165, 113]),
    ('heldout_depth3_slice', 'held-out depth-3 slice (n = 500)', [0.488, 0.272]),
    ('heldout_len6', 'held-out greedy, 6-line bin', [0.686, 0.584]),
    ('heldout_len6_nopattern', 'held-out greedy, 6-line no-pattern', [0.818, 0.834]),
    ('heldout_overall', 'held-out greedy, overall', [0.9088, 0.8968]),
]

# ---------- figure 1: the eight null cells per quantity, as ratios to their own mean ----------
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ylab = []
for i, (q, lab, ctrl) in enumerate(ROWS):
    f = F.get(q)
    if not f:
        continue
    y = len(ROWS) - i
    m = f['mean']
    vals = f['values']
    ax.hlines(y, f['min'] / m, f['max'] / m, color='#d7d6d0', lw=6, zorder=1)
    for k, v in vals.items():
        pre = k.endswith('_s0') or k.endswith('_s1')          # the pre-registered 4 x 2 design
        ax.plot(v / m, y, marker='o' if pre else 'o', ms=6 if pre else 4,
                mfc=BLUE if pre else 'none', mec=BLUE if pre else BLUE, mew=0.6 if pre else 1.0,
                color=BLUE, ls='', zorder=3 if pre else 2)
    for c in ctrl:
        ax.plot(c / m, y, marker='|', ms=15, color=ORANGE, mew=2.0, zorder=4)
    g = S['gap_closers'].get('la_frozen_dsg_g1_s0')
    if q == 'ladder_frozen_transfer_solved' and g:
        ax.plot(g['transfer_solved'] / m, y, marker='D', ms=6, mfc='none', mec=AQUA, mew=1.4, zorder=4)
        ax.plot(170 / m, y, marker='D', ms=6, mfc='none', mec=AQUA, mew=1.4, zorder=4)
    ylab.append((y, f'{lab}\n{f["n_cells"]} cells: mean {m:.4g}, sd {f["sd"]:.3g}, max/min {f["max_over_min"]:.2f}×'))
ax.axvline(1.0, color='#c9c8c2', lw=0.8, zorder=0)
ax.set_yticks([y for y, _ in ylab]); ax.set_yticklabels([t for _, t in ylab], fontsize=7.5)
ax.set_xlabel('value ÷ the mean of that quantity\'s null cells  (everything on this chart is noise by construction)')
ax.grid(axis='x', color='#e6e5e0', lw=0.6)
h = [plt.Line2D([], [], marker='o', ls='', color=BLUE, label='null cell, Stage-1 seed 0 or 1 (pre-registered 4 × 2)'),
     plt.Line2D([], [], marker='o', ls='', ms=4, mfc='none', mec=BLUE, label='null cell, Stage-1 seed 2–12 (addenda 1–2, held-out only)'),
     plt.Line2D([], [], marker='|', ls='', color=ORANGE, mew=2, label="C0 control's published value (both seeds)"),
     plt.Line2D([], [], marker='D', ls='', mfc='none', mec=AQUA, label='ds-generator G1 (fifth independent draw)')]
ax.legend(handles=h, frameon=False, fontsize=7.5, loc='lower right')
fig.suptitle('noise-floor: four independent null pools × up to thirteen Stage-1 seeds — same generator settings, same composition, same schedule',
             fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('figures/nf_cells.png', dpi=160); plt.close(fig)

# ---------- figure 2: one row per standing finding, against the floor of its own quantity ----------
import nf_retro

RET = nf_retro.rows()
LABEL = dict((q, l) for q, l, _ in ROWS)
fig, ax = plt.subplots(figsize=(13.2, 6.4))
ticks = []
order = sorted(range(len(RET)), key=lambda i: (LABEL.get(RET[i]['q'], RET[i]['q']), -RET[i]['effect_pct']))
for row, i in enumerate(order):
    r = RET[i]
    y = len(order) - row
    fl, eff = r['mdd_pct'], r['effect_pct']
    ax.barh(y, fl, height=0.62, color='#e2e1db', edgecolor='#c9c8c2', lw=0.6, zorder=1)
    surv = r['verdict'] == 'survives'
    ax.plot(eff, y, marker='o', ms=8, mfc=AQUA if surv else 'none', mec=AQUA if surv else ORANGE,
            mew=1.8, zorder=3)
    ticks.append((y, f"{r['run']} — {r['claim']}  ({r['orig']})"))
    ax.text(max(fl, eff) * 1.12, y, LABEL.get(r['q'], r['q']), va='center', fontsize=6.2, color=MUTED)
ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([t for _, t in ticks], fontsize=6.9)
ax.set_ylim(0.3, len(order) + 0.7)
ax.set_xscale('log'); ax.set_xlim(0.8, 1500)
ax.set_xlabel('difference as % of the control value (log scale)\n'
              'grey bar: the smallest difference resolvable at 80 % power   •   point: the difference reported', fontsize=8)
ax.grid(axis='x', color='#e6e5e0', lw=0.6)
h = [plt.Line2D([], [], marker='o', ls='', ms=8, mfc='none', mec=ORANGE, mew=1.8,
                label='reported difference is inside the floor — not resolvable by that run'),
     plt.Line2D([], [], marker='o', ls='', ms=8, color=AQUA, label='reported difference clears the floor — survives')]
ax.legend(handles=h, frameon=False, fontsize=7.5, loc='lower right')
fig.suptitle('noise-floor: every standing finding this run can score, against the floor of the quantity it was read off',
             fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('figures/nf_resolvable.png', dpi=160); plt.close(fig)
print('figures/nf_cells.png figures/nf_resolvable.png')
