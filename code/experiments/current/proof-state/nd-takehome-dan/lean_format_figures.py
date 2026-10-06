#!/usr/bin/env python3
"""Figures for run lean-format, drawn from artifacts/lf/summary.json (lean_format_analysis.py) -> figures/lean_format.png.
Categorical colours = the dataviz reference palette slots 1-3 in fixed order (token blue, lean_seq orange, lean_rand aqua);
identity is never colour-alone: every series is direct-labelled or in the legend, frozen controls are dashed."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = json.load(open('artifacts/lf/summary.json'))
COL = {'token': '#2a78d6', 'seq': '#eb6834', 'rand': '#1baf7a'}
NAME = {'token': 'token format (abs)', 'seq': 'Lean, ordered names (lean_seq)', 'rand': 'Lean, random labels (lean_rand)'}
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={'width_ratios': [1.15, 1, 1]})

# (a) depth-3 acquisition by round
a = ax[0]; P2 = S['P2_depth3']
for fmt in ('token', 'seq', 'rand'):
    for kind, ls, lw in (('ei', '-', 2), ('frozen', (0, (4, 2)), 1.3)):
        for s in (0, 1):
            v = P2.get(f'{fmt}_{kind}_s{s}')
            if not v:
                continue
            y = v['acq_by_round']
            a.plot(range(1, len(y) + 1), y, color=COL[fmt], ls=ls, lw=lw, marker='o' if kind == 'ei' else None, ms=3.5, label=(NAME[fmt] if (kind == 'ei' and s == 0) else None))
ends = {fmt: [P2[f'{fmt}_ei_s{s}']['acq'] for s in (0, 1) if f'{fmt}_ei_s{s}' in P2] for fmt in ('token', 'seq', 'rand')}
ypos = {'seq': 0.492, 'rand': 0.447, 'token': 0.35}
for fmt, v in ends.items():
    if v:
        a.text(8.15, ypos[fmt], ' / '.join(f'{x:.3f}' for x in v), fontsize=7.5, color=INK, va='center')
a.plot([], [], color=MUTED, ls=(0, (4, 2)), lw=1.3, label='frozen control (same attempts)')
a.axhspan(0.271, 0.364, color='#2a78d6', alpha=0.08, lw=0, label='range of all 8 token f = 0 arms (0.271–0.364)')
a.set_xlabel('EI round (32 attempts per target each)'); a.set_ylabel('depth-3 acquisition (fraction of 1,000 targets)')
a.set_title('(a) Depth-3 f = 0 dial: targets solved with a depth-3 proof', fontsize=9, loc='left'); a.grid(axis='y', color='#e6e5e0', lw=0.6); a.set_xlim(0.8, 9.6); a.set_ylim(-0.01, 0.62); a.legend(frameon=False, fontsize=7.5, loc='upper left')

# (b) ladder: transfer theorems solved by L_true bin
b = ax[1]; P3 = S['P3_ladder']
TOK = {'s0': {7: 140, 8: 137, 9: 300, 10: 34, 11: 1, 12: 0}, 's1': {7: 139, 8: 138, 9: 312, 10: 34, 11: 0, 12: 0}}     # ladder.md (branch dan_ladder_a), la_T1_s0 / s1
NB = {7: 300, 8: 300, 9: 1010, 10: 451, 11: 99, 12: 102}
bins = [7, 8, 9, 10, 11, 12]
for s in ('s0', 's1'):
    b.plot(bins, [TOK[s][L] / NB[L] for L in bins], color=COL['token'], lw=2, marker='o', ms=3.5, label=NAME['token'] + ' T1' if s == 's0' else None)
for fmt in ('seq', 'rand'):
    for s in (0, 1):
        v = P3.get(f'{fmt}_T1_s{s}')
        if v:
            bb = {int(k): c for k, c in v['transfer_by_bin'].items()}
            b.plot(bins, [bb.get(L, 0) / NB[L] for L in bins], color=COL[fmt], lw=2, marker='o', ms=3.5, label=NAME[fmt] + f" T1 (L* {v['lstar_transfer']})" if s == 0 else None)
        v = P3.get(f'{fmt}_frozen_s{s}')
        if v:
            bb = {int(k): c for k, c in v['transfer_by_bin'].items()}
            b.plot(bins, [bb.get(L, 0) / NB[L] for L in bins], color=COL[fmt], lw=1.3, ls=(0, (4, 2)))
b.set_xlabel('minimal proof length L_true of the transfer theorem'); b.set_ylabel('fraction solved in 256 attempts')
b.set_title('(b) Ladder rung T1, transfer pool (2,285; never trained on)', fontsize=9, loc='left'); b.grid(axis='y', color='#e6e5e0', lw=0.6); b.set_ylim(-0.02, 0.72); b.plot([], [], color=MUTED, ls=(0, (4, 2)), lw=1.3, label='frozen Lean controls (token frozen: L* 7)'); b.legend(frameon=False, fontsize=7.5, loc='upper right')

# (c) mechanism: distinct verified proofs by written length, Stage-1 base model, pass@16 on the take-home transfer set
c = ax[2]; P4 = S['P4_mechanism']
series = [('token_abs', 'token abs', COL['token'], None), ('token_absfixed', 'token abs-fixed', COL['token'], '//'), ('seq', 'lean_seq', COL['seq'], None),
          ('seqfixed', 'lean_seq no offset', COL['seq'], '//'), ('rand', 'lean_rand', COL['rand'], None)]
series = [x for x in series if x[0] in P4]
L = [6, 7, 8, 9]; w = 0.8 / max(1, len(series))
for i, (k, lab, col, hatch) in enumerate(series):
    wh = {int(kk): vv for kk, vv in P4[k]['written_hist'].items()}
    xs = [j + (i - (len(series) - 1) / 2) * w for j in range(len(L))]
    bars = c.bar(xs, [wh.get(l, 0) for l in L], width=w * 0.86, color=col if not hatch else SURF, edgecolor=col, hatch=hatch, lw=1, label=lab)
    for x, l in zip(xs, L):
        c.text(x, wh.get(l, 0) + 4, str(wh.get(l, 0)), ha='center', fontsize=6.5, color=INK)
c.set_xticks(range(len(L))); c.set_xticklabels([f'{l} lines' for l in L]); c.set_ylabel('distinct verified proofs (1,638 theorems × 16 samples)')
c.set_title('(c) Base model (trained on ≤ 6 lines): lengths written', fontsize=9, loc='left'); c.grid(axis='y', color='#e6e5e0', lw=0.6); c.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('figures/lean_format.png', dpi=170)
print('figures/lean_format.png')
