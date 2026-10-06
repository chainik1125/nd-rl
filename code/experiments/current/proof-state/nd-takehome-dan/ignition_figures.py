#!/usr/bin/env python3
"""Figures for the ignition study (called from ignition_analysis.py --figs; summary.json in, PNGs out).
Palette: dataviz reference instance (categorical slots 1-3 blue/orange/aqua; sequential blue ramp for base rate)."""
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAT = ['#2a78d6', '#eb6834', '#1baf7a']
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
GREY = '#9a9a96'; INK = '#0b0b0b'; INK2 = '#52514e'
ZERO_X = 3e-7   # where r = 0 is drawn on the log axis (left edge)


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis='y', color='#e6e5e1', linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=9)


def _x(r):
    return ZERO_X if not r else r


def _seq_color(r, rates):
    pos = [v for v in rates if v]
    if not r or not pos:
        return SEQ[0]
    lo, hi = math.log10(min(pos)), math.log10(max(pos))
    t = 0.0 if hi == lo else (math.log10(r) - lo) / (hi - lo)
    return SEQ[min(len(SEQ) - 1, int(round(1 + t * (len(SEQ) - 2))))]


def acq_vs_base(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (sname, S) in zip(axes, summary.items()):
        _style(ax); used = {}
        for t in S['table']:
            if t['final_acq'] is None or t['base_rate'] is None:
                continue
            ign = t['ignition_round']
            mk = 'o' if (ign is not None and ign <= 4) else ('s' if ign is not None else 'x')
            ax.plot(_x(t['base_rate']), t['final_acq'], mk, color=CAT[0], markersize=7, markerfacecolor=CAT[0] if mk != 's' else 'white', markeredgewidth=1.5)
            key = (round(math.log10(_x(t['base_rate'])), 1), round(t['final_acq'], 2)); k = used.get(key, 0); used[key] = k + 1
            ax.annotate(f"s{t['seed']}", (_x(t['base_rate']), t['final_acq']), textcoords='offset points', xytext=(5, 3 + 9 * k), fontsize=8, color=INK2)
            for iv, v in (t.get('interventions') or {}).items():
                if v['final_acq'] is not None:
                    ax.plot(_x(t['base_rate']), v['final_acq'], marker='$%s$' % iv, color=CAT[1], markersize=8, linestyle='none')
        if S['plateau']:
            ax.axhline(S['plateau'], color=GREY, linestyle='--', linewidth=1)
            ax.text(ZERO_X * 1.2, S['plateau'] + 0.01, 'follow-up f = 0 mean 0.34', fontsize=8, color=INK2)
        ax.set_xscale('log'); ax.set_xlim(ZERO_X / 1.5, 3e-3)
        ax.set_xticks([ZERO_X, 1e-6, 1e-5, 1e-4, 1e-3]); ax.set_xticklabels(['0', '1e-6', '1e-5', '1e-4', '1e-3'])
        ax.set_xlabel(f"pre-RL base rate of {S['pattern']} proofs per sample (300 targets × 2,000)", color=INK2, fontsize=9)
        ax.set_ylabel('round-8 acquisition (fraction of targets)', color=INK2, fontsize=9)
        ax.set_title(f"{sname}: ● ignited ≤ r4   □ ignited r5–8   × never   K/T/S = interventions", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def round_vs_base(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (sname, S) in zip(axes, summary.items()):
        _style(ax); used = {}
        N, k = S['n_targets_pool'], S['k']
        xs = [10 ** (i / 20) for i in range(-130, -50)]
        ax.plot(xs, [1 / (r * k * N) for r in xs], color=GREY, linewidth=1.5, label='1 / (r·k·N)')
        done_lbl = set()
        for t in S['table']:
            if t['base_rate'] is None or t['rounds_done'] is None:
                continue
            x = _x(t['base_rate'])
            fr = t['first_pattern_round']; ig = t['ignition_round']
            ax.plot(x, fr if fr is not None else 9, 'o', color=CAT[0], markersize=7, label='first pattern proof' if 'f' not in done_lbl else None); done_lbl.add('f')
            ax.plot(x, ig if ig is not None else 9, 's', color=CAT[1], markersize=7, markerfacecolor='white', markeredgewidth=1.5, label='ignition (≥ 2 % of targets)' if 'i' not in done_lbl else None); done_lbl.add('i')
            key = (round(math.log10(x), 1), ig); k = used.get(key, 0); used[key] = k + 1
            ax.annotate(f"s{t['seed']}", (x, ig if ig is not None else 9), textcoords='offset points', xytext=(5 + 16 * k, 3), fontsize=8, color=INK2)
        ax.set_xscale('log'); ax.set_xlim(ZERO_X / 1.5, 3e-3); ax.set_ylim(0.5, 9.6)
        ax.set_yticks(range(1, 10)); ax.set_yticklabels([str(i) for i in range(1, 9)] + ['never'])
        ax.set_xticks([ZERO_X, 1e-6, 1e-5, 1e-4, 1e-3]); ax.set_xticklabels(['0', '1e-6', '1e-5', '1e-4', '1e-3'])
        ax.set_xlabel('pre-RL base rate r', color=INK2, fontsize=9); ax.set_ylabel('round', color=INK2, fontsize=9)
        ax.set_title(f"{sname} (N = {N}, k = {k})", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def curves(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (sname, S) in zip(axes, summary.items()):
        _style(ax)
        rates = [t['base_rate'] for t in S['table'] if t['base_rate'] is not None]
        for t in S['table']:
            if not t['per_round']:
                continue
            ys = t['per_round']; xs = list(range(1, len(ys) + 1))
            c = _seq_color(t['base_rate'], rates) if t['base_rate'] is not None else GREY
            ax.plot(xs, ys, '-', color=c, linewidth=2, alpha=0.95)
            ax.annotate(f"s{t['seed']}", (xs[-1], ys[-1]), textcoords='offset points', xytext=(4, -3), fontsize=7, color=INK2)
        ax.axhline(S['ignition_threshold'], color=GREY, linestyle=':', linewidth=1)
        ax.text(1, S['ignition_threshold'] * 1.15, 'ignition threshold (2 %)', fontsize=8, color=INK2)
        ax.set_yscale('symlog', linthresh=10); ax.set_ylim(bottom=-0.5); ax.set_xticks(range(1, 9))
        ax.set_xlabel('round', color=INK2, fontsize=9); ax.set_ylabel(f"target theorems with a {S['pattern']} proof (cumulative)", color=INK2, fontsize=9)
        ax.set_title(f"{sname}: one line per arm, darker = higher pre-RL base rate (grey = not sampled)", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def interventions(summary, out):
    panels = []
    for sname, S in summary.items():
        for t in S['table']:
            if t.get('interventions'):
                panels.append((sname, S, t))
    if not panels:
        return
    n = len(panels); cols = min(4, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    names = {'K': 'k = 128 (r5)', 'T': 'T = 1.0 (r5)', 'S': 'sibling transfer (outside data)'}
    for ax, (sname, S, t) in zip(axes.flat, panels):
        _style(ax)
        ys = t['per_round']; ax.plot(range(1, len(ys) + 1), ys, '-', color=INK2, linewidth=2, label='parent arm')
        for i, iv in enumerate('KTS'):
            v = t['interventions'].get(iv)
            if not v: continue
            yv = v['per_round']; x0 = 5
            ax.plot([4] + list(range(x0, x0 + len(yv))), [ys[3]] + yv, '-', color=CAT[i], linewidth=2, label=names[iv])
        ax.axhline(S['ignition_threshold'], color=GREY, linestyle=':', linewidth=1)
        ax.set_yscale('symlog', linthresh=10); ax.set_ylim(bottom=-0.5); ax.set_xticks(range(1, 9))
        ax.set_title(f"{sname} s{t['seed']}  r = {t['base_rate']:.1e}" if t['base_rate'] is not None else f"{sname} s{t['seed']}", fontsize=9, color=INK)
        ax.set_xlabel('round', fontsize=8, color=INK2); ax.set_ylabel('pattern theorems', fontsize=8, color=INK2)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def make_all(summary, figdir):
    os.makedirs(figdir, exist_ok=True)
    acq_vs_base(summary, f'{figdir}/ignition_acq_vs_base.png')
    round_vs_base(summary, f'{figdir}/ignition_round_vs_base.png')
    curves(summary, f'{figdir}/ignition_curves.png')
    interventions(summary, f'{figdir}/ignition_interventions.png')
