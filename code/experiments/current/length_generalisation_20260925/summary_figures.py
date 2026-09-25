#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Two summary figures from the results files (no sampling):

    uv run --script summary_figures.py

report/fig_attempts.png     share of samples with at least L_true written lines, and median written lines, by test
                            length and arm (standard sets; report/results.json 'attempts')
report/fig_composition.png  joined ANDI pass@1 against the product of the two parts' pass@1, by joined length:
                            pretrain / A / B / C s101 on the pre-final draw, and C_s101 vs D_s101 on the final draw
                            (composition/results.json 'step1' and 'dc')
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'pretrain': '#8a8984', 'A': '#86b6ef', 'B': '#2a78d6', 'C': '#0d366b', 'D': '#eb6834'}
MK = {'pretrain': 'o', 'A': 's', 'B': '^', 'C': 'D', 'D': 'v'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'lines.linewidth': 2})


def attempts():
    r = json.load(open(os.path.join(HERE, 'report', 'results.json')))['attempts']
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for arm in ('pretrain', 'A', 'B', 'C', 'D'):
        Ls = sorted(int(k.split('|')[2]) for k in r if k.startswith(f'{arm}|std|'))
        if not Ls:
            continue
        sh = [100 * r[f'{arm}|std|{L}']['share_ge_L'] for L in Ls]
        md = [r[f'{arm}|std|{L}']['median'] for L in Ls]
        lab = {'pretrain': 'pretrain (max 6)', 'A': 'A (max 7)', 'B': 'B (max 9)', 'C': 'C (max 11)', 'D': 'D (max 11, + constructed)'}[arm]
        axs[0].plot(Ls, sh, color=COL[arm], marker=MK[arm], ms=4, label=lab)
        axs[1].plot(Ls, md, color=COL[arm], marker=MK[arm], ms=4, label=lab)
    Ls = list(range(6, 19))
    axs[1].plot(Ls, Ls, color=INK2, ls=':', lw=1, label='L_true (lines needed)')
    axs[0].set_xlabel('test length L_true'); axs[0].set_ylabel('samples with >= L_true written lines (%)')
    axs[1].set_xlabel('test length L_true'); axs[1].set_ylabel('median written lines per sample')
    for ax in axs:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8, loc='upper left')
    fig.suptitle('Attempts stop at 6-10 lines whatever the length needed (standard sets, k = 32, all seeds)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, 'report', 'fig_attempts.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


def composition():
    r = json.load(open(os.path.join(HERE, 'composition', 'results.json')))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for arm in ('pretrain', 'A', 'B', 'C'):
        s = r['step1'].get(arm, {})
        Ls = [L for L in range(6, 19) if s.get(str(L))]
        axs[0].plot(Ls, [100 * s[str(L)]['joined'] for L in Ls], color=COL[arm], marker=MK[arm], ms=4, label=f'{arm}: joined')
        axs[0].plot(Ls, [100 * s[str(L)]['prod'] for L in Ls], color=COL[arm], ls='--', lw=1.3, label=f'{arm}: product of parts')
    dc = r.get('dc') or {}
    for lab, arm in (('C_s101', 'C'), ('D_s101', 'D')):
        Ls = [L for L in range(6, 15) if f'{lab}|{L}' in dc.get('ratio', {})]
        axs[1].plot(Ls, [100 * dc['ratio'][f'{lab}|{L}']['joined'] for L in Ls], color=COL[arm], marker=MK[arm], ms=4, label=f'{arm}: joined')
        axs[1].plot(Ls, [100 * dc['ratio'][f'{lab}|{L}']['prod'] for L in Ls], color=COL[arm], ls='--', lw=1.3, label=f'{arm}: product of parts')
    axs[0].set_title('pre-final draw, one seed per arm (s101)', fontsize=9, color=INK2)
    axs[1].set_title('final draw: arm D (constructed targets in RL) vs C, s101', fontsize=9, color=INK2)
    for ax in axs:
        ax.set_xlabel('joined ANDI length L_true')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7, ncol=2)
    axs[0].set_ylabel('pass@1 (%)')
    fig.suptitle('Joined ANDI theorem vs the product of its two parts solved alone (100 theorems per length, k = 32)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(HERE, 'report', 'fig_composition.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    attempts()
    composition()
    print('wrote report/fig_attempts.png, report/fig_composition.png')
