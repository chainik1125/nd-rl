#!/usr/bin/env python3
"""Figures for the writeup. Reads artifacts/<arm>/round_*.json and eval summaries.

  python plots.py --rl ei_s0 --control frozen_s0 [--rl2 ei_s1 --control2 frozen_s1] --stage1 artifacts/stage1_heldout_summary.json
"""
import argparse, json, glob, os, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_RL, C_CTRL, C_S1 = '#1f5fbf', '#8a8a8a', '#d1782f'


def rounds(arm):
    fs = sorted(glob.glob(f'artifacts/{arm}/round_*.json'), key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return [json.load(open(f)) for f in fs]


def err(ci, rate):
    return [[rate - ci[0]], [ci[1] - rate]]


def per_length(ax, summ, label, color, ls='-'):
    ks = sorted(int(k) for k in summ['by_len'] if k != 'None')
    ys = [summ['by_len'][str(k)]['rate'] for k in ks]
    lo = [summ['by_len'][str(k)]['rate'] - summ['by_len'][str(k)]['ci'][0] for k in ks]
    hi = [summ['by_len'][str(k)]['ci'][1] - summ['by_len'][str(k)]['rate'] for k in ks]
    ax.errorbar(ks, ys, yerr=[lo, hi], label=label, color=color, ls=ls, marker='o', ms=4, capsize=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rl', required=True)
    ap.add_argument('--control', required=True)
    ap.add_argument('--rl2', default=None)
    ap.add_argument('--control2', default=None)
    ap.add_argument('--stage1', default=None)
    ap.add_argument('--arms', nargs='*', default=None, help='arms to compare at their final common round (figures/arms.png)')
    a = ap.parse_args()
    os.makedirs('figures', exist_ok=True)
    R, C = rounds(a.rl), rounds(a.control)
    R2 = rounds(a.rl2) if a.rl2 else None
    C2 = rounds(a.control2) if a.control2 else None
    n = min(len(R), len(C))
    k = R[0]['k']

    # Fig 1: transfer solve rate by generating length, final round, RL vs frozen control (same attempts)
    fig, ax = plt.subplots(figsize=(7, 4))
    per_length(ax, R[n - 1]['transfer_cum'], f'expert iteration, cumulative {n*k} attempts', C_RL)
    per_length(ax, C[n - 1]['transfer_cum'], f'frozen Stage-1 model, cumulative {n*k} attempts', C_CTRL)
    if R2 and C2:
        m = min(len(R2), len(C2))
        per_length(ax, R2[m - 1]['transfer_cum'], f'expert iteration, seed 1, cumulative {m*k} attempts', C_RL, ls='--')
        per_length(ax, C2[m - 1]['transfer_cum'], f'frozen, seed 1, cumulative {m*k} attempts', C_CTRL, ls='--')
    ax.set_xlabel('generating proof length of transfer theorem (upper bound on shortest proof)')
    ax.set_ylabel('fraction solved (verified proof of prompted sequent)')
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f'Transfer set (never sampled for training), n={R[n-1]["transfer_cum"]["n"]}, 95% Wilson CIs')
    fig.tight_layout(); fig.savefig('figures/transfer_by_length.png', dpi=150); plt.close(fig)

    # Fig 2: found-proof-length histogram across rounds (RL targets, written length), plus control final
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in ((axes[0], 'written_hist', 'written length'), (axes[1], 'pruned_hist', 'dependency-pruned length')):
        Ls = list(range(2, 25))
        for i, r in enumerate(R[:n]):
            h = r['targets_cum'][key]
            ax.plot(Ls, [h.get(str(L), 0) for L in Ls], marker='o', ms=3, color=plt.cm.Blues(0.35 + 0.65 * i / max(1, n - 1)), label=f'EI round {r["round"]}')
        h = C[n - 1]['targets_cum'][key]
        ax.plot(Ls, [h.get(str(L), 0) for L in Ls], marker='s', ms=3, color=C_CTRL, ls='--', label=f'frozen, {n*k} attempts')
        ax.axhline(5, color='k', lw=.8, ls=':'); ax.axvline(6.5, color=C_S1, lw=.8, ls=':')
        ax.set_xlabel(title + ' of distinct verified proofs (RL targets, cumulative)'); ax.set_yscale('symlog'); ax.set_xlim(1.5, 13.5); ax.grid(alpha=.3)
    axes[0].set_ylabel('number of distinct verified proofs'); axes[1].legend(fontsize=7)
    fig.suptitle('Found-proof-length histogram; dotted line = 5 proofs (robust-frontier threshold), orange = training cap 6')
    fig.tight_layout(); fig.savefig('figures/found_length_hist.png', dpi=150); plt.close(fig)

    # Fig 3: per-round curves: target solve (cum), transfer solve (cum), heldout greedy, frontier
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    xs = [r['round'] for r in R[:n]]
    def curve(ax, key, sub, label, color, ls='-', arm=None):
        arm = arm or R
        ys = [r[key][sub] for r in arm[:n]]
        ax.plot(xs, ys, marker='o', ms=4, color=color, ls=ls, label=label)
    curve(axes[0], 'targets_cum', 'rate', 'RL targets, EI (cumulative)', C_RL)
    curve(axes[0], 'targets_cum', 'rate', 'RL targets, frozen (cumulative)', C_CTRL, '--', C)
    curve(axes[0], 'transfer_cum', 'rate', 'transfer, EI (cumulative)', C_RL, '-.')
    curve(axes[0], 'transfer_cum', 'rate', 'transfer, frozen (cumulative)', C_CTRL, ':', C)
    axes[0].set_ylabel('fraction of theorems solved'); axes[0].set_xlabel(f'round (each = {k} attempts per theorem)'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3); axes[0].set_ylim(0, 1)
    curve(axes[1], 'transfer_greedy', 'rate', 'transfer greedy pass@1, EI', C_RL)
    curve(axes[1], 'transfer_greedy', 'rate', 'transfer greedy pass@1, frozen', C_CTRL, '--', C)
    curve(axes[1], 'heldout_greedy', 'rate', 'Stage-1 held-out greedy, EI', C_S1)
    curve(axes[1], 'heldout_greedy', 'rate', 'Stage-1 held-out greedy, frozen', C_S1, '--', C)
    axes[1].set_xlabel('round'); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3); axes[1].set_ylim(0, 1)
    for key, sub, lab, col in (('transfer_cum', 'frontier_written', 'transfer, written', C_RL), ('transfer_cum', 'frontier_pruned', 'transfer, pruned', C_RL)):
        axes[2].plot(xs, [r[key][sub] for r in R[:n]], marker='o', ms=4, color=col, ls='-' if 'written' in sub else '-.', label=lab + ' (EI)')
        axes[2].plot(xs, [r[key][sub] for r in C[:n]], marker='s', ms=4, color=C_CTRL, ls='--' if 'written' in sub else ':', label=lab + ' (frozen)')
    axes[2].axhline(6, color=C_S1, ls=':', lw=.8); axes[2].set_xlabel('round'); axes[2].set_ylabel('robust frontier L (≥5 distinct proofs)'); axes[2].legend(fontsize=7); axes[2].grid(alpha=.3)
    def ge(r, n_):
        return sum(v for kk, v in r['transfer_cum']['written_hist'].items() if int(kk) >= n_)
    for n_, ls in ((7, '-'), (8, '--'), (9, ':')):
        axes[3].plot(xs, [ge(r, n_) for r in R[:n]], marker='o', ms=4, color=C_RL, ls=ls, label=f'EI, written ≥{n_}')
        axes[3].plot(xs, [ge(r, n_) for r in C[:n]], marker='s', ms=4, color=C_CTRL, ls=ls, label=f'frozen, written ≥{n_}')
    axes[3].set_yscale('symlog'); axes[3].set_xlabel('round'); axes[3].set_ylabel('distinct verified transfer proofs (cumulative)'); axes[3].legend(fontsize=7); axes[3].grid(alpha=.3)
    fig.tight_layout(); fig.savefig('figures/rounds.png', dpi=150); plt.close(fig)

    # Fig 4: Stage-1 held-out by length
    if a.stage1:
        s = json.load(open(a.stage1))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        per_length(ax, s, 'Stage-1 greedy', C_S1)
        ax.set_xlabel('held-out proof length'); ax.set_ylabel('greedy solve rate'); ax.set_ylim(0, 1); ax.grid(alpha=.3)
        ax.set_title(f'Stage-1 held-out (n={s["n"]}), 95% Wilson CIs')
        fig.tight_layout(); fig.savefig('figures/stage1_heldout.png', dpi=150); plt.close(fig)
    if a.arms:
        arms = [(nm, rounds(nm)) for nm in a.arms]
        m = min(len(r) for _, r in arms)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        xs = list(range(len(arms)))
        names = [nm for nm, _ in arms]
        cols = [C_CTRL if 'frozen' in nm else C_RL for nm in names]
        axes[0].bar(xs, [r[m - 1]['transfer_cum']['rate'] for _, r in arms], color=cols)
        axes[0].set_ylabel(f'transfer solved, cumulative {m*k} attempts'); axes[0].set_ylim(0, 1)
        axes[1].bar(xs, [r[m - 1]['transfer_greedy']['rate'] for _, r in arms], color=cols)
        axes[1].set_ylabel('transfer greedy pass@1'); axes[1].set_ylim(0, 1)
        for n_, hatch in ((8, ''), (9, '//')):
            axes[2].bar([x + (0.2 if n_ == 9 else -0.2) for x in xs], [sum(v for kk, v in r[m - 1]['transfer_cum']['written_hist'].items() if int(kk) >= n_) for _, r in arms], width=0.4, color=cols, hatch=hatch, label=f'written ≥{n_}')
        axes[2].set_yscale('symlog'); axes[2].set_ylabel('distinct verified transfer proofs'); axes[2].legend(fontsize=7)
        for ax in axes:
            ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, fontsize=7); ax.grid(alpha=.3, axis='y')
        fig.suptitle(f'Arms at round {m} (transfer set, n={arms[0][1][m-1]["transfer_cum"]["n"]}); blue = trained arms, grey = frozen controls', fontsize=9)
        fig.tight_layout(); fig.savefig('figures/arms.png', dpi=150); plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    main()
