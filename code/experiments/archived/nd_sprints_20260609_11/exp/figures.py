"""Make figures from results. Run from code/ dir; reads ../results, ../runs.

Figures (PNG, saved to ../results/figures):
  fig1_cliff.png      solve rate vs target proof length, per L_train cutoff
  fig2_ruleclass.png  same, L8 model, split by discharging-rule usage
  fig3_bootstrap.png  transfer pass@8 vs bootstrap round, RL vs controls
  fig4_textbook.png   textbook-target solves vs round
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'results')
RES = os.path.abspath(RES)
RUNS = os.path.abspath(os.path.join(RES, '..', 'runs'))
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def wilson(s, n, z=1.96):
    if n == 0:
        return 0, 0
    p = s / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return c - half, c + half


def rate_by_len(results, binw=2, lo=2, hi=30, pred=None):
    g = defaultdict(lambda: [0, 0])
    for r in results:
        if r.get('n_lines') is None:
            continue
        if pred and not pred(r):
            continue
        b = (r['n_lines'] - lo) // binw
        g[b][0] += r['solved']
        g[b][1] += 1
    xs, ys, es = [], [], []
    for b in sorted(g):
        s, n = g[b]
        if n >= 10:
            xs.append(lo + b * binw + (binw - 1) / 2)
            ys.append(s / n)
            cl, ch = wilson(s, n)
            es.append((max(0.0, s / n - cl), max(0.0, ch - s / n)))
    return xs, ys, es


def fig_cliff(metric='greedy'):
    conds = [('L8', 8, 'tab:blue'), ('L12', 12, 'tab:green'),
             ('L16', 16, 'tab:red')]
    plt.figure(figsize=(7, 4.5))
    for cond, cut, color in conds:
        rs = []
        for ev in ['eval_iid', 'eval_ood']:
            p = os.path.join(RES, f'cliff_{cond}_{ev}_{metric}.jsonl')
            if os.path.exists(p):
                rs += load_jsonl(p)
        if not rs:
            continue
        xs, ys, es = rate_by_len(rs)
        yerr = list(zip(*es)) if es else None
        plt.errorbar(xs, ys, yerr=yerr, fmt='o-', color=color, capsize=2,
                     label=f'trained on proofs $\\leq$ {cut} lines')
        plt.axvline(cut, color=color, ls=':', alpha=0.5)
    # 2x data condition as dashed
    p8 = []
    for ev in ['eval_iid', 'eval_ood']:
        p = os.path.join(RES, f'cliff_L8_2xtrue_{ev}_{metric}.jsonl')
        if os.path.exists(p):
            p8 += load_jsonl(p)
    if p8:
        xs, ys, es = rate_by_len(p8)
        plt.errorbar(xs, ys, yerr=list(zip(*es)) if es else None, fmt='s--',
                     color='tab:blue', alpha=0.5, capsize=2,
                     label='$\\leq$ 8 lines, 2$\\times$ more data')
    lbl = 'greedy solve rate' if metric == 'greedy' else 'pass@8 solve rate'
    plt.xlabel('held-out target difficulty: length of its generating proof (lines)')
    plt.ylabel(f'{lbl} (verifier-checked)')
    plt.title('Imitation hits a cliff at the training length cap\n'
              '(dotted verticals = caps; bars = 95% Wilson CIs)', fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f'fig1_cliff_{metric}.png'), dpi=160)
    plt.close()


def fig_cliff_local(metric='greedy'):
    """Cliff re-plotted on the local-rules-only pool: same generator settings
    at every length (discharging rules disabled), so the x-axis is not
    confounded by rule-composition shift."""
    conds = [('L8', 8, 'tab:blue'), ('L12', 12, 'tab:green'),
             ('L16', 16, 'tab:red')]
    plt.figure(figsize=(7, 4.5))
    for cond, cut, color in conds:
        p = os.path.join(RES, f'cliff_{cond}_eval_local_{metric}.jsonl')
        if not os.path.exists(p):
            continue
        rs = load_jsonl(p)
        xs, ys, es = rate_by_len(rs)
        plt.errorbar(xs, ys, yerr=list(zip(*es)) if es else None, fmt='o-',
                     color=color, capsize=2,
                     label=f'trained on proofs $\\leq$ {cut} lines')
        plt.axvline(cut, color=color, ls=':', alpha=0.5)
    lbl = 'greedy solve rate' if metric == 'greedy' else 'pass@8 solve rate'
    plt.xlabel('held-out target difficulty: length of its generating proof (lines)')
    plt.ylabel(f'{lbl} (verifier-checked)')
    plt.title('Same cliff on a composition-stable pool (local rules only,\n'
              'no discharging rules at any length; bars = 95% Wilson CIs)',
              fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f'fig1b_cliff_local_{metric}.png'), dpi=160)
    plt.close()


def fig_ruleclass(cond='L8', metric='greedy'):
    rs = []
    for ev in ['eval_iid', 'eval_ood']:
        p = os.path.join(RES, f'cliff_{cond}_{ev}_{metric}.jsonl')
        if os.path.exists(p):
            rs += load_jsonl(p)
    if not rs:
        return
    plt.figure(figsize=(7, 4.5))
    xs, ys, es = rate_by_len(rs, pred=lambda r: not r.get('discharge'))
    plt.errorbar(xs, ys, yerr=list(zip(*es)) if es else None, fmt='o-',
                 color='tab:green', capsize=2, label='local rules only')
    xs, ys, es = rate_by_len(rs, pred=lambda r: r.get('discharge'))
    plt.errorbar(xs, ys, yerr=list(zip(*es)) if es else None, fmt='s-',
                 color='tab:red', capsize=2, label='uses discharging rules')
    plt.axvline(8, color='k', ls=':', alpha=0.5)
    plt.xlabel('target proof length (lines, generating proof)')
    plt.ylabel('greedy solve rate' if metric == 'greedy' else 'pass@8 solve rate')
    plt.title(f'Length generalization by rule class ({cond} model)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f'fig2_ruleclass_{cond}_{metric}.png'), dpi=160)
    plt.close()


def fig_bootstrap():
    modes = [('ei_rl', 'RL self-training, fixed line numbers', 'tab:blue'),
             ('ei_rl_off', 'RL self-training, offset line numbers', 'tab:orange'),
             ('ei_prim', 'control: same tokens of extra short proofs', 'tab:gray'),
             ('ei_oracle', 'control: same tokens of generator long proofs', 'tab:green')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for d, label, color in modes:
        p = os.path.join(RUNS, d, 'history.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            h = json.load(f)
        rounds = [x['round'] for x in h]
        axes[0].plot(rounds, [x['transfer_passk'] for x in h], 'o-',
                     color=color, label=label)
        axes[1].plot(rounds, [x['transfer_pass1'] for x in h], 'o-',
                     color=color, label=label)
    for ax, ttl in zip(axes, ['pass@8 (T=0.8)', 'pass@1 (greedy)']):
        ax.set_xlabel('round')
        ax.set_ylabel(f'held-out transfer solve rate, {ttl}')
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Held-out transfer (500 theorems, 9-24 lines, never sampled '
                 'during RL) across rounds', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_bootstrap.png'), dpi=160)
    plt.close()


def fig_textbook():
    p = os.path.join(RUNS, 'ei_rl', 'history.json')
    if not os.path.exists(p):
        return
    with open(p) as f:
        h = json.load(f)
    plt.figure(figsize=(6, 4))
    rounds = [x['round'] for x in h]
    vals = [x.get('textbook_pass16') for x in h]
    plt.plot(rounds, vals, 'o-', color='tab:purple')
    plt.xlabel('bootstrap round')
    plt.ylabel('textbook targets solved (pass@16, frac of 36)')
    plt.title('Textbook-style held-out sequents')
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig4_textbook.png'), dpi=160)
    plt.close()


def fig_frontier():
    """Found-proof-length histograms: the token wall made visible.
    L8 vs L8off round-0 on eval_ood (pass@8 for more solves)."""
    from collections import Counter
    plt.figure(figsize=(7, 4.5))
    specs = [('cliff_L8_eval_ood_pass8.jsonl',
              'fixed line numbers (every training proof starts at N1)',
              'tab:blue', -0.2),
             ('cliff_L8off_eval_ood_pass8.jsonl',
              'random-offset line numbers (same proofs, shifted)',
              'tab:orange', 0.2)]
    for fn, label, color, dx in specs:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        sv = [r for r in load_jsonl(p) if r['solved']]
        c = Counter(r['best_len'] for r in sv)
        xs = sorted(c)
        plt.bar([x + dx for x in xs], [c[x] for x in xs], width=0.38,
                color=color, label=f'{label}: {len(sv)} theorems solved')
    plt.axvline(8.5, color='k', ls='--', alpha=0.7)
    plt.text(8.6, plt.ylim()[1] * 0.45, 'training proofs\nend here (8 lines)',
             fontsize=9)
    plt.xlabel('length of the proof the model wrote (lines)')
    plt.ylabel('held-out theorems solved (count)')
    plt.title('Before any RL, only the offset-trained model can write\n'
              'proofs longer than its training data (8 samples/theorem)',
              fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig_frontier.png'), dpi=160)
    plt.close()


def fig_frontier_rounds(run_dir='ei_rl_off', fname='fig_frontier_rounds.png'):
    """Per-round discovered-proof-length histograms for a chain: the ratchet test."""
    from collections import Counter
    rows = []
    rnd = 1
    while os.path.isdir(os.path.join(RUNS, run_dir, f'round{rnd}')):
        p = os.path.join(RUNS, run_dir, f'round{rnd}', 'discoveries.jsonl')
        if os.path.exists(p):
            rows.append((rnd, [r['n_lines'] for r in load_jsonl(p)]))
        rnd += 1
    if not rows:
        return
    plt.figure(figsize=(7, 4.5))
    cmap = plt.cm.viridis
    for i, (rnd, lens) in enumerate(rows):
        c = Counter(lens)
        xs = sorted(c)
        plt.semilogy(xs, [c[x] for x in xs], 'o-',
                     color=cmap(i / max(1, len(rows) - 1)),
                     label=f'after round {rnd} ({len(lens)} theorems total)')
    plt.axvline(8.5, color='k', ls='--', alpha=0.7)
    plt.text(8.6, 2.2, 'training proofs\nend here (8 lines)', fontsize=9)
    plt.xlabel('length of the proof RL discovered (lines)')
    plt.ylabel('theorems discovered so far (count, log scale)')
    plt.title('The beyond-cap tail of discovered proofs thickens each round\n'
              'and reaches 12 lines at round 3 (cumulative counts)', fontsize=11)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close()


def fig_discovery():
    """Cumulative discoveries vs round: chains vs never-retrained static controls."""
    plt.figure(figsize=(7, 4.5))
    chains = [('ei_rl', 'fixed line numbers, retrained each round', 'tab:blue', '-'),
              ('ei_rl_off', 'offset line numbers, retrained each round', 'tab:orange', '-')]
    for d, label, color, ls in chains:
        xs, ys = [], []
        rnd = 1
        while os.path.isdir(os.path.join(RUNS, d, f'round{rnd}')):
            p = os.path.join(RUNS, d, f'round{rnd}', 'discoveries.jsonl')
            if os.path.exists(p):
                xs.append(rnd)
                ys.append(len(load_jsonl(p)))
            rnd += 1
        if xs:
            plt.plot(xs, ys, 'o' + ls, color=color, label=label)
    statics = [('static_accum.json', 'fixed, NEVER retrained (resampling only)', 'tab:blue'),
               ('static_accum_off.json', 'offset, NEVER retrained (resampling only)', 'tab:orange')]
    for fn, label, color in statics:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            h = json.load(f)
        plt.plot([x['round'] for x in h], [x['cum'] for x in h], 's--',
                 color=color, alpha=0.6, label=label)
    plt.xlabel('sampling round (32 attempts per unsolved theorem each round)')
    plt.ylabel('cumulative theorems proven (of 1,200 held-out)')
    plt.title('Retraining on verified discoveries beats pure resampling —\n'
              'and much more so once the index format allows longer proofs',
              fontsize=11)
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig_discovery.png'), dpi=160)
    plt.close()


if __name__ == '__main__':
    for m in ['greedy', 'pass8']:
        fig_cliff(m)
        fig_cliff_local(m)
        for c in ['L8', 'L12', 'L16']:
            fig_ruleclass(c, m)
    fig_bootstrap()
    fig_textbook()
    fig_frontier()
    fig_frontier_rounds()
    fig_discovery()
    print('figures written to', FIG)
