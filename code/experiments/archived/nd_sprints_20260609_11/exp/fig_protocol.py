"""Protocol comparison figure: four arms from one checkpoint on one instance.

X-axis: cumulative verified rollout samples (EI round k consumed
32 x unsolved_k samples; GRPO logs cum_samples directly; the oracle arm
consumes no samples — it is plotted at the HER arm's x-positions because its
per-round token budget was matched to HER's).

Panels: (a) held-out transfer pass@8, (b) in-distribution accuracy.

Usage: python fig_protocol.py <her_hist> <van_hist> <grpo_hist> <oracle_hist> <out.png>
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ei_xs(hist, n_targets=1200, k=32):
    """Cumulative samples at each round's eval (rounds sample unsolved only)."""
    xs, cum = [0], 0
    prev_buf = 0
    for x in hist[1:]:
        unsolved = n_targets - prev_buf
        cum += unsolved * k
        xs.append(cum)
        prev_buf = x['buffer_size']
    return xs


def main(her_p, van_p, grpo_p, oracle_p, out):
    her = json.load(open(her_p))
    van = json.load(open(van_p))
    grpo = json.load(open(grpo_p))['log']
    grpo_ev = [x for x in grpo if x.get('EVAL')]
    oracle = json.load(open(oracle_p)) if oracle_p != '-' else None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    her_x = ei_xs(her)
    van_x = ei_xs(van)

    series = [
        (van_x, van, 'vanilla expert iteration', 'tab:blue', 'o-'),
        (her_x, her, 'hindsight-EI (+ verified prefix harvest)', 'tab:orange', 's-'),
    ]
    if oracle is not None:
        series.append((her_x, oracle,
                       'oracle: external long proofs, HER-matched tokens',
                       'tab:green', '^-'))
    for xs, hist, label, color, fmt in series:
        axes[0].plot(xs[:len(hist)], [h['transfer_passk'] for h in hist],
                     fmt, color=color, label=label)
        axes[1].plot(xs[:len(hist)], [h.get('iid_pass1') for h in hist],
                     fmt, color=color, label=label)
    axes[0].plot([g['cum_samples'] for g in grpo_ev],
                 [g['transfer_pass8'] for g in grpo_ev], 'd-',
                 color='tab:red', label='GRPO (on-policy, KL-anchored)')
    axes[1].plot([g['cum_samples'] for g in grpo_ev],
                 [g['iid_pass1'] for g in grpo_ev], 'd-', color='tab:red',
                 label='GRPO (on-policy, KL-anchored)')

    axes[0].set_ylabel('held-out transfer pass@8 (n=500, never sampled)')
    axes[1].set_ylabel('in-distribution accuracy (greedy)')
    for ax in axes:
        ax.set_xlabel('cumulative rollout samples (oracle: token-matched positions)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Four protocols, one starting model, one data instance',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:6])
