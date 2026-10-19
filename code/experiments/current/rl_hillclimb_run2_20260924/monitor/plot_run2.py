#!/usr/bin/env python3
"""Run 2 progress: every node's dev and frontier against wall-clock, with the two-seed running best of the score.

  python plot_run2.py --out /scratch/cowifornia/nd-rl/ab2/monitor/fig_run2_progress.png
"""
import argparse, json, os, datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AB = '/scratch/cowifornia/nd-rl/ab2'
LAUNCH = dt.datetime(2026, 9, 24, 2, 26, tzinfo=dt.timezone.utc)


def load_nodes():
    out = []
    for nid in sorted(os.listdir(f'{AB}/nodes')):
        m = f'{AB}/nodes/{nid}/meta.json'
        if not os.path.isfile(m):
            continue
        d = json.load(open(m))
        if d.get('dev') is None:
            continue
        t = d.get('created') or d.get('recorded') or ''
        try:
            when = dt.datetime.fromisoformat(str(t).replace('Z', '+00:00'))
        except Exception:
            when = dt.datetime.fromtimestamp(os.path.getmtime(m), dt.timezone.utc)
        out.append({'id': nid, 'min': (when - LAUNCH).total_seconds() / 60, 'dev': d.get('dev'),
                    'front': d.get('frontier') or 0, 'score': d.get('metric'), 'seeds': d.get('metric_seeds') or [],
                    'status': d.get('status'), 'role': d.get('role')})
    return sorted(out, key=lambda r: r['min'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    nodes = load_nodes()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.scatter([n['min'] for n in nodes], [n['dev'] for n in nodes], s=18, color='#1f77b4', alpha=0.6, label='dev (lengths 8-11, of 1,200), one run')
    ax.scatter([n['min'] for n in nodes], [n['front'] for n in nodes], s=18, color='#d62728', alpha=0.6, label='frontier (lengths 12-14, of 1,000), one run')
    ax.axhline(185, color='#1f77b4', ls='--', lw=1, label='seed recipe dev (185)')
    ax.set_ylabel('theorems solved')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    ax.set_title('Run 2 (2026-09-24): 7 GEAR workers + auditor, 20-minute experiments, 5 capped GPUs')
    ax = axes[1]
    best, bx, by = -1, [], []
    for n in nodes:
        two = len(n['seeds']) >= 2
        ax.scatter([n['min']], [n['score']], s=30 if two else 12, color='#2ca02c' if two else '#999999', alpha=0.8)
        if two and n['score'] and n['score'] > best:
            best = n['score']
        bx.append(n['min']); by.append(best if best > 0 else None)
    xs = [x for x, y in zip(bx, by) if y]; ys = [y for y in by if y]
    ax.step(xs, ys, where='post', color='#2ca02c', lw=2, label='running best, two-seed confirmed')
    ax.scatter([], [], s=30, color='#2ca02c', label='two-seed node'); ax.scatter([], [], s=12, color='#999999', label='single-seed node')
    ax.set_ylabel('score = dev + 4 x frontier'); ax.set_xlabel('minutes since launch (02:26 UTC)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=150)
    print('wrote', a.out, 'nodes', len(nodes))


if __name__ == '__main__':
    main()
