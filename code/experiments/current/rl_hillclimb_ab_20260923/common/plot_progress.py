#!/usr/bin/env python3
"""Dev score vs wall-clock for both arms, with running best. Reads runs/*/summary.json under each arm.

  python plot_progress.py --out /scratch/cowifornia/nd-rl/ab/logs/final/fig_dev_vs_time.png
"""
import argparse, glob, json, os, datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AB = '/scratch/cowifornia/nd-rl/ab'
LAUNCH = dt.datetime(2026, 9, 23, 1, 16, tzinfo=dt.timezone.utc)
DEADLINE = dt.datetime(2026, 9, 23, 4, 20, tzinfo=dt.timezone.utc)


def runs(arm):
    out = []
    for f in glob.glob(f'{AB}/{arm}/*/runs/*/summary.json'):
        if 'smoke' in f:
            continue
        s = json.load(open(f))
        if s.get('status') != 'ok' or s.get('metric') is None:
            continue
        t = s.get('started_utc') or ''
        try:
            t0 = dt.datetime.fromisoformat(t.replace('Z', '+00:00'))
        except Exception:
            t0 = dt.datetime.fromtimestamp(os.path.getmtime(f), dt.timezone.utc)
        secs = (s.get('rl_secs') or 0) + (s.get('score_secs') or 0)
        done = t0 + dt.timedelta(seconds=secs)
        out.append((done, s['metric'], f.split('/runs/')[-1].split('/')[0], s.get('heldout_greedy')))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'gear': '#1f77b4', 'orch': '#d62728'}
    labels = {'gear': 'GEAR-Evolve (2 workers)', 'orch': 'orchestration (executor + reviewer)'}
    for arm in ('gear', 'orch'):
        rs = runs(arm)
        xs = [(d - LAUNCH).total_seconds() / 60 for d, *_ in rs]
        ys = [m for _, m, *_ in rs]
        ax.scatter(xs, ys, s=22, color=colors[arm], alpha=0.6, label=f'{labels[arm]}: experiments ({len(rs)})')
        best, bx, by = -1, [], []
        for x, y in zip(xs, ys):
            if y > best:
                best = y
            bx.append(x); by.append(best)
        ax.step(bx, by, where='post', color=colors[arm], lw=2, label=f'{labels[arm]}: running best')
    ax.axhline(238, color='gray', ls='--', lw=1, label='seed pipeline, one 600 s run (238)')
    ax.axhline(7, color='black', ls=':', lw=1, label='pretrain checkpoint (7)')
    ax.axvline((DEADLINE - LAUNCH).total_seconds() / 60, color='gray', lw=0.8)
    ax.set_xlabel('minutes since launch (01:16 UTC)')
    ax.set_ylabel('dev score: theorems solved of 1,610 (pass@32, length 7-9)')
    ax.set_title('RL pipeline hill-climb A/B, 2026-09-23: 600 s experiments, 2 GPUs per arm')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
