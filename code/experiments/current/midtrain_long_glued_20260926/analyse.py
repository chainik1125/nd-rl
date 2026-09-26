#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Mid-training experiment: report/report.md, report/results.json and figures from eval/, runs/ and ckpt/.

    uv run --script analyse.py

Checkpoints (one seed): pre039 = the released 039 s0 pretrain; mt_long / mt_short / mt_more = after mid-training,
before RL; base / long / short / more = after arm-C RL (runs/<arm>_s101/final.pt). Sets: std L6-18 (natural at 6-14,
constructed ANDI + chain at 15-18), matched L6-14 (constructed), long L22-26 (held-out glued ANDI, longer than any
training proof). Headline groups: natural L12-14 (std, never glued), chain 15-18 (std units, families chaink1/chaink2:
a glue type absent from mid-training), glued 22-26 (long). Differences are paired by theorem with a 95% theorem
bootstrap (one seed: seed-to-seed spread is not measured here).
"""
import collections, datetime, gzip, json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
LE = '/scratch/cowifornia/nd-rl/ab3/lengthexp'
sys.path.insert(0, LE)
from common import pass_at_k, read_jsonl  # noqa: E402

EVAL, REPORT, TESTSETS = (os.path.join(HERE, d) for d in ('eval', 'report', 'testsets'))
LABELS = ['pre039', 'mt_long', 'mt_short', 'mt_more', 'base', 'long', 'short', 'more']
PRETTY = {'pre039': '039 s0 (no RL)', 'mt_long': 'mid-trained long (no RL)', 'mt_short': 'mid-trained short (no RL)',
          'mt_more': 'mid-trained more (no RL)', 'base': 'base + RL', 'long': 'long + RL', 'short': 'short + RL', 'more': 'more + RL'}
COL = {'pre039': '#8a8984', 'base': '#0d366b', 'long': '#eb6834', 'short': '#1baf7a', 'more': '#4a3aa7',
       'mt_long': '#eb6834', 'mt_short': '#1baf7a', 'mt_more': '#4a3aa7'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BOOT = 2000
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'lines.linewidth': 2})


def load():
    fam = {}
    cur = {}
    for f in ('std', 'matched', 'long'):
        fn = os.path.join(TESTSETS, f'{f}.jsonl')
        if os.path.isfile(fn):
            rs = read_jsonl(fn)
            for r in rs:
                fam[r['name']] = r.get('family') or ('standard' if r.get('source') in ('gen', 'textbook') else r.get('source'))
            by = collections.defaultdict(list)
            for r in rs:
                by[r['L_true']].append(r)
            cur.update({(f, L): {r['name'] for r in v} for L, v in by.items()})
    ev = {}
    for lab in LABELS:
        d = os.path.join(EVAL, lab)
        if not os.path.isdir(d):
            continue
        units = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.summary.json'):
                u = fn[:-len('.summary.json')]
                s, _, L = u.rpartition('_L')
                rows = read_jsonl(os.path.join(d, u + '.jsonl'))
                if cur.get((s, int(L))) != {r['name'] for r in rows}:
                    continue                                 # stale draw
                units[(s, int(L))] = rows
        ev[lab] = units
    return ev, fam


def pk(rows, k):
    return float(np.mean([pass_at_k(r['k'], r['n_ok'], k) for r in rows])) if rows else float('nan')


def lines_of(lab, s, L):
    fn = os.path.join(EVAL, lab, 'samples', f'{s}_L{L}.jsonl.gz')
    if not os.path.isfile(fn):
        return None
    with gzip.open(fn, 'rt') as f:
        return np.array([json.loads(x)['body'].count(';') for x in f], float)


def fail_positions(rows):
    """lengthexp/analyse.py's hazard bookkeeping: failures and samples at risk per line position."""
    fails, risk = collections.Counter(), collections.Counter()
    for r in rows:
        for n in r['ok_lines']:
            for p in range(1, n + 2):
                risk[p] += 1
        for kind, line, reason, start in r['fails']:
            if kind == 'bad_line':
                for p in range(1, line + 1):
                    risk[p] += 1
                fails[line] += 1
            elif kind == 'early_qed':
                for p in range(1, line + 2):
                    risk[p] += 1
                fails[line + 1] += 1
            elif kind == 'no_qed':
                for p in range(1, line + 1):
                    risk[p] += 1
    return fails, risk


def group_rows(units, fam, group):
    if group == 'natural L12-14':
        return [r for L in (12, 13, 14) for r in units.get(('std', L), [])]
    if group == 'chain 15-18':
        return [r for L in (15, 16, 17, 18) for r in units.get(('std', L), []) if fam.get(r['name']) in ('chaink1', 'chaink2')]
    if group == 'ANDI 15-18':
        return [r for L in (15, 16, 17, 18) for r in units.get(('std', L), []) if fam.get(r['name']) == 'andi']
    if group == 'glued 22-26':
        return [r for L in (22, 24, 26) for r in units.get(('long', L), [])]
    if group == 'in-distribution L6-11':
        return [r for L in range(6, 12) for r in units.get(('std', L), [])]
    raise KeyError(group)


def paired(ev, fam, a, b, group, rng):
    ra = {r['name']: r['n_ok'] / r['k'] for r in group_rows(ev.get(a, {}), fam, group)}
    rb = {r['name']: r['n_ok'] / r['k'] for r in group_rows(ev.get(b, {}), fam, group)}
    names = sorted(set(ra) & set(rb))
    if not names:
        return None
    x = np.array([ra[n] for n in names]); y = np.array([rb[n] for n in names])
    d = x - y
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    return {'n': len(names), 'a': float(x.mean()), 'b': float(y.mean()), 'diff': float(d.mean()),
            'ci': (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))}


def pct(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{100 * x:.{nd}f}%'


def md(h, rows):
    return '\n'.join(['| ' + ' | '.join(h) + ' |', '|' + '|'.join('---' for _ in h) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def main():
    os.makedirs(REPORT, exist_ok=True)
    ev, fam = load()
    labs = [l for l in LABELS if l in ev]
    rng = np.random.default_rng(0)
    res = {'pass': {}, 'attempts': {}, 'groups': {}, 'headline': {}, 'hazard': {}, 'train': {}, 'midtrain': {}}
    units_all = sorted({u for l in labs for u in ev[l]}, key=lambda u: ({'std': 0, 'matched': 1, 'long': 2}[u[0]], u[1]))
    for l in labs:
        for u in units_all:
            rows = ev[l].get(u)
            if rows:
                lines = lines_of(l, *u)
                res['pass'][f'{l}|{u[0]}|{u[1]}'] = {'n': len(rows), 'p1': pk(rows, 1), 'p4': pk(rows, 4), 'p32': pk(rows, 32)}
                if lines is not None:
                    res['attempts'][f'{l}|{u[0]}|{u[1]}'] = {'median': float(np.median(lines)), 'p90': float(np.percentile(lines, 90)),
                                                             'share_ge_L': float((lines >= u[1]).mean())}
        for g in ('in-distribution L6-11', 'natural L12-14', 'chain 15-18', 'ANDI 15-18', 'glued 22-26'):
            rows = group_rows(ev[l], fam, g)
            if rows:
                res['groups'][f'{l}|{g}'] = {'n': len(rows), 'p1': pk(rows, 1), 'p4': pk(rows, 4)}
        for g, sel in (('std L15-18', [('std', L) for L in (15, 16, 17, 18)]), ('long L22-26', [('long', L) for L in (22, 24, 26)])):
            rows = [r for u in sel for r in ev[l].get(u, [])]
            if rows:
                c = collections.Counter(reason if kind == 'bad_line' else kind for r in rows for kind, line, reason, start in r['fails'])
                nf = sum(c.values())
                res.setdefault('reasons', {})[f'{l}|{g}'] = {'failed': nf, 'share': {k: v / nf for k, v in c.most_common()} if nf else {}}
                f_, r_ = fail_positions(rows)
                res['hazard'][f'{l}|{g}'] = {str(p): {'risk': r_[p], 'fails': f_.get(p, 0), 'hazard': f_.get(p, 0) / r_[p]} for p in sorted(r_) if r_[p] >= 100}
    for a, b in (('long', 'base'), ('long', 'short'), ('long', 'more'), ('mt_long', 'pre039'), ('mt_long', 'mt_short'), ('mt_long', 'mt_more')):
        for g in ('natural L12-14', 'chain 15-18', 'ANDI 15-18', 'glued 22-26', 'in-distribution L6-11'):
            r = paired(ev, fam, a, b, g, rng)
            if r:
                res['headline'][f'{a} vs {b}|{g}'] = r
    for arm in ('base', 'long', 'short', 'more'):
        fn = os.path.join(HERE, 'runs', f'{arm}_s101', 'summary.json')
        if os.path.isfile(fn):
            s = json.load(open(fn))
            res['train'][arm] = {'status': s['status'], 'rl_secs': s.get('rl_secs'), 'rounds': s['rl_stats'].get('rounds'),
                                 'samples': s['rl_stats'].get('total_samples'), 'heldout_greedy': s.get('heldout_greedy'),
                                 'compute_frac': (s.get('compute') or {}).get('frac'), 'pretrain_sha256': s['shas'].get('pretrain.pt'),
                                 'rl_py_sha256': s['shas'].get('rl.py')}
    for arm in ('long', 'short', 'more'):
        fn = os.path.join(HERE, 'ckpt', f'mt_{arm}.json')
        if os.path.isfile(fn):
            res['midtrain'][arm] = json.load(open(fn))
    res['data'] = data_stats()
    with open(os.path.join(REPORT, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    figs = figures(res, labs)
    write_md(res, labs, units_all, figs)


def figures(res, labs):
    out = []
    groups = [('std', range(6, 19), 'standard sets (natural 6-14, constructed 15-18)'), ('matched', range(6, 15), 'matched constructed 6-14'),
              ('long', (22, 24, 26), 'held-out glued 22-26')]
    for tag, sel in (('rl', ['pre039', 'base', 'long', 'short', 'more']), ('mt', ['pre039', 'mt_long', 'mt_short', 'mt_more'])):
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), gridspec_kw={'width_ratios': [13, 9, 3]})
        for j, (s, Ls, title) in enumerate(groups):
            for l in sel:
                if l not in labs:
                    continue
                xs = [L for L in Ls if f'{l}|{s}|{L}' in res['pass']]
                if not xs:
                    continue
                y = [max(res['pass'][f'{l}|{s}|{L}']['p1'], 1e-5) for L in xs]
                a = [100 * res['attempts'].get(f'{l}|{s}|{L}', {}).get('share_ge_L', float('nan')) for L in xs]
                ls = '--' if l == 'pre039' else '-'
                axs[0][j].plot(xs, y, ls, color=COL[l], marker='o', ms=3, label=PRETTY[l])
                axs[1][j].plot(xs, a, ls, color=COL[l], marker='o', ms=3, label=PRETTY[l])
            axs[0][j].set_yscale('log'); axs[0][j].set_title(title, fontsize=9, color=INK2)
            axs[1][j].set_xlabel('L_true')
            for i in (0, 1):
                axs[i][j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axs[0][0].set_ylabel('pass@1 (log; 1e-5 = none)'); axs[1][0].set_ylabel('attempts >= L_true lines (%)')
        axs[0][0].legend(fontsize=7)
        fig.suptitle('After RL (arm-C recipe, one seed)' if tag == 'rl' else 'After mid-training, before RL', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fn = f'fig_{tag}.png'
        fig.savefig(os.path.join(REPORT, fn), dpi=130, bbox_inches='tight'); plt.close(fig)
        out.append(fn)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for j, g in enumerate(('std L15-18', 'long L22-26')):
        for l in ('pre039', 'base', 'long', 'short', 'more', 'mt_long'):
            h = res['hazard'].get(f'{l}|{g}')
            if not h:
                continue
            ps = sorted(int(p) for p in h)
            axs[j].plot(ps, [h[str(p)]['hazard'] for p in ps], '--' if l in ('pre039', 'mt_long') else '-', color=COL[l], marker='o', ms=3,
                        label=PRETTY[l])
        axs[j].set_title(g, fontsize=9, color=INK2); axs[j].set_xlabel('line position p')
        axs[j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[0].set_ylabel('P(fail at p | lines 1..p-1 valid)'); axs[1].legend(fontsize=7)
    fig.suptitle('Per-line failure hazard on the long sets (positions with >= 100 samples at risk)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(REPORT, 'fig_hazard_long.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    out.append('fig_hazard_long.png')
    return out


def data_stats():
    """Mean prompt tokens (one token per whitespace symbol, as the tokenizer) and 3-part share per training file / test set."""
    import random as _r
    out = {}
    files = {'sft_long (mid-training)': os.path.join(HERE, 'data', 'sft_long.jsonl'),
             'sft_short (mid-training)': os.path.join(HERE, 'data', 'sft_short.jsonl'),
             'RL pool targets_L6-11 (all arms)': '/scratch/cowifornia/nd-rl/ab3/frozen_039/data/targets_L6-11.jsonl'}
    for name, fn in files.items():
        rs = read_jsonl(fn)
        out[name] = {'n': len(rs), 'prompt_tokens_mean': float(np.mean([len(r['prompt'].split()) for r in rs])),
                     'k3_share': (sum(r.get('family') == 'andi_k3' for r in rs) / len(rs)) if 'sft' in name else None,
                     'written_lines': [int(min(r['n_lines'] for r in rs)), int(max(r['n_lines'] for r in rs))]}
    cap6 = read_jsonl('/scratch/cowifornia/nd-rl/ab3/frozen_039/data/train_cap6.jsonl')
    idx = _r.Random(0).sample(range(len(cap6)), 10000)
    out['cap-6 records in the mixes (10,000)'] = {'n': 10000, 'prompt_tokens_mean': float(np.mean([len(cap6[i]['prompt'].split()) for i in idx])),
                                                   'k3_share': None, 'written_lines': [int(min(cap6[i]['n_lines'] for i in idx)), int(max(cap6[i]['n_lines'] for i in idx))]}
    fam = {}
    for f in ('std', 'matched', 'long'):
        rs = read_jsonl(os.path.join(TESTSETS, f'{f}.jsonl'))
        groups = {'std': [('std natural L6-11', lambda r: r['L_true'] <= 11), ('std natural L12-14', lambda r: 12 <= r['L_true'] <= 14),
                          ('std chain L15-18', lambda r: r['L_true'] >= 15 and r.get('family') in ('chaink1', 'chaink2')),
                          ('std ANDI L15-18', lambda r: r['L_true'] >= 15 and r.get('family') == 'andi')],
                  'matched': [('matched L6-14', lambda r: True)], 'long': [('long L22-26 (glued, 3-part)', lambda r: True)]}[f]
        for name, sel in groups:
            sub = [r for r in rs if sel(r)]
            out[name] = {'n': len(sub), 'prompt_tokens_mean': float(np.mean([len(r['prompt'].split()) for r in sub])),
                         'k3_share': (sum(r.get('family') == 'andi_k3' for r in sub) / len(sub)) if f == 'long' else None,
                         'written_lines': [int(min(r['L_true'] for r in sub)), int(max(r['L_true'] for r in sub))]}
    return out


CAVEATS = """- **test_long is close to in-distribution for the long arm.** It is built exactly like sft_long: glued ANDI joins, all
  of them 3-part, from held-out parts. So the main evidence of generalisation is the natural theorems at L12-14 (never
  glued) and chain 15-18, a glue type absent from mid-training. Those are shown first; glued 22-26 comes last.
- **The 3-part share is tied to length.** sft_short has 3-part joins only at L9-11; sft_long is two thirds 3-part and
  test_long is all 3-part (shares in the data table below). The long-vs-short gap on test_long partly measures
  experience with the 3-part join shape, not only length.
- **Prompts are longer than in run 2** (mean prompt tokens per file and test set in the data table below). Part of
  what the long mix teaches is long prompts.
- **test_long labels** are constructed upper bounds with a proved lower bound of 15; additivity is checked only up to
  17 lines. pass@1 here means that the verifier accepted a proof of the prompted sequent. That does not depend on the
  label. Only the attempt-length share (written lines >= L_true) uses it.
- **No truncation.** Mid-training does not go through `lib/train.load`, so its line cap does not apply. It uses the
  seed recipe's own encode-and-pack path, which never cuts a record: every record is at most 800 tokens, and the
  packer raises an error above 1,024.
- **One seed.** Differences are given with a theorem bootstrap, which does not include seed-to-seed variation. For
  scale, arm C's three seeds in lengthexp spanned 3.1-3.9% pass@1 at L12.
"""


def write_md(res, labs, units_all, figs):
    L = []
    w = L.append
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    w('---\nwritten_on: 2026-09-26\n---\n# Mid-training on long glued proofs: report\n')
    w(f'Generated {now} by `analyse.py`. One seed (RL seed 101). k = 32 samples per theorem at T = 0.8, max_new 1,024; '
      'pass@k are unbiased estimates averaged over theorems. Checkpoints evaluated: ' + ', '.join(f'{l} ({PRETTY[l]})' for l in labs) + '.\n')
    cf = os.path.join(REPORT, 'conclusion.md')
    w('## Conclusion\n\n' + (open(cf).read().strip() if os.path.isfile(cf) else '(to be written)') + '\n')
    rows = []
    for g in ('natural L12-14', 'chain 15-18', 'ANDI 15-18', 'glued 22-26', 'in-distribution L6-11'):
        for pair in ('long vs base', 'long vs short', 'long vs more'):
            r = res['headline'].get(f'{pair}|{g}')
            if r:
                rows.append([g, pair, r['n'], pct(r['a'], 2), pct(r['b'], 2), f"{100 * r['diff']:+.2f} pts [{100 * r['ci'][0]:+.2f}, {100 * r['ci'][1]:+.2f}]"])
    w('## Caveats\n\n' + CAVEATS)
    w('## Headline comparisons (after RL)\n\npass@1 (a valid proof of the prompted sequent) pooled over the group; difference paired by '
      'theorem with a 95% theorem bootstrap. Groups in order of evidential weight: natural L12-14, chain 15-18 (held-out glue '
      'type), ANDI 15-18 (2-part joins, test construction), glued 22-26 (the mid-training construction).\n\n'
      + md(['group', 'comparison', 'theorems', 'first', 'second', 'difference [95% CI]'], rows) + '\n')
    rows = []
    for g in ('natural L12-14', 'chain 15-18', 'ANDI 15-18', 'glued 22-26', 'in-distribution L6-11'):
        for pair in ('mt_long vs pre039', 'mt_long vs mt_short', 'mt_long vs mt_more'):
            r = res['headline'].get(f'{pair}|{g}')
            if r:
                rows.append([g, pair, r['n'], pct(r['a'], 2), pct(r['b'], 2), f"{100 * r['diff']:+.2f} pts [{100 * r['ci'][0]:+.2f}, {100 * r['ci'][1]:+.2f}]"])
    w('## The same before RL (mid-trained checkpoints)\n\n' + md(['group', 'comparison', 'theorems', 'first', 'second', 'difference [95% CI]'], rows) + '\n')
    for k in ('p1', 'p4'):
        for s in ('std', 'matched', 'long'):
            Ls = [u[1] for u in units_all if u[0] == s]
            if not Ls:
                continue
            rows = [[PRETTY[l]] + [pct(res['pass'].get(f'{l}|{s}|{x}', {}).get(k), 2) if f'{l}|{s}|{x}' in res['pass'] else '' for x in Ls] for l in labs]
            w(f"## pass@{k[1:]}, {s} sets\n\n" + md(['checkpoint'] + [f'L{x}' for x in Ls], rows) + '\n')
    rows = []
    for l in labs:
        for g in ('natural L12-14', 'chain 15-18', 'ANDI 15-18', 'glued 22-26'):
            r = res['groups'].get(f'{l}|{g}')
    grp = ('in-distribution L6-11', 'natural L12-14', 'chain 15-18', 'ANDI 15-18', 'glued 22-26')
    rows = [[PRETTY[l]] + [f"{pct(res['groups'][f'{l}|{g}']['p1'], 2)} / {pct(res['groups'][f'{l}|{g}']['p4'], 2)}" if f'{l}|{g}' in res['groups'] else '' for g in grp] for l in labs]
    w('## pass@1 / pass@4 by group\n\n' + md(['checkpoint'] + list(grp), rows) + '\n')
    for s in ('std', 'matched', 'long'):
        Ls = [u[1] for u in units_all if u[0] == s]
        if not Ls:
            continue
        rows = [[PRETTY[l]] + [(f"{res['attempts'][f'{l}|{s}|{x}']['median']:.0f} / {pct(res['attempts'][f'{l}|{s}|{x}']['share_ge_L'], 0)}"
                                if f'{l}|{s}|{x}' in res['attempts'] else '') for x in Ls] for l in labs]
        w(f'## Attempt lengths, {s} sets\n\nCell: median written lines / share of samples with at least L_true written lines.\n\n'
          + md(['checkpoint'] + [f'L{x}' for x in Ls], rows) + '\n')
    for g in ('std L15-18', 'long L22-26'):
        pos = sorted({int(p) for l in labs for p in res['hazard'].get(f'{l}|{g}', {})})
        if pos:
            rows = [[PRETTY[l]] + [f"{res['hazard'][f'{l}|{g}'][str(p)]['hazard']:.2f}" if str(p) in res['hazard'].get(f'{l}|{g}', {}) else '' for p in pos]
                    for l in labs if f'{l}|{g}' in res['hazard']]
            w(f'## Per-line failure hazard, {g}\n\nP(line p fails | lines 1..p-1 valid), positions with at least 100 samples at risk.\n\n'
              + md(['checkpoint'] + [f'p{p}' for p in pos], rows) + '\n')
    for g in ('std L15-18', 'long L22-26'):
        keys = ['premise', 'parse', 'rule', 'cite']
        rows = []
        for l in labs:
            v = res.get('reasons', {}).get(f'{l}|{g}')
            if v:
                other = 1 - sum(v['share'].get(k, 0) for k in keys)
                rows.append([PRETTY[l]] + [pct(v['share'].get(k, 0), 0) for k in keys] + [pct(other, 0)])
        w(f'## First-error reasons of failed samples, {g}\n\nShare of failed samples by the reason their first line is rejected: premise = '
          'the premise lines are wrong (a premise mis-copied, skipped or out of order); parse = a malformed line; rule = an '
          'invalid inference; cite = a citation out of scope.\n\n' + md(['checkpoint'] + keys + ['other'], rows) + '\n')
    rows = []
    for arm in ('long', 'short', 'more'):
        m = res['midtrain'].get(arm)
        if m:
            rows.append([f'mt_{arm}', m['n_records'], ', '.join(f'{k} {v}' for k, v in m['by_source'].items()), m['steps_done'], m['tokens_total'],
                         f"{m['wall_secs']:.0f} s", f"{m['loss_first']:.3f} -> {m['loss_last']:.3f}", f"{m['heldout_greedy_before']:.3f} -> {m['heldout_greedy_after']:.3f}"])
    w('## Mid-training\n\nSame steps, batch, schedule and cap-6 records in all three arms (`midtrain.py`, `ckpt/mt_<arm>.json`).\n\n'
      + md(['checkpoint', 'records', 'mix', 'steps', 'tokens', 'wall', 'loss (first -> last logged step)', 'held-out greedy (before -> after)'], rows) + '\n')
    rows = [[arm, t['status'], t['rounds'], t['samples'], f"{t['heldout_greedy']:.3f}", f"{t['compute_frac']:.3f}", f"{t['rl_secs'] / 60:.1f} min",
             (t['pretrain_sha256'] or '')[:12]] for arm, t in res['train'].items()]
    w('## RL runs (arm-C recipe, seed 101, DEFAULT_FLOPS)\n\n' + md(['arm', 'status', 'rounds', 'samples', 'held-out greedy', 'compute used', 'wall', 'pretrain.pt sha256'], rows) + '\n')
    ds = res.get('data', {})
    rows = [[k, v['n'], f"{v['prompt_tokens_mean']:.0f}", f"{v['written_lines'][0]}-{v['written_lines'][1]}", pct(v['k3_share'], 0) if v['k3_share'] is not None else '-']
            for k, v in ds.items()]
    w('## Data: prompt size and join shape\n\nMean prompt tokens (one token per symbol), length range (written lines for training files, '
      'L_true for test sets) and share of 3-part joins.\n\n' + md(['file / set', 'records', 'mean prompt tokens', 'lines', '3-part share'], rows) + '\n')
    w('## Figures\n\n' + '\n\n'.join(f'![{f}]({f})' for f in figs) + '\n')
    with open(os.path.join(REPORT, 'report.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', os.path.join(REPORT, 'report.md'))


if __name__ == '__main__':
    main()
