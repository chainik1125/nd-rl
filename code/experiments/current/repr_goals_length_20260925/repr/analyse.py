#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Representation experiment analysis: report/report.md, report/*.png, report/results.json.

    uv run --script analyse.py [--eval-root eval] [--out report] [--boot 200] [--no-samples]

Arms are the variants (abs, rel, posrand, nope, posline), each with pretraining seeds 0-2 (RL seeds 101-103),
all trained on L_true 6-11 (arm C). Statistics come from lengthexp/analyse.py (imported, not copied): the
grouped-binomial logistic fit, Welch t, exact permutation tests. Sections:
  0 headline; 1 reproduction check (abs re-pretrain vs released 039); 2 pass@1 / pass@4 / pass@32 per length;
  3 log-odds drop per extra line past 11 (slope, seed t-interval, tests against abs); 4 attempt length (share of
  attempts with >= L_true written lines, median written lines); 5 first-error position; 6 in-distribution checks;
  7 training side; 8 rel: citation distance coverage and the split of long-length results by whether the
  reference proof needs a citation beyond the trained range; 9 pos-rand random-position check; 10 matched set.
Arm C of lengthexp (the older frozen/pretrain.pt, same RL recipe) is shown as a reference row where useful.
"""
import argparse, collections, datetime, gzip, importlib.util, json, math, os, re, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LE = '/scratch/cowifornia/nd-rl/ab3/lengthexp'
_spec = importlib.util.spec_from_file_location('le_analyse', os.path.join(LE, 'analyse.py'))
LA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LA)          # lengthexp's statistics (fit_logistic, welch, fail_positions, md_table, ...)
sys.path.insert(0, HERE)
from common_repr import (DATA, EVAL, ORDER, PRETRAIN_DIR, REPORT, RUNS, SEEDS, TRAIN_CAP6, TRAIN_MAX, label,  # noqa: E402
                         parse_label, read_jsonl)

LENGTHS = tuple(range(6, 19))
LONG = tuple(range(12, 19))
NAMES = {'abs': 'abs (baseline)', 'rel': 'rel (relative citations)', 'posrand': 'pos-rand (random positions)',
         'nope': 'nope (no positional encoding)', 'posline': 'pos-line (line + offset rotary)', 'C_le': 'arm C (lengthexp, older pretrain)'}
COLORS = {'abs': '#2a78d6', 'rel': '#eb6834', 'posrand': '#1baf7a', 'nope': '#eda100', 'posline': '#e87ba4', 'C_le': '#8a8984'}
MARK = {'abs': 'o', 'rel': 's', 'posrand': '^', 'nope': 'D', 'posline': 'v', 'C_le': 'x'}
INK, INK2 = '#0b0b0b', '#52514e'
T975 = LA.T975
REF_SOURCES = ['/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl', '/scratch/cowifornia/nd-rl/ab2/sealed/sealed_L12-14.jsonl',
               '/scratch/cowifornia/nd-rl/ab3/sealed/sealed_L15-16.jsonl', '/scratch/cowifornia/nd-rl/ab3/sealed/sealed_L17-18.jsonl']


# ------------------------------------------------------------------ loading
def load_evals(root):
    ev = {}
    if not os.path.isdir(root):
        return ev
    for lab in sorted(os.listdir(root)):
        d = os.path.join(root, lab)
        if not os.path.isdir(d):
            continue
        kind = ''
        base = lab
        if lab.endswith('_randpos'):
            kind, base = 'randpos', lab[:-len('_randpos')]
        v, s = parse_label(base)
        if v is None:
            continue
        units, summ = {}, {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.summary.json'):
                u = fn[:-len('.summary.json')]
                st, _, L = u.rpartition('_L')
                if os.path.isfile(os.path.join(d, u + '.jsonl')):
                    units[(st, int(L))] = read_jsonl(os.path.join(d, u + '.jsonl'))
                    summ[(st, int(L))] = json.load(open(os.path.join(d, fn)))
        if units:
            ev[lab] = {'variant': v, 'seed': s, 'kind': kind, 'units': units, 'summ': summ, 'dir': d}
    return ev


def load_le_armC():
    out = {}
    for s in (101, 102, 103):
        d = os.path.join(LE, 'eval', f'C_s{s}')
        if not os.path.isdir(d):
            continue
        units, summ = {}, {}
        for fn in os.listdir(d):
            if fn.endswith('.summary.json'):
                u = fn[:-len('.summary.json')]
                st, _, L = u.rpartition('_L')
                units[(st, int(L))] = read_jsonl(os.path.join(d, u + '.jsonl'))
                summ[(st, int(L))] = json.load(open(os.path.join(d, fn)))
        out[f'C_le_s{s}'] = {'variant': 'C_le', 'seed': s, 'kind': '', 'units': units, 'summ': summ, 'dir': d}
    return out


def unit_shas(ev):
    shas = collections.defaultdict(set)
    for e in ev.values():
        for key, sm in e.get('summ', {}).items():
            shas[key].add(sm.get('unit_sha'))
    return shas


def keep_matching(le, shas):
    """Drop reference-arm units evaluated on another theorem draw than ours (unit_sha differs)."""
    dropped = 0
    for e in le.values():
        for key in list(e['units']):
            ours = shas.get(key)
            if ours and e['summ'][key].get('unit_sha') not in ours:
                del e['units'][key]
                dropped += 1
    return dropped


def runs_of(ev, v, kind=''):
    return [e for e in ev.values() if e['variant'] == v and e['kind'] == kind]


def mean_rng(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, None
    return float(np.mean(xs)), float(min(xs)), float(max(xs))


def tint(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    n = len(xs)
    if n < 2:
        return (float(np.mean(xs)) if xs else float('nan')), float('nan'), float('nan')
    m, sd = float(np.mean(xs)), float(np.std(xs, ddof=1))
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, m - h, m + h


def pct(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:.{nd}f}%'


def f2(x, nd=2):
    return '-' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def md(header, rows):
    return LA.md_table(header, rows)


# ------------------------------------------------------------------ pass@k
def pass_table(ev, variants, k, st='std', kind=''):
    tab = {}
    for v in variants:
        for L in LENGTHS:
            vals = [LA.passk(e['units'][(st, L)], k) for e in runs_of(ev, v, kind) if (st, L) in e['units']]
            tab[(v, L)] = mean_rng(vals)
    return tab


def pass_md(tab, variants, lengths, runs_n):
    rows = []
    for v in variants:
        cells = []
        for L in lengths:
            m, lo, hi = tab.get((v, L), (None, None, None))
            cells.append('' if m is None else (pct(m) + (f' [{pct(lo)}, {pct(hi)}]' if runs_n.get(v, 0) > 1 else '')))
        rows.append([NAMES[v]] + cells)
    return md(['variant'] + [f'L{L}' for L in lengths], rows)


def pooled_long(e, lo=12, hi=18, st='std'):
    rows = [r for L in range(lo, hi + 1) for r in e['units'].get((st, L), [])]
    return LA.passk(rows, 1) if rows else float('nan')


def pooled_rng(e, lo, hi, st='std'):
    rows = [r for L in range(lo, hi + 1) for r in e['units'].get((st, L), [])]
    return LA.passk(rows, 1) if rows else float('nan')


# ------------------------------------------------------------------ slopes
def fit_run(e, lo, hi, boot, rng):
    pts = LA.run_points({'units': e['units']}, range(lo, hi + 1))
    f = LA.fit_logistic([p[0] - TRAIN_MAX for p in pts], [p[1] for p in pts], [p[2] for p in pts])
    f['boot_se'] = float('nan')
    if f['ok'] and boot:
        bs = []
        for _ in range(boot):
            ss, nn = [], []
            for (_, _, _, nok, kk) in pts:
                ix = rng.integers(0, len(nok), len(nok))
                ss.append(nok[ix].sum())
                nn.append(kk[ix].sum())
            g = LA.fit_logistic([p[0] - TRAIN_MAX for p in pts], ss, nn)
            if g['ok']:
                bs.append(g['b'])
        if len(bs) > 10:
            f['boot_se'] = float(np.std(bs, ddof=1))
    return f


def perm_p(x, y):
    import itertools
    if len(x) < 2 or len(y) < 2 or not np.all(np.isfinite(list(x) + list(y))):
        return float('nan')
    allv = np.array(list(x) + list(y))
    obs = abs(np.mean(x) - np.mean(y))
    cnt = tot = 0
    for c in itertools.combinations(range(len(allv)), len(x)):
        rest = [i for i in range(len(allv)) if i not in c]
        tot += 1
        cnt += abs(allv[list(c)].mean() - allv[rest].mean()) >= obs - 1e-12
    return cnt / tot if tot else float('nan')


def section_slopes(ev, variants, boot):
    rng = np.random.default_rng(0)
    res = {}
    for wname, lo, hi in (('d>=1 (L12-18)', 12, 18), ('d>=0 (L11-18)', 11, 18)):
        per = {}
        for v in variants:
            per[v] = []
            for e in sorted(runs_of(ev, v), key=lambda e: e['seed']):
                f = fit_run(e, lo, hi, boot, rng)
                per[v].append({'seed': e['seed'], **f})
        res[wname] = per
    return res


def slopes_md(res, variants):
    out = []
    for wname, per in res.items():
        rows = []
        base = [f['b'] for f in per.get('abs', []) if f['ok']]
        for v in variants:
            fs = per.get(v, [])
            ok = [f['b'] for f in fs if f['ok']]
            m, lo, hi = tint(ok)
            cells = ', '.join(f2(f['b']) if f['ok'] else 'n/a' for f in fs)
            if v != 'abs' and len(ok) >= 2 and len(base) >= 2:
                t, p = LA.welch(ok, base)
                cmp = f'{m - np.mean(base):+.2f} (Welch p {LA.fmt_p(p)}, perm p {LA.fmt_p(perm_p(ok, base))})'
            else:
                cmp = '-'
            bse = [f['boot_se'] for f in fs if f['ok'] and np.isfinite(f['boot_se'])]
            rows.append([NAMES[v], cells, f2(m) if ok else '-', f'[{f2(lo)}, {f2(hi)}]' if len(ok) >= 2 else '-',
                         f2(np.mean(bse)) if bse else '-', cmp])
        out.append(f'**Window {wname}** (d = L - 11)\n\n' + md(
            ['variant', 'slope per run (log-odds per extra line)', 'mean', '95% t-interval over seeds', 'mean bootstrap SE',
             'difference from abs'], rows))
    return '\n\n'.join(out)


# ------------------------------------------------------------------ samples: attempt length, rel citations
LINE_RE = re.compile(r'^N(-?\d+)$')


def body_lines(body):
    toks = body.split()
    if 'QED' in toks:
        toks = toks[:toks.index('QED')]
    lines, cur = [], []
    for t in toks:
        cur.append(t)
        if t == ';':
            lines.append(cur)
            cur = []
    return lines


def cite_dists(line):
    """Citation distances (line index - cited index) of one decoded line, or [] if unparseable."""
    if not line or not LINE_RE.match(line[0]) or ':' not in line:
        return []
    i0 = int(line[0][1:])
    j = line.index(':')
    out = []
    for t in line[j + 2:]:
        m = LINE_RE.match(t)
        if m:
            out.append(i0 - int(m.group(1)))
    return out


def scan_samples(ev, variants):
    """Per (variant, L): written lines of every attempt. For rel runs at L >= 12: the incremental checker's first
    error of every failed sample (lengthexp/ndcheck.classify) and the largest citation distance on that line, and
    the citation distances of every line written."""
    sys.path.insert(0, LE)
    from ndcheck import classify, sequent
    att = collections.defaultdict(list)          # (v, L) -> [written lines]
    first_bad_k = collections.defaultdict(list)  # (label, L) -> [(kind, max citation distance at the first bad line)]
    emitted = collections.defaultdict(collections.Counter)   # label -> citation distances in all samples (L >= 12)
    for lab, e in ev.items():
        if e['kind'] or e['variant'] not in variants:
            continue
        for L in LENGTHS:
            fn = os.path.join(e['dir'], 'samples', f'std_L{L}.jsonl.gz')
            if not os.path.isfile(fn):
                continue
            deep = e['variant'] == 'rel' and L >= 12
            seqs = {}
            with gzip.open(fn, 'rt') as f:
                for l in f:
                    s = json.loads(l)
                    ls = body_lines(s['body'])
                    att[(e['variant'], L)].append(len(ls))
                    if not deep:
                        continue
                    for ln in ls:
                        emitted[lab].update(cite_dists(ln))
                    pr = PROMPTS.get(s['name'])
                    if pr is None:            # a theorem not in repr/testsets (another draw): not classified
                        SKIPPED['unknown_theorem'] += 1
                        continue
                    if pr not in seqs:
                        seqs[pr] = sequent(pr)
                    c = classify(pr, s['body'], seqs[pr])
                    if c['ok']:
                        continue
                    k = None
                    if c['kind'] == 'bad_line' and 1 <= c['line'] <= len(ls):
                        d = cite_dists(ls[c['line'] - 1])
                        k = max(d) if d else None
                    first_bad_k[(lab, L)].append((c['kind'], k))
    return att, first_bad_k, emitted


PROMPTS = {}
SKIPPED = collections.Counter()


def attempt_md(att, variants):
    rows, rows2 = [], []
    for v in variants:
        c1, c2 = [], []
        for L in LENGTHS:
            xs = att.get((v, L), [])
            c1.append(pct(np.mean([x >= L for x in xs])) if xs else '')
            c2.append(f'{np.median(xs):.0f}' if xs else '')
        rows.append([NAMES[v]] + c1)
        rows2.append([NAMES[v]] + c2)
    h = ['variant'] + [f'L{L}' for L in LENGTHS]
    return ('Share of attempts (all samples) with at least L_true written lines:\n\n' + md(h, rows)
            + '\n\nMedian written lines per attempt:\n\n' + md(h, rows2))


# ------------------------------------------------------------------ first error
def first_error(ev, variants):
    tab = {}
    for v in variants:
        for L in LENGTHS:
            rows = [r for e in runs_of(ev, v) for r in e['units'].get(('std', L), [])]
            if not rows:
                continue
            bad, idx, eq, nq, fails, risk = LA.fail_positions(rows)
            nf = sum(fails.values())
            mean_pos = sum(p * c for p, c in fails.items()) / nf if nf else float('nan')
            med = float(np.median([p for p, c in fails.items() for _ in range(c)])) if nf else float('nan')
            tot = nf + nq
            tab[(v, L)] = {'mean': mean_pos, 'median': med, 'early_qed': sum(eq.values()) / tot if tot else float('nan'),
                           'no_qed': nq / tot if tot else float('nan'), 'n': tot, 'fails': dict(fails), 'risk': dict(risk)}
    return tab


def hazard_md(tab, variants, Ls=LONG, min_risk=100, P=18):
    """P(fail at line p | lines 1..p-1 valid), pooled over seeds and over the lengths Ls; '' where fewer than min_risk
    samples are still at risk. Bold = the position of the largest hazard."""
    rows = []
    for v in variants:
        fails, risk = collections.Counter(), collections.Counter()
        for L in Ls:
            d = tab.get((v, L))
            if d:
                fails.update(d['fails'])
                risk.update(d['risk'])
        hz = {p: fails.get(p, 0) / risk[p] for p in range(1, P + 1) if risk.get(p, 0) >= min_risk}
        top = max(hz, key=hz.get) if hz else None
        rows.append([NAMES[v]] + [('' if p not in hz else (f'**{hz[p]:.2f}**' if p == top else f'{hz[p]:.2f}')) for p in range(1, P + 1)])
    return md(['variant'] + [f'p{p}' for p in range(1, P + 1)], rows)


def fig_hazard(tab, variants, out, min_risk=100):
    style()
    Ls = [12, 14, 16, 18]
    fig, axs = plt.subplots(1, len(Ls), figsize=(3.0 * len(Ls) + 1, 3.3), sharey=True, squeeze=False)
    for j, L in enumerate(Ls):
        ax = axs[0][j]
        for v in variants:
            d = tab.get((v, L))
            if not d:
                continue
            ps = [p for p in sorted(d['risk']) if d['risk'][p] >= min_risk]
            ax.plot(ps, [d['fails'].get(p, 0) / d['risk'][p] for p in ps], color=COLORS[v], marker=MARK[v], ms=3, lw=1.6,
                    label=NAMES[v])
        ax.axvline(TRAIN_MAX + 1, color=INK2, lw=0.9, ls='--')
        ax.set_title(f'test L{L}', fontsize=9)
        ax.set_xlabel('line position p')
    axs[0][0].set_ylabel('P(fail at p | lines 1..p-1 valid)')
    axs[0][0].legend(fontsize=7)
    fig.suptitle(f'Per-line failure hazard on long tests (positions with >= {min_risk} samples at risk; dashed = training max + 1 = 12)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fn = os.path.join(out, 'fig_hazard.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    return os.path.basename(fn)


def first_error_md(tab, variants):
    rows = []
    for v in variants:
        rows.append([NAMES[v]] + [(f"{tab[(v, L)]['mean']:.1f} / {tab[(v, L)]['median']:.0f}" if (v, L) in tab else '') for L in LENGTHS])
    return md(['variant'] + [f'L{L}' for L in LENGTHS], rows)


# ------------------------------------------------------------------ rel citation coverage
def rel_counts(proofs):
    """Counter of citation distances in abs-text proofs (current line index - cited index)."""
    c = collections.Counter()
    for p in proofs:
        for ln in body_lines(p):
            c.update(cite_dists(ln))
    return c


def reference_proofs():
    ref = {}
    for fn in REF_SOURCES:
        if not os.path.isfile(fn):
            continue
        for r in read_jsonl(fn):
            p = r.get('minlen_proof') or r.get('constructed_proof') or r.get('gen_proof')
            if p and p != 'None':
                ref[r['name']] = p
    return ref


def coverage_md(pre, per_run):
    ks = sorted(set(pre) | {k for c in per_run.values() for k in c})
    ks = [k for k in ks if k >= 1]
    rows = [['pretraining (cap-6, all 154,990 records)'] + [str(pre.get(k, 0)) for k in ks]]
    for lab, c in sorted(per_run.items()):
        rows.append([f'RL-trained proofs, {lab} (distinct found proofs <= 11 lines)'] + [str(c.get(k, 0)) for k in ks])
    return md(['source'] + [f'B{k}' for k in ks], rows)


# ------------------------------------------------------------------ figures
def style():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'lines.linewidth': 2})


def fig_pass(tab, variants, out, ref=None):
    style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in variants + (['C_le'] if ref else []):
        t = ref if v == 'C_le' else tab
        Ls = [L for L in LENGTHS if t.get((v, L), (None,))[0] is not None]
        if not Ls:
            continue
        m = np.array([t[(v, L)][0] for L in Ls])
        lo = np.array([t[(v, L)][1] for L in Ls])
        hi = np.array([t[(v, L)][2] for L in Ls])
        ls = '--' if v == 'C_le' else '-'
        axs[0].plot(Ls, m, ls, color=COLORS[v], marker=MARK[v], ms=5, label=NAMES[v])
        axs[0].fill_between(Ls, lo, hi, color=COLORS[v], alpha=0.12, lw=0)
        eps = 1e-4
        nz = m > 0                         # log-odds of an estimated 0 is -inf: zeros are left out of this panel
        mc = np.clip(m[nz], eps, 1 - eps)
        axs[1].plot(np.array(Ls)[nz], np.log(mc / (1 - mc)), ls, color=COLORS[v], marker=MARK[v], ms=5, label=NAMES[v])
    for ax in axs:
        ax.axvline(TRAIN_MAX + 0.5, color=INK2, lw=1, ls=':')
        ax.set_xlabel('test length L_true (lines)')
        ax.set_xticks(LENGTHS)
    axs[0].set_ylabel('pass@1')
    axs[0].set_title('pass@1 (band: range over seeds)', fontsize=10, color=INK)
    axs[1].set_ylabel('log-odds of pass@1 (lengths with pass@1 = 0 omitted)')
    axs[1].set_title('log-odds; dotted line = training max (11)', fontsize=10, color=INK)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(out, 'fig_pass.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    return os.path.basename(fn)


def fig_attempts(att, fe, variants, out):
    style()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for v in variants:
        Ls = [L for L in LENGTHS if att.get((v, L))]
        if Ls:
            axs[0].plot(Ls, [np.mean([x >= L for x in att[(v, L)]]) for L in Ls], color=COLORS[v], marker=MARK[v], ms=5, label=NAMES[v])
            axs[1].plot(Ls, [np.median(att[(v, L)]) for L in Ls], color=COLORS[v], marker=MARK[v], ms=5, label=NAMES[v])
        Ls = [L for L in LENGTHS if (v, L) in fe]
        if Ls:
            axs[2].plot(Ls, [fe[(v, L)]['mean'] for L in Ls], color=COLORS[v], marker=MARK[v], ms=5, label=NAMES[v])
    axs[1].plot(LENGTHS, LENGTHS, ':', color=INK2, lw=1, label='L_true')
    axs[0].set_title('share of attempts with >= L_true lines', fontsize=10)
    axs[1].set_title('median written lines per attempt', fontsize=10)
    axs[2].set_title('mean first-error line (failed samples)', fontsize=10)
    for ax in axs:
        ax.set_xlabel('test length L_true')
        ax.set_xticks(LENGTHS)
        ax.axvline(TRAIN_MAX + 0.5, color=INK2, lw=1, ls=':')
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(out, 'fig_attempts.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    return os.path.basename(fn)


def fig_rel_split(split, variants, out):
    style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.38
    xs = np.arange(len(variants))
    for j, (g, hatch) in enumerate((('within', None), ('beyond', '//'))):
        vals = [split.get((v, g), (None,))[0] or 0 for v in variants]
        ax.bar(xs + (j - 0.5) * w, vals, w * 0.95, color=[COLORS[v] for v in variants], hatch=hatch,
               edgecolor='white', linewidth=0.5, label=f'reference proof needs a citation {g} the trained range')
        for x, val in zip(xs + (j - 0.5) * w, vals):
            ax.text(x, val, pct(val), ha='center', va='bottom', fontsize=7, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([v for v in variants])
    ax.set_ylabel('pass@1, L12-18 pooled')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fn = os.path.join(out, 'fig_rel_split.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    return os.path.basename(fn)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-root', default=EVAL)
    ap.add_argument('--runs-root', default=RUNS)
    ap.add_argument('--pretrain-root', default=PRETRAIN_DIR)
    ap.add_argument('--out', default=REPORT)
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--no-samples', action='store_true', help='skip the per-sample scans (attempt length, rel first errors)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = load_evals(a.eval_root)
    incomplete = sorted(lab for lab, e in ev.items() if any(('std', Lx) not in e['units'] for Lx in LENGTHS))
    for lab in incomplete:                       # a run enters the analysis only with every standard length evaluated
        del ev[lab]
    variants = [v for v in ORDER if runs_of(ev, v)]
    runs_n = {v: len(runs_of(ev, v)) for v in variants}
    le = load_le_armC()
    shas = unit_shas(ev)
    le_dropped = keep_matching(le, shas)
    for r in read_jsonl(os.path.join(HERE, 'testsets', 'std.jsonl')):
        PROMPTS[r['name']] = r['prompt']
    L = []
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L.append(f'---\nwritten_on: {datetime.date.today().isoformat()}\n---\n\n# Representation and length generalisation: report\n\n'
             f'Generated {now} by `analyse.py` from `{a.eval_root}`. Every variant: 039 pretraining recipe (2,911 fixed steps), '
             f'then arm C RL (L_true 6-11, proofs <= 11 lines, DEFAULT_FLOPS), then k = 32 samples per theorem at T = 0.8 '
             f'(max_new 1024) at every length. pass@k are unbiased estimates averaged over theorems; brackets are the '
             f'[min, max] over seeds. Runs evaluated: ' + ', '.join(f'{v} {runs_n[v]}' for v in variants) + '.'
             + (f' Left out (evaluation incomplete): {", ".join(incomplete)}.' if incomplete else '') + '\n')

    # ---- pass tables
    tabs = {k: pass_table(ev, variants, k) for k in (1, 4, 32)}
    ref1 = pass_table(le, ['C_le'], 1) if le else None
    # ---- headline
    fits = section_slopes(ev, variants, a.boot)
    head = []
    for v in variants:
        rs = runs_of(ev, v)
        id_ = mean_rng([pooled_rng(e, 6, 11) for e in rs])
        lg = mean_rng([pooled_rng(e, 12, 18) for e in rs])
        l12 = tabs[1].get((v, 12), (None,))[0]
        ok = [f['b'] for f in fits['d>=1 (L12-18)'][v] if f['ok']]
        m, lo, hi = tint(ok)
        ok0 = [f['b'] for f in fits['d>=0 (L11-18)'][v] if f['ok']]
        m0, lo0, hi0 = tint(ok0)
        rl = [json.load(open(os.path.join(a.runs_root, label(v, e['seed']), 'summary.json'))).get('heldout_greedy')
              for e in rs if os.path.isfile(os.path.join(a.runs_root, label(v, e['seed']), 'summary.json'))]
        head.append([NAMES[v], str(len(rs)), pct(id_[0]), pct(tabs[1].get((v, 11), (None,))[0]), pct(l12), pct(lg[0], 2),
                     (f'{f2(m)} [{f2(lo)}, {f2(hi)}]' if len(ok) >= 2 else (f2(m) if ok else 'n/a')),
                     (f'{f2(m0)} [{f2(lo0)}, {f2(hi0)}]' if len(ok0) >= 2 else (f2(m0) if ok0 else 'n/a')),
                     pct(mean_rng(rl)[0])])
    L.append('## Headline\n\n' + md(['variant', 'runs', 'pass@1 L6-11 pooled', 'pass@1 L11', 'pass@1 L12', 'pass@1 L12-18 pooled',
                                    'slope past 11, L12-18 [95% t]', 'slope from 11, L11-18 [95% t]', 'held-out greedy after RL'], head))

    # ---- 1 reproduction
    rep = []
    for s in SEEDS:
        for kind, d in (('released 039', f'ref039_s{s}'), ('abs re-pretrain (ours)', label('abs', s))):
            fn = os.path.join(a.pretrain_root, d, 'summary.json')
            if os.path.isfile(fn):
                j = json.load(open(fn))
                rep.append([kind, str(s), str(j.get('steps') or j.get('released_summary', {}).get('steps')), f2(j.get('val_loss'), 4),
                            pct(j.get('heldout_greedy_039'), 2), pct(j.get('heldout_greedy'), 2),
                            f2(j.get('train_secs'), 0) if j.get('train_secs') else f"{j.get('released_summary', {}).get('secs', 0):.0f} (A40)"])
    L.append('\n## 1. Reproduction check: abs re-pretrain against the released 039 checkpoints\n\n'
             'Both measured with the same code on our cards. val loss = 039\'s val_loss (train records [0, 1024), no shift); held-out '
             'greedy (039) = 039 harness slice heldout_cap6[::2][:2000]; held-out greedy (frozen) = run-3 harness slice [::5][:1000].\n\n'
             + md(['checkpoint', 'seed', 'steps', 'val loss', 'held-out greedy (039)', 'held-out greedy (frozen)', 'train secs'], rep))
    pre_rows = []
    for v in ORDER:
        for s in SEEDS:
            fn = os.path.join(a.pretrain_root, label(v, s), 'summary.json')
            if os.path.isfile(fn):
                j = json.load(open(fn))
                pre_rows.append([NAMES[v], str(s), f2(j.get('val_loss'), 4), pct(j.get('heldout_greedy_039'), 2), pct(j.get('heldout_greedy'), 2),
                                 (f"{f2(j.get('val_loss_randpos'), 4)} / {pct(j.get('heldout_greedy_randpos'), 2)}" if 'val_loss_randpos' in j else '-'),
                                 f"{j.get('steps_per_sec', 0):.2f}"])
    L.append('\nAll pretrained checkpoints (val loss is per proof token and not comparable across tokenizers: rel has no line-index '
             'tokens):\n\n' + md(['variant', 'seed', 'val loss', 'held-out greedy (039)', 'held-out greedy (frozen)',
                                  'random positions: val loss / greedy (039)', 'steps/s'], pre_rows))

    # ---- 2 pass tables
    L.append('\n## 2. pass@k by variant and length (standard sets)\n')
    for k in (1, 4, 32):
        L.append(f'**pass@{k}**\n\n' + pass_md(tabs[k], variants, LENGTHS, runs_n) + '\n')
    if ref1:
        L.append('Reference: lengthexp arm C (the older frozen/pretrain.pt, same RL recipe, seeds 101-103), pass@1:\n\n'
                 + pass_md(ref1, ['C_le'], LENGTHS, {'C_le': 3}) + '\n')
    f_pass = fig_pass(tabs[1], variants, a.out, ref1)
    L.append(f'![pass@1]({f_pass})\n')

    # ---- 3 slopes
    L.append('\n## 3. Log-odds drop per extra line past the training max (11)\n\nGrouped-binomial logistic fit of pass@1 per run '
             '(lengthexp/analyse.py fit_logistic) against d = L - 11. The slope is the change in log-odds per extra line (more '
             'negative = steeper fall-off). "n/a" = successes at fewer than two lengths in the window, so the slope is not estimable. '
             'Tests against abs: Welch t and the exact permutation test over the 3 + 3 runs (smallest possible p = 0.10).\n\n'
             + slopes_md(fits, variants))

    # ---- differences from abs (seed-level)
    def metric_vals(v, fn):
        return [fn(e) for e in sorted(runs_of(ev, v), key=lambda e: e['seed'])]
    mets = [('pass@1 L6-11 pooled', lambda e: pooled_rng(e, 6, 11)), ('pass@1 L11', lambda e: pooled_rng(e, 11, 11)),
            ('pass@1 L12', lambda e: pooled_rng(e, 12, 12)), ('pass@1 L13', lambda e: pooled_rng(e, 13, 13)),
            ('pass@1 L12-18 pooled', lambda e: pooled_rng(e, 12, 18))]
    drows = []
    base = {m: metric_vals('abs', f) for m, f in mets} if 'abs' in variants else {}
    for v in variants:
        if v == 'abs' or not base:
            continue
        for m, f in mets:
            x, y = metric_vals(v, f), base[m]
            t, p = LA.welch(x, y) if len(x) >= 2 and len(y) >= 2 else (float('nan'), float('nan'))
            d = np.mean(x) - np.mean(y)
            pp = perm_p(x, y)
            drows.append([NAMES[v], m, f'{pct(np.mean(x), 2)} vs {pct(np.mean(y), 2)}', f'{100 * d:+.2f} points',
                          LA.fmt_p(p) if np.isfinite(p) else '-', LA.fmt_p(pp) if np.isfinite(pp) else '-'])
    if drows:
        L.append('\n**Differences from abs (seed means; Welch t and exact permutation p over 3 + 3 runs, smallest possible 0.10)**\n\n'
                 + md(['variant', 'measure', 'variant vs abs', 'difference', 'Welch p', 'perm p'], drows))

    # ---- 4/5 attempts and first errors
    fe = first_error(ev, variants)
    att = first_bad = emitted = None
    if not a.no_samples:
        att, first_bad, emitted = scan_samples(ev, variants)
        L.append('\n## 4. Attempt length\n\n' + attempt_md(att, variants))
    L.append('\n## 5. First-error position\n\nCell: mean / median position of the first failure among failed samples (first line the '
             'incremental checker rejects; early QED counts at the QED slot).\n\n' + first_error_md(fe, variants))
    if att:
        f_att = fig_attempts(att, fe, variants, a.out)
        L.append(f'\n![attempts and first errors]({f_att})\n')
    f_hz = fig_hazard(fe, variants, a.out)
    L.append('\n**Per-position failure hazard on the long sets** (L12-18 pooled over lengths and seeds): P(first failure at line p '
             '| lines 1..p-1 valid); an early QED counts at its QED slot; blank = fewer than 100 samples still at risk; bold = the '
             'largest. A spike near p = 12 (training max + 1) would mean attempts die where the training data ends (a '
             'length-specific circuit); a flat hazard means errors are spread over lines.\n\n' + hazard_md(fe, variants)
             + '\n\nThe same on L15-18 only (the constructed long sets):\n\n' + hazard_md(fe, variants, Ls=(15, 16, 17, 18))
             + f'\n\n![hazard]({f_hz})\n')

    # ---- 6 in-distribution
    idr = []
    for v in variants:
        pre = [json.load(open(os.path.join(a.pretrain_root, label(v, s), 'summary.json'))).get('heldout_greedy_039')
               for s in SEEDS if os.path.isfile(os.path.join(a.pretrain_root, label(v, s), 'summary.json'))]
        rl = [json.load(open(os.path.join(a.runs_root, label(v, e['seed']), 'summary.json'))).get('heldout_greedy')
              for e in runs_of(ev, v) if os.path.isfile(os.path.join(a.runs_root, label(v, e['seed']), 'summary.json'))]
        p611 = mean_rng([pooled_rng(e, 6, 11) for e in runs_of(ev, v)])
        m6 = mean_rng([pooled_rng(e, 6, 11, 'matched') for e in runs_of(ev, v)])
        idr.append([NAMES[v], pct(mean_rng(pre)[0], 2), pct(mean_rng(rl)[0], 2), f'{pct(p611[0])} [{pct(p611[1])}, {pct(p611[2])}]',
                    pct(m6[0])])
    L.append('\n## 6. In-distribution checks\n\n' + md(['variant', 'held-out greedy after pretraining (039 slice)',
                                                       'held-out greedy after RL (harness)', 'pass@1 L6-11 pooled (std)',
                                                       'pass@1 L6-11 pooled (matched)'], idr))

    # ---- 7 training side
    tr = []
    for v in variants:
        for e in sorted(runs_of(ev, v), key=lambda e: e['seed']):
            fn = os.path.join(a.runs_root, label(v, e['seed']), 'summary.json')
            if not os.path.isfile(fn):
                continue
            h = json.load(open(fn))
            rs = h.get('rl_stats', {})
            last = (rs.get('per_round') or [{}])[-1]
            cum = last.get('solved_cum_all_by_len', {})
            tb = rs.get('targets_by_len', {})
            sol = ', '.join(f'L{k} {cum.get(k, 0) / tb[k]:.2f}' for k in sorted(tb, key=int) if tb.get(k))
            tr.append([label(v, e['seed']), h['status'], str(rs.get('rounds')), f"{rs.get('total_samples', 0):,}",
                       f"{h.get('rl_secs', 0):.0f}", f"{h['compute']['frac']:.3f}", str(rs.get('stop_reason')), sol])
    L.append('\n## 7. Training side\n\n' + md(['run', 'status', 'rounds', 'samples', 'rl secs', 'compute used', 'stop', 'solved at the end, by L_true'], tr))

    # ---- 8 rel citations
    if 'rel' in variants:
        pre = rel_counts([r['proof'] for r in read_jsonl(TRAIN_CAP6)])
        per_run, kmax = {}, {}
        for vv in ('rel', 'abs'):
            for e in runs_of(ev, vv):
                fn = os.path.join(a.runs_root, label(vv, e['seed']), 'found_targets.jsonl')
                if os.path.isfile(fn):
                    c = rel_counts([r['proof'] for r in read_jsonl(fn) if r['written'] <= TRAIN_MAX])
                    per_run[label(vv, e['seed'])] = c
                    if vv == 'rel':
                        kmax[e['seed']] = max([k for k in set(c) | set(pre) if (c.get(k, 0) + pre.get(k, 0)) > 0])
        K = min(kmax.values()) if kmax else max(pre)
        ref = reference_proofs()
        need = {n: max([max(cite_dists(ln) or [0]) for ln in body_lines(p)] or [0]) for n, p in ref.items()}
        split = {}
        for v in variants:
            for g in ('within', 'beyond'):
                vals = []
                for e in runs_of(ev, v):
                    rows = [r for Lx in LONG for r in e['units'].get(('std', Lx), []) if r['name'] in need
                            and ((need[r['name']] > K) == (g == 'beyond'))]
                    if rows:
                        vals.append(LA.passk(rows, 1))
                split[(v, g)] = mean_rng(vals)
        n_th = {g: sum(1 for Lx in LONG for r in next(iter(runs_of(ev, 'rel')))['units'].get(('std', Lx), [])
                       if r['name'] in need and ((need[r['name']] > K) == (g == 'beyond'))) for g in ('within', 'beyond')}
        srows = [[NAMES[v], pct(split[(v, 'within')][0], 2), pct(split[(v, 'beyond')][0], 2)] for v in variants]
        BINS = [(1, 5), (6, 6), (7, 8), (9, 10), (11, 64)]
        some = next(iter(runs_of(ev, 'rel')))
        brow_n = []
        for lo_, hi_ in BINS:
            brow_n.append(sum(1 for Lx in LONG for r in some['units'].get(('std', Lx), []) if r['name'] in need and lo_ <= need[r['name']] <= hi_))
        brows = []
        for v in variants:
            cells = []
            for lo_, hi_ in BINS:
                vals = [LA.passk(rows, 1) for e in runs_of(ev, v)
                        for rows in [[r for Lx in LONG for r in e['units'].get(('std', Lx), []) if r['name'] in need and lo_ <= need[r['name']] <= hi_]] if rows]
                cells.append(pct(mean_rng(vals)[0], 2))
            brows.append([NAMES[v]] + cells)
        bin_md = md(['variant'] + [f'needs B{lo_}' + (f'-B{hi_}' if hi_ != lo_ else '') + f' ({n} thm)' for (lo_, hi_), n in zip(BINS, brow_n)], brows)
        rsn = []
        for v in ('abs', 'rel'):
            c = collections.Counter()
            for e in runs_of(ev, v):
                for Lx in LONG:
                    for r in e['units'].get(('std', Lx), []):
                        c.update(f'{kind}:{reason}' for kind, line, reason, start in r['fails'])
            tot = sum(c.values())
            if tot:
                rsn.append([NAMES[v]] + [', '.join(f'{k} {pct(n / tot)}' for k, n in c.most_common(6))])
        rsn_md = md(['variant', 'first-error kinds at L12-18 (share of failed samples)'], rsn)
        fb = []
        if first_bad:
            for lab in sorted({l for l, _ in first_bad}):
                xs = [x for Lx in LONG for x in first_bad.get((lab, Lx), [])]
                bl = [k for kind, k in xs if kind == 'bad_line']
                fb.append([lab, str(len(xs)), pct(len(bl) / len(xs) if xs else float('nan')),
                           pct(np.mean([k is not None and k > kmax.get(parse_label(lab)[1], K) for k in bl]) if bl else float('nan'), 2)])
        em = ''
        if emitted:
            em = '\n\nCitation distances emitted by rel samples at L12-18 (all lines of all samples), share beyond the run\'s trained range: ' + \
                 ', '.join(f"{lab} {pct(sum(c for k, c in cnt.items() if k > kmax.get(parse_label(lab)[1], K)) / max(1, sum(cnt.values())), 2)}"
                           for lab, cnt in sorted(emitted.items()))
        f_split = fig_rel_split(split, variants, a.out)
        L.append('\n## 8. rel: citation-distance coverage and the long-length split\n\n'
                 'Citation distance k = citing line index - cited line index (rel writes it as the token B<k>). Counts of '
                 'citations by k in the pretraining data and in the proofs each rel run trained on (abs runs for comparison: the same '
                 'distances written with N<j> tokens):\n\n' + coverage_md(pre, per_run)
                 + f'\n\nTrained range per rel run (largest k seen at least once in pretraining or its RL-trained proofs): '
                 + ', '.join(f'seed {s}: B{k}' for s, k in sorted(kmax.items())) + f'. The split uses K = B{K} (the smallest).\n\n'
                 f'Theorems at L12-18 split by the largest citation distance their reference proof needs (minlen_proof, else '
                 f'constructed_proof; one reference proof per theorem, so "needs" means this proof needs it). Within = <= {K} '
                 f'({n_th["within"]} theorems), beyond = > {K} ({n_th["beyond"]} theorems). Every variant is scored on the same '
                 f'split, so abs is the control for the difficulty of the "beyond" theorems.\n\n'
                 + md(['variant', f'pass@1, needs <= B{K}', f'pass@1, needs > B{K}'], srows)
                 + '\n\nThe same theorems in bins of the largest citation distance their reference proof needs (pass@1, L12-18 pooled):\n\n'
                 + bin_md + '\n\nWhat the first error is (incremental checker reason codes) for abs and rel at L12-18:\n\n' + rsn_md
                 + (('\n\nrel failed samples at L12-18: share whose first error is a bad line, and share of those first bad lines '
                     'that cite beyond the run\'s trained range:\n\n' + md(['run', 'failed samples', 'bad line', 'first bad line cites > trained range'], fb)) if fb else '')
                 + em + f'\n\n![rel split]({f_split})')

    # ---- 9 posrand check
    if runs_of(ev, 'posrand', 'randpos'):
        t_r = pass_table(ev, ['posrand'], 1, kind='randpos')
        rows = [['contiguous positions (primary)'] + [pct(tabs[1].get(('posrand', Lx), (None,))[0]) for Lx in LENGTHS],
                ['random positions (check)'] + [pct(t_r.get(('posrand', Lx), (None,))[0]) for Lx in LENGTHS]]
        L.append('\n## 9. pos-rand: evaluation on random positions\n\n' + md(['pos-rand eval'] + [f'L{Lx}' for Lx in LENGTHS], rows))
    else:
        L.append('\n## 9. pos-rand: evaluation on random positions\n\nNot run.')

    # ---- 10 matched
    tm = pass_table(ev, variants, 1, st='matched')
    L.append('\n## 10. Matched-construction control (pass@1)\n\n' + pass_md(tm, variants, range(6, 15), runs_n))

    # ---- checks
    multi = {f'{k[0]}_L{k[1]}': len(v) for k, v in shas.items() if len(v) > 1}
    dis = collections.Counter()
    hit = n_samp = 0
    for e in ev.values():
        for sm in e['summ'].values():
            dis.update(sm.get('checker_disagreements') or {})
            hit += sm.get('n_hit_max_new', 0)
            n_samp += sm['n'] * sm['k']
    L.append('\n## Checks\n\n'
             f'- Theorem draw: every evaluated unit has one theorem list (unit_sha) across all labels: '
             f'{"yes" if not multi else "NO: " + str(multi)}. Test sets = repr/testsets (frozen copy of lengthexp\'s final draw, '
             f'2026-09-25 02:27 UTC). Reference arm C units on another draw dropped: {le_dropped}.\n'
             f'- Incremental checker vs nd_verify disagreements over {n_samp:,} samples: '
             + (', '.join(f'{k} {v}' for k, v in dis.items()) if dis else 'none') + f'.\n- Samples that hit max_new: {hit:,}.'
             + (f'\n- Samples skipped by the per-sample scan (theorem not in repr/testsets): {SKIPPED["unknown_theorem"]:,}.' if SKIPPED else ''))
    with open(os.path.join(a.out, 'report.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    js = {'pass': {k: {f'{v}|{Lx}': val for (v, Lx), val in t.items()} for k, t in tabs.items()},
          'slopes': {w: {v: [{kk: f[kk] for kk in ('seed', 'b', 'se_b', 'ok', 'boot_se', 'why')} for f in fs] for v, fs in per.items()}
                     for w, per in fits.items()},
          'first_error': {f'{v}|{Lx}': {k: x for k, x in d.items() if k not in ('fails', 'risk')} for (v, Lx), d in fe.items()}}
    if att:
        js['attempt_share_ge_L'] = {f'{v}|{Lx}': float(np.mean([x >= Lx for x in xs])) for (v, Lx), xs in att.items() if xs}
        js['attempt_median_lines'] = {f'{v}|{Lx}': float(np.median(xs)) for (v, Lx), xs in att.items() if xs}
    with open(os.path.join(a.out, 'results.json'), 'w') as f:
        json.dump(js, f, indent=1, default=str)
    print('wrote', os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
