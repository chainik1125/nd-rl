#!/usr/bin/env python3
"""Small extracts for the repository record of repr/ and goals/ (CPU only; no model is run).

    python record_extracts.py            # -> report/record/{runs_summary,pretrain_summary,derived}.json

runs_summary.json     every RL run of repr/ (abs, rel, nope, posline) and goals/: status, rounds, samples, optimizer
                      steps, compute used, wall time, held-out greedy, final.pt / source rl.py / harness.py sha256
pretrain_summary.json every pretrain of repr/ (5 variants x 3 seeds) and goals/ (3 seeds), and the released 039
                      checkpoints measured with the same code: steps, speed, val loss, held-out greedy, pretrain.pt sha256
derived.json          numbers used in the summary that the generated reports do not print directly:
                      - repr GPU time by stage (pretrain train + held-out secs, RL harness total secs, evaluation
                        sample + check secs per unit)
                      - shares of citation distances in the cap-6 pretraining proofs (from report section 8's counts)
                      - goals minus abs paired differences over the three seed pairs (95% t-interval, t = 4.303),
                        from goals/report/results.json
"""
import glob, hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOALS = '/scratch/cowifornia/nd-rl/ab3/goals'
OUT = os.path.join(HERE, 'report', 'record')
T975_2 = 4.302652729911275


def sha(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jl(fn):
    return json.load(open(fn))


def run_rec(d, label, variant, pseed, harness):
    s = jl(os.path.join(d, 'summary.json'))
    rs = s.get('rl_stats', {})
    return {'label': label, 'variant': variant, 'pretrain_seed': pseed, 'rl_seed': s.get('seed'), 'status': s['status'],
            'rounds': rs.get('rounds'), 'samples': rs.get('total_samples'), 'optimizer_steps': s['compute']['optimizer_steps'],
            'compute_frac': s['compute']['frac'], 'used_flops': s['compute']['used_flops'], 'rl_secs': s.get('rl_secs'),
            'total_secs': s.get('total_secs'), 'heldout_greedy': s.get('heldout_greedy'), 'stop_reason': rs.get('stop_reason'),
            'gpu': s.get('gpu'), 'final_pt_sha256': sha(os.path.join(d, 'final.pt')),
            'source_rl_py_sha256': sha(os.path.join(d, 'source', 'rl.py')) if os.path.isfile(os.path.join(d, 'source', 'rl.py')) else None,
            'harness_py_sha256': sha(harness) if os.path.isfile(harness) else None,
            'shas_recorded_by_harness': {k: v for k, v in (s.get('shas') or {}).items() if k in ('rl.py', 'harness.py', 'pretrain.pt', 'meter.py')}}


def t_int(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = T975_2 * sd / math.sqrt(len(xs))
    return {'per_pair': xs, 'mean': m, 'ci95': [m - h, m + h]}


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for d in sorted(glob.glob(os.path.join(HERE, 'runs', '*_s[0-9]'))):
        lab = os.path.basename(d)
        v, _, s = lab.rpartition('_s')
        runs.append(run_rec(d, lab, v, int(s), os.path.join(HERE, 'frozen', lab, 'harness.py')))
    for d in sorted(glob.glob(os.path.join(GOALS, 'runs', 'goals_s[0-9]'))):
        lab = os.path.basename(d)
        runs.append(run_rec(d, lab, 'goals', int(lab[-1]), os.path.join(GOALS, 'frozen', lab, 'harness.py')))
    json.dump(runs, open(os.path.join(OUT, 'runs_summary.json'), 'w'), indent=1)

    pre = []
    keys = ('steps', 'train_secs', 'steps_per_sec', 'val_loss', 'val_loss_lines', 'val_loss_randpos', 'heldout_greedy_039',
            'heldout_greedy', 'heldout_greedy_randpos', 'n_params', 'cfg', 'tok_mode', 'gpu', 'adamw_from_step')
    for root, pat in ((HERE, 'pretrain/*_s[0-9]'), (GOALS, 'pretrain/goals_s[0-9]')):
        for d in sorted(glob.glob(os.path.join(root, pat))):
            if os.path.basename(d).startswith('ref039'):
                continue                      # the released checkpoints: below
            s = jl(os.path.join(d, 'summary.json'))
            if s.get('smoke') in (True, 'True'):
                continue
            pre.append({'label': os.path.basename(d), **{k: s.get(k) for k in keys if k in s},
                        'pretrain_pt_sha256': sha(os.path.join(d, 'pretrain.pt'))})
    for d in sorted(glob.glob(os.path.join(HERE, 'pretrain', 'ref039_s[0-9]'))):
        s = jl(os.path.join(d, 'summary.json'))
        pre.append({'label': os.path.basename(d), 'released': True,
                    **{k: s.get(k) for k in ('ckpt', 'val_loss', 'heldout_greedy_039', 'heldout_greedy', 'released_summary')},
                    'pretrain_pt_sha256': sha(s['ckpt'])})
    json.dump(pre, open(os.path.join(OUT, 'pretrain_summary.json'), 'w'), indent=1)

    # ---- derived numbers
    der = {}
    pt = sum(r.get('train_secs', 0) for r in pre if not r.get('released') and not r['label'].startswith('goals'))
    pt_held = 0.0
    for d in glob.glob(os.path.join(HERE, 'pretrain', '*_s[0-9]')):
        if os.path.basename(d).startswith('ref039'):
            continue
        s = jl(os.path.join(d, 'summary.json'))
        pt_held += s.get('heldout_secs', 0)
    rl = sum(r['total_secs'] for r in runs if r['variant'] != 'goals')
    ev = 0.0
    for d in glob.glob(os.path.join(HERE, 'eval', '*_s[0-9]')):
        for f in glob.glob(os.path.join(d, '*.summary.json')):
            s = jl(f)
            ev += s['sample_secs'] + s['check_secs']
    der['repr_gpu_hours'] = {'pretrain_15': (pt + pt_held) / 3600, 'rl_12': rl / 3600, 'eval_12': ev / 3600,
                             'total': (pt + pt_held + rl + ev) / 3600,
                             'method': 'pretrain train_secs + heldout_secs; RL harness total_secs; evaluation sample_secs + '
                                       'check_secs per unit. Excludes model loading, lock waits, tests, smokes and diagnostics.'}
    res = jl(os.path.join(HERE, 'report', 'results.json'))
    cnt = {}
    rep = open(os.path.join(HERE, 'report', 'report.md')).read()
    m = re.search(r'\| pretraining \(cap-6, all 154,990 records\) \| ([^\n]+)', rep)
    if m:
        vals = [int(x.strip()) for x in m.group(1).strip(' |').split('|')]
        tot = sum(vals)
        cnt = {f'B{i + 1}': {'count': c, 'share': c / tot} for i, c in enumerate(vals)}
    der['pretraining_citation_distance'] = {'source': 'repr report section 8 (pretraining row)', 'total_citations': sum(v['count'] for v in cnt.values()), **cnt}
    beyond = {}
    for m in re.finditer(r'\| RL-trained proofs, (\w+_s\d) \(distinct found proofs <= 11 lines\) \| ([^\n]+)', rep):
        vals = [int(x.strip()) for x in m.group(2).strip(' |').split('|')]
        beyond[m.group(1)] = {'citations': sum(vals), 'share_beyond_B5': sum(vals[5:]) / sum(vals)}
    der['rl_trained_citations_beyond_B5'] = {'source': 'repr report section 8 (RL-trained proofs rows)', **beyond}
    m = re.search(r'\| variant \| (needs B1-B5 [^\n]+)', rep)
    if m:
        bins = [(c.split(' (')[0].strip(), int(c.split('(')[1].split(' thm')[0])) for c in m.group(1).strip(' |').split('|')]
        tot = sum(n for _, n in bins)
        der['L12_18_theorems_by_longest_needed_jump'] = {'source': 'repr report section 8 (bins header)', 'theorems': tot,
                                                         **{b: {'theorems': n, 'share': n / tot} for b, n in bins}}
    g = jl(os.path.join(GOALS, 'report', 'results.json'))
    pairs = []
    for L in (12, 13):
        pairs.append((f'pass@1 L{L}', [g['pass'][f'goals|{s}|{L}'][0] - g['pass'][f'abs|{s}|{L}'][0] for s in (0, 1, 2)]))
    for w in ('12-14', '12-18'):
        pairs.append((f'pass@1 pooled L{w}', [a - b for a, b in zip(g['pooled'][f'goals|{w}'], g['pooled'][f'abs|{w}'])]))
    pairs.append(('pass@1 pooled L6-11', [a - b for a, b in zip(g['indist']['goals']['pass1_L6_11'], g['indist']['abs']['pass1_L6_11'])]))
    der['goals_minus_abs_paired'] = {k: t_int(v) for k, v in pairs}
    der['goals_minus_abs_paired']['source'] = 'goals/report/results.json (pass, pooled, indist); seed pairs goals s_i vs repr abs s_i'
    der['repr_results_json_keys'] = sorted(res.keys())
    json.dump(der, open(os.path.join(OUT, 'derived.json'), 'w'), indent=1)
    print('wrote', OUT, len(runs), 'runs,', len(pre), 'pretrains')


if __name__ == '__main__':
    main()
