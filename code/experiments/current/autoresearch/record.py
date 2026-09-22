#!/usr/bin/env python3
"""Post-run record: results/<NNN>_<slug>/results.json from artifacts/autoresearch/<NNN>_<slug>/summary.json.

  python3 record.py <NNN>_<slug> keep|discard|crash|timeout "surprise text"

Checks the pre-registered params.json first: no TODOs, and the harness sha, pretrain sha and budget
that actually ran must equal what was registered (otherwise the run is not the experiment that was
pre-registered). Copies the loss curve, per-round RL numbers and the eval histograms into git.
"""
import json, os, sys, time

from params import HERE, check, git

ART = f'{HERE}/../../../../artifacts/autoresearch'


def results_from_summary(s, status, surprise=None):
    pre, rl, ev = s.get('pretrain', {}), s.get('rl', {}), s.get('eval', {})
    r = {
        'commit': git('rev-parse', '--short', 'HEAD').strip(), 'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'status': status, 'metric': s['metric'],
        'metric_rl': s.get('metric_rl') or (rl['rounds'][-1]['targets_solved'] if rl.get('rounds') else None),
        'harness_sha': s['harness_sha'], 'pretrain_sha': s['pretrain_sha'], 'rules': s['rules'],
        'gpu': s.get('gpu'), 'total_secs': round(s['total_secs']), 'reason': s.get('reason'),
        'pretrain': {**pre, 'secs': round(pre['secs'], 1) if 'secs' in pre else None},
        'rl_rounds': [{k: v for k, v in rd.items() if k not in ('sampling', 'finetune')}
                      | {'samples_per_sec': round(rd['sampling']['samples_per_sec']), 'secs': round(rd['secs'])}
                      for rd in rl.get('rounds', [])],
        'eval': {k: ev[k] for k in ('solved', 'n', 'ge', 'lstar', 'heldout_greedy', 'max_pruned', 'by_bin') if k in ev},
        'holdout_metric': None,
        'surprise': surprise,
    }
    if r['status'] == 'ok' and 'curve' not in pre:
        print('WARNING: pretrain stats have no `curve`; pretrain.py must return one (see its docstring)', file=sys.stderr)
    return r


def main():
    name, status, surprise = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None
    assert status in ('keep', 'discard', 'crash', 'timeout')
    p = check(name)
    s = json.load(open(f'{ART}/{name}/summary.json'))
    if s['status'] != 'ok':
        assert status in ('crash', 'timeout'), f"harness status {s['status']}"
    for k in ('harness_sha', 'pretrain_sha'):
        assert s[k] == p[k], f'{k}: ran {s[k]} but registered {p[k]}'
    b = p['protocol']['budget']
    ran = s['rules']
    assert (ran['TOTAL_SECS'], ran['PRETRAIN_SECS'], ran['PARAM_CAP']) == (b['total_secs'], b['pretrain_secs'], b['param_cap']), \
        f'budget ran {ran} but registered {b}'
    r = results_from_summary(s, status, surprise)
    json.dump(r, open(f'{HERE}/results/{name}/results.json', 'w'), indent=2)
    print(json.dumps({k: r[k] for k in ('status', 'metric', 'metric_rl', 'total_secs')}
                     | {'pretrain_secs': r['pretrain'].get('secs'), 'greedy': r['pretrain'].get('heldout_greedy'),
                        'r1_solved': r['rl_rounds'][0]['targets_solved'] if r['rl_rounds'] else None}))


if __name__ == '__main__':
    main()
