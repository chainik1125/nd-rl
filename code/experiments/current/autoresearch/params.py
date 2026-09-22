#!/usr/bin/env python3
"""Pre-registration for one experiment: results/<NNN>_<slug>/params.json + pretrain.patch, written BEFORE the run.

  python3 params.py new <NNN> <slug>      # after editing pretrain.py; fills the mechanical fields, leaves TODOs
  python3 params.py check <NNN>_<slug>    # no TODOs left, config.n_params numeric; record.py runs this too

Mechanical (never hand-typed): base_commit, harness/pretrain sha, the patch, and `protocol` = the
harness constants plus the pool sizes and proof-length histograms from pools/<fork-commit>.json.
Hand-filled by the agent: provider/model, title/description/hypothesis/prediction, and `config`
(what pretrain.py builds: arch, tokenizer, optimizer, batch, data, precision, seed).

Shared helpers (harness_rules, fork_commit, sha) are imported by record.py and pools.py.
"""
import hashlib, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TODO = 'TODO'


def sha(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()[:12]


def git(*args):
    return subprocess.run(['git', *args], cwd=HERE, capture_output=True, text=True, check=True).stdout


def fork_commit():
    return re.search(r'^FORK_COMMIT=([0-9a-f]+)', open(f'{HERE}/remote.sh').read(), re.M).group(1)


def harness_rules():
    """The UPPER_CASE constants at the top of harness.py, parsed from text (harness.py imports torch and the fork)."""
    src = open(f'{HERE}/harness.py').read().split('\nDEV = ')[0]
    ns = {}
    for line in src.splitlines():
        if re.match(r'^[A-Z_][A-Z_0-9, ]* = ', line):
            try:
                exec(line.split('  #')[0], {'os': os, '__file__': f'{HERE}/harness.py'}, ns)
            except Exception:   # a constant that needs torch or the fork; not a rule we record
                pass
    return {k: v for k, v in ns.items() if isinstance(v, (int, float))}


def protocol():
    H, fc = harness_rules(), fork_commit()
    pools_fn = f'{HERE}/pools/{fc[:7]}.json'
    if not os.path.exists(pools_fn):
        raise SystemExit(f'{pools_fn} missing: remote.sh pins fork {fc[:7]}; run `python3 pools.py <fork-checkout>` once')
    P = json.load(open(pools_fn))
    return {
        'fork_commit': fc,
        'budget': {'total_secs': H['TOTAL_SECS'], 'pretrain_secs': H['PRETRAIN_SECS'], 'grace_secs': H['PRETRAIN_GRACE'],
                   'param_cap': H['PARAM_CAP']},
        'pretrain_data': {'max_lines': 6, **P['pretrain']},
        'pretrain_heldout': P['pretrain_heldout'],
        'rl': {'rounds': H['RL_ROUNDS'], 'n_targets': H['N_TARGETS'], 'rollouts_per_target': H['K'], 'temp': H['TEMP'],
               'max_new_tokens': H['MAX_NEW'], 'ft_steps': H['FT_STEPS'], 'ft_bs': H['FT_BS'], 'ft_lr': H['FT_LR'],
               'retain_pretrain_records': H['RETAIN'], 'max_proofs_per_thm': H['MAX_PER_THM'], 'rl_weight': H['RL_WEIGHT'],
               'target_pool': P['rl_targets']['pool'], 'targets_sampled': P['rl_targets']['sampled']},
        'eval': {'metric': 'dev-transfer theorems solved at L_true >= 7', 'attempts_per_theorem': H['EVAL_K'],
                 'temp': H['TEMP'], 'pool': P['transfer_dev']},
        'holdout': P['transfer_holdout'],
        'seed': 0,
    }


CONFIG_TEMPLATE = {
    'arch': {'type': 'gpt (fork model.py)', 'n_layers': TODO, 'd_model': TODO, 'n_heads': TODO, 'd_ff': TODO,
             'pos': 'rope', 'ctx': 1024, 'norm': 'layernorm', 'act': 'gelu', 'dropout': 0.0, 'tied_emb': False,
             'n_params': TODO},
    'tokenizer': {'codec': TODO, 'maxn': 64},
    'optimizer': {'name': TODO, 'lr': TODO, 'schedule': TODO, 'warmup_steps': TODO, 'min_lr_frac': TODO,
                  'wd': TODO, 'betas': TODO, 'grad_clip': TODO},
    'batch': {'size': TODO, 'padding': TODO, 'buckets': TODO},
    'data': {'filter': None, 'augmentation': TODO, 'order': TODO, 'val_slice': TODO},
    'precision': TODO, 'compile': TODO, 'init': 'fork default (normal 0.02)', 'seed': 0,
}


def new(nnn, slug):
    name = f'{nnn}_{slug}'
    d = f'{HERE}/results/{name}'
    if os.path.exists(d):
        raise SystemExit(f'{d} exists')
    os.makedirs(d)
    patch = git('diff', 'HEAD', '--', 'pretrain.py')
    open(f'{d}/pretrain.patch', 'w').write(patch)
    p = {
        'id': nnn, 'slug': slug, 'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'provider': TODO, 'model': TODO,
        'base_commit': git('rev-parse', '--short', 'HEAD').strip(),
        'harness_sha': sha(f'{HERE}/harness.py'), 'pretrain_sha': sha(f'{HERE}/pretrain.py'),
        'patch_lines': sum(1 for l in patch.splitlines() if l[:1] in '+-' and l[:3] not in ('+++', '---')),
        'title': TODO, 'description': TODO, 'tags': [TODO], 'hypothesis': TODO,
        'p_keep': TODO, 'expected_delta': TODO, 'baseline_id': TODO, 'baseline_metric': TODO,
        'config': CONFIG_TEMPLATE,
        'protocol': protocol(),
    }
    json.dump(p, open(f'{d}/params.json', 'w'), indent=2)
    print(f'{d}/params.json  (patch {p["patch_lines"]} lines; fill every TODO, then commit before running)')


def check(name):
    p = json.load(open(f'{HERE}/results/{name}/params.json'))
    todos = [k for k, v in walk(p) if v == TODO]
    if todos:
        raise SystemExit(f'{name}/params.json has TODOs: {todos}')
    if not isinstance(p['config']['arch']['n_params'], int):
        raise SystemExit(f'{name}: config.arch.n_params must be an int')
    return p


def walk(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f'{path}[{i}]')
    else:
        yield path, x


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'new':
        new(sys.argv[2], sys.argv[3])
    elif cmd == 'check':
        check(sys.argv[2])
        print('ok')
    else:
        raise SystemExit(__doc__)
