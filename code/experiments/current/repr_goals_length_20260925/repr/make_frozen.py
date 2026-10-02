#!/usr/bin/env python3
"""Build the frozen-copy harness of one variant x seed: repr/frozen/<variant>_s<seed>/ (the pattern of
ab3/frozen_039 = ab3/frozen with pretrain.pt replaced), and, with --template, one without pretrain.pt for tests.

    python make_frozen.py --label abs_s0 [--pretrain pretrain/abs_s0/pretrain.pt]
    python make_frozen.py --all                      # every variant x seed whose pretrain summary is ok

Contents: ab3/frozen's files unchanged (meter.py, metrics.py, gpulock.py, score*.py, SPEC.md, test_*.py, ref/),
data/ = symlinks to ab3/frozen/data/*, lib/ = repr/lib (the variant library), pretrain.pt = the variant's
checkpoint, and harness.py = ab3/frozen/harness.py with the three edits of patch_harness() (tokenizer mode and
positional variant read from pretrain.pt's tok_mode / cfg instead of the hard-coded abs / plain GPT). The edits
are checked textually (each must apply exactly once) and recorded with sha256s in <dir>/REPR_MANIFEST.json.
"""
import argparse, difflib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_repr import FROZEN0, FROZEN_DIR, LIB, ORDER, PRETRAIN_DIR, SEEDS, label, sha256_file, utc, write_json

EDITS = [
    ('from model import GPT  # noqa: E402\n',
     'from model import GPT, variant_kwargs  # noqa: E402  (repr/lib: GPT with positional variants)\n'),
    ('''def load_pretrained(device):
    tok = Tokenizer('abs')
    ck = torch.load(os.path.join(FROZEN, 'pretrain.pt'), map_location='cpu')
    assert ck.get('tok_mode', 'abs') == 'abs', ck.get('tok_mode')
    model = GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD)
    model.load_state_dict(ck['state'])
    return model.to(device), tok
''', '''VARIANT = {}    # repr: the checkpoint's tokenizer mode and positional variant (lib/model.py), set by load_pretrained


def load_pretrained(device):
    # repr edit: tokenizer mode and model variant from the checkpoint (frozen: Tokenizer('abs'), plain GPT)
    ck = torch.load(os.path.join(FROZEN, 'pretrain.pt'), map_location='cpu')
    tok = Tokenizer(ck.get('tok_mode') or 'abs')
    VARIANT.update(tok_mode=tok.mode, model=variant_kwargs(ck.get('cfg')))
    model = GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD, **VARIANT['model'])
    bad = {k: v for k, v in (ck.get('cfg') or {}).items() if model.cfg.get(k) != v}
    assert not bad, f'pretrain.pt cfg does not match the model built from it: {bad}'
    model.load_state_dict(ck['state'])
    return model.to(device), tok
'''),
    ('''        if not isinstance(model, GPT) or dict(model.cfg) != dict(GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD).cfg):''',
     '''        if not isinstance(model, GPT) or dict(model.cfg) != dict(GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD, **VARIANT['model']).cfg):'''),
]
HEADER = ('#!/usr/bin/env python3\n# REPR COPY (ab3/repr/make_frozen.py): ab3/frozen/harness.py with three edits so that the tokenizer mode and\n'
          '# the positional variant come from pretrain.pt (see REPR_MANIFEST.json for the diff). Nothing else differs.\n')


def patch_harness(src):
    assert src.startswith('#!/usr/bin/env python3\n')
    out = src
    for a, b in EDITS:
        if out.count(a) != 1:
            raise SystemExit(f'harness edit does not apply exactly once: {a[:70]!r}')
        out = out.replace(a, b)
    return HEADER + out[len('#!/usr/bin/env python3\n'):]


def build(dest, pretrain=None):
    tmp = dest + '.tmp'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for f in sorted(os.listdir(FROZEN0)):
        s = os.path.join(FROZEN0, f)
        if f in ('lib', 'data', '__pycache__', 'pretrain.pt', 'harness.py'):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(tmp, f))
        else:
            shutil.copy2(s, os.path.join(tmp, f))
    src = open(os.path.join(FROZEN0, 'harness.py')).read()
    new = patch_harness(src)
    with open(os.path.join(tmp, 'harness.py'), 'w') as f:
        f.write(new)
    os.chmod(os.path.join(tmp, 'harness.py'), 0o775)
    shutil.copytree(LIB, os.path.join(tmp, 'lib'), ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    os.makedirs(os.path.join(tmp, 'data'))
    for f in sorted(os.listdir(os.path.join(FROZEN0, 'data'))):
        if f == 'SHA256SUMS':
            continue
        os.symlink(os.path.join(FROZEN0, 'data', f), os.path.join(tmp, 'data', f))
    if pretrain:
        shutil.copy2(pretrain, os.path.join(tmp, 'pretrain.pt'))
    diff = list(difflib.unified_diff(src.splitlines(), new.splitlines(), 'ab3/frozen/harness.py', 'harness.py', lineterm='', n=1))
    man = {'built_utc': utc(), 'from': FROZEN0, 'pretrain': os.path.abspath(pretrain) if pretrain else None,
           'pretrain_sha256': sha256_file(pretrain) if pretrain else None, 'harness_diff': diff,
           'sha256': {p: sha256_file(os.path.join(tmp, p)) for p in ('harness.py', 'meter.py', 'metrics.py', 'gpulock.py')},
           'lib_sha256': {f: sha256_file(os.path.join(tmp, 'lib', f)) for f in sorted(os.listdir(os.path.join(tmp, 'lib')))
                          if f.endswith('.py')}}
    write_json(os.path.join(tmp, 'REPR_MANIFEST.json'), man)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label')
    ap.add_argument('--pretrain', default=None)
    ap.add_argument('--dest', default=None)
    ap.add_argument('--template', action='store_true', help='no pretrain.pt (tests copy one in)')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    labels = [label(v, s) for v in ORDER for s in SEEDS] if a.all else [a.label]
    for lab in labels:
        pt = a.pretrain or os.path.join(PRETRAIN_DIR, lab, 'pretrain.pt')
        if a.all:
            sj = os.path.join(PRETRAIN_DIR, lab, 'summary.json')
            if not (os.path.isfile(sj) and json.load(open(sj)).get('status') == 'ok' and not json.load(open(sj)).get('smoke')):
                continue
        d = build(a.dest or os.path.join(FROZEN_DIR, lab), None if a.template else pt)
        print('built', d)


if __name__ == '__main__':
    main()
