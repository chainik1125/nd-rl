#!/usr/bin/env python3
"""Build the frozen-copy harness of one goals seed: goals/frozen/goals_s<seed>/ (the pattern of ab3/frozen_039 and
repr/make_frozen.py: ab3/frozen with pretrain.pt replaced), or with --template one without pretrain.pt (tests).

    python make_frozen.py --seed 0 [--pretrain pretrain/goals_s0/pretrain.pt]
    python make_frozen.py --all                     # every seed whose pretrain summary is ok

Contents: ab3/frozen's files unchanged (meter.py, metrics.py, gpulock.py, score*.py, SPEC.md, test_*.py, ref/);
lib/ = ab3/frozen/lib unchanged plus goals.py and gtokenizer.py (goals/lib); data/ = symlinks to ab3/frozen/data/*
plus train_cap6_goals.jsonl -> goals/data/train_cap6_goals.jsonl; pretrain.pt = the seed's goals checkpoint; and
harness.py = ab3/frozen/harness.py with the edits of EDITS (DESIGN.md 5):
  1. imports: the goals tokenizer factory and the annotation library
  2. TRAIN_CAP6_FILE = the annotated cap-6 file (the retention records rl.py mixes in)
  3. held-out greedy check: annotations stripped before verification, max_new 768 (frozen 512)
  4. load_pretrained: tokenizer from pretrain.pt's tok_mode (abs_goals), GPT(tok.vocab_size, 4, 256, 8)
Each edit must apply exactly once; the diff and sha256s go into <dir>/GOALS_MANIFEST.json.
"""
import argparse, difflib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_goals import DATA, FROZEN0, FROZEN_DIR, GLIB, PRETRAIN_DIR, SEEDS, label, sha256_file, utc, write_json

EDITS = [
    ('from tokenizer import Tokenizer  # noqa: E402\n',
     'from tokenizer import Tokenizer  # noqa: E402\n'
     'from gtokenizer import make_tokenizer  # noqa: E402  (goals: abs + GOAL tokenizer)\n'
     'import goals as _goals  # noqa: E402  (goals: annotation strip)\n'),
    ("TRAIN_CAP6_FILE = 'train_cap6.jsonl'     # cap-6 supervised records (retention)\n",
     "TRAIN_CAP6_FILE = 'train_cap6_goals.jsonl'  # goals: the cap-6 records with canonical goal annotations (retention)\n"),
    ('GREEDY_BATCH, GREEDY_MAX_NEW = 384, 512\n',
     'GREEDY_BATCH, GREEDY_MAX_NEW = 384, 768   # goals: 768 (annotated proofs are longer; frozen 512)\n'),
    ("    ok = sum(verify_text(r['prompt'] + ' ' + p)[0] for r, p in zip(recs, outs))\n",
     "    ok = sum(verify_text(r['prompt'] + ' ' + _goals.strip(p))[0] for r, p in zip(recs, outs))   # goals: stripped\n"),
    ('''def load_pretrained(device):
    tok = Tokenizer('abs')
    ck = torch.load(os.path.join(FROZEN, 'pretrain.pt'), map_location='cpu')
    assert ck.get('tok_mode', 'abs') == 'abs', ck.get('tok_mode')
    model = GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD)
''', '''def load_pretrained(device):
    # goals edit: the tokenizer from the checkpoint's tok_mode (abs_goals: abs + GOAL, vocab 100)
    ck = torch.load(os.path.join(FROZEN, 'pretrain.pt'), map_location='cpu')
    tok = make_tokenizer(ck.get('tok_mode') or 'abs')
    assert tok.mode == 'abs_goals', tok.mode
    model = GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD)
'''),
]
HEADER = ('#!/usr/bin/env python3\n# GOALS COPY (ab3/goals/make_frozen.py): ab3/frozen/harness.py with five small edits (goals tokenizer from\n'
          '# pretrain.pt, annotated retention file, stripped held-out check with max_new 768). See GOALS_MANIFEST.json.\n')
LIB_EXTRA = ('goals.py', 'gtokenizer.py')


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
    shutil.copytree(os.path.join(FROZEN0, 'lib'), os.path.join(tmp, 'lib'), ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for f in LIB_EXTRA:
        shutil.copy2(os.path.join(GLIB, f), os.path.join(tmp, 'lib', f))
    # goals.py puts ab3/frozen/lib on sys.path for nd_verify; the copy's own lib (same files) comes first anyway
    os.makedirs(os.path.join(tmp, 'data'))
    for f in sorted(os.listdir(os.path.join(FROZEN0, 'data'))):
        if f == 'SHA256SUMS':
            continue
        os.symlink(os.path.join(FROZEN0, 'data', f), os.path.join(tmp, 'data', f))
    os.symlink(os.path.join(DATA, 'train_cap6_goals.jsonl'), os.path.join(tmp, 'data', 'train_cap6_goals.jsonl'))
    if pretrain:
        shutil.copy2(pretrain, os.path.join(tmp, 'pretrain.pt'))
    diff = list(difflib.unified_diff(src.splitlines(), new.splitlines(), 'ab3/frozen/harness.py', 'harness.py', lineterm='', n=1))
    man = {'built_utc': utc(), 'from': FROZEN0, 'pretrain': os.path.abspath(pretrain) if pretrain else None,
           'pretrain_sha256': sha256_file(pretrain) if pretrain else None, 'harness_diff': diff,
           'sha256': {p: sha256_file(os.path.join(tmp, p)) for p in ('harness.py', 'meter.py', 'metrics.py', 'gpulock.py')},
           'frozen_harness_sha256': sha256_file(os.path.join(FROZEN0, 'harness.py')),
           'lib_sha256': {f: sha256_file(os.path.join(tmp, 'lib', f)) for f in sorted(os.listdir(os.path.join(tmp, 'lib')))
                          if f.endswith('.py')},
           'train_cap6_goals_sha256': sha256_file(os.path.join(DATA, 'train_cap6_goals.jsonl'))}
    write_json(os.path.join(tmp, 'GOALS_MANIFEST.json'), man)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int)
    ap.add_argument('--pretrain', default=None)
    ap.add_argument('--dest', default=None)
    ap.add_argument('--template', action='store_true', help='no pretrain.pt (tests copy one in)')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    seeds = SEEDS if a.all else [a.seed]
    for s in seeds:
        lab = label(s)
        pt = a.pretrain or os.path.join(PRETRAIN_DIR, lab, 'pretrain.pt')
        if a.all:
            sj = os.path.join(PRETRAIN_DIR, lab, 'summary.json')
            if not (os.path.isfile(sj) and json.load(open(sj)).get('status') == 'ok' and not json.load(open(sj)).get('smoke')):
                continue
        d = build(a.dest or os.path.join(FROZEN_DIR, lab), None if a.template else pt)
        print('built', d)


if __name__ == '__main__':
    main()
