"""Shared constants and helpers for the length-generalisation experiment (ab3/lengthexp).

No torch here: build_testsets.py, analyse.py and launch.py import this without a GPU stack.
"""
import gzip, hashlib, itertools, json, math, os, sys

AB2 = '/scratch/cowifornia/nd-rl/ab2'
AB3 = '/scratch/cowifornia/nd-rl/ab3'
FROZEN = f'{AB3}/frozen'
LE = os.path.dirname(os.path.abspath(__file__))           # .../ab3/lengthexp
PY = '/home/cowifornia/code/research/nd-rl/code/.venv/bin/python'

TESTSETS = f'{LE}/testsets'
EVAL = f'{LE}/eval'
RUNS = f'{LE}/runs'
REPORT = f'{LE}/report'
PRETRAIN = f'{FROZEN}/pretrain.pt'
SEED_RL = f'{AB3}/seed/rl.py'

# ---- the design (fixed by the monitor; change only after asking)
ARMS = {'A': 7, 'B': 9, 'C': 11}          # arm -> max_len_train (true proof length of the RL targets, 6..max)
PRETRAIN_MAX = 6                           # pretraining proofs have at most 6 lines
SEEDS = (101, 102, 103)
LENGTHS = tuple(range(6, 19))              # evaluated test lengths
K = 32                                     # samples per theorem
TEMPERATURE = 0.8                          # the scorer's T
MAX_NEW = 1024                             # 18 lines x up to ~43 tokens/line (+ slack); see README
PER_LEN = 300                              # theorems per length (at most)
DRAW_SEED = 20260925                       # test-set draws
TEXTBOOK_SHARE = 0.4                       # same rule as ab2's assemble.py draw(): textbook share <= 40%
SAMPLE_SEED = 0                            # base seed of the evaluation sampler

# ---- test-set sources: (set, file, lengths it may supply, how to pick records)
# 'std' is the main curve; 'matched' is the construction-style control (evaluated separately when present).
SOURCES = [
    dict(set='std', file=f'{AB2}/data/reserve.jsonl', lengths=(6, 7), origin_prefix='ab2/'),
    dict(set='std', file=f'{AB2}/data/dev_L8-11.jsonl', lengths=(8, 9, 10, 11)),
    dict(set='std', file=f'{AB2}/sealed/sealed_L12-14.jsonl', lengths=(12, 13, 14)),
    dict(set='std', file=f'{AB3}/sealed/sealed_L15-16.jsonl', lengths=(15, 16)),
    dict(set='std', file=f'{AB3}/sealed/sealed_L17-18.jsonl', lengths=(17, 18)),
    dict(set='matched', file=f'{AB3}/sealed/matched_L6-14.jsonl', lengths=tuple(range(6, 15))),
]
# theorems the test sets must avoid (pkey classes): the RL pool and the pretraining data
TRAIN_FILES = {
    'targets_ab2': f'{AB2}/data/targets_L6-11.jsonl',
    'targets_ab3_frozen': f'{FROZEN}/data/targets_L6-11.jsonl',
    'train_cap6': f'{FROZEN}/data/train_cap6.jsonl',
    'heldout_cap6': f'{FROZEN}/data/heldout_cap6.jsonl',
    'targets_comp': f'{AB3}/data_comp/targets_L6-11.jsonl',    # arm D's RL pool (standard + constructed)
}

if FROZEN + '/lib' not in sys.path:
    sys.path.insert(0, FROZEN + '/lib')
from keys import canon_key  # noqa: E402  (frozen/lib/keys.py, stdlib only)


def read_jsonl(fn):
    op = gzip.open if fn.endswith('.gz') else open
    with op(fn, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]


def iter_jsonl(fn):
    with open(fn) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def write_jsonl(fn, rows):
    tmp = fn + '.tmp'
    op = gzip.open if fn.endswith('.gz') else open
    with op(tmp, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, fn)


def write_json(fn, obj):
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, fn)


def sha256_file(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def split_thm(thm):
    lhs, rhs = thm.strip().split('|-')
    prem = [p.strip() for p in lhs.split(' , ')] if lhs.strip() else []
    return prem, rhs.strip()


def pkey(thm):
    """Premise-order-invariant atom-renaming class (same as ab2/data/tools/keys2.py)."""
    prem, concl = split_thm(thm)
    best = None
    for perm in itertools.permutations(prem):
        k = canon_key((' , '.join(perm) + ' |- ' + concl).strip())
        if best is None or k < best:
            best = k
    return best


def thm_of(rec):
    """'thm' field, or rebuilt from the prompt ('THM a , b SEQ c PRF')."""
    if rec.get('thm'):
        return rec['thm']
    t = rec['prompt'].split()
    return ' '.join(t[1:-1]).replace(' SEQ ', ' |- ').replace('SEQ ', '|- ')


def unit_name(set_name, L):
    return f'{set_name}_L{L}'


def pass_at_k(n, c, k):
    """Unbiased pass@k from c correct of n samples (Chen et al. 2021)."""
    if k > n:
        return float('nan')
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def arm_of_label(label):
    """'pretrain' -> ('pretrain', None); 'A_s101' -> ('A', 101); anything else -> (label, None)."""
    if label == 'pretrain':
        return 'pretrain', None
    a, _, s = label.partition('_s')
    if a in ARMS and s.isdigit():
        return a, int(s)
    return label, None


def train_max(arm):
    return PRETRAIN_MAX if arm == 'pretrain' else ARMS.get(arm)
