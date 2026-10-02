"""Shared constants and helpers for the representation experiment (ab3/repr). No torch at import time.

Variants (README "Design"): every one uses the 039 pretraining recipe for PRETRAIN_STEPS fixed steps, the arm-C
RL recipe (rl_repr.py) and the lengthexp evaluation (evaluate.py here). Only the representation differs:
  abs      frozen format: abs tokenizer (N<i> line tokens, random index shift in training), RoPE
  rel      rel tokenizer (line numbers dropped, citation = B<lines back>), RoPE
  posrand  abs tokenizer, RoPE on a sorted random subset of range(1024) per training sequence (Ruoss et al.
           2023); sampling / evaluation on positions 0, 1, 2, ...
  nope     abs tokenizer, no positional encoding (Kazemnejad et al. 2023)
  posline  abs tokenizer, two-part rotary: 8 of 16 rotary pairs by line index, 8 by offset within the line
"""
import json, os, sys, time

REPR = os.path.dirname(os.path.abspath(__file__))
AB3 = '/scratch/cowifornia/nd-rl/ab3'
FROZEN0 = f'{AB3}/frozen'                 # the run-3 frozen harness (never modified)
LE = f'{AB3}/lengthexp'                   # the length experiment (read only)
LIB = f'{REPR}/lib'                       # variant library (frozen/lib + positional variants)
PY = '/home/cowifornia/code/research/nd-rl/code/.venv/bin/python'
DATA = f'{FROZEN0}/data'
TRAIN_CAP6 = f'{DATA}/train_cap6.jsonl'   # = the 039 fork's data/train.jsonl (sha256 9c024cd9...)
HELDOUT_CAP6 = f'{DATA}/heldout_cap6.jsonl'
RL_C = f'{LE}/rl_C.py'                    # arm C: seed rl.py with max_len_train = max_proof_lines_train = 11
REF039 = '/scratch/cowifornia/nd-rl/checkpoints/runs/039_muon-adamw-finish-5pct_s{seed}'

PRETRAIN_DIR = f'{REPR}/pretrain'
FROZEN_DIR = f'{REPR}/frozen'
RUNS = f'{REPR}/runs'
EVAL = f'{REPR}/eval'
REPORT = f'{REPR}/report'
LOGS = f'{REPR}/logs'

PRETRAIN_STEPS = 2911                     # 039_s0's step count (300 s on an A40); fixed for every variant and seed
SEEDS = (0, 1, 2)                         # pretraining seeds (= the released 039 seeds)
RL_SEED = {0: 101, 1: 102, 2: 103}        # RL seed paired with each pretraining seed (arm C's seeds)
TRAIN_MAX = 11                            # arm C: max_len_train = max_proof_lines_train = 11

VARIANTS = {
    'abs': dict(tok='abs', model={}),
    'rel': dict(tok='rel', model={}),
    'posrand': dict(tok='abs', model={'pos_train': 'rand', 'pos_range': 1024}),
    'nope': dict(tok='abs', model={'pe': 'none'}),
    'posline': dict(tok='abs', model={'pe': 'line', 'line_pairs': 8, 'seps': [';', 'PRF']}),
}
ORDER = ('abs', 'rel', 'posrand', 'nope', 'posline')


def lib_path():
    """Put repr/lib first on sys.path (lengthexp/common.py and ndcheck.py insert frozen/lib) and check that the
    model / tokenizer / sample modules already imported, if any, are repr/lib's."""
    while LIB in sys.path:
        sys.path.remove(LIB)
    sys.path.insert(0, LIB)
    for name in ('model', 'sample', 'tokenizer', 'posenc'):
        m = sys.modules.get(name)
        if m is not None and not os.path.abspath(m.__file__).startswith(LIB + os.sep):
            raise ImportError(f'{name} was imported from {m.__file__}, not {LIB}')


def build_model(variant, device='cpu'):
    """(model, tok) for a fresh variant model (torch's global RNG draws the init, as in 039)."""
    lib_path()
    from model import GPT
    from tokenizer import Tokenizer
    v = VARIANTS[variant]
    tok = Tokenizer(v['tok'])
    kw = dict(v['model'])
    if 'seps' in kw:
        kw['seps'] = [tok.stoi[s] for s in kw['seps']]
    return GPT(tok.vocab_size, 4, 256, 8, **kw).to(device), tok


def load_model(path, device='cpu'):
    """(model, tok, ckpt dict) from a checkpoint in lib.model.save_ckpt / harness layout (cfg optional)."""
    lib_path()
    import torch
    from model import GPT
    from tokenizer import Tokenizer
    ck = torch.load(path, map_location='cpu', weights_only=True)
    tok = Tokenizer(ck.get('tok_mode') or 'abs')
    cfg = dict(ck.get('cfg') or dict(vocab=tok.vocab_size, n_layer=4, d=256, n_head=8))
    model = GPT(**cfg)
    state = {k.removeprefix('_orig_mod.').removeprefix('module.'): v for k, v in ck['state'].items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), tok, ck


def label(variant, seed):
    return f'{variant}_s{seed}'


def parse_label(lab):
    v, _, s = lab.rpartition('_s')
    return (v, int(s)) if v in VARIANTS and s.isdigit() else (None, None)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read_jsonl(fn):
    import gzip
    op = gzip.open if fn.endswith('.gz') else open
    with op(fn, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(fn, obj):
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, fn)


def sha256_file(fn):
    import hashlib
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Lease:
    """--gpu auto / N: an exclusive frozen/gpulock.py lease (FIFO queue), exported as CUDA_VISIBLE_DEVICES.
    --gpu env: use CUDA_VISIBLE_DEVICES as it is. Must be taken before CUDA initialises."""

    def __init__(self, run):
        self.run, self.gpu, self.holder = run, None, None

    def take(self, gpu='auto', gpus='0,1,2,3,4', wait=12 * 3600):
        if gpu == 'env':
            return
        if FROZEN0 not in sys.path:
            sys.path.insert(0, FROZEN0)
        import atexit, gpulock
        cand = gpulock.parse_gpus(gpus) if gpu == 'auto' else [int(gpu)]
        t0 = time.time()
        g, h = gpulock.acquire(cand, wait=wait, run=self.run, owner=os.getpid(), log=log)
        if g is None:
            raise SystemExit(f'no free GPU among {cand} after {wait:.0f}s')
        self.gpu, self.holder = g, h
        atexit.register(self.drop)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(g)
        log(f'gpulock: holding GPU {g} (holder {h}) after {time.time() - t0:.0f}s')

    def drop(self):
        if self.gpu is None:
            return
        import gpulock
        g, self.gpu = self.gpu, None
        ok = gpulock.release(g, self.holder)
        log(f'gpulock: released GPU {g}' + ('' if ok else ' (lock still busy?)'))
