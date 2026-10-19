"""Shared constants and helpers for the goals variant (ab3/goals). No torch at import time.

The variant: the abs representation of the repr/ baseline (frozen GPT, RoPE, abs tokenizer) plus self-written goal
annotations (lib/goals.py, lib/gtokenizer.py; DESIGN.md). Pipeline constants are repr/'s (common_repr.py):
PRETRAIN_STEPS 2911, pretraining seeds 0-2 paired with RL seeds 101-103, arm-C RL, the lengthexp evaluation.
"""
import hashlib, json, os, sys, time

GOALS = os.path.dirname(os.path.abspath(__file__))
AB3 = '/scratch/cowifornia/nd-rl/ab3'
FROZEN0 = f'{AB3}/frozen'                  # the run-3 frozen harness (never modified)
FROZEN_LIB = f'{FROZEN0}/lib'
LE = f'{AB3}/lengthexp'
REPR = f'{AB3}/repr'
GLIB = f'{GOALS}/lib'
PY = '/home/cowifornia/code/research/nd-rl/code/.venv/bin/python'
DATA = f'{GOALS}/data'
TRAIN_CAP6 = f'{DATA}/train_cap6_goals.jsonl'      # annotated frozen/data/train_cap6.jsonl (same order)
HELDOUT_CAP6 = f'{FROZEN0}/data/heldout_cap6.jsonl'
RL_C = f'{LE}/rl_C.py'

PRETRAIN_DIR = f'{GOALS}/pretrain'
FROZEN_DIR = f'{GOALS}/frozen'
RUNS = f'{GOALS}/runs'
EVAL = f'{GOALS}/eval'
REPORT = f'{GOALS}/report'
LOGS = f'{GOALS}/logs'

PRETRAIN_STEPS = 2911                      # = repr/common_repr.py PRETRAIN_STEPS (039_s0's step count)
SEEDS = (0, 1, 2)
RL_SEED = {0: 101, 1: 102, 2: 103}         # = repr's pairing
TRAIN_MAX = 11
GREEDY_MAX_NEW = 768                       # held-out greedy checks (DESIGN.md 4-5; frozen: 512)
N_LAYER, D_MODEL, N_HEAD = 4, 256, 8


def lib_path():
    for p in (FROZEN_LIB, GLIB):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, FROZEN_LIB)
    sys.path.insert(0, GLIB)


def build_model(device='cpu'):
    lib_path()
    from model import GPT
    from gtokenizer import GoalTokenizer
    tok = GoalTokenizer()
    return GPT(tok.vocab_size, N_LAYER, D_MODEL, N_HEAD).to(device), tok


def load_model(path, device='cpu'):
    """(model, tok, ckpt) from a checkpoint in lib.model.save_ckpt / harness layout."""
    lib_path()
    import torch
    from model import GPT
    from gtokenizer import make_tokenizer
    ck = torch.load(path, map_location='cpu', weights_only=True)
    tok = make_tokenizer(ck.get('tok_mode') or 'abs')
    cfg = dict(ck.get('cfg') or dict(vocab=tok.vocab_size, n_layer=4, d=256, n_head=8))
    model = GPT(**cfg)
    state = {k.removeprefix('_orig_mod.').removeprefix('module.'): v for k, v in ck['state'].items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), tok, ck


def label(seed):
    return f'goals_s{seed}'


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
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Lease:
    """--gpu auto / N: an exclusive frozen/gpulock.py lease (FIFO queue), exported as CUDA_VISIBLE_DEVICES.
    --gpu env: use CUDA_VISIBLE_DEVICES as it is. Must be taken before CUDA initialises. (repr/common_repr.py's.)"""

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
