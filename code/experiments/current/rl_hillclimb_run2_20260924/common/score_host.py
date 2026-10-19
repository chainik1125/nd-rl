#!/usr/bin/env python3
"""HOST-SIDE scorer for the RL hill-climb, run 2 (ab2). Run by the monitor or by score_daemon.py only.

  python score_host.py --ckpt <path> --set dev|frontier|sealed_L12-14|sealed_L15-16|textbook [--k K]
                       [--seed 0] [--temperature 0.8] [--smoke] [--out result.json] [--per-theorem rows.jsonl]

Sets (file locations are the constants below; AB2_POOL_DIR / AB2_SEALED_DIR / AB2_TEXTBOOK override
them for tests):
  dev            dev_L8-11.jsonl       the metric (agents see the aggregate counts; daemon-served)
  frontier       frontier_L12-14.jsonl the frontier count (agents see the aggregate counts; daemon-served)
  sealed_L12-14  sealed_L12-14.jsonl   monitor only, never served by the daemon
  sealed_L15-16  sealed_L15-16.jsonl   monitor only
  textbook       ab/sealed/textbook_dev.jsonl  monitor only
k is limited to 1..256 (default: 32 for dev, 64 otherwise). The official numbers are dev k = 32 and frontier
k = 64, seed 0, temperature 0.8, not --smoke. max_new is 640 for frontier and the sealed L12-16 sets (long
proofs), 512 for dev and textbook.

Samples k proofs per sealed theorem with lib.sample.generate (temperature 0.8, max_new per set (MAX_NEW_BY_SET), batch 384,
fixed seed), verifies each with nd_verify.verify_text(prompt + ' ' + proof); a theorem is solved if any
sample verifies. Prints one aggregate line on stderr and the aggregate JSON on stdout (and to --out):

  {set, n, solved, by_len: {"7": [solved, n], ...}, k, seed, temperature, ckpt, ckpt_sha256,
   set_sha256, cfg, arch_standard, secs, sample_secs, samples_per_sec, status}

Only aggregate counts leave this process by default. `--per-theorem` writes {name, L_true, solved, n_ok}
per theorem; it is for the MONITOR'S OWN final analysis only. score_daemon.py never passes it, and it
must never be used for a request coming from an arm or written anywhere under an arm directory.

Checkpoints are loaded with torch.load(weights_only=True): they come from agent-controlled containers,
so arbitrary pickles are refused. A 'cfg' key is used if present, otherwise GPT(vocab, 4, 256, 8).
--smoke uses the first 16 theorems of the set.
"""
import argparse, collections, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import torch  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402
from sample import generate  # noqa: E402
from nd_verify import verify_text  # noqa: E402

# ---- set files. Change only here (or with the env overrides, for tests).
AB2 = '/scratch/cowifornia/nd-rl/ab2'
POOL_DIR = os.environ.get('AB2_POOL_DIR', f'{AB2}/data')          # dev / frontier pools (scorer-only files)
SEALED_DIR = os.environ.get('AB2_SEALED_DIR', f'{AB2}/sealed')    # monitor-only sealed sets
TEXTBOOK = os.environ.get('AB2_TEXTBOOK', '/scratch/cowifornia/nd-rl/ab/sealed/textbook_dev.jsonl')
DEV_FILE, FRONTIER_FILE = 'dev_L8-11.jsonl', 'frontier_L12-14.jsonl'
SETS = {'dev': [os.path.join(POOL_DIR, DEV_FILE)],
        'frontier': [os.path.join(POOL_DIR, FRONTIER_FILE)],
        # the pool builder may leave the sealed files in data/ until they are moved to sealed/
        'sealed_L12-14': [os.path.join(SEALED_DIR, 'sealed_L12-14.jsonl'), os.path.join(POOL_DIR, 'sealed_L12-14.jsonl')],
        'sealed_L15-16': [os.path.join(SEALED_DIR, 'sealed_L15-16.jsonl'), os.path.join(POOL_DIR, 'sealed_L15-16.jsonl')],
        'textbook': [TEXTBOOK]}
DAEMON_SETS = ('dev', 'frontier')     # the only sets score_daemon.py may request
OFFICIAL_K = {'dev': 32, 'frontier': 64}   # official k per daemon set; also the default --k
OFFICIAL_SEED, OFFICIAL_T = 0, 0.8
DEFAULT_K = 64                              # default --k for the monitor-only sets
MAX_K = 256
STANDARD_ARCH = dict(n_layer=4, d=256, n_head=8)
BATCH, SMOKE_N = 384, 16
MAX_NEW_BY_SET = {'dev': 512, 'textbook': 512, 'frontier': 512, 'sealed_L12-14': 640, 'sealed_L15-16': 640}
# frontier is scored on a fixed stratified subset (per L_true), drawn once with FRONTIER_SUBSET_SEED, to keep
# per-experiment scoring near 4 minutes; the full set is scored only by the monitor via sealed_L12-14 / --set frontier --full
FRONTIER_SUBSET = {'12': 700, '13': 200, '14': 100}
FRONTIER_SUBSET_SEED = 20260924
# --per-theorem output may not land where an agent works: all of ab2 except these monitor-only dirs, and run-1 arms
PER_THEOREM_OK = [os.path.realpath(f'{AB2}/{d}') for d in ('sealed', 'monitor')]
ARM_DIRS = [os.path.realpath(AB2)] + [os.path.realpath(f'/scratch/cowifornia/nd-rl/ab/{d}') for d in ('gear', 'orch', 'homes', 'dash')]


def set_path(name):
    for p in SETS[name]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'set {name}: none of {SETS[name]} exists')


def under(path, d):
    return path == d or path.startswith(d + os.sep)


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ck, dict) or 'state' not in ck:
        raise ValueError('checkpoint is not a dict with a "state" key')
    tok = Tokenizer(ck.get('tok_mode', 'abs'))
    cfg = ck.get('cfg')
    if cfg:
        cfg = dict(cfg)
        cfg['vocab'] = cfg.get('vocab', tok.vocab_size)
        if cfg['vocab'] != tok.vocab_size:
            raise ValueError(f'cfg vocab {cfg["vocab"]} != tokenizer vocab {tok.vocab_size}')
    else:
        cfg = dict(vocab=tok.vocab_size, **STANDARD_ARCH)
    model = GPT(**cfg)
    state = {k.removeprefix('_orig_mod.').removeprefix('module.'): v for k, v in ck['state'].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    arch_ok = all(cfg.get(k) == v for k, v in STANDARD_ARCH.items())
    return model, tok, cfg, arch_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--set', required=True, choices=sorted(SETS))
    ap.add_argument('--k', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--full', action='store_true', help='frontier: score the full set instead of the fixed subset (monitor use)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--per-theorem', default=None,
                    help='MONITOR ONLY: per-theorem rows (name, L_true, solved, n_ok). Never from an arm.')
    a = ap.parse_args()
    if a.k is None:
        a.k = OFFICIAL_K.get(a.set, DEFAULT_K)
    max_new = MAX_NEW_BY_SET[a.set]
    if not (1 <= a.k <= MAX_K):
        ap.error(f'--k must be in 1..{MAX_K}')
    if a.per_theorem:
        rp = os.path.realpath(a.per_theorem)
        if any(under(rp, arm) for arm in ARM_DIRS) and not any(under(rp, ok) for ok in PER_THEOREM_OK):
            ap.error('--per-theorem output may not be written under an arm directory')

    t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    set_file = set_path(a.set)
    recs = [json.loads(l) for l in open(set_file) if l.strip()]
    if a.set == 'frontier' and not a.full:
        import random as _random
        by = {}
        for r in recs:
            by.setdefault(str(r.get('L_true', r.get('n_lines'))), []).append(r)
        rng = _random.Random(FRONTIER_SUBSET_SEED)
        sub = []
        for L, n in FRONTIER_SUBSET.items():
            pool = sorted(by.get(L, []), key=lambda r: r['name'])
            rng.shuffle(pool)
            sub += pool[:n]
        recs = sub
    if a.smoke:
        recs = recs[:SMOKE_N]
    model, tok, cfg, arch_ok = load_model(a.ckpt, dev)

    prompts = [r['prompt'] for r in recs for _ in range(a.k)]
    ts = time.time()
    flat = generate(model, tok, prompts, greedy=False, temperature=a.temperature, batch=BATCH,
                    seed=a.seed, max_new=max_new)
    sample_secs = time.time() - ts

    by = collections.defaultdict(lambda: [0, 0])
    rows = []
    for i, r in enumerate(recs):
        n_ok = 0
        for p in flat[i * a.k:(i + 1) * a.k]:
            ok, _, _ = verify_text(r['prompt'] + ' ' + p)
            n_ok += bool(ok)
        L = str(r['L_true'])
        by[L][0] += n_ok > 0
        by[L][1] += 1
        if a.per_theorem:
            rows.append({'name': r['name'], 'L_true': r['L_true'], 'solved': n_ok > 0, 'n_ok': n_ok})
    solved = sum(v[0] for v in by.values())
    secs = time.time() - t0
    out = {'set': a.set, 'n': len(recs), 'solved': solved,
           'by_len': {L: by[L] for L in sorted(by, key=int)},
           'k': a.k, 'seed': a.seed, 'temperature': a.temperature, 'smoke': a.smoke,
           'official': (a.set in DAEMON_SETS and a.k == OFFICIAL_K[a.set] and a.seed == OFFICIAL_SEED
                        and a.temperature == OFFICIAL_T and not a.smoke),
           'ckpt': os.path.abspath(a.ckpt), 'ckpt_sha256': sha256(a.ckpt), 'set_sha256': sha256(set_file), 'set_file': os.path.basename(set_file),
           'cfg': cfg, 'arch_standard': arch_ok, 'max_new': max_new, 'batch': BATCH,
           'secs': round(secs, 2), 'sample_secs': round(sample_secs, 2),
           'samples_per_sec': round(len(prompts) / max(sample_secs, 1e-9), 1),
           'gpu_name': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', 'status': 'ok'}
    lens = ' '.join(f'L{L} {s}/{n}' for L, (s, n) in out['by_len'].items())
    print(f'{a.set} pass@{a.k}: solved {solved}/{len(recs)} | {lens} | secs {secs:.0f}', file=sys.stderr, flush=True)
    if a.out:
        tmp = a.out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, a.out)
    if a.per_theorem:
        with open(a.per_theorem, 'w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
