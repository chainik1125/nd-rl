#!/usr/bin/env python3
"""HOST-SIDE sealed scorer for the RL hill-climb A/B. Never copied into an arm, never run in a container.

  python score_host.py --ckpt <path> --set dev|test10|test11|textbook [--k 32] [--seed 0]
                       [--temperature 0.8] [--smoke] [--out result.json] [--per-theorem rows.jsonl]

Samples k proofs per sealed theorem with lib.sample.generate (temperature 0.8, max_new 512, batch 384,
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

SEALED = '/scratch/cowifornia/nd-rl/ab/sealed'
SETS = {'dev': 'dev_L7-9.jsonl', 'test10': 'test_L10.jsonl', 'test11': 'test_L11-14.jsonl',
        'textbook': 'textbook_dev.jsonl'}
STANDARD_ARCH = dict(n_layer=4, d=256, n_head=8)
MAX_NEW, BATCH, SMOKE_N = 512, 384, 16
ARM_DIRS = [os.path.realpath(f'/scratch/cowifornia/nd-rl/ab/{d}') for d in ('gear', 'orch', 'homes', 'dash')]


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
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--per-theorem', default=None,
                    help='MONITOR ONLY: per-theorem rows (name, L_true, solved, n_ok). Never from an arm.')
    a = ap.parse_args()
    if a.k < 1:
        ap.error('--k must be >= 1')
    if a.per_theorem:
        rp = os.path.realpath(a.per_theorem)
        for arm in ARM_DIRS:
            if rp == arm or rp.startswith(arm + os.sep):
                ap.error('--per-theorem output may not be written under an arm directory')

    t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    set_path = os.path.join(SEALED, SETS[a.set])
    recs = [json.loads(l) for l in open(set_path) if l.strip()]
    if a.smoke:
        recs = recs[:SMOKE_N]
    model, tok, cfg, arch_ok = load_model(a.ckpt, dev)

    prompts = [r['prompt'] for r in recs for _ in range(a.k)]
    ts = time.time()
    flat = generate(model, tok, prompts, greedy=False, temperature=a.temperature, batch=BATCH,
                    seed=a.seed, max_new=MAX_NEW)
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
           'official': a.set == 'dev' and a.k == 32 and a.seed == 0 and a.temperature == 0.8 and not a.smoke,
           'ckpt': os.path.abspath(a.ckpt), 'ckpt_sha256': sha256(a.ckpt), 'set_sha256': sha256(set_path),
           'cfg': cfg, 'arch_standard': arch_ok, 'max_new': MAX_NEW, 'batch': BATCH,
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
