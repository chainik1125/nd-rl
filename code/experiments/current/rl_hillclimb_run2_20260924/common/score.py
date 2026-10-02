#!/usr/bin/env python3
"""FROZEN score CLIENT (run 2). The real scorer runs host-side (score_daemon.py -> score_host.py).

    python frozen/score.py --ckpt runs/<name>/final.pt [--set dev|frontier] [--k 32|64] [--seed 0]
                           [--tag free-text] [--smoke]

Sets: `dev` (default; dev_L8-11, the metric) and `frontier` (frontier_L12-14, the frontier count).
Protocol (SPEC.md "Scoring"):
  arm root = the nearest ancestor of the checkpoint (or of the cwd) that contains `frozen/`.
  1. write  <arm root>/score_requests/<id>.json  {id, set, ckpt, k, seed, gpu, cwd, tag, smoke, t}
            (ckpt is the absolute host path; gpu is CUDA_VISIBLE_DEVICES, a host GPU index)
  2. poll   <arm root>/score_results/<id>.json   every 3 s, timeout 55 min
  3. print  `<set> pass@<k>: solved T/N | L8 a/n8 L9 b/n9 ... | secs S`
  4. write  score.json (dev) or score_<set>.json next to the checkpoint; append the result to
            <arm root>/logs/scores.jsonl
Exit status is non-zero on a result with status "error" or on timeout.
This file holds no evaluation data and does not verify anything itself.
"""
import argparse, hashlib, json, os, sys, time, uuid

SETS = ('dev', 'frontier')      # the only sets the daemon serves
OFFICIAL_K = {'dev': 32, 'frontier': 64}   # official k per set (default --k)
OFFICIAL_SEED = 0
POLL_SECS = 3
TIMEOUT_SECS = 55 * 60


def arm_root_for(path):
    d = os.path.abspath(path)
    if not os.path.isdir(d):
        d = os.path.dirname(d)
    while True:
        if os.path.isdir(os.path.join(d, 'frozen')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def file_sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def by_len_counts(res):
    """Normalise the result's by_len into ({'8': solved, ...}, {'8': n, ...}), keys sorted by length.
    Accepts {'8': int} (denominators from res['n_by_len']), {'8': [solved, n]} or {'8': {'solved', 'n'}}."""
    solved, n = {}, {str(k): int(v) for k, v in (res.get('n_by_len') or {}).items()}
    for L, v in (res.get('by_len') or {}).items():
        L = str(L)
        if isinstance(v, dict):
            solved[L] = int(v.get('solved', 0))
            if 'n' in v:
                n[L] = int(v['n'])
        elif isinstance(v, (list, tuple)):
            solved[L], n[L] = int(v[0]), int(v[1])
        else:
            solved[L] = int(v)
    for L in n:
        solved.setdefault(L, 0)
    key = lambda L: (int(L) if L.isdigit() else 10 ** 9, L)
    return {L: solved[L] for L in sorted(solved, key=key)}, {L: n.get(L, 0) for L in sorted(solved, key=key)}


def stub_result(req, ckpt):
    """--local-stub: a fake all-zero result so harness plumbing can be tested without the daemon."""
    n = ({'8': 4, '9': 4, '10': 4, '11': 4} if req['set'] == 'dev' else {'12': 6, '13': 5, '14': 5}) if req['smoke'] \
        else ({'8': 100, '9': 100, '10': 100, '11': 100} if req['set'] == 'dev' else {'12': 100, '13': 100, '14': 100})
    return {'id': req['id'], 'status': 'ok', 'stub': True, 'ckpt': req['ckpt'], 'ckpt_sha256': file_sha256(ckpt),
            'k': req['k'], 'seed': req['seed'], 'smoke': req['smoke'], 'solved': 0,
            'by_len': {L: 0 for L in n}, 'n_by_len': n, 'n': sum(n.values()), 'secs': 0.0, 'samples_per_sec': 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--set', default='dev', choices=SETS, help='dev (default; the metric) or frontier (L12-14 count)')
    ap.add_argument('--k', type=int, default=None, help='samples per theorem (default: the official k, dev 32 / frontier 64)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--smoke', action='store_true', help='score a 16-theorem subset of the set (plumbing check)')
    ap.add_argument('--timeout', type=float, default=TIMEOUT_SECS, help=argparse.SUPPRESS)
    ap.add_argument('--local-stub', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.k is None:
        a.k = OFFICIAL_K[a.set]

    ckpt = os.path.abspath(a.ckpt)
    if not os.path.isfile(ckpt):
        sys.exit(f'score.py: no checkpoint at {ckpt}')
    root = arm_root_for(ckpt) or arm_root_for(os.getcwd())
    if root is None:
        sys.exit(f'score.py: could not find the arm root (a directory containing frozen/) above {ckpt}')
    req_dir, res_dir, log_dir = (os.path.join(root, d) for d in ('score_requests', 'score_results', 'logs'))
    for d in (req_dir, res_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    rid = uuid.uuid4().hex
    req = {'id': rid, 'set': a.set, 'ckpt': ckpt, 'k': a.k, 'seed': a.seed, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
           'cwd': os.getcwd(), 'tag': a.tag, 'smoke': bool(a.smoke), 't': time.time()}
    res_path = os.path.join(res_dir, f'{rid}.json')
    write_json_atomic(os.path.join(req_dir, f'{rid}.json'), req)
    if a.local_stub:
        write_json_atomic(res_path, stub_result(req, ckpt))

    t0 = time.time()
    res = None
    while time.time() - t0 < a.timeout:
        if os.path.exists(res_path):
            try:
                with open(res_path) as f:
                    res = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                pass   # partially written; try again
        time.sleep(POLL_SECS)
    if res is None:
        sys.exit(f'score.py: timed out after {a.timeout:.0f}s waiting for {res_path} (is the scoring daemon running?)')
    if res.get('status', 'ok') != 'ok':
        sys.exit(f'score.py: scorer error for request {rid}: {res.get("error", res)}')

    solved_len, n_len = by_len_counts(res)
    n = int(res.get('n') or sum(n_len.values()))
    solved = int(res.get('solved', sum(solved_len.values())))
    secs = float(res.get('secs', 0.0))
    lens = ' '.join(f'L{L} {solved_len[L]}/{n_len[L]}' for L in solved_len)
    print(f'{a.set} pass@{a.k}: solved {solved}/{n} | {lens} | secs {secs:.0f}', flush=True)

    out = {**res, 'id': rid, 'set': a.set, 'ckpt': ckpt, 'k': a.k, 'seed': a.seed, 'tag': a.tag, 'smoke': bool(a.smoke),
           'metric': solved, 'solved': solved, 'n': n, 'by_len': solved_len, 'n_by_len': n_len, 'secs': secs,
           'official': (a.k == OFFICIAL_K[a.set] and a.seed == OFFICIAL_SEED and not a.smoke and not res.get('stub')),
           'wait_secs': time.time() - t0, 't': time.time()}
    fn = 'score.json' if a.set == 'dev' else f'score_{a.set}.json'
    write_json_atomic(os.path.join(os.path.dirname(ckpt), fn), out)
    log_rec = dict(out)
    summ = os.path.join(os.path.dirname(ckpt), 'summary.json')
    if os.path.exists(summ):   # harness runs write a provisional summary with the file hashes before scoring
        try:
            with open(summ) as f:
                s = json.load(f)
            log_rec['shas'] = {k: v for k, v in (s.get('shas') or {}).items() if not k.startswith('data/')}
            log_rec['run_seed'] = s.get('seed')
        except (json.JSONDecodeError, OSError):
            pass
    with open(os.path.join(log_dir, 'scores.jsonl'), 'a') as f:
        f.write(json.dumps(log_rec) + '\n')


if __name__ == '__main__':
    main()
