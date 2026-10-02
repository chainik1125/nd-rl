#!/usr/bin/env python3
"""FROZEN dev-score CLIENT. The real scorer runs host-side (score_daemon.py -> score_host.py).

    python frozen/score.py --ckpt runs/<name>/final.pt [--k 32] [--seed 0] [--tag free-text] [--smoke]

Protocol (SPEC.md addendum):
  arm root = the nearest ancestor of the checkpoint (or of the cwd) that contains `frozen/`
             (`/work` inside a container, the arm directory on the host).
  1. write  <arm root>/score_requests/<id>.json  {id, ckpt, k, seed, gpu, cwd, tag, smoke, t}
  2. poll   <arm root>/score_results/<id>.json   every 3 s, timeout 30 min
  3. print  `dev pass@<k>: solved T/N | L7 a/300 L8 b/300 L9 c/1010 | secs S`
  4. write  score.json next to the checkpoint; append the result to <arm root>/logs/scores.jsonl
Exit status is non-zero on a result with status "error" or on timeout.
This file holds no evaluation data and does not verify anything itself.
"""
import argparse, hashlib, json, os, sys, time, uuid

DEFAULT_N = {'7': 300, '8': 300, '9': 1010}
POLL_SECS = 3
TIMEOUT_SECS = 30 * 60


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
    """Normalise the result's by_len into ({'7': solved, ...}, {'7': n, ...}).
    Accepts {'7': int} (denominators from res['n_by_len'] or the full dev set) or {'7': {'solved', 'n'}}."""
    solved, n = {}, dict(res.get('n_by_len') or {})
    for L, v in (res.get('by_len') or {}).items():
        if isinstance(v, dict):
            solved[str(L)] = int(v.get('solved', 0))
            if 'n' in v:
                n[str(L)] = int(v['n'])
        else:
            solved[str(L)] = int(v)
    for L in DEFAULT_N:
        solved.setdefault(L, 0)
        if L not in n:
            n[L] = DEFAULT_N[L] if not res.get('smoke') else 0
    return solved, {str(k): int(v) for k, v in n.items()}


def stub_result(req, ckpt):
    """--local-stub: a fake all-zero result so harness plumbing can be tested without the daemon."""
    n = {'7': 5, '8': 5, '9': 6} if req['smoke'] else dict(DEFAULT_N)
    return {'id': req['id'], 'status': 'ok', 'stub': True, 'ckpt': req['ckpt'], 'ckpt_sha256': file_sha256(ckpt),
            'k': req['k'], 'seed': req['seed'], 'smoke': req['smoke'], 'solved': 0,
            'by_len': {L: 0 for L in n}, 'n_by_len': n, 'n': sum(n.values()), 'secs': 0.0, 'samples_per_sec': 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--smoke', action='store_true', help='score a 16-theorem dev subset (plumbing check)')
    ap.add_argument('--timeout', type=float, default=TIMEOUT_SECS, help=argparse.SUPPRESS)
    ap.add_argument('--local-stub', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()

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
    req = {'id': rid, 'ckpt': ckpt, 'k': a.k, 'seed': a.seed, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
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
    print(f'dev pass@{a.k}: solved {solved}/{n} | L7 {solved_len["7"]}/{n_len["7"]} L8 {solved_len["8"]}/{n_len["8"]} '
          f'L9 {solved_len["9"]}/{n_len["9"]} | secs {secs:.0f}', flush=True)

    out = {**res, 'id': rid, 'ckpt': ckpt, 'k': a.k, 'seed': a.seed, 'tag': a.tag, 'smoke': bool(a.smoke),
           'metric': solved, 'solved': solved, 'n': n, 'by_len': solved_len, 'n_by_len': n_len, 'secs': secs,
           'official': (a.k == 32 and a.seed == 0 and not a.smoke and not res.get('stub')),
           'wait_secs': time.time() - t0, 't': time.time()}
    write_json_atomic(os.path.join(os.path.dirname(ckpt), 'score.json'), out)
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
