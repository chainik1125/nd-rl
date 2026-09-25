#!/usr/bin/env python3
"""HOST-SIDE scoring daemon for the RL hill-climb, run 2 (run by the monitor; agents never run it).

  python score_daemon.py --arm ab2:/scratch/cowifornia/nd-rl/ab2:0,1,2,3,4 [--poll 2] [--per-arm 5]
                         [--per-gpu 1] [--gpu-map host|index] [--log PATH]

For each arm `name:armdir:hostgpus` it watches `<armdir>/score_requests/*.json`. A request is
{id, set, ckpt, k, seed, gpu, cwd, tag, smoke, t}.
  set   'dev' (default) or 'frontier'. Anything else (the sealed sets) is refused: those are scored by
        the monitor with score_host.py directly, never through the daemon.
  ckpt  a host-absolute path under <armdir> (run 2: in-process agents, no containers), or a legacy
        container path under /work/ (mapped to <armdir>/...); after resolving symlinks it must stay
        inside <armdir>.
  gpu   --gpu-map host (default): a host GPU index (the requester's CUDA_VISIBLE_DEVICES, i.e. the GPU
        it holds through gpulock), which must be in the arm's list. --gpu-map index (run 1): an index
        into the arm's list.
  k     1..256.
Each request runs `score_host.py --set <set>` as its own subprocess with CUDA_VISIBLE_DEVICES = host GPU,
at most --per-arm concurrently per arm and --per-gpu concurrently per host GPU (a request whose GPU is
busy waits; others overtake it). Results (request fields + aggregate scorer output with by_len
{"8": solved} and n_by_len {"8": n} as frozen/score.py expects; never per-theorem data) go to
<armdir>/score_results/<id>.json (atomic, mode 0644); the request file moves to score_requests/done/;
one line is appended to --log (default <ab2>/logs/daemon_scores.jsonl) with `arm`.
Only dev k=32 / frontier k=64, seed=0, non-smoke is official (score_host decides) (`official` in the result), but any k/seed runs.
A malformed request yields an error result (if an id can be derived) and never stops the daemon.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, threading, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
AB = os.path.dirname(HERE)
SCORES_LOG = os.path.join(AB, 'logs', 'daemon_scores.jsonl')   # replaced by --log
DAEMON_SETS = ('dev', 'frontier')
DEFAULT_PY = 'uv run --project /home/cowifornia/code/research/nd-rl/code --no-sync python'
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
MAX_K = 256
PARTIAL_GRACE = 10      # seconds a request file may stay unparsable (client mid-write) before it is rejected
_log_lock = threading.Lock()
_gpu_busy = {}          # host gpu -> running requests (all arms)
_gpu_lock = threading.Lock()


def log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def write_json_atomic(path, obj, mode=0o644):
    tmp = f'{path}.tmp{os.getpid()}.{threading.get_ident()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def append_scores(rec):
    with _log_lock:
        os.makedirs(os.path.dirname(SCORES_LOG), exist_ok=True)
        with open(SCORES_LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')


def scrub(text, n=3000):
    """stderr tail for error results: drop any line that could carry a theorem or proof."""
    keep = [l for l in text.splitlines() if not re.search(r'\b(THM|SEQ|PRF|QED)\b', l)]
    return '\n'.join(keep)[-n:]


class Arm:
    def __init__(self, spec, gpu_map='host'):
        try:
            name, d, gpus = spec.split(':')
            self.gpus = [int(g) for g in gpus.split(',') if g.strip()]
        except ValueError:
            raise SystemExit(f'bad --arm {spec!r}; want name:dir:gpu[,gpu]')
        self.name, self.dir, self.gpu_map = name, os.path.realpath(d), gpu_map
        self.req = os.path.join(self.dir, 'score_requests')
        self.done = os.path.join(self.req, 'done')
        self.res = os.path.join(self.dir, 'score_results')
        for p in (self.req, self.done, self.res):
            os.makedirs(p, exist_ok=True)
            try:
                os.chmod(p, 0o777)
            except OSError as e:
                log(f'[{name}] chmod {p}: {e}')
        self.running = {}   # request filename -> thread
        self.first_bad = {}  # filename -> first time it failed to parse

    def map_ckpt(self, p):
        if not isinstance(p, str) or not p:
            raise ValueError('ckpt missing')
        if p == '/work' or p.startswith('/work/'):
            host = os.path.join(self.dir, p[len('/work/'):])
        elif os.path.isabs(p):
            host = p
        else:
            raise ValueError(f'ckpt must be an absolute path: {p!r}')
        real = os.path.realpath(host)
        if not real.startswith(self.dir + os.sep):
            raise ValueError(f'ckpt resolves outside the arm directory: {p!r}')
        if not os.path.isfile(real):
            raise ValueError(f'ckpt not found: {p!r}')
        return real

    def map_gpu(self, g):
        if isinstance(g, str):   # the client sends CUDA_VISIBLE_DEVICES verbatim; use its first entry
            g = g.split(',')[0].strip()
            g = int(g) if g.isdigit() else g
        if not isinstance(g, int) or isinstance(g, bool):
            raise ValueError(f'gpu {g!r} is not an index')
        if self.gpu_map == 'host':
            if g not in self.gpus:
                raise ValueError(f'gpu {g} is not one of this arm\'s host GPUs {self.gpus}')
            return g
        if not (0 <= g < len(self.gpus)):
            raise ValueError(f'gpu {g!r} not in container range 0..{len(self.gpus) - 1}')
        return self.gpus[g]


def finish(arm, fname, result):
    rid = result.get('id') or os.path.splitext(fname)[0]
    write_json_atomic(os.path.join(arm.res, f'{rid}.json'), result)
    src = os.path.join(arm.req, fname)
    if os.path.exists(src):
        dst = os.path.join(arm.done, fname)
        if os.path.exists(dst):
            dst = os.path.join(arm.done, f'{os.path.splitext(fname)[0]}.{int(time.time())}.json')
        shutil.move(src, dst)
    append_scores({'arm': arm.name, **result})
    log(f'[{arm.name}] {rid}: set={result.get("set")} status={result.get("status")} solved={result.get("solved")}/{result.get("n")} '
        f'k={result.get("k")} seed={result.get("seed")} secs={result.get("secs")} '
        f'{result.get("error", "")[:200] if result.get("status") != "ok" else ""}')


def run_request(arm, fname, req, a):
    rid = req['id']
    t_start = time.time()
    base = {**req, 'arm': arm.name, 'daemon_received_t': t_start}
    try:
        which = req.get('set', 'dev')
        base['set'] = which
        if which not in DAEMON_SETS:
            raise ValueError(f'set {which!r} is not served (only {list(DAEMON_SETS)})')
        k, seed = int(req.get('k', {'dev': 32, 'frontier': 64}[which])), int(req.get('seed', 0))
        if not (1 <= k <= MAX_K):
            raise ValueError(f'k={k} outside 1..{MAX_K}')
        smoke = bool(req.get('smoke', False))
        host_ckpt = arm.map_ckpt(req.get('ckpt'))
        host_gpu = arm.map_gpu(req.get('gpu', 0))
    except Exception as e:
        finish(arm, fname, {**base, 'status': 'error', 'error': f'bad request: {e}'})
        return
    base.update(host_ckpt=host_ckpt, host_gpu=host_gpu)
    out_tmp = os.path.join(a.tmpdir, f'{arm.name}.{rid}.{int(t_start * 1000)}.json')
    cmd = a.python.split() + [os.path.join(HERE, 'score_host.py'), '--ckpt', host_ckpt, '--set', which,
                              '--k', str(k), '--seed', str(seed), '--out', out_tmp]
    if smoke:
        cmd.append('--smoke')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(host_gpu)}
    log(f'[{arm.name}] {rid}: start set={which} ckpt={req.get("ckpt")} k={k} seed={seed} smoke={smoke} '
        f'gpu {req.get("gpu")} -> host {host_gpu} tag={req.get("tag")!r}')
    try:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                             cwd=HERE, start_new_session=True)
        try:
            _, stderr = p.communicate(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            raise
        if p.returncode == 0 and os.path.exists(out_tmp):
            sc = json.load(open(out_tmp))
            sc['score_host_ckpt'] = sc.pop('ckpt', None)
            sc.pop('set_file', None)
            # client format (frozen/score.py, harness summary.json): by_len {"7": solved}, n_by_len {"7": n}
            pairs = sc.get('by_len') or {}
            sc['by_len'] = {L: v[0] for L, v in pairs.items()}
            sc['n_by_len'] = {L: v[1] for L, v in pairs.items()}
            sc['metric'] = sc.get('solved')
            res = {**base, **sc, 'ckpt': req.get('ckpt'), 'id': rid}
        else:
            res = {**base, 'status': 'error', 'returncode': p.returncode,
                   'error': scrub(stderr) or 'scorer failed without stderr'}
    except subprocess.TimeoutExpired:
        res = {**base, 'status': 'error', 'error': f'scorer timed out after {a.timeout}s'}
    except Exception:
        res = {**base, 'status': 'error', 'error': scrub(traceback.format_exc())}
    finally:
        if os.path.exists(out_tmp):
            os.remove(out_tmp)
    res['daemon_secs'] = round(time.time() - t_start, 2)
    finish(arm, fname, res)


def worker(arm, fname, req, a, gpu):
    try:
        run_request(arm, fname, req, a)
    except Exception:
        log(f'[{arm.name}] {fname}: internal error\n{traceback.format_exc()}')
    finally:
        if gpu is not None:
            with _gpu_lock:
                _gpu_busy[gpu] -= 1


def request_gpu(arm, req):
    """The host GPU a request will run on, or None if the request is invalid (run_request reports it)."""
    try:
        return arm.map_gpu(req.get('gpu', 0))
    except Exception:
        return None


def parse_request(arm, fname):
    """Returns a request dict, None (skip for now), or raises ValueError (reject)."""
    path = os.path.join(arm.req, fname)
    try:
        with open(path) as f:
            req = json.load(f)
        if not isinstance(req, dict):
            raise ValueError('request is not a JSON object')
    except (OSError, ValueError) as e:
        first = arm.first_bad.setdefault(fname, time.time())
        if time.time() - first < PARTIAL_GRACE:
            return None   # maybe still being written
        raise ValueError(f'unparsable request: {e}')
    arm.first_bad.pop(fname, None)
    rid = str(req.get('id') or os.path.splitext(fname)[0])
    if not ID_RE.match(rid):
        raise ValueError(f'bad id {rid!r}')
    req['id'] = rid
    return req


def poll_arm(arm, a):
    for fname, th in list(arm.running.items()):
        if not th.is_alive():
            del arm.running[fname]
    try:
        names = sorted((f for f in os.listdir(arm.req) if f.endswith('.json')),
                       key=lambda f: os.path.getmtime(os.path.join(arm.req, f)))
    except FileNotFoundError:
        names = []
    for fname in names:
        if len(arm.running) >= a.per_arm:
            break
        if fname in arm.running:
            continue
        try:
            req = parse_request(arm, fname)
        except ValueError as e:
            stem = os.path.splitext(fname)[0]
            rid = stem if ID_RE.match(stem) else f'bad-{int(time.time() * 1000)}'
            arm.first_bad.pop(fname, None)
            finish(arm, fname, {'id': rid, 'arm': arm.name, 'status': 'error', 'error': str(e),
                                'daemon_received_t': time.time()})
            continue
        if req is None:
            continue
        gpu = request_gpu(arm, req)
        if gpu is not None:
            with _gpu_lock:
                if _gpu_busy.get(gpu, 0) >= a.per_gpu:
                    continue   # that GPU is scoring already; leave the request queued
                _gpu_busy[gpu] = _gpu_busy.get(gpu, 0) + 1
        th = threading.Thread(target=worker, args=(arm, fname, req, a, gpu), daemon=True)
        arm.running[fname] = th
        th.start()


def main():
    global SCORES_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, help='name:armdir:hostgpu[,hostgpu]')
    ap.add_argument('--poll', type=float, default=2)
    ap.add_argument('--per-arm', type=int, default=5)
    ap.add_argument('--per-gpu', type=int, default=1, help='concurrent scorer subprocesses per host GPU')
    ap.add_argument('--gpu-map', choices=('host', 'index'), default='host',
                    help='host: request gpu is a host GPU index (run 2); index: index into the arm list (run 1)')
    ap.add_argument('--log', default=SCORES_LOG, help='JSONL log of every result')
    ap.add_argument('--timeout', type=int, default=3600, help='per-request scorer timeout (s)')
    ap.add_argument('--python', default=DEFAULT_PY, help='command prefix that runs python with torch')
    ap.add_argument('--tmpdir', default=os.path.join(AB, 'logs', 'score_tmp'))
    a = ap.parse_args()
    SCORES_LOG = os.path.abspath(a.log)
    os.makedirs(a.tmpdir, exist_ok=True)
    try:   # resolve the interpreter once (no uv wrapper per request, clean kill on timeout)
        exe = subprocess.run(a.python.split() + ['-c', 'import torch, sys; print(sys.executable)'],
                             capture_output=True, text=True, check=True, timeout=300).stdout.split()[-1]
        a.python = exe
        log(f'scorer python: {exe}')
    except Exception as e:
        raise SystemExit(f'cannot run python with torch via {a.python!r}: {e}')
    arms = [Arm(s, a.gpu_map) for s in a.arm]
    for arm in arms:
        log(f'watching arm {arm.name}: {arm.req} (host GPUs {arm.gpus}, gpu-map {a.gpu_map}, sets {list(DAEMON_SETS)}, '
            f'log {SCORES_LOG})')
    while True:
        for arm in arms:
            try:
                poll_arm(arm, a)
            except Exception:
                log(f'[{arm.name}] poll error\n{traceback.format_exc()}')
        time.sleep(a.poll)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('interrupted; exiting (running scorer subprocesses are abandoned)')
        sys.exit(0)
