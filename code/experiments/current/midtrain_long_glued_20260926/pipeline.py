#!/usr/bin/env python3
"""The mid-training experiment end to end (run in the background; everything on the GPU goes through the lock).

    nohup python pipeline.py > logs/pipeline.out 2>&1 &

1. data/SHA256SUMS must verify; build testsets/long.jsonl (build_long.py).
2. midtrain.py --arm long, short, more (one after another; about a minute each).
3. make_frozen.py per arm (frozen/<arm>: frozen_039 with only pretrain.pt replaced, verified by sha256).
4. RL (arm-C recipe rl_C.py, seed 101, DEFAULT_FLOPS, --no-score) for long, short, more through frozen/<arm>/harness.py.
   (base was launched separately through ../frozen_039/harness.py.)
5. Evaluations (mt_evaluate.py, sets std, matched, long): pre039 (only its missing long units), mt_long, mt_short,
   mt_more at once; each RL arm's final.pt as soon as its run is ok.
6. analyse.py.
Idempotent: finished steps are skipped (checkpoints, frozen copies, runs with status ok, cached evaluation units).
Log: logs/pipeline.log.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/cowifornia/code/research/nd-rl/code/.venv/bin/python'
UV = '/home/cowifornia/.local/bin/uv'
ENV = dict(os.environ, AB_WORKER='midtrain')


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}Z] {msg}'
    print(line, flush=True)
    with open(os.path.join(HERE, 'logs', 'pipeline.log'), 'a') as f:
        f.write(line + '\n')


def run(cmd, logf=None, cwd=HERE):
    log('run: ' + ' '.join(cmd))
    out = open(os.path.join(HERE, logf), 'a') if logf else None
    r = subprocess.run(cmd, cwd=cwd, env=ENV, stdout=out, stderr=subprocess.STDOUT if out else None)
    if r.returncode != 0:
        log(f'FAILED ({r.returncode}): {" ".join(cmd)}')
        sys.exit(1)


def spawn(cmd, logf):
    log('spawn: ' + ' '.join(cmd))
    return subprocess.Popen(cmd, cwd=HERE, env=ENV, stdout=open(os.path.join(HERE, logf), 'a'), stderr=subprocess.STDOUT,
                            start_new_session=True)


def status(arm):
    try:
        return json.load(open(os.path.join(HERE, 'runs', f'{arm}_s101', 'summary.json'))).get('status')
    except (OSError, ValueError):
        return None


def running(arm):
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                c = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if 'harness.py' in c and f'runs/{arm}_s101 ' in c + ' ':
                return True
    return False


EVAL_SETS = ['--sets', 'std,matched,long', '--lengths', '6-18,22,24,26', '--gpu', 'auto', '--gpu-wait', '43200']
EXPECTED = [f'std_L{L}' for L in range(6, 19)] + [f'matched_L{L}' for L in range(6, 15)] + [f'long_L{L}' for L in (22, 24, 26)]
CKPT = {'pre039': '/scratch/cowifornia/nd-rl/ab3/frozen_039/pretrain.pt', 'mt_long': 'ckpt/mt_long.pt', 'mt_short': 'ckpt/mt_short.pt',
        'mt_more': 'ckpt/mt_more.pt', 'base': 'runs/base_s101/final.pt', 'long': 'runs/long_s101/final.pt',
        'short': 'runs/short_s101/final.pt', 'more': 'runs/more_s101/final.pt'}


def gpulock_free(i):
    sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/frozen_039')
    import gpulock
    return gpulock.is_free(i)


def missing_units(label):
    d = os.path.join(HERE, 'eval', label)
    return [u for u in EXPECTED if not os.path.isfile(os.path.join(d, f'{u}.summary.json'))]


def eval_running(label):
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                c = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if 'mt_evaluate.py' in c and f'--label {label} ' in c:
                return True
    return False


def clean_gpus():
    """GPUs with no compute process of another user right now (another user's scheduler runs jobs outside our lock;
    a shared card would slow an RL run towards the harness's wall cap). Falls back to all five."""
    try:
        uu = dict(l.split(', ') for l in subprocess.run(['nvidia-smi', '--query-gpu=uuid,index', '--format=csv,noheader'],
                                                        capture_output=True, text=True).stdout.strip().splitlines())
        apps = [l.split(', ') for l in subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                                                      capture_output=True, text=True).stdout.strip().splitlines() if l.strip()]
        busy = set()
        for pid, u in apps:
            try:
                uid = int(open(f'/proc/{pid}/status').read().split('Uid:')[1].split()[0])
            except (OSError, IndexError, ValueError):
                continue
            if uid != os.getuid():
                busy.add(int(uu[u]))
        clean = [g for g in range(5) if g not in busy]
        return ','.join(map(str, clean or range(5)))
    except Exception as e:  # noqa: BLE001
        log(f'clean_gpus failed ({e}); using all GPUs')
        return '0,1,2,3,4'


def main():
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    run(['sha256sum', '-c', 'SHA256SUMS'], 'logs/data_check.log', cwd=os.path.join(HERE, 'data'))
    if not os.path.isfile(os.path.join(HERE, 'testsets', 'long.jsonl')):
        run([PY, 'build_long.py'], 'logs/build_long.log')
    for arm in ('long', 'short', 'more'):
        if not os.path.isfile(os.path.join(HERE, 'ckpt', f'mt_{arm}.pt')):
            run([PY, 'midtrain.py', '--arm', arm, '--gpu-wait', '43200'], f'logs/midtrain_{arm}.log')
        if not os.path.isdir(os.path.join(HERE, 'frozen', arm)):
            run([PY, 'make_frozen.py', '--arm', arm], f'logs/make_frozen_{arm}.log')
    procs = {}
    # evaluations: a label is (re)launched when its checkpoint exists (RL arms: run status ok), units are missing and
    # no evaluation of that label is running (units already done are skipped by evaluate.py's cache)
    todo_eval = set(CKPT)
    while todo_eval:
        # RL: a pending arm is launched only when a GPU is both lock-free and free of other users' processes, pinned
        # to that GPU, so an RL ticket never waits at the head of the FIFO queue blocking the evaluations behind it
        for arm in ('long', 'short', 'more'):
            if status(arm) is None and not running(arm):
                free = [g for g in clean_gpus().split(',') if gpulock_free(int(g))]
                if free:
                    log(f'train {arm}: lock-free GPUs without other users\' processes: {free}; pinning to {free[0]}')
                    procs[f'train {arm}'] = spawn([PY, f'frozen/{arm}/harness.py', '--rl', 'rl_C.py', '--out', f'runs/{arm}_s101',
                                                   '--seed', '101', '--gpu', 'auto', '--gpus', free[0], '--no-score',
                                                   '--gpu-wait', '43200'], f'runs/{arm}_s101.log')
                    time.sleep(20)
        for label in sorted(todo_eval):
            arm = label if label in ('base', 'long', 'short', 'more') else None
            if arm:
                st = status(arm)
                if st in ('crash', 'timeout') and not running(arm):
                    log(f'{arm}_s101 finished: {st}; not evaluated'); todo_eval.discard(label); continue
                if st != 'ok' or running(arm):
                    continue
            if not missing_units(label):
                log(f'eval {label}: complete'); todo_eval.discard(label); continue
            if not eval_running(label):
                procs[f'eval {label} {time.time():.0f}'] = spawn([PY, 'mt_evaluate.py', '--ckpt', CKPT[label], '--label', label] + EVAL_SETS,
                                                                 f'eval/{label}.log')
                time.sleep(3)
        for name, p in procs.items():
            if p.poll() is not None and not getattr(p, '_seen', False):
                p._seen = True
                log(f'{name}: exit {p.returncode}')
        time.sleep(30)
    run([UV, 'run', '--script', 'analyse.py'], 'logs/analyse.log')
    log('pipeline done')


if __name__ == '__main__':
    main()
