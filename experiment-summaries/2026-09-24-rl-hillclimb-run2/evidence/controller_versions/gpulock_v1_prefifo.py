#!/usr/bin/env python3
"""FROZEN GPU lock for the ab2 run (7 workers + 1 auditor share host GPUs 0-4; stdlib only).

    python frozen/gpulock.py acquire [--gpus 0,1,2,3,4] [--wait SECS] [--run NAME] [--owner PID]
        -> prints the GPU index it locked (exit 0), or exits 1 after --wait seconds with none free
    python frozen/gpulock.py release <i>
    python frozen/gpulock.py status

One lock per GPU: an exclusive flock(2) on <LOCK_DIR>/gpu<i>. The flock is held by a small detached
holder process (its own session, stdio closed), so the lock outlives the shell that ran `acquire`.
The holder's pid is written to <LOCK_DIR>/gpu<i>.pid, and the lock file itself holds one JSON line
{gpu, worker (env AB_WORKER), run, holder_pid, owner_pid, since, host}.

The kernel drops an flock when its holder dies, so a lock whose holder pid is dead is free: `acquire`
simply takes it and overwrites the stale info. With --owner PID (the harness passes its own pid) the
holder also exits, freeing the GPU, within HOLDER_POLL seconds of that process dying, which covers
SIGKILL of the harness. Without --owner the lock is held until `release <i>` (or the holder is killed).

`acquire` tries the GPUs in the order given and takes the first free one; if none is free it retries
every RETRY seconds until --wait runs out (default 0: one pass).

Python API (used by harness.py --gpu auto):
    gpu, holder_pid = acquire(gpus, wait=0, run='', owner=None)   # gpu is None if none was free
    release(gpu, holder_pid=None)
    status() -> list of dicts
Env: AB_GPULOCK_DIR overrides LOCK_DIR (tests only).
"""
import argparse, errno, fcntl, json, os, select, signal, socket, subprocess, sys, time

LOCK_DIR = os.environ.get('AB_GPULOCK_DIR', '/scratch/cowifornia/nd-rl/ab2/locks')
DEFAULT_GPUS = (0, 1, 2, 3, 4)
HOLDER_POLL = 2.0     # seconds between the holder's owner-alive checks
RETRY = 5.0           # seconds between acquire passes while waiting
HANDSHAKE_SECS = 30   # how long acquire waits for the holder's answer on one pass
HOLD_PASSES = 4       # passes over the GPU list inside one holder before it answers 'none'


def lock_path(i):
    return os.path.join(LOCK_DIR, f'gpu{i}')


def pid_path(i):
    return os.path.join(LOCK_DIR, f'gpu{i}.pid')


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM
    try:   # a zombie is dead for our purposes
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return True


def is_holder(pid):
    """True if pid is a live gpulock holder process (guards `release` against pid reuse)."""
    if not pid_alive(pid):
        return False
    try:
        with open(f'/proc/{int(pid)}/cmdline', 'rb') as f:
            argv = f.read().split(b'\0')
        return any(a.endswith(b'gpulock.py') for a in argv) and b'_hold' in argv
    except OSError:
        return False


def read_info(i):
    try:
        with open(lock_path(i)) as f:
            txt = f.read().strip()
        return json.loads(txt) if txt else {}
    except (OSError, ValueError):
        return {}


def is_free(i):
    """Non-destructive probe: can gpu<i> be flocked right now?"""
    os.makedirs(LOCK_DIR, exist_ok=True)
    fd = os.open(lock_path(i), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


# ---------------------------------------------------------------- holder (runs detached)
def hold(gpus, run, owner, worker, answer_fd):
    """Lock the first free GPU of `gpus`, report its index on answer_fd, then sleep holding it."""
    os.makedirs(LOCK_DIR, exist_ok=True)
    os.chdir('/')
    got, fd = None, None
    for attempt in range(HOLD_PASSES):   # a few quick passes: another process's is_free() probe holds a
        for i in gpus:                   # lock for microseconds, which must not read as "busy"
            fd = os.open(lock_path(i), os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            got = i
            break
        if got is not None:
            break
        time.sleep(0.05)
    if got is None:
        os.write(answer_fd, b'none\n')
        os.close(answer_fd)
        return 1
    info = {'gpu': got, 'worker': worker, 'run': run, 'holder_pid': os.getpid(), 'owner_pid': owner,
            'since': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 't': time.time(), 'host': socket.gethostname()}
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, (json.dumps(info) + '\n').encode())
    os.fsync(fd)
    tmp = pid_path(got) + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(f'{os.getpid()}\n')
    os.replace(tmp, pid_path(got))
    os.write(answer_fd, f'{got}\n'.encode())
    os.close(answer_fd)

    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    while not stop:
        if owner and not pid_alive(owner):
            break
        time.sleep(HOLDER_POLL)
    # clear the info while still holding the lock, so `status` never shows a released GPU as held
    try:
        os.ftruncate(fd, 0)
        with open(pid_path(got)) as f:
            if f.read().strip() == str(os.getpid()):
                os.remove(pid_path(got))
    except OSError:
        pass
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    return 0


# ---------------------------------------------------------------- client side
def _spawn_holder(gpus, run, owner):
    r, w = os.pipe()
    cmd = [sys.executable, os.path.abspath(__file__), '_hold', '--gpus', ','.join(map(str, gpus)),
           '--run', run or '', '--answer-fd', str(w)]
    if owner:
        cmd += ['--owner', str(owner)]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         pass_fds=(w,), start_new_session=True, close_fds=True, env=dict(os.environ))
    os.close(w)
    buf, t0 = b'', time.time()
    while b'\n' not in buf and time.time() - t0 < HANDSHAKE_SECS:
        ready, _, _ = select.select([r], [], [], 1.0)
        if ready:
            chunk = os.read(r, 64)
            if not chunk:
                break
            buf += chunk
    os.close(r)
    ans = buf.decode().strip()
    if ans.isdigit():
        p.returncode = 0   # detached on purpose; keep Popen's destructor quiet
        return int(ans), p.pid
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
    return None, None


def acquire(gpus=DEFAULT_GPUS, wait=0.0, run='', owner=None, log=None):
    gpus = [int(g) for g in gpus]
    t0 = time.time()
    warned = False
    while True:
        # probe first so the common "all busy" case does not fork a holder every pass
        if any(is_free(i) for i in gpus):
            gpu, pid = _spawn_holder(gpus, run, owner)
            if gpu is not None:
                return gpu, pid
        left = wait - (time.time() - t0)
        if left <= 0:
            return None, None
        if log and not warned:
            log(f'gpulock: all of GPUs {gpus} are held; waiting up to {wait:.0f}s')
            warned = True
        time.sleep(min(RETRY, left))


def release(i, holder_pid=None, timeout=15.0):
    """Stop gpu<i>'s holder and wait until the lock is free. Returns True if the GPU is free afterwards."""
    i = int(i)
    pid = holder_pid
    if pid is None:
        try:
            pid = int(open(pid_path(i)).read().strip())
        except (OSError, ValueError):
            pid = read_info(i).get('holder_pid')
    if pid and is_holder(pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pid:
            try:   # reap the holder if it is our own child (the harness spawned it), so it leaves no zombie
                os.waitpid(int(pid), os.WNOHANG)
            except (ChildProcessError, OSError):
                pass
        if is_free(i):
            return True
        time.sleep(0.1)
    return is_free(i)


def status(gpus=DEFAULT_GPUS):
    out = []
    for i in gpus:
        free = is_free(i)
        info = {} if free else read_info(i)
        row = {'gpu': i, 'held': not free}
        if not free:
            row.update({k: info.get(k) for k in ('worker', 'run', 'holder_pid', 'owner_pid', 'since')})
            if info.get('t'):
                row['held_secs'] = round(time.time() - info['t'])
        out.append(row)
    return out


def parse_gpus(s):
    return [int(g) for g in str(s).split(',') if g.strip() != '']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('acquire')
    a1.add_argument('--gpus', default=','.join(map(str, DEFAULT_GPUS)))
    a1.add_argument('--wait', type=float, default=0.0, help='seconds to keep retrying if every GPU is held')
    a1.add_argument('--run', default='', help='run name stored in the lock file')
    a1.add_argument('--owner', type=int, default=None, help='free the GPU when this pid exits')
    a2 = sub.add_parser('release')
    a2.add_argument('gpu', type=int)
    a3 = sub.add_parser('status')
    a3.add_argument('--gpus', default=','.join(map(str, DEFAULT_GPUS)))
    a3.add_argument('--json', action='store_true')
    h = sub.add_parser('_hold')
    h.add_argument('--gpus', required=True)
    h.add_argument('--run', default='')
    h.add_argument('--owner', type=int, default=None)
    h.add_argument('--answer-fd', type=int, required=True)
    a = ap.parse_args(argv)

    if a.cmd == '_hold':
        return hold(parse_gpus(a.gpus), a.run, a.owner, os.environ.get('AB_WORKER', ''), a.answer_fd)
    if a.cmd == 'acquire':
        gpu, _ = acquire(parse_gpus(a.gpus), a.wait, a.run, a.owner,
                         log=lambda m: print(m, file=sys.stderr, flush=True))
        if gpu is None:
            print(f'gpulock: no free GPU among {a.gpus}', file=sys.stderr)
            return 1
        print(gpu, flush=True)
        return 0
    if a.cmd == 'release':
        info = read_info(a.gpu)
        me = os.environ.get('AB_WORKER', '')
        if info.get('worker') and me and info['worker'] != me:
            print(f'gpulock: note: GPU {a.gpu} was held by worker {info["worker"]!r}, releasing as {me!r}', file=sys.stderr)
        ok = release(a.gpu)
        print(json.dumps({'gpu': a.gpu, 'released': ok}))
        return 0 if ok else 1
    rows = status(parse_gpus(a.gpus))
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            if r['held']:
                print(f"gpu{r['gpu']}  held  worker={r.get('worker') or '-'}  run={r.get('run') or '-'}  "
                      f"holder={r.get('holder_pid')}  owner={r.get('owner_pid') or '-'}  since={r.get('since')} "
                      f"({r.get('held_secs', '?')}s)")
            else:
                print(f"gpu{r['gpu']}  free")
    return 0


if __name__ == '__main__':
    sys.exit(main())
