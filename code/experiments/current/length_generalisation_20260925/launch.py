#!/usr/bin/env python3
"""Launcher for the length-generalisation experiment: arm files, 9 training runs through the GPU lock,
evaluation of every checkpoint, analysis.

    python launch.py prepare [--seed-rl ../seed/rl.py] [--no-proof-cap]
                                                           make rl_A.py / rl_B.py / rl_C.py: only max_len_train and
                                                           max_proof_lines_train differ, both set to the arm's value
                                                           (--no-proof-cap: max_len_train only)
    python launch.py smoke [--arm A]                       harness --smoke --no-score on one arm file (plumbing check)
    python launch.py train [--dry-run] [--allow-unmetered] 9 runs: harness.py --rl rl_<arm>.py --out runs/<arm>_s<seed>
                                                           --seed <seed> --gpu auto --no-score (all queued on the lock)
    python launch.py eval [--labels pretrain,A_s101,...]   evaluate.py for pretrain + every finished run (through the lock)
    python launch.py analyse                               uv run --script analyse.py
    python launch.py all [--allow-unmetered]               prepare, then train with each run's evaluation queued as soon
                                                           as the run finishes (and pretrain's at the start), then analyse
    python launch.py status                                runs, evaluations and GPU locks

Everything runs from ab3/lengthexp with AB_WORKER=lengthexp. `train` refuses to start while the harness's
DEFAULT_FLOPS is None (the run-3 budget is not final) unless --allow-unmetered. Runs whose summary.json says
status ok are skipped; a failed run's directory is moved aside (runs/<label>.failed.<time>) and re-run.
Logs: runs/<label>.log, eval/<label>.log, logs/launch.log, logs/launch_status.json.
"""
import argparse, ast, datetime, difflib, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ARMS, EVAL, FROZEN, LE, PRETRAIN, PY, RUNS, SEED_RL, SEEDS, sha256_file, write_json  # noqa: E402

UV = shutil.which('uv') or '/home/cowifornia/.local/bin/uv'
HARNESS = os.path.join(FROZEN, 'harness.py')
LOGS = os.path.join(LE, 'logs')
KNOB = 'max_len_train'                  # targets: L_true <= the arm's value
PROOF_KNOB = 'max_proof_lines_train'    # fine-tune mix: found proofs of <= the arm's value written lines
GPU_WAIT = 12 * 3600
SPACING = 3.0      # seconds between launches, so the FIFO lock queue keeps the launch order


def log(msg):
    line = f'[{datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}Z] {msg}'
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, 'launch.log'), 'a') as f:
        f.write(line + '\n')


def env():
    e = os.environ.copy()
    e['AB_WORKER'] = 'lengthexp'
    return e


def default_flops():
    """DEFAULT_FLOPS as written in frozen/harness.py (None while the budget is not calibrated)."""
    tree = ast.parse(open(HARNESS).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'DEFAULT_FLOPS' for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return ast.unparse(node.value)
    return 'missing'


def arm_file(arm, root=LE):
    return os.path.join(root, f'rl_{arm}.py')


def labels_all():
    return [f'{arm}_s{s}' for s in SEEDS for arm in ARMS]     # interleaved: A101 B101 C101 A102 ...


# ------------------------------------------------------------------ prepare
def knob_span(src, knob=KNOB):
    """(start, end) character offsets of the value of CONFIG's `knob` entry, found with the AST.
    CONFIG must be a top-level `CONFIG = dict(..., knob=V, ...)` or `CONFIG = {..., 'knob': V}`."""
    tree = ast.parse(src)
    vals = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'CONFIG' for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, 'id', None) == 'dict':
                vals += [k.value for k in v.keywords if k.arg == knob]
            elif isinstance(v, ast.Dict):
                vals += [val for key, val in zip(v.keys, v.values) if isinstance(key, ast.Constant) and key.value == knob]
    if len(vals) != 1:
        raise SystemExit(f'prepare: expected exactly one {knob} entry in a top-level CONFIG, found {len(vals)}')
    v = vals[0]
    lines = src.splitlines(keepends=True)
    off = [0]
    for l in lines:
        off.append(off[-1] + len(l))
    # col offsets are in UTF-8 bytes; convert on the line
    def pos(lineno, col):
        line = lines[lineno - 1]
        return off[lineno - 1] + len(line.encode()[:col].decode())
    return pos(v.lineno, v.col_offset), pos(v.end_lineno, v.end_col_offset)


def make_arms(seed_rl, root=LE, proof_cap=True):
    """rl_<arm>.py = the seed with CONFIG's max_len_train and (proof_cap) max_proof_lines_train set to the arm's value:
    a one-line diff per knob (two lines with the proof cap), checked."""
    src = open(seed_rl).read()
    knobs = [KNOB] + ([PROOF_KNOB] if proof_cap else [])
    spans = {}
    for knob in knobs:
        spans[knob] = knob_span(src, knob)
        reads = len(re.findall(r"""(?:\[\s*['"]""" + knob + r"""['"]\s*\]|\.get\(\s*['"]""" + knob + r"""['"])""", src))
        if reads < 1:
            raise SystemExit(f'prepare: {seed_rl} sets {knob} but never reads it (cfg[{knob!r}] / .get({knob!r}))')
    lines = {k: src.count('\n', 0, s0) + 1 for k, (s0, _) in spans.items()}
    if len(set(lines.values())) != len(lines):
        raise SystemExit(f'prepare: the knobs {knobs} must be on separate lines of CONFIG ({lines})')
    info = {'seed_rl': os.path.abspath(seed_rl), 'seed_sha256': sha256_file(seed_rl), 'knobs': knobs,
            'seed_value': src[slice(*spans[KNOB])], 'knob_line': lines[KNOB],
            'seed_values': {k: src[slice(*sp)] for k, sp in spans.items()}, 'knob_lines': lines, 'arms': {}}
    for arm, v in ARMS.items():
        out = src
        for knob, (s0, s1) in sorted(spans.items(), key=lambda kv: -kv[1][0]):    # from the end: offsets stay valid
            out = out[:s0] + str(v) + out[s1:]
        ast.parse(out)
        diff = [l for l in difflib.unified_diff(src.splitlines(), out.splitlines(), lineterm='', n=0)
                if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
        n_changed = sum(1 for k in knobs if info['seed_values'][k] != str(v))
        if len(diff) != 2 * n_changed or not all(any(k in l for k in knobs) for l in diff):
            raise SystemExit(f'prepare: arm {arm} should differ from the seed in exactly the {n_changed} knob line(s): {diff}')
        fn = arm_file(arm, root)
        with open(fn, 'w') as f:
            f.write(out)
        info['arms'][arm] = {'file': fn, 'max_len_train': v, PROOF_KNOB: v if proof_cap else info['seed_values'].get(PROOF_KNOB),
                             'sha256': sha256_file(fn), 'diff': diff}
        log(f'prepare: {fn}: ' + ', '.join(f'{k} {info["seed_values"][k]} -> {v} (line {lines[k]})' for k in knobs))
    write_json(os.path.join(root, 'arms.json'), info)
    return info


def check_arms():
    fn = os.path.join(LE, 'arms.json')
    if not os.path.isfile(fn):
        raise SystemExit('run `launch.py prepare` first')
    info = json.load(open(fn))
    for arm, a in info['arms'].items():
        if sha256_file(a['file']) != a['sha256']:
            raise SystemExit(f'{a["file"]} changed since prepare; re-run prepare')
    if os.path.isfile(info['seed_rl']) and sha256_file(info['seed_rl']) != info['seed_sha256']:
        raise SystemExit(f'the seed {info["seed_rl"]} changed since prepare; re-run prepare')
    return info


# ------------------------------------------------------------------ processes
def run_status(label):
    fn = os.path.join(RUNS, label, 'summary.json')
    if not os.path.isfile(fn):
        return None
    try:
        return json.load(open(fn)).get('status')
    except (OSError, ValueError):
        return 'unreadable'


def live_procs(*needles):
    """pids of live processes whose command line contains every needle (guards against double launches)."""
    out = []
    for p in os.listdir('/proc'):
        if not p.isdigit() or int(p) == os.getpid():
            continue
        try:
            cmd = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if all(n in cmd for n in needles):
            out.append(int(p))
    return out


def start_train(label, allow_unmetered):
    arm, _, s = label.partition('_s')
    out = os.path.join(RUNS, label)
    st = run_status(label)
    if st == 'ok':
        log(f'train {label}: already ok, skipped')
        return None
    pids = live_procs('harness.py', f'runs/{label}/') + live_procs('harness.py', f'runs/{label} ')
    if pids:
        log(f'train {label}: already running (pids {pids}); not relaunched (wait for it, then re-run)')
        return None
    if os.path.exists(out):
        aside = f'{out}.failed.{time.strftime("%Y%m%d-%H%M%S")}'
        os.rename(out, aside)
        log(f'train {label}: previous attempt (status {st}) moved to {aside}')
    cmd = [PY, HARNESS, '--rl', os.path.basename(arm_file(arm)), '--out', os.path.relpath(out, LE) + '/', '--seed', s,
           '--gpu', 'auto', '--no-score', '--gpu-wait', str(GPU_WAIT)]
    os.makedirs(RUNS, exist_ok=True)
    fh = open(os.path.join(RUNS, f'{label}.log'), 'a')
    log(f'train {label}: ' + ' '.join(cmd))
    return subprocess.Popen(cmd, cwd=LE, env=env(), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def start_eval(label, extra=()):
    ckpt = PRETRAIN if label == 'pretrain' else os.path.join(RUNS, label, 'final.pt')
    if not os.path.isfile(ckpt):
        log(f'eval {label}: no checkpoint {ckpt}, skipped')
        return None
    pids = live_procs('evaluate.py', f'--label {label} ')
    if pids:
        log(f'eval {label}: already running (pids {pids}); not relaunched')
        return None
    cmd = [PY, os.path.join(LE, 'evaluate.py'), '--ckpt', ckpt, '--label', label, '--gpu', 'auto',
           '--gpu-wait', str(GPU_WAIT)] + list(extra)   # (--label is followed by another argument: the live-process guard relies on it)
    os.makedirs(EVAL, exist_ok=True)
    fh = open(os.path.join(EVAL, f'{label}.log'), 'a')
    log(f'eval {label}: ' + ' '.join(cmd))
    return subprocess.Popen(cmd, cwd=LE, env=env(), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def wait_all(procs, on_done=None):
    """procs: {name: Popen}. Polls until all exit; on_done(name, returncode) may add more to procs."""
    t0 = time.time()
    last = 0
    while any(p.poll() is None for p in procs.values()):
        for name, p in list(procs.items()):
            if p.poll() is not None and not getattr(p, '_reported', False):
                p._reported = True
                log(f'{name}: exited {p.returncode}' + (f' (run status {run_status(name.split()[-1])})' if name.startswith('train') else ''))
                if on_done:
                    on_done(name, p.returncode)
        if time.time() - last > 60:
            last = time.time()
            write_json(os.path.join(LOGS, 'launch_status.json'),
                       {'t': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'elapsed_secs': time.time() - t0,
                        'running': [n for n, p in procs.items() if p.poll() is None],
                        'finished': {n: p.returncode for n, p in procs.items() if p.poll() is not None}})
        time.sleep(5)
    for name, p in list(procs.items()):
        if not getattr(p, '_reported', False):
            p._reported = True
            log(f'{name}: exited {p.returncode}')
            if on_done:
                on_done(name, p.returncode)
    if any(p.poll() is None for p in procs.values()):   # on_done added more
        wait_all(procs, on_done)


# ------------------------------------------------------------------ commands
def cmd_prepare(a):
    make_arms(a.seed_rl, proof_cap=not a.no_proof_cap)
    fl = default_flops()
    log(f'prepare: harness DEFAULT_FLOPS = {fl}' + (' (not final: `train` will refuse without --allow-unmetered)' if fl is None else ''))


def guard_budget(a):
    fl = default_flops()
    if fl is None and not a.allow_unmetered:
        raise SystemExit('train: frozen/harness.py has DEFAULT_FLOPS = None (budget not calibrated). '
                         'Wait for the final harness, or pass --allow-unmetered for a test.')
    log(f'train: DEFAULT_FLOPS = {fl}')


def cmd_smoke(a):
    check_arms()
    out = os.path.join(LE, 'runs_smoke', f'{a.arm}_smoke')
    if os.path.exists(out):
        shutil.rmtree(out)
    cmd = [PY, HARNESS, '--rl', os.path.basename(arm_file(a.arm)), '--out', os.path.relpath(out, LE), '--seed', '101',
           '--gpu', 'auto', '--no-score', '--smoke', '--gpu-wait', '1800']
    log('smoke: ' + ' '.join(cmd))
    r = subprocess.run(cmd, cwd=LE, env=env())
    log(f'smoke: exit {r.returncode}, status {json.load(open(os.path.join(out, "summary.json"))).get("status") if os.path.isfile(os.path.join(out, "summary.json")) else None}')
    return r.returncode


def cmd_train(a):
    check_arms()
    guard_budget(a)
    labels = [l for l in labels_all() if not a.labels or l in a.labels.split(',')]
    if a.dry_run:
        for l in labels:
            arm, _, s = l.partition('_s')
            print(f'cd {LE} && AB_WORKER=lengthexp {PY} {HARNESS} --rl {os.path.basename(arm_file(arm))} --out runs/{l} '
                  f'--seed {s} --gpu auto --no-score --gpu-wait {GPU_WAIT}')
        return 0
    procs = {}
    for l in labels:
        p = start_train(l, a.allow_unmetered)
        if p:
            procs[f'train {l}'] = p
            time.sleep(SPACING)
    wait_all(procs)
    bad = [l for l in labels if run_status(l) != 'ok']
    log(f'train: done; not ok: {bad or "none"}')
    return 1 if bad else 0


def cmd_eval(a):
    labels = a.labels.split(',') if a.labels else ['pretrain'] + [l for l in labels_all() if run_status(l) == 'ok']
    procs = {}
    for l in labels:
        p = start_eval(l)
        if p:
            procs[f'eval {l}'] = p
            time.sleep(SPACING)
    wait_all(procs)
    return 0 if all(p.returncode == 0 for p in procs.values()) else 1


def cmd_analyse(a):
    cmd = [UV, 'run', '--script', os.path.join(LE, 'analyse.py')]
    log('analyse: ' + ' '.join(cmd))
    return subprocess.run(cmd, cwd=LE).returncode


def cmd_all(a):
    if not a.skip_prepare:
        make_arms(a.seed_rl, proof_cap=not a.no_proof_cap)
    check_arms()
    guard_budget(a)
    labels = labels_all()
    procs = {}
    for l in labels:
        p = start_train(l, a.allow_unmetered)
        if p:
            procs[f'train {l}'] = p
            time.sleep(SPACING)
        elif run_status(l) == 'ok':
            q = start_eval(l)
            if q:
                procs[f'eval {l}'] = q
    p = start_eval('pretrain')
    if p:
        procs['eval pretrain'] = p

    def on_done(name, rc):
        kind, l = name.split()
        if kind == 'train':
            if run_status(l) == 'ok':
                q = start_eval(l)
                if q:
                    procs[f'eval {l}'] = q
            else:
                log(f'{l}: training did not finish ok (status {run_status(l)}); not evaluated')
    wait_all(procs, on_done)
    bad = [l for l in labels if run_status(l) != 'ok']
    log(f'all: training not ok: {bad or "none"}; evaluation exit codes: '
        f'{ {n: p.returncode for n, p in procs.items() if n.startswith("eval")} }')
    cmd_analyse(a)
    return 0


def cmd_status(a):
    print(f'harness DEFAULT_FLOPS = {default_flops()}')
    info = json.load(open(os.path.join(LE, 'arms.json'))) if os.path.isfile(os.path.join(LE, 'arms.json')) else None
    print('arms:', {k: (v['max_len_train'], v.get(PROOF_KNOB)) for k, v in info['arms'].items()} if info else 'not prepared',
          '(max_len_train, max_proof_lines_train)' if info else '',
          f"(seed {info['seed_rl']} {info['seed_sha256'][:12]})" if info else '')
    for l in labels_all():
        st = run_status(l)
        s = {}
        fn = os.path.join(RUNS, l, 'summary.json')
        if st:
            s = json.load(open(fn))
        n_units = len([f for f in os.listdir(os.path.join(EVAL, l)) if f.endswith('.summary.json')]) if os.path.isdir(os.path.join(EVAL, l)) else 0
        print(f'  {l:8s} train {st or "-":8s} rl_secs {s.get("rl_secs") or 0:7.0f} heldout {s.get("heldout_greedy") or 0:.3f}  eval units {n_units}')
    n_units = len([f for f in os.listdir(os.path.join(EVAL, 'pretrain')) if f.endswith('.summary.json')]) if os.path.isdir(os.path.join(EVAL, 'pretrain')) else 0
    print(f'  pretrain eval units {n_units}')
    subprocess.run([PY, os.path.join(FROZEN, 'gpulock.py'), 'status'])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'smoke', 'train', 'eval', 'analyse', 'all', 'status'):
        p = sub.add_parser(name)
        p.add_argument('--seed-rl', default=SEED_RL)
        p.add_argument('--allow-unmetered', action='store_true')
        p.add_argument('--dry-run', action='store_true')
        p.add_argument('--labels', default='')
        p.add_argument('--arm', default='A')
        p.add_argument('--skip-prepare', action='store_true')
        p.add_argument('--no-proof-cap', action='store_true', help='prepare: set max_len_train only (no proof-line cap)')
    a = ap.parse_args()
    return {'prepare': cmd_prepare, 'smoke': cmd_smoke, 'train': cmd_train, 'eval': cmd_eval, 'analyse': cmd_analyse,
            'all': cmd_all, 'status': cmd_status}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main() or 0)
