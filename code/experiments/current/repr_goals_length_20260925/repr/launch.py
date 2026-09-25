#!/usr/bin/env python3
"""Pipeline launcher for the representation experiment: per variant x seed, pretrain -> frozen copy -> RL (arm C
recipe, DEFAULT_FLOPS, --no-score) -> evaluation; then the analysis.

    python launch.py status
    python launch.py run [--variants abs,rel,...] [--seeds 0,1,2] [--hold-rl posrand] [--max-inflight 4] [--dry-run]
    python launch.py analyse

Every job is a subprocess that takes its own GPU through frozen/gpulock.py (FIFO queue), so other approved jobs
interleave: at most --max-inflight of these jobs are holding or waiting for a GPU at any time, and each job is
<= ~35 min (pretrain ~13 min, RL ~30 min, evaluation ~14 min). Ready jobs start in the order (seed, later stage
first, variant), so seed 0 of every variant finishes first. A stage is done when its output says so
(pretrain/<l>/summary.json ok, runs/<l>/summary.json ok, eval/<l>/ has every unit), so `run` can be restarted;
a failed job is logged and not retried (its directory is left for inspection). --hold-rl V stops variant V after
pretraining (pos-rand: sampling-time positions await the monitor's decision). The pos-rand random-position
evaluation check (std set, --pos-eval rand, label <l>_randpos) runs after each pos-rand evaluation.
Logs: logs/launch.log, and each job's log in its own directory.
"""
import argparse, datetime, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_repr import (EVAL, FROZEN_DIR, LOGS, ORDER, PRETRAIN_DIR, PY, RL_SEED, RUNS, SEEDS, label,  # noqa: E402
                         write_json)

GPU_WAIT = 12 * 3600
N_UNITS = {'std': 13, 'matched': 9}
STAGES = ('pretrain', 'rl', 'eval', 'randpos')


def log(msg):
    line = f'[{datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}Z] {msg}'
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, 'launch.log'), 'a') as f:
        f.write(line + '\n')


def status_of(path):
    try:
        return json.load(open(path)).get('status')
    except (OSError, ValueError):
        return None


def n_units(d, sets=('std', 'matched')):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.endswith('.summary.json') and f.split('_L')[0] in sets)


def done(stage, lab):
    if stage == 'pretrain':
        return status_of(os.path.join(PRETRAIN_DIR, lab, 'summary.json')) == 'ok'
    if stage == 'rl':
        return status_of(os.path.join(RUNS, lab, 'summary.json')) == 'ok'
    if stage == 'eval':
        return n_units(os.path.join(EVAL, lab)) >= sum(N_UNITS.values())
    if stage == 'randpos':
        return n_units(os.path.join(EVAL, lab + '_randpos'), ('std',)) >= N_UNITS['std']


def command(stage, v, s):
    lab = label(v, s)
    if stage == 'pretrain':
        out = os.path.join(PRETRAIN_DIR, lab)
        return [PY, os.path.join(HERE, 'pretrain.py'), '--variant', v, '--seed', str(s), '--out', out, '--gpu', 'auto',
                '--gpu-wait', str(GPU_WAIT)], out
    if stage == 'rl':
        return [PY, os.path.join(FROZEN_DIR, lab, 'harness.py'), '--rl', os.path.join(HERE, 'rl_repr.py'), '--out',
                os.path.join(RUNS, lab), '--seed', str(RL_SEED[s]), '--gpu', 'auto', '--no-score', '--gpu-wait',
                str(GPU_WAIT)], os.path.join(RUNS, lab)
    if stage == 'eval':
        return [PY, os.path.join(HERE, 'evaluate.py'), '--ckpt', os.path.join(RUNS, lab, 'final.pt'), '--label', lab,
                '--gpu', 'auto', '--gpu-wait', str(GPU_WAIT)], os.path.join(EVAL, lab)
    if stage == 'randpos':
        return [PY, os.path.join(HERE, 'evaluate.py'), '--ckpt', os.path.join(RUNS, lab, 'final.pt'), '--label',
                lab + '_randpos', '--sets', 'std', '--pos-eval', 'rand', '--gpu', 'auto', '--gpu-wait', str(GPU_WAIT)], \
            os.path.join(EVAL, lab + '_randpos')


def ready(stage, v, s, held):
    lab = label(v, s)
    if done(stage, lab):
        return False
    if stage == 'pretrain':
        return True
    if stage == 'rl':
        return done('pretrain', lab) and v not in held
    if stage == 'eval':
        return done('rl', lab)
    if stage == 'randpos':
        return v == 'posrand' and done('eval', lab)


def live_pid(stage, v, s):
    """pid of a live process already doing this job (started by an earlier launcher or by hand), else None."""
    lab = label(v, s)
    needles = {'pretrain': ['pretrain.py', f'--variant {v} --seed {s} '], 'rl': ['harness.py', f'runs/{lab} '],
               'eval': ['evaluate.py', f'--label {lab} '], 'randpos': ['evaluate.py', f'--label {lab}_randpos ']}[stage]
    for p in os.listdir('/proc'):
        if not p.isdigit() or int(p) == os.getpid():
            continue
        try:
            cmd = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace') + ' '
        except OSError:
            continue
        if cmd.startswith(PY) and all(n in cmd for n in needles):
            return int(p)
    return None


class Adopted:
    """A job found running at start-up, tracked by pid (Popen-like poll / returncode)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        if self.returncode is None and not os.path.exists(f'/proc/{self.pid}'):
            self.returncode = 0
        return self.returncode


def start(stage, v, s, dry):
    lab = label(v, s)
    if stage == 'rl':
        if not os.path.isfile(os.path.join(FROZEN_DIR, lab, 'pretrain.pt')):
            subprocess.run([PY, os.path.join(HERE, 'make_frozen.py'), '--label', lab], check=True)
        out = os.path.join(RUNS, lab)
        if os.path.exists(out):
            aside = f'{out}.failed.{time.strftime("%Y%m%d-%H%M%S")}'
            os.rename(out, aside)
            log(f'rl {lab}: previous attempt moved to {aside}')
    cmd, out = command(stage, v, s)
    log(f'start {stage} {lab}: ' + ' '.join(cmd))
    if dry:
        return None
    os.makedirs(out, exist_ok=True)
    logf = os.path.join(out, 'log.txt') if stage != 'rl' else os.path.join(RUNS, f'{lab}.log')
    env = dict(os.environ, AB_WORKER='repr', OMP_NUM_THREADS='8')
    return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=open(logf, 'a'), stderr=subprocess.STDOUT, start_new_session=True)


def cmd_run(a):
    variants = [v for v in ORDER if v in a.variants.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    held = set(filter(None, a.hold_rl.split(',')))
    stages = [st for st in STAGES if st in a.stages.split(',')]
    jobs = [(st, v, s) for s in seeds for v in variants for st in stages]
    rank = {st: i for i, st in enumerate(STAGES)}
    running, failed = {}, set()
    log(f'run: variants {variants}, seeds {seeds}, stages {stages}, hold RL {sorted(held) or "none"}, max in flight {a.max_inflight}')
    last_status = 0
    while True:
        for key, p in list(running.items()):
            if p.poll() is not None:
                st, v, s = key
                ok = done(st, label(v, s))
                log(f'end {st} {label(v, s)}: exit {p.returncode}, {"done" if ok else "NOT DONE"}')
                if not ok:
                    failed.add(key)
                del running[key]
        todo = [j for j in jobs if j not in running and j not in failed and ready(*j, held)]
        todo.sort(key=lambda j: (j[2], -rank[j[0]], ORDER.index(j[1])))
        for j in list(todo):                       # adopt jobs that are already running (they count as in flight)
            pid = live_pid(*j)
            if pid:
                running[j] = Adopted(pid)
                todo.remove(j)
                log(f'adopt {j[0]} {label(j[1], j[2])}: already running (pid {pid})')
        while todo and len(running) < a.max_inflight:
            j = todo.pop(0)
            p = start(*j, a.dry_run)
            if a.dry_run:
                failed.add(j)
                continue
            running[j] = p
            time.sleep(3)          # launch spacing keeps the FIFO order
        if time.time() - last_status > 60:
            last_status = time.time()
            write_json(os.path.join(LOGS, 'launch_status.json'),
                       {'t': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'running': [f'{st} {label(v, s)}' for st, v, s in running],
                        'failed': [f'{st} {label(v, s)}' for st, v, s in failed],
                        'done': [f'{st} {label(v, s)}' for st, v, s in jobs if done(st, label(v, s))]})
        if not running and not todo:
            break
        time.sleep(10)
    rest = [f'{st} {label(v, s)}' for st, v, s in jobs if not done(st, label(v, s)) and (st != 'randpos' or v == 'posrand')]
    log(f'run: finished; failed {[f"{st} {label(v, s)}" for st, v, s in failed] or "none"}; not done {rest or "none"}')
    return 1 if failed else 0


def cmd_status(a):
    for v in ORDER:
        for s in SEEDS:
            lab = label(v, s)
            pt = json.load(open(os.path.join(PRETRAIN_DIR, lab, 'summary.json'))) if done('pretrain', lab) else {}
            rl = json.load(open(os.path.join(RUNS, lab, 'summary.json'))) if done('rl', lab) else {}
            print(f'{lab:12s} pretrain {"ok" if pt else "-":3s} heldout039 {pt.get("heldout_greedy_039", 0):.3f}  '
                  f'rl {"ok" if rl else "-":3s} rl_secs {rl.get("rl_secs") or 0:5.0f} heldout {rl.get("heldout_greedy") or 0:.3f}  '
                  f'eval units {n_units(os.path.join(EVAL, lab))}' + (f' randpos {n_units(os.path.join(EVAL, lab + "_randpos"), ("std",))}' if v == 'posrand' else ''))
    subprocess.run([PY, '/scratch/cowifornia/nd-rl/ab3/frozen/gpulock.py', 'status'])


def cmd_analyse(a):
    uv = '/home/cowifornia/.local/bin/uv'
    return subprocess.run([uv, 'run', '--script', os.path.join(HERE, 'analyse.py')], cwd=HERE).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'status', 'analyse'))
    ap.add_argument('--variants', default=','.join(ORDER))
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--hold-rl', default='')
    ap.add_argument('--max-inflight', type=int, default=4)
    ap.add_argument('--stages', default=','.join(STAGES), help='restrict to these stages (default: all)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    return {'run': cmd_run, 'status': cmd_status, 'analyse': cmd_analyse}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main() or 0)
