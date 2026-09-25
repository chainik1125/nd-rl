#!/usr/bin/env python3
"""Run the goals variant end to end: per pretraining seed s (0, 1, 2) one chain of GPU jobs, each through the gpulock
FIFO queue (every job takes its own lock; at most --chains chains at a time, so at most that many of this
experiment's jobs hold or wait on the lock):

  1. pretrain.py --seed s                                   -> pretrain/goals_s<s>/{pretrain.pt, summary.json}
  2. make_frozen.py --seed s (CPU)                          -> frozen/goals_s<s>/ (harness copy + pretrain.pt)
  3. frozen/goals_s<s>/harness.py --rl rl_goals.py --seed RL_SEED[s] --no-score --gpu auto
                                                            -> runs/goals_s<s>/ (final.pt, metrics.jsonl, summary.json)
  4. evaluate.py --ckpt runs/goals_s<s>/final.pt --label goals_s<s>          -> eval/goals_s<s>/
  and at the end: evaluate.py on pretrain/goals_s0/pretrain.pt --label pretrain_goals_s0 (the pretrained anchor)

A step whose output says ok is skipped (re-running resumes). Logs: logs/<step>_<label>.log; progress lines are
appended to STATUS.md.

    python launch.py all [--chains 2] [--dry-run]
    python launch.py status
"""
import argparse, datetime, json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_goals import EVAL, FROZEN_DIR, LOGS, PRETRAIN_DIR, PY, RL_SEED, RUNS, SEEDS, label

LOCK = threading.Lock()
DRY = []          # set by --dry-run: status lines are printed only


def status_line(msg):
    line = f'- {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")} {msg}\n'
    with LOCK:
        if not DRY:
            with open(os.path.join(HERE, 'STATUS.md'), 'a') as f:
                f.write(line)
    print(line, end='', flush=True)


def ok_json(fn, key='status', val='ok'):
    try:
        return json.load(open(fn)).get(key) == val
    except (OSError, ValueError):
        return False


def eval_done(lab, n_units=22):
    d = os.path.join(EVAL, lab)
    return os.path.isdir(d) and sum(1 for f in os.listdir(d) if f.endswith('.summary.json')) >= n_units


def run(cmd, logname, dry):
    if dry:
        print('DRY', ' '.join(cmd), '>', logname)
        return 0, 0.0
    t0 = time.time()
    with open(os.path.join(LOGS, logname), 'a') as f:
        f.write(f'\n### {datetime.datetime.now(datetime.timezone.utc).isoformat()} {" ".join(cmd)}\n')
        f.flush()
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, AB_WORKER='goals'))
    return p.returncode, time.time() - t0


def chain(s, dry):
    lab = label(s)
    pt_dir = os.path.join(PRETRAIN_DIR, lab)
    if not ok_json(os.path.join(pt_dir, 'summary.json')):
        status_line(f'{lab}: pretraining started (queued on the lock)')
        rc = run([PY, 'pretrain.py', '--seed', str(s), '--gpu', 'auto', '--gpu-wait', '43200'], f'pretrain_{lab}.log', dry)
        if rc[0] != 0 or (not dry and not ok_json(os.path.join(pt_dir, 'summary.json'))):
            status_line(f'{lab}: pretraining FAILED (exit {rc[0]}); see logs/pretrain_{lab}.log')
            return
        if not dry:
            sm = json.load(open(os.path.join(pt_dir, 'summary.json')))
            status_line(f"{lab}: pretraining done in {rc[1] / 60:.0f} min ({sm['steps']} steps, {sm['steps_per_sec']:.2f}/s, "
                        f"val {sm['val_loss']:.4f}, heldout_greedy {sm['heldout_greedy']:.3f}, 039 slice {sm['heldout_greedy_039']:.3f})")
    fz = os.path.join(FROZEN_DIR, lab)
    if not os.path.isfile(os.path.join(fz, 'pretrain.pt')) and not dry:
        subprocess.run([PY, 'make_frozen.py', '--seed', str(s)], cwd=HERE, check=True)
    run_dir = os.path.join(RUNS, lab)
    if not ok_json(os.path.join(run_dir, 'summary.json')):
        if os.path.isdir(run_dir) and not dry:
            os.replace(run_dir, run_dir + f'.failed_{int(time.time())}')
        status_line(f'{lab}: RL started (arm C, seed {RL_SEED[s]}, queued on the lock)')
        rc = run([PY, os.path.join(fz, 'harness.py'), '--rl', os.path.join(HERE, 'rl_goals.py'), '--out', run_dir,
                  '--seed', str(RL_SEED[s]), '--no-score', '--gpu', 'auto', '--gpu-wait', '43200'], f'rl_{lab}.log', dry)
        if not dry and not ok_json(os.path.join(run_dir, 'summary.json')):
            status_line(f'{lab}: RL FAILED (exit {rc[0]}); see logs/rl_{lab}.log')
            return
        if not dry:
            sm = json.load(open(os.path.join(run_dir, 'summary.json')))
            rs = sm.get('rl_stats', {})
            status_line(f"{lab}: RL done in {rc[1] / 60:.0f} min ({rs.get('rounds')} rounds, {rs.get('total_samples')} samples, "
                        f"{sm['compute']['frac']:.1%} of the compute, rl_secs {sm['rl_secs']:.0f}, heldout_greedy {sm['heldout_greedy']:.3f})")
    if not eval_done(lab):
        status_line(f'{lab}: evaluation started (queued on the lock)')
        rc = run([PY, 'evaluate.py', '--ckpt', os.path.join(run_dir, 'final.pt'), '--label', lab, '--gpu', 'auto',
                  '--gpu-wait', '43200'], f'eval_{lab}.log', dry)
        if not dry:
            status_line(f'{lab}: evaluation {"done" if eval_done(lab) else "FAILED"} in {rc[1] / 60:.0f} min')


def eval_anchor(dry):
    lab = 'pretrain_goals_s0'
    if eval_done(lab):
        return
    status_line(f'{lab}: evaluation of the pretrained checkpoint (anchor) started (queued on the lock)')
    rc = run([PY, 'evaluate.py', '--ckpt', os.path.join(PRETRAIN_DIR, label(0), 'pretrain.pt'), '--label', lab,
              '--gpu', 'auto', '--gpu-wait', '43200'], f'eval_{lab}.log', dry)
    if not dry:
        status_line(f'{lab}: evaluation {"done" if eval_done(lab) else "FAILED"} in {rc[1] / 60:.0f} min')


def cmd_all(a):
    os.makedirs(LOGS, exist_ok=True)
    if a.dry_run:
        DRY.append(True)
    sem = threading.Semaphore(a.chains)
    threads = []

    def guarded(fn, *args):
        with sem:
            fn(*args)

    for s in SEEDS:
        t = threading.Thread(target=guarded, args=(chain, s, a.dry_run))
        t.start()
        threads.append(t)
        time.sleep(2)
    for t in threads:
        t.join()
    if os.path.isfile(os.path.join(PRETRAIN_DIR, label(0), 'pretrain.pt')) or a.dry_run:
        eval_anchor(a.dry_run)       # the pretrained-checkpoint anchor, last (lowest priority)
    status_line('all chains finished')


def cmd_status(a):
    for s in SEEDS:
        lab = label(s)
        print(lab, 'pretrain', ok_json(os.path.join(PRETRAIN_DIR, lab, 'summary.json')),
              'rl', ok_json(os.path.join(RUNS, lab, 'summary.json')), 'eval', eval_done(lab))
    print('anchor eval', eval_done('pretrain_goals_s0'))
    subprocess.run([PY, '/scratch/cowifornia/nd-rl/ab3/frozen/gpulock.py', 'status'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('all', 'status'))
    ap.add_argument('--chains', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    {'all': cmd_all, 'status': cmd_status}[a.cmd](a)


if __name__ == '__main__':
    main()
