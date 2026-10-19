#!/usr/bin/env python3
"""Arm D follow-up: when each D training run (runs/D_s10x, frozen_comp harness) finishes with status ok, queue its
evaluation (evaluate.py, all units) and, for D_s101, the composition part evaluation on composition/final.
Everything goes through the GPU lock (--gpu auto). Log: logs/watch_d.log.

    nohup python watch_d.py > logs/watch_d.log 2>&1 &
"""
import json, os, subprocess, sys, time

LE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/cowifornia/code/research/nd-rl/code/.venv/bin/python'
LABELS = ['D_s101', 'D_s102', 'D_s103']


def status(l):
    try:
        return json.load(open(os.path.join(LE, 'runs', l, 'summary.json'))).get('status')
    except (OSError, ValueError):
        return None


def running(l):
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                c = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if 'harness.py' in c and f'runs/{l} ' in c + ' ':
                return True
    return False


def launch(cmd, log):
    env = dict(os.environ, AB_WORKER='lengthexp')
    print(time.strftime('%H:%M:%S'), 'launch', ' '.join(cmd), flush=True)
    return subprocess.Popen(cmd, cwd=LE, env=env, stdout=open(os.path.join(LE, log), 'a'), stderr=subprocess.STDOUT,
                            start_new_session=True)


done = set()
while len(done) < len(LABELS):
    for l in LABELS:
        if l in done:
            continue
        st = status(l)
        if st in ('ok', 'timeout', 'crash') and not running(l):
            done.add(l)
            print(time.strftime('%H:%M:%S'), l, 'finished with status', st, flush=True)
            if st == 'ok':
                ck = os.path.join('runs', l, 'final.pt')
                launch([PY, 'evaluate.py', '--ckpt', ck, '--label', l, '--gpu', 'auto', '--gpu-wait', '43200'], f'eval/{l}.log')
                if l == 'D_s101':
                    time.sleep(3)
                    launch([PY, 'composition/run_eval.py', '--dir', 'composition/final', '--ckpt', ck, '--label', l,
                            '--gpu', 'auto', '--gpu-wait', '43200'], f'composition/logs/final_{l}.log')
    time.sleep(20)
print(time.strftime('%H:%M:%S'), 'all D runs handled', flush=True)
