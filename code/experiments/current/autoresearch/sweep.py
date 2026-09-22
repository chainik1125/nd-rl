#!/usr/bin/env python3
"""Run every (config, seed) pair on its own pod, all at once. One pod per run: up, run, down.

  python3 sweep.py <sweep-name> <seeds> <config.py>...      e.g.  python3 sweep.py s20 0,1,2 sweep/*.py

Runs land in artifacts/autoresearch/<sweep-name>_<config>_s<seed>/ and each pod's driver log in
artifacts/autoresearch/<sweep-name>_<config>_s<seed>.driver.log. Pods are terminated in `finally`.
"""
import os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, '..', '..', '..', '..', 'artifacts', 'autoresearch')


def one(sweep, cfg, seed):
    name = f'{sweep}_{os.path.basename(cfg)[:-3]}_s{seed}'
    log = open(os.path.join(ART, f'{name}.driver.log'), 'w')
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, 'pod.py'), *a, '--pod', name], stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        if run('up').returncode != 0:
            print(f'{name}: pod up FAILED', flush=True)
            return
        run('run', name, '--pretrain', cfg, '--seed', str(seed))
        print(f'{name}: done after {(time.time() - t0) / 60:.0f} min', flush=True)
    finally:
        run('down')


def main():
    sweep, seeds, cfgs = sys.argv[1], [int(s) for s in sys.argv[2].split(',')], sys.argv[3:]
    os.makedirs(ART, exist_ok=True)
    threads = [threading.Thread(target=one, args=(sweep, c, s)) for c in cfgs for s in seeds]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(3)   # stagger the RunPod API calls
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
