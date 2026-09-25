#!/usr/bin/env python3
"""Run a command on one GPU taken through frozen/gpulock.py (FIFO queue), then release it.

    $PY tests/with_gpu.py [--name NAME] -- cmd args...
The command sees CUDA_VISIBLE_DEVICES=<gpu>; harness / evaluate / pretrain calls inside it use --gpu env.
"""
import argparse, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_repr import Lease  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='repr_tests')
ap.add_argument('cmd', nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
os.environ.setdefault('AB_WORKER', 'repr')
lease = Lease(a.name)
try:
    lease.take('auto')
    rc = subprocess.run(cmd, env=os.environ.copy()).returncode
finally:
    lease.drop()
sys.exit(rc)
