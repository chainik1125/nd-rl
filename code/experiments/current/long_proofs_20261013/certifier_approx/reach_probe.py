"""Reach probe: deepen one hunter on each prompt in a jsonl, reporting per-level secs and memo.
   python reach_probe.py IN.jsonl LEM LO HI CAP SECS"""
import json, sys, os, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hunter import hunt
path, lem, lo, hi, cap, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
for line in open(path):
    d = json.loads(line)
    r = hunt(d['prompt'], lo, hi, secs=secs, lem=lem, memo_cap=cap)
    print(json.dumps({'key': d['key'], 'lem': lem, 'found': r['found'], 'exh': r['exh'], 'stop': r['stop'], 'secs': r['secs'],
                      'memo': r['memo'], 'per_level': r['per_level'], 'maxrss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 20}), flush=True)
