#!/usr/bin/env python3
"""Replay every attempt of a pool file (pool rows only) and count mismatches: per-step state sha1, steps, done, and
the assembled nd byte for byte for done attempts.   python check_replay.py <pool.jsonl> [...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL
for fn in sys.argv[1:]:
    n = bad = nd = 0
    for row in PL.read_jsonl(fn):
        for att in row['attempts']:
            n += 1; nd += att['done']
            if not PL.check_attempt(row['prompt'], att):
                bad += 1
    r = dict(pool=os.path.basename(fn), attempts=n, done=nd, mismatches=bad)
    json.dump(r, open(fn[:-6] + ".replay.json", "w"))
    print(json.dumps(r), flush=True)
