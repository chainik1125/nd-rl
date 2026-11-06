"""Run minlen at certify's bound (n_prem+floor-1) on {"prompt"} jsonl; append min_body/timeout/secs/|S|.
    python screen.py IN.jsonl OUT.jsonl SECS [PROCS] [FLOOR]
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import minlen, parse_thm, Search

src, dst, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
procs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
floor = int(sys.argv[5]) if len(sys.argv) > 5 else 20


def work(r):
    prem, concl = parse_thm(r['prompt'])
    t0 = time.time()
    m = minlen(r['prompt'], bound=len(prem) + floor - 1, time_limit=secs)
    mb = m['min_lines_ub'] - len(prem) if m['min_lines_ub'] else None
    return {**r, 'n_prem': len(prem), 'S': len(Search(prem, concl).S), 'min_body': mb, 'timeout': m['timeout'],
            'secs': round(time.time() - t0, 2), 'calls': m['calls'], 'proof': m['proof']}


if __name__ == '__main__':
    rs = [json.loads(l) for l in open(src)]
    with ProcessPoolExecutor(procs) as ex, open(dst, 'w') as f:
        for x in ex.map(work, rs, chunksize=1):
            f.write(json.dumps(x) + '\n'); f.flush()
            print(f"S={x['S']:3d} np={x['n_prem']} min_body={x['min_body']} to={x['timeout']} {x['secs']:6.1f}s {x['prompt'][:100]}", flush=True)
