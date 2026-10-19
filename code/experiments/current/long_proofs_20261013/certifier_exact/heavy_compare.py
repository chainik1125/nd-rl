"""One heavy theorem, same load: minlen_sem, minlen, certify_fast back to back.
    python heavy_compare.py OUT 'THM ...' FLOOR CAP_SECS [--skip-minlen]   (minlen can need > 4 GB)"""
import json, os, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'approach_d_theorem_first'))
from certify_fast import run_fast
from minlen import minlen, parse_thm
from minlen_sem import minlen_sem
out, p, floor, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
n_prem = len(parse_thm(p)[0])
rows = {}
t = time.time(); x = run_fast(p, floor, cap, 2048); rows['fast'] = [round(time.time() - t, 3), x.get('body'), x['status']]
print(rows, flush=True)
t = time.time(); m = minlen_sem(p, bound=n_prem + floor - 1, time_limit=cap)
rows['minlen_sem'] = [round(time.time() - t, 1), m['min_lines_ub'] and m['min_lines_ub'] - n_prem, m['timeout'], m.get('memo')]
print(rows, 'maxrss_GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, flush=True)
if '--skip-minlen' in sys.argv[5:]:
    m = {'min_lines_ub': None, 'timeout': None, 'calls': None}; t = time.time()
else:
    t = time.time(); m = minlen(p, bound=n_prem + floor - 1, time_limit=cap)
rows['minlen'] = [round(time.time() - t, 1), m['min_lines_ub'] and m['min_lines_ub'] - n_prem, m['timeout'], m['calls']]
t = time.time(); x = run_fast(p, floor, cap, 2048); rows['fast_again'] = [round(time.time() - t, 3), x.get('body'), x['status']]
print(rows, 'maxrss_GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, flush=True)
json.dump({'prompt': p, 'floor': floor, **rows}, open(out, 'w'))
