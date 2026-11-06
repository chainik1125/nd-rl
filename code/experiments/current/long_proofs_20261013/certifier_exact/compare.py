"""Same-load timing: for each theorem run minlen (certify.py's search), minlen_sem and certify_fast back to back in
one process, same floor.  python compare.py OUT.jsonl N [--secs 120]"""
import argparse, json, os, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'approach_d_theorem_first'))
from certify_fast import run_fast
from minlen import minlen, parse_thm
from minlen_sem import minlen_sem
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('n', type=int)
ap.add_argument('--secs', type=float, default=120); a = ap.parse_args()
rs = [json.loads(l) for l in open(os.path.join(HERE, 'logs', 'val_v1.jsonl'))]
pick = [r for r in rs if r['verdict'] == 'ok' and 5 <= r['minlen_secs'] <= 45]
pick.sort(key=lambda r: r['minlen_secs'])
pick = pick[::max(1, len(pick) // a.n)][:a.n]
with open(a.out, 'w') as fo:
    for r in pick:
        p, floor = r['prompt'], r['floor']
        n_prem = len(parse_thm(p)[0])
        row = {'prompt': p, 'floor': floor, 'label': r['label']}
        t = time.time(); m = minlen(p, bound=n_prem + floor - 1, time_limit=a.secs)
        row['minlen'] = [round(time.time() - t, 2), m['min_lines_ub'] and m['min_lines_ub'] - n_prem, m['timeout'], m['calls']]
        t = time.time(); m = minlen_sem(p, bound=n_prem + floor - 1, time_limit=a.secs)
        row['minlen_sem'] = [round(time.time() - t, 2), m['min_lines_ub'] and m['min_lines_ub'] - n_prem, m['timeout'], m.get('memo')]
        t = time.perf_counter(); x = run_fast(p, floor, a.secs, 2048)
        row['fast'] = [round(max(time.perf_counter() - t, 1e-5), 5), x.get('body'), x['status'], x['stats']['entries']]
        row['speedup_vs_minlen'] = round(row['minlen'][0] / row['fast'][0], 1)
        row['speedup_vs_sem'] = round(row['minlen_sem'][0] / row['fast'][0], 1)
        fo.write(json.dumps(row) + '\n'); fo.flush()
        print(row, flush=True)
