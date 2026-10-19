"""Per-level cost of certify_fast on given theorems:  python probe.py FLOOR SECS MEM_MB 'THM ...' ['THM ...' ...]
Prints, for each body budget b, the cumulative seconds and memo entries after exhausting every proof with <= b body lines."""
import sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from certify_fast import run_fast

floor, secs, mem = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for p in sys.argv[4:]:
    r = run_fast(p, floor, secs, mem)
    st = r['stats']
    print(f"{r['status']} body={r.get('body')} secs={r['secs']:.2f} |S|={st['S']} nw={st['nw']} abits={st['abits']} "
          f"entries={st['entries']} evictions={st['evictions']} calls={st['calls']} {p}", flush=True)
    for b, t, e in st['levels'][-6:]:
        print(f'   b<={b:2d} {t:8.2f}s entries={e}', flush=True)
