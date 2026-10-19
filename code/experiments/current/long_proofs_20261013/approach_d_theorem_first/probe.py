"""Deepen minlen level by level on each prompt in a file (one per line), up to body MAXBODY, giving up when
time per prompt exceeds SECS or the memo exceeds MAXMEMO entries. Reports per-level time, memo size, result.
    python probe.py PROMPTS.txt MAXBODY SECS MAXMEMO"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import Search, parse_thm, Timeout
path, maxbody, secs, maxmemo = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
for prompt in open(path):
    prompt = prompt.strip()
    if not prompt or prompt.startswith('#'):
        continue
    prem, concl = parse_thm(prompt)
    s = Search(prem, concl, deadline=time.time() + secs)
    print(f'|S|={len(s.S)} np={len(prem)} {prompt}', flush=True)
    t0 = time.time(); out = 'exhausted'
    try:
        for body in range(1, maxbody + 1):
            t1 = time.time()
            r = s.best(concl, frozenset(prem), body)
            print(f'  body<={body:2d}: {time.time()-t1:7.2f}s memo={len(s.memo):9d} found={r[0] if r else None}', flush=True)
            if r:
                out = f'min_body={r[0]}'; break
            if len(s.memo) > maxmemo:
                out = 'memo cap'; break
    except Timeout:
        out = 'timeout'
    print(f'  => {out} total {time.time()-t0:.1f}s memo={len(s.memo)}', flush=True)
    del s
