"""minlen with semantic pruning: a state (G, A) where A does not entail G (truth table over the theorem's atoms)
can have no proof, so it fails immediately instead of being searched to the budget. Sound because ND is sound.
Same search space and the same proofs as vendor/fork/minlen.py otherwise (subclass; only `best` is changed).
NOT the official scoreboard: certify.py numbers come from the unmodified minlen.

    from minlen_sem import minlen_sem   # same signature/return as minlen.minlen
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C  # noqa: E402  (sets up nd_verify atoms, imports minlen)
from minlen import Search, Timeout, parse_thm  # noqa: E402
from nd_verify import verify_text  # noqa: E402


class SemSearch(Search):
    def __init__(self, prem, concl, **kw):
        super().__init__(prem, concl, **kw)
        names = sorted(set().union(*[C.atoms(f, set()) for f in prem + [concl]]))
        self.mask = {}
        rows = []
        for i in range(1 << len(names)):
            rows.append({n: bool(i >> j & 1) for j, n in enumerate(names)})
        self.rows = rows
        self.full = (1 << len(rows)) - 1
        self.amask = {}
        self.pruned = 0
        self.aid, self.gid = {}, {}   # intern A sets / goals so memo keys are one small int (less memory)

    def fmask(self, f):
        m = self.mask.get(f)
        if m is None:
            if f[0] == 'not':
                m = self.full & ~self.fmask(f[1])
            else:
                m = 0
                for i, val in enumerate(self.rows):
                    if C.ev(f, val):
                        m |= 1 << i
            self.mask[f] = m
        return m

    def entails(self, A, G):
        ma = self.amask.get(A)
        if ma is None:
            ma = self.full
            for f in A:
                ma &= self.fmask(f)
            self.amask[A] = ma
        return ma & ~self.fmask(G) == 0

    def best(self, G, A, b, d=0, H=frozenset()):
        if b <= 0:
            return None
        if self.track_h:
            key = (G, A, b, d, H)
        else:
            ai = self.aid.get(A)
            if ai is None:
                ai = self.aid[A] = len(self.aid)
            gi = self.gid.get(G)
            if gi is None:
                gi = self.gid[G] = len(self.gid)
            key = ((ai << 20 | gi) << 7 | b) << 4 | d
        if key in self.memo:
            return self.memo[key]
        self.calls += 1
        if self.deadline and (self.calls & 1023) == 0 and time.time() > self.deadline:
            raise Timeout()
        if G in A:
            res = (1, ('R', G))
        elif not self.entails(A, G):
            self.pruned += 1
            res = None
        else:
            res = self._search(G, A, b, d, H)
        self.memo[key] = res
        return res


def minlen_sem(prompt, bound=8, time_limit=20.0, **kw):
    prem, concl = parse_thm(prompt)
    s = SemSearch(prem, concl, deadline=time.time() + time_limit if time_limit else None, **kw)
    try:
        r = s.run(bound)
    except Timeout:
        return {'min_lines_ub': None, 'proof': None, 'timeout': True, 'bound': bound, 'calls': s.calls, 'memo': len(s.memo)}
    if r is None:
        return {'min_lines_ub': None, 'proof': None, 'timeout': False, 'bound': bound, 'calls': s.calls, 'memo': len(s.memo)}
    n, text = r
    ok, reason, nl = verify_text(prompt + ' ' + text)
    if not ok or nl != n:
        return {'min_lines_ub': None, 'proof': text, 'timeout': False, 'bound': bound, 'error': f'{reason} nl={nl} n={n}', 'calls': s.calls}
    return {'min_lines_ub': n, 'proof': text, 'timeout': False, 'bound': bound, 'calls': s.calls, 'memo': len(s.memo)}


if __name__ == '__main__':
    # per-level probe, like probe.py:  python minlen_sem.py 'THM ...' MAXBODY [MAXMEMO]
    prompt, maxbody = sys.argv[1], int(sys.argv[2])
    maxmemo = int(sys.argv[3]) if len(sys.argv) > 3 else 8_000_000
    prem, concl = parse_thm(prompt)
    s = SemSearch(prem, concl)
    print(f'|S|={len(s.S)} np={len(prem)} {prompt}', flush=True)
    t0 = time.time()
    for body in range(1, maxbody + 1):
        t1 = time.time()
        r = s.best(concl, frozenset(prem), body)
        print(f'  body<={body:2d}: {time.time()-t1:7.2f}s memo={len(s.memo):9d} pruned={s.pruned} found={r[0] if r else None}', flush=True)
        if r or len(s.memo) > maxmemo:
            break
    print(f'  total {time.time()-t0:.1f}s', flush=True)
    if r:
        n, text = len(prem) + r[0], s.to_text(r[1])
        print('  verify:', verify_text(prompt + ' ' + text)[:2], 'lines', n)
