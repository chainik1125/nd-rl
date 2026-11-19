# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import itertools, sys, random
from collections import Counter
import census as C
from matrices import B_, C_, I_, W_, K_


# all 3-element and sampled 4-element matrices validating B, C, I, W and closed under MP, but not K
def kfree(n, tables):
    V = range(n)
    out = []
    for flat in tables:
        i = [flat[k * n : (k + 1) * n] for k in range(n)]
        for r in range(1, n):
            for D in itertools.combinations(V, r):
                D = set(D)
                if not all(not (a in D and i[a][b] in D) or b in D for a in V for b in V):
                    continue
                if not all(i[a][a] in D for a in V):
                    continue
                if not all(W_(i, 0, a, b) in D for a in V for b in V):
                    continue
                if not all(C_(i, 0, a, b, c) in D for a in V for b in V for c in V):
                    continue
                if not all(B_(i, 0, a, b, c) in D for a in V for b in V for c in V):
                    continue
                if all(K_(i, 0, a, b) in D for a in V for b in V):
                    continue
                out.append((i, D))
    return out


M3 = kfree(3, itertools.product(range(3), repeat=9))
print("3-element K-free BCIW matrices:", len(M3))
rng = random.Random(0)
M4 = kfree(4, (tuple(rng.randrange(4) for _ in range(16)) for _ in range(400000)))
print("4-element K-free BCIW matrices from 400k random tables:", len(M4))
ALL = M3 + M4


def refuted(f, na):
    def ev(f, v, i):
        return v[f] if not isinstance(f, tuple) else i[ev(f[0], v, i)][ev(f[1], v, i)]

    for i, D in ALL:
        n = len(i)
        if any(ev(f, v, i) not in D for v in itertools.product(range(n), repeat=na)):
            return True
    return False


C.refuted_relevant = refuted
NMAX = int(sys.argv[1])
tot = Counter()
ex = {}
for n in range(2, NMAX + 1):
    cnt = Counter()
    for sh in C.shapes(n):
        for lab in C.labelings(n, 4):
            f = C.fill(sh, iter(lab))
            na = max(lab) + 1
            if not C.classical(f, na):
                cnt["not a tautology"] += 1
                continue
            if not C.int_valid(f):
                c = "classical only"
            else:
                lin = C.prove(f, "BCI", 7)
                aff = C.prove(f, "BCK", 7)
                rel = C.prove(f, "BCIW", 7)
                if lin:
                    c = "needs neither (BCI)"
                elif aff and rel:
                    c = "K alone or W alone suffices"
                elif aff:
                    c = "needs K only" if refuted(f, na) else "needs K only (uncertified)"
                elif rel:
                    c = "needs W only"
                else:
                    c = "needs K and W" if refuted(f, na) else "needs W, K uncertified"
            cnt[c] += 1
            ex.setdefault(c, []).append(f)
    tot += cnt
    print(n, dict(sorted(cnt.items())), flush=True)
print("TOTAL", dict(sorted(tot.items())))
for c, fs in ex.items():
    print(f"{c:32s}", " | ".join(C.show(f) for f in fs[:3]))
