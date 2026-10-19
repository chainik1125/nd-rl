"""pkey: premise-order-invariant class key = min over premise permutations of keys.canon_key."""
import itertools, sys
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab/common/lib')
from keys import canon_key


def split_thm(thm):
    lhs, rhs = thm.strip().split('|-')
    prem = [p.strip() for p in lhs.split(' , ')] if lhs.strip() else []
    return prem, rhs.strip()


def pkey(thm):
    prem, concl = split_thm(thm)
    assert len(prem) <= 6, thm
    best = None
    for perm in itertools.permutations(prem):
        k = canon_key((' , '.join(perm) + ' |- ' + concl).strip())
        if best is None or k < best:
            best = k
    return best
