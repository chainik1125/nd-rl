"""Signature lower bound: a sound lower bound on the body length of every Fitch proof of a sequent.

CLAIMED SPACE (the "nu-system"). nu rewrites every ~A to (A > F) (Lean's definitional unfolding of Not). A proof is a
Fitch proof whose lines satisfy nd_verify's rule shapes on nu-formulas (NEGE read as IMPE with major (A > F), NEGI as
IMPI, DN as "from ((G > F) > F) infer G") and in which every cited line AND every cited box ends before the citing
line. This contains (1) every nd_verify-accepted proof whose boxes are cited after they close and (2) minlen's /
certify_fast's whole search space. (3) The bound also holds for every proof the Lean gate accepts (nd2lean at
nd-takehome 51604b3 + Lean), by the same induction run over the EMITTED TERM instead of the source lines: each term
node is a rule application labelled by one source line, with that line's (rule, principal nu-formula) (NEGE is
emitted as plain application = IMPE with major refs[1]); Lean typing is checking modulo nu (Not is the only unfolding
among these Props); boxes are lambdas, so scoping and backward citation hold in the term. A box cited twice repeats
its signatures; boxes the translator drops are unchecked extra source lines. (The gate is looser than the
nu-system: it accepts dead boxes with invalid lines and boxes cited from outside their parent context; Codex review
2026-10-14.) The bound does NOT hold for all nd_verify-accepted proofs: box_citable has no "box before the citing
line" check, and accepts the circular
  THM P SEQ Q PRF N1 P : PR ; N2 ( P > Q ) : IMPI N3 N4 ; N3 | P : AS ; N4 | Q : IMPE N2 N3 ; N5 Q : IMPE N2 N1 ; QED
(an invalid sequent "proved" in 4 lines; Codex found the same trick beats the bound on valid sequents).

ARGUMENT. Fix a non-atomic nu-subformula X of the sequent and a fresh parameter z. I interprets every nu-formula
classically (I(F) = 0) except X, whose value is the perturbation below applied to I(children) and z; formulas
containing X stay compositional over I(X). Each rule's truth preservation uses the clause of one principal formula:
  ANDI: I(A)&I(B) <= I(A&B) [conclusion]        ANDE1/2: I(A&B) <= I(A) / I(B) [major]
  IMPE (+NEGE): I(A>B)&I(A) <= I(B) [major]      IMPI (+NEGI): ~I(A) v I(B) <= I(A>B) [conclusion]
  ORI1/2: I(A) / I(B) <= I(AvB) [conclusion]     ORE: I(AvB) <= I(A) v I(B) [major]
  DN: I((G>F)>F) <= I(G) [((G>F)>F), and the clause of (G>F)]   BOTE: I(F) = 0   R/PR/AS: nothing
A line's signature is (rule, principal nu-formula); a line has at most one. perturbations() lists, per X, the
signatures whose inequality can fail (brute-force checked by test_sigbound.py); all other rule instances stay
truth-preserving.
LEMMA. If the perturbed sequent is invalid (truth table over atoms + z), every proof has a line whose signature is in
the kill set. Proof: else, by induction on line index, every line is I-true at every valuation satisfying the
premises and the hypotheses of the boxes open at that line (cited lines have prefix contexts; a cited box (s, e) is
closed, its parent context is a prefix of the citing line's, and ctx(e) = ctx(s); all cited lines and boxes are
EARLIER, which is where backward citation is used). The last line is at depth 0, so I(premises) |= I(conclusion).
THEOREM. With P a family of pairwise-disjoint kill sets (greedy packing) and H the distinct nu-formulas A, B for
singleton requirements {ORE (A v B)} and A for {IMPI (A > B)}: every proof has >= |P| + |H| body lines. Each set of
P is hit by its own line (one signature per line); a required ORE / IMPI line cites boxes whose AS lines have those
nu-formulas; AS lines have no signature; distinct formulas are distinct lines.
Ignored (only costs tightness): a signature needed in several boxes counts once (E's recycled hypothesis atoms),
BOTE lines, AS lines behind non-singleton kill sets, IMPE minor premises.

    from sigbound import sig_bound
    b = sig_bound(prem, concl)      # b['lb']: every proof in the nu-system has >= lb body lines
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import certify as C  # noqa: E402  (widens nd_verify atoms)
from minlen import parse_thm  # noqa: E402

BOT = ('bot',)


def subformulas(f, acc):
    acc.add(f)
    if f[0] == 'not':
        subformulas(f[1], acc)
    elif f[0] in ('and', 'or', 'imp'):
        subformulas(f[1], acc)
        subformulas(f[2], acc)
    return acc


def nu(f):
    """Normal form modulo Lean's definitional unfolding ~A := (A > F): every negation becomes an implication into F.
    Lean (nd2lean) accepts any line whose formulas agree up to this unfolding, and nd_verify's NEGE / NEGI / DN are
    IMPE / IMPI / DN on the unfolded formulas, so the bound is computed on nu(sequent) with NEGE -> IMPE, NEGI -> IMPI."""
    k = f[0]
    if k == 'not':
        return ('imp', nu(f[1]), BOT)
    if k in ('and', 'or', 'imp'):
        return (k, nu(f[1]), nu(f[2]))
    return f


# Perturbations (on nu-formulas: no negations). Each maps (a, b, z, full) -> I(X), given the children's values a, b
# and the fresh atom z as row bitmasks, together with the signatures it makes unsound. Signatures: (RULE, principal
# nu-formula). Principal: ANDI / IMPI (incl. NEGI) / ORI1 / ORI2 -> the line's own formula; ANDE1 / ANDE2 / IMPE
# (incl. NEGE: its negation premise) / ORE -> the major premise; DN -> the cited ((G > F) > F). DN uses the clauses of
# both ((G > F) > F) and (G > F). Soundness of every row: module docstring and test_sigbound.py (brute force).
def perturbations(X):
    k = X[0]
    if k == 'and':
        return [
            ('e1', lambda a, b, z, m: (a | z) & b, [('ANDE1', X)]),
            ('e2', lambda a, b, z, m: a & (b | z), [('ANDE2', X)]),
            ('i', lambda a, b, z, m: a & b & z, [('ANDI', X)]),
        ]
    if k == 'imp':
        A, B = X[1], X[2]
        dn_e = [('DN', X)] if B == BOT and A[0] == 'imp' and A[2] == BOT else []   # X = ((G > F) > F): I(X) = G v z
        dn_i = [('DN', ('imp', X, BOT))] if B == BOT else []                     # X = (G > F): I((X > F)) = G v ~z
        return [
            ('e', lambda a, b, z, m: (m & ~a) | b | z, [('IMPE', X)] + dn_e),
            ('i', lambda a, b, z, m: ((m & ~a) | b) & z, [('IMPI', X)] + dn_i),
        ]
    if k == 'or':
        return [
            ('e', lambda a, b, z, m: a | b | z, [('ORE', X)]),
            ('i1', lambda a, b, z, m: (a & z) | b, [('ORI1', X)]),
            ('i2', lambda a, b, z, m: a | (b & z), [('ORI2', X)]),
        ]
    return []


def _masks(names):
    """Row bitmasks for the atoms plus z (the last variable)."""
    n = len(names) + 1
    rows = 1 << n
    full = (1 << rows) - 1
    m = {}
    for i, a in enumerate(names + ['#z']):
        bits = 0
        block = 1 << i
        # rows r with bit i set: build by repetition
        pat = ((1 << block) - 1) << block          # block zeros then block ones
        period = 2 * block
        rep = 0
        for s in range(0, rows, period):
            rep |= pat << s
        bits = rep & full
        m[a] = bits
    return m, full


def _eval(f, m, full, X, pert, cache):
    r = cache.get(f)
    if r is not None:
        return r
    k = f[0]
    if k == 'atom':
        v = m[f[1]]
    elif k == 'bot':
        v = 0
    else:
        a = _eval(f[1], m, full, X, pert, cache)
        b = _eval(f[2], m, full, X, pert, cache) if k != 'not' else 0
        if f == X:
            v = pert(a, b, m['#z'], full) & full
        elif k == 'not':
            v = full & ~a
        elif k == 'and':
            v = a & b
        elif k == 'or':
            v = a | b
        else:
            v = (full & ~a) | b
    cache[f] = v
    return v


def _valid(prem, concl, m, full, X, pert):
    cache = {}
    g = full
    for p in prem:
        g &= _eval(p, m, full, X, pert, cache)
    return (g & ~_eval(concl, m, full, X, pert, cache) & full) == 0


def requirements(prem, concl):
    """-> list of (X, kind, kill-signature list) whose perturbed sequent is invalid (X, signatures: nu-formulas)."""
    prem, concl = [nu(p) for p in prem], nu(concl)
    names = sorted(set().union(*[C.atoms(f, set()) for f in prem + [concl]]))
    m, full = _masks(names)
    sub = set()
    for f in prem + [concl]:
        subformulas(f, sub)
    out = []
    for X in sorted(sub, key=C.fstr):
        for kind, pert, kills in perturbations(X):
            if not _valid(prem, concl, m, full, X, pert):
                out.append((X, kind, kills))
    return out


def sig_bound(prem, concl):
    """Lower bound on body lines of every proof of prem |- concl in the claimed space (module docstring).
    lb = (packing of pairwise-disjoint kill sets) + (distinct hypothesis formulas forced by singleton
    requirements: ORE (A v B) needs AS lines A and B, IMPI (A > B) or NEGI ~A needs AS A; distinct
    nu-formulas are distinct lines)."""
    _, valid = C.truth_table(prem, concl)
    assert valid, 'invalid sequent'
    reqs = requirements(prem, concl)
    sets = sorted({frozenset(k) for _, _, k in reqs}, key=len)
    used, packed = set(), []
    for s in sets:                   # greedy packing; any packing is <= the minimum hitting set
        if not (s & used):
            used |= s
            packed.append(s)
    hyps = set()
    for s in sets:
        if len(s) != 1:
            continue
        (rule, X), = s
        if rule == 'ORE':
            hyps |= {X[1], X[2]}
        elif rule == 'IMPI':                 # IMPI or NEGI: its box opens with an AS line whose nu-formula is X[1]
            hyps.add(X[1])
    return {'lb': len(packed) + len(hyps), 'n_sig': len(packed), 'n_hyp': len(hyps), 'n_req': len(reqs)}


def sig_bound_prompt(prompt):
    prem, concl = parse_thm(prompt)
    return sig_bound(prem, concl)


if __name__ == '__main__':
    for p in ['THM ( P > Q ) , P SEQ Q PRF', 'THM ( P & Q ) SEQ ( Q & P ) PRF', 'THM ( P v Q ) SEQ ( Q v P ) PRF',
              'THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF', 'THM ( ~ ( ~ P ) ) SEQ P PRF']:
        print(sig_bound_prompt(p), p)
