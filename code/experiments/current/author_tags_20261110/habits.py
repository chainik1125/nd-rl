#!/usr/bin/env python3
"""How much room is there for a checker-neutral proof *style* in K12?  (author-tags, 2026-11-10;
CPU only.)

For each candidate habit this measures, over the 155,000 K12 pretraining proofs
(`train_k12.jsonl.gz`):
  room    : the proof has at least one point where an author could do it either way and the checker
  accepts both;
  split   : how the existing (single-generator) corpus is split at those points;
  cost    : the change in proof lines of the habit's rewrite;
and, on a seeded sample of rewritten proofs per habit, Dan's checks: `nd_verify`,
`nd2lean.translate` + Lean 4 core
(`nd2lean.lean_check`), and `state_env.decompose(canon=True)` (the step format the models see).
The decomposition is
also used to ask whether the habit leaves a trace in the *state* at later steps (the models see the
state, not the
token history): a rewritten proof's post-divergence state is "novel" if no state of the other proof
has the same
hypothesis formulas in the same order and the same goal (names stripped).

  python habits.py all                  # everything below, outputs in
  /scratch/cowifornia/nd-rl/author_tags/habits/
  python habits.py scan | lean | search | dan | table      # one stage (each reads the previous
  stages' JSON)

The proof is handled as a tree in Lean's shape: every box (AS ... last line) is attached to the
line that discharges it
(IMPI / NEGI: one box; ORE: two), which is where `nd2lean` renders it.  A *unit* is one non-AS line
together with the
boxes it discharges; a *scope* is the top level or one box body.  Every rewrite renders back to the
spec.md ND format.
Habit definitions: see HABITS.md (same folder) and the docstrings of the `h_*` functions.
"""

import argparse
import collections
import gzip
import itertools
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DAN = "/scratch/cowifornia/nd-rl/knockout_danbase/dan_code_13e41bbd"
# Dan's proof-strategies fork (strategies.py), same base code
PSFORK = "/scratch/cowifornia/nd-rl/ext/nd-takehome-ps"
CORPUS = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl.gz"
OUT = "/scratch/cowifornia/nd-rl/author_tags/habits"
sys.path.insert(0, DAN)
from nd_verify import verify_text  # noqa: E402
from nd_verify.verify import ParseError, parse_proof_tokens  # noqa: E402
from patterns import fstr  # noqa: E402

BOT = ("bot",)
BOXRULES = ("IMPI", "NEGI", "ORE")
HABITS = ("order", "conjunct", "reit", "reductio", "disjunct", "detour", "exfalso", "redsplit")


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


class Unsupported(Exception):
    pass


# ============================================================================ tree
class Ln:
    """one non-AS line plus the boxes it discharges (a unit)."""

    __slots__ = ("id", "f", "rule", "refs", "boxes")

    def __init__(self, id, f, rule, refs, boxes=()):
        self.id, self.f, self.rule, self.refs, self.boxes = id, f, rule, list(refs), list(boxes)

    def line_refs(self):
        """citations of single lines (box citations excluded)."""
        if self.rule in ("IMPI", "NEGI"):
            return []
        if self.rule == "ORE":
            return [self.refs[0]]
        return list(self.refs)

    def copy(self):
        return Ln(self.id, self.f, self.rule, self.refs, [b.copy() for b in self.boxes])


class Box:
    __slots__ = ("hid", "hyp", "body")

    def __init__(self, hid, hyp, body):
        self.hid, self.hyp, self.body = hid, hyp, body

    def last(self):
        return self.body[-1].id if self.body else self.hid

    def copy(self):
        return Box(self.hid, self.hyp, [u.copy() for u in self.body])


class Proof:
    __slots__ = ("prem", "body", "next_id")

    def __init__(self, prem, body, next_id):
        self.prem, self.body, self.next_id = prem, body, next_id

    def copy(self):
        return Proof([u.copy() for u in self.prem], [u.copy() for u in self.body], self.next_id)

    def fresh(self):
        self.next_id += 1
        return self.next_id


def build(proof_text):
    """ND proof text -> Proof (tree), or raise Unsupported."""
    try:
        lines = parse_proof_tokens(proof_text.split())
    except ParseError as e:
        raise Unsupported(f"parse {e}") from e
    pos = 0
    prem = []
    while pos < len(lines) and lines[pos]["rule"] == "PR":
        ln = lines[pos]
        prem.append(Ln(ln["idx"], ln["formula"], "PR", []))
        pos += 1

    def scope(pos, d):
        units, pending = [], {}
        while pos < len(lines):
            ln = lines[pos]
            if ln["depth"] < d or (d >= 1 and ln["rule"] == "AS" and ln["depth"] == d):
                break
            if ln["rule"] == "AS":
                if ln["depth"] != d + 1:
                    raise Unsupported("AS depth")
                body, pos2 = scope(pos + 1, d + 1)
                pending[ln["idx"]] = Box(ln["idx"], ln["formula"], body)
                pos = pos2
                continue
            if ln["depth"] != d or ln["rule"] == "PR":
                raise Unsupported("depth/PR")
            u = Ln(ln["idx"], ln["formula"], ln["rule"], ln["refs"])
            if u.rule in BOXRULES:
                pairs = (
                    [(u.refs[0], u.refs[1])]
                    if u.rule != "ORE"
                    else [(u.refs[1], u.refs[2]), (u.refs[3], u.refs[4])]
                )
                for s, e in pairs:
                    b = pending.pop(s, None)
                    if b is None:
                        raise Unsupported("box cited twice or from another scope")
                    if b.last() != e:
                        raise Unsupported("box end")
                    u.boxes.append(b)
            units.append(u)
            pos += 1
        if pending:
            raise Unsupported("undischarged box")
        return units, pos

    body, pos = scope(pos, 0)
    if pos != len(lines):
        raise Unsupported("trailing")
    return Proof(prem, body, max(x["idx"] for x in lines))


def render(P, want_idx=False):
    """Proof -> ND proof text (N1.. numbering; boxes emitted immediately before the line that
    discharges them).
    want_idx: also return the map line id -> rendered index."""
    toks, idx = [], {}

    def emit(depth, f, rule, refs):
        n = len(idx_order) + 1
        idx_order.append(n)
        toks.extend(
            [f"N{n}"] + ["|"] * depth + [fstr(f), ":", rule] + [f"N{r}" for r in refs] + [";"]
        )
        return n

    idx_order = []

    def scope(units, d):
        for u in units:
            be = []
            for b in u.boxes:
                s = emit(d + 1, b.hyp, "AS", [])
                idx[b.hid] = s
                scope(b.body, d + 1)
                be.append((s, idx[b.last()]))
            if u.rule in ("IMPI", "NEGI"):
                refs = [be[0][0], be[0][1]]
            elif u.rule == "ORE":
                refs = [idx[u.refs[0]], be[0][0], be[0][1], be[1][0], be[1][1]]
            else:
                refs = [idx[r] for r in u.refs]
            idx[u.id] = emit(d, u.f, u.rule, refs)

    for p in P.prem:
        idx[p.id] = emit(0, p.f, "PR", [])
    scope(P.body, 0)
    toks.append("QED")
    return (" ".join(toks), idx) if want_idx else " ".join(toks)


def n_lines(P):
    def cnt(units):
        return sum(1 + sum(1 + cnt(b.body) for b in u.boxes) for u in units)

    return len(P.prem) + cnt(P.body)


def max_depth(P):
    def md(units, d):
        return max([d] + [md(b.body, d + 1) for u in units for b in u.boxes])

    return md(P.body, 0)


# ---------------------------------------------------------------- tree analysis helpers
def subtree_ids(u):
    s = {u.id}
    for b in u.boxes:
        s.add(b.hid)
        for v in b.body:
            s |= subtree_ids(v)
    return s


def subtree_refs(u):
    """every line id cited anywhere in u's subtree (box citations excluded)."""
    r = set(u.line_refs())
    for b in u.boxes:
        for v in b.body:
            r |= subtree_refs(v)
    return r


def ext_refs(u):
    return subtree_refs(u) - subtree_ids(u)


def scopes(P):
    """yield (units list, hyp id or None, depth, owner unit or None) for the top level and every
    box body."""
    out = [(P.body, None, 0, None)]

    def rec(units, d):
        for u in units:
            for b in u.boxes:
                out.append((b.body, b.hid, d + 1, u))
                rec(b.body, d + 1)

    rec(P.body, 0)
    return out


def all_lines(P):
    """id -> (Ln or ('AS', formula)), plus id -> scope depth and id -> chain of enclosing box hyp
    ids."""
    info, chain = {}, {}
    for p in P.prem:
        info[p.id] = p
        chain[p.id] = ()

    def rec(units, ch):
        for u in units:
            info[u.id] = u
            chain[u.id] = ch
            for b in u.boxes:
                info[b.hid] = ("AS", b.hyp)
                chain[b.hid] = ch + (b.hid,)
                rec(b.body, ch + (b.hid,))

    rec(P.body, ())
    return info, chain


def formula_of(info, i):
    x = info[i]
    return x[1] if isinstance(x, tuple) else x.f


def deps_in_scope(units):
    """unit index -> set of unit indices (same scope) it cites directly (anywhere in its
    subtree)."""
    pos = {u.id: k for k, u in enumerate(units)}
    return [{pos[r] for r in ext_refs(u) if r in pos} for u in units]


def topo(units, prio, extra=()):
    """stable Kahn: a dependency-respecting order of `units` choosing, among the ready ones, the
    minimal prio(k).
    extra: additional edges (a, b) meaning a must come before b."""
    n = len(units)
    dep = deps_in_scope(units)
    for a, b in extra:
        dep[b].add(a)
    indeg = [len(d) for d in dep]
    users = [[] for _ in range(n)]
    for k, d in enumerate(dep):
        for j in d:
            users[j].append(k)
    ready = [k for k in range(n) if indeg[k] == 0]
    out = []
    while ready:
        k = min(ready, key=prio)
        ready.remove(k)
        out.append(k)
        for w in users[k]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
    if len(out) != n:
        raise Unsupported("cycle")
    return [units[k] for k in out]


def topo_alap(units, prio):
    """reverse Kahn: build the order from the end, choosing among units whose users are all placed
    the minimal prio(k)
    (so units with small prio end up as LATE as possible)."""
    n = len(units)
    dep = deps_in_scope(units)
    users = [set() for _ in range(n)]
    for k, d in enumerate(dep):
        for j in d:
            users[j].add(k)
    outdeg = [len(u) for u in users]
    ready = [k for k in range(n) if outdeg[k] == 0]
    out = []
    while ready:
        k = min(ready, key=prio)
        ready.remove(k)
        out.append(k)
        for j in dep[k]:
            outdeg[j] -= 1
            if outdeg[j] == 0:
                ready.append(j)
    if len(out) != n:
        raise Unsupported("cycle")
    return [units[k] for k in reversed(out)]


def linear_extensions(units, cap=10**9):
    """number of dependency-respecting orders of one scope (DP over subsets; exact up to 20
    units)."""
    n = len(units)
    if n <= 1:
        return 1
    if n > 20:
        return cap
    dep = deps_in_scope(units)
    mask = [sum(1 << j for j in d) for d in dep]
    f = {0: 1}
    for _size in range(n):
        g = collections.defaultdict(int)
        for S, c in f.items():
            for k in range(n):
                if not S >> k & 1 and mask[k] & S == mask[k]:
                    g[S | 1 << k] += c
        f = g
    return min(f.get((1 << n) - 1, 0), cap)


def is_boxunit(u):
    return bool(u.boxes)


def ancestors(units):
    """unit index -> set of unit indices (same scope) it depends on, transitively."""
    dep = deps_in_scope(units)
    anc = []
    for k in range(len(units)):
        seen, st = set(), list(dep[k])
        while st:
            j = st.pop()
            if j not in seen:
                seen.add(j)
                st.extend(dep[j])
        anc.append(seen)
    return anc


# ============================================================================ habit 1: step order
def hoist_all(P):
    """premise-driven placement: move every unit that does not depend on its box's hypothesis
    (transitively, within
    the box) out of the box, to just before the unit that discharges the box; repeat to a fixpoint.
     The box's last
    line never moves (the box needs it).  Returns the number of moves."""
    moves = 0
    changed = True
    while changed:
        changed = False
        for units, hid, _d, owner in scopes(P):
            if hid is None:
                continue
            body = units
            if len(body) <= 1:
                continue
            bad = {hid}
            hoist = []
            for u in body[:-1]:
                if ext_refs(u) & bad:
                    bad.add(u.id)
                else:
                    hoist.append(u)
            if not hoist:
                continue
            # find the parent scope list that holds `owner`
            for punits, _phid, _pd, _pown in scopes(P):
                if any(x is owner for x in punits):
                    k = next(i for i, x in enumerate(punits) if x is owner)
                    hs = set(id(x) for x in hoist)
                    body[:] = [x for x in body if id(x) not in hs]
                    punits[k:k] = hoist
                    moves += len(hoist)
                    changed = True
                    break
            if changed:
                break
    return moves


def sink_all(P):
    """goal-driven placement: move every unit whose citers all lie inside ONE box of ONE later unit
    of its scope into
    that box (at the start of the body; the order pass then places it); repeat to a fixpoint.
    Returns moves."""
    moves = 0
    changed = True
    while changed:
        changed = False
        for units, _hid, _d, _owner in scopes(P):
            if len(units) <= 1:
                continue
            for k in range(len(units) - 2, -1, -1):
                u = units[k]
                target = None
                ok = True
                for w in units[k + 1 :]:
                    if u.id in w.line_refs():
                        ok = False
                        break
                    hit = [b for b in w.boxes if any(u.id in subtree_refs(v) for v in b.body)]
                    if not hit:
                        continue
                    if len(hit) > 1 or target is not None:
                        ok = False
                        break
                    target = hit[0]
                if ok and target is not None:
                    units.pop(k)
                    target.body.insert(0, u)
                    moves += 1
                    changed = True
                    break
            if changed:
                break
    return moves


def order_all(P, mode):
    """re-order every scope.  'pd' (premise-driven): flat units (no box) as early as possible,
    boxes after;
    'gd' (goal-driven): boxes as early as possible, flat units as late as possible (just before
    their first use);
    'orig': stable (identity on a corpus proof)."""
    for units, _hid, _d, _owner in scopes(P):
        if len(units) <= 1:
            continue
        if mode == "pd":
            new = topo(units, lambda k, us=units: (is_boxunit(us[k]), k))
        elif mode == "gd":
            new = topo_alap(units, lambda k, us=units: (is_boxunit(us[k]), -k))
        else:
            new = topo(units, lambda k: k)
        units[:] = new
    return P


def normal_form(P0, mode):
    P = P0.copy()
    if mode == "pd":
        hoist_all(P)
    else:
        sink_all(P)
    return order_all(P, mode)


def h_order(P, text):
    """Step order.  room = the premise-driven normal form (hoist + flat-first) and the goal-driven
    one (sink + boxes
    first, flat units just in time) render to different proofs.  Also: number of within-scope
    orders, and the
    pairwise split (flat unit before an incomparable box unit of the same scope =
    premise-driven)."""
    n_orders = 1
    for units, _hid, _d, _owner in scopes(P):
        n_orders *= linear_extensions(units)
    pairs = pd_pairs = 0
    for units, _hid, _d, _owner in scopes(P):
        anc = ancestors(units)
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                if a in anc[b]:  # b depends on a: no choice
                    continue
                fa, fb = is_boxunit(units[a]), is_boxunit(units[b])
                if fa == fb:
                    continue
                pairs += 1
                pd_pairs += not fa  # a comes first; premise-driven iff the earlier one is flat
    Q = P.copy()
    hm = hoist_all(Q)
    R = P.copy()
    sm = sink_all(R)
    pd = render(order_all(Q, "pd"))
    gd = render(order_all(R, "gd"))
    return {
        "n_orders": n_orders,
        "room_perm": n_orders > 1,
        "hoistable": hm,
        "sinkable": sm,
        "room": pd != gd,
        "is_pd": text == pd,
        "is_gd": text == gd,
        "pairs": pairs,
        "pd_pairs": pd_pairs,
    }


def v_order(P, text):
    """variants: the two normal forms."""
    return {"pd": render(normal_form(P, "pd")), "gd": render(normal_form(P, "gd"))}


# ============================================================================ habit 2: conjunct
# order
def conj_sites(P):
    """(units, a_idx, b_idx, kind) for every ANDI whose two cited lines are distinct units of the
    ANDI's own scope that
    do not depend on each other, and every ANDE1/ANDE2 pair on the same conjunction in one scope
    (independent)."""
    out = []
    for units, _hid, _d, _owner in scopes(P):
        pos = {u.id: k for k, u in enumerate(units)}
        anc = ancestors(units)
        for u in units:
            if (
                u.rule == "ANDI"
                and u.refs[0] != u.refs[1]
                and u.refs[0] in pos
                and u.refs[1] in pos
            ):
                a, b = pos[u.refs[0]], pos[u.refs[1]]
                if a not in anc[b] and b not in anc[a]:
                    out.append((units, a, b, "andi"))
        by = collections.defaultdict(dict)
        for k, u in enumerate(units):
            if u.rule in ("ANDE1", "ANDE2"):
                by[u.refs[0]].setdefault(u.rule, k)
        for _c, m in by.items():
            if "ANDE1" in m and "ANDE2" in m:
                a, b = m["ANDE1"], m["ANDE2"]
                if a not in anc[b] and b not in anc[a]:
                    out.append((units, a, b, "ande"))
    return out


def h_conjunct(P, text):
    """Conjunct order.  site = an ANDI whose two conjunct lines are independent units of its scope,
    or an ANDE1/ANDE2
    pair on one conjunction in one scope.  split: left conjunct (or ANDE1) first."""
    s = conj_sites(P)
    return {
        "room": bool(s),
        "sites": len(s),
        "left_first": sum(a < b for _, a, b, _ in s),
        "andi_sites": sum(k == "andi" for *_, k in s),
        "ande_sites": sum(k == "ande" for *_, k in s),
    }


def v_conjunct(P, text):
    """variant: the first site swapped (artificial edge right-before-left, otherwise the original
    order)."""
    Q = P.copy()
    s = conj_sites(Q)
    if not s:
        return {}
    units, a, b, kind = s[0]
    first, second = (a, b) if a < b else (b, a)
    units[:] = topo(units, lambda k: k, extra=[(second, first)])
    return {"swap": render(Q)}


# ============================================================================ habit 3: reiteration
def r_lines(P):
    """classify every R line: 'required' = the last line of a scope (a box, or the top level) whose
    source is neither
    the box's own assumption nor the line just before it -- the box must end in that formula and
    the step format's
    `exact` only closes on the frame's last statement; otherwise 'optional' (its citers can cite
    the source)."""
    out = []
    for units, hid, _d, _owner in scopes(P):
        for k, u in enumerate(units):
            if u.rule != "R":
                continue
            src = u.refs[0]
            last = k == len(units) - 1
            if last and not (
                (hid is not None and src == hid) or (k > 0 and units[k - 1].id == src)
            ):
                out.append((units, k, "required"))
            else:
                out.append((units, k, "optional"))
    return out


def reit_sites(P):
    """(box units, box hyp id, outer line id, reiterated?) for every outer line used at a box's own
    level: cited by a
    non-R line of that box, or restated there by an optional R line."""
    info, chain = all_lines(P)
    req = {id(units[k]) for units, k, c in r_lines(P) if c == "required"}
    sites = {}
    for units, hid, _d, _owner in scopes(P):
        if hid is None:
            continue
        mine = {hid} | {u.id for u in units}
        for u in units:
            for r in u.line_refs():
                if r in mine:
                    continue
                key = (id(units), r)
                via = u.rule == "R" and id(u) not in req
                if u.rule == "R" and id(u) in req:
                    continue
                prev = sites.get(key)
                sites[key] = (units, hid, r, (prev[3] if prev else False) or via)
    return list(sites.values())


def h_reit(P, text):
    rl = r_lines(P)
    s = reit_sites(P)
    return {
        "room": bool(s),
        "sites": len(s),
        "reiterated": sum(x[3] for x in s),
        "r_required": sum(c == "required" for *_, c in rl),
        "r_optional": sum(c == "optional" for *_, c in rl),
        "r_opt_toplevel": sum(c == "optional" and units is P.body for units, k, c in rl),
    }


def redirect(P, old, new, within=None):
    """replace citations of line `old` by `new` (in the whole proof, or only in the given units'
    subtrees)."""

    def fix(units):
        for u in units:
            if u.rule not in ("IMPI", "NEGI"):
                lr = u.line_refs()
                if u.rule == "ORE":
                    if u.refs[0] == old:
                        u.refs[0] = new
                else:
                    u.refs = [new if r == old else r for r in lr]
            for b in u.boxes:
                fix(b.body)

    fix(within if within is not None else P.body)


def v_reit(P, text):
    """variants: 'rfree' = every optional R removed (citers cite the source); 'reit' = every outer
    line a box uses at
    its own level is restated once by R right after the box's assumption (textbook reiteration),
    citers redirected."""
    out = {}
    Q = P.copy()
    changed = True
    while changed:
        changed = False
        for units, k, c in r_lines(Q):
            if c == "optional":
                u = units.pop(k)
                redirect(Q, u.id, u.refs[0])
                changed = True
                break
    out["rfree"] = render(Q)
    Q = P.copy()
    sites = [x for x in reit_sites(Q) if not x[3]]
    for units, _hid, r, _via in sites:
        nid = Q.fresh()
        info, chain = all_lines(Q)
        f = formula_of(info, r)
        # redirect only the citations made at this box's own level (not inside deeper boxes, which
        # reiterate for themselves)
        for u in units:
            if u.rule == "ORE" and u.refs[0] == r:
                u.refs[0] = nid
            elif u.rule not in BOXRULES:
                u.refs = [nid if x == r else x for x in u.refs]
        units.insert(0, Ln(nid, f, "R", [r]))
    out["reit"] = render(Q)
    return out


# ============================================================================ habit 4: idle
# reductio
def h_reductio(P, text):
    """room: every proof (the conclusion can always be wrapped).  The corpus counts (template /
    working reductios) come
    from Dan's classifier in the `dan` stage."""
    return {"room": True}


def v_reductio(P, text):
    """'outside': after the conclusion G: | ~G : AS ; | F : NEGE G ~G ; ~~G : NEGI ; G : DN   (+4
    lines);
    'inside': the whole derivation of G inside the ~G box, then the same three closing lines   (+4
    lines, depth +1)."""
    out = {}
    G = P.body[-1].f
    Q = P.copy()
    g = Q.body[-1].id
    h, fl, nn, dn = Q.fresh(), Q.fresh(), Q.fresh(), Q.fresh()
    box = Box(h, ("not", G), [Ln(fl, BOT, "NEGE", [g, h])])
    Q.body.append(Ln(nn, ("not", ("not", G)), "NEGI", [h, fl], [box]))
    Q.body.append(Ln(dn, G, "DN", [nn]))
    out["outside"] = render(Q)
    Q = P.copy()
    g = Q.body[-1].id
    h, fl, nn, dn = Q.fresh(), Q.fresh(), Q.fresh(), Q.fresh()
    box = Box(h, ("not", G), Q.body + [Ln(fl, BOT, "NEGE", [g, h])])
    Q.body = [Ln(nn, ("not", ("not", G)), "NEGI", [h, fl], [box]), Ln(dn, G, "DN", [nn])]
    out["inside"] = render(Q)
    return out


# ============================================================================ habit 5: disjunct
# choice
def available(P):
    """for every unit: the set of formulas citable from it (premises, enclosing hypotheses, earlier
    units of every
    enclosing scope).  -> dict id(unit) -> {formula: line id}"""
    av = {}

    def rec(units, base):
        cur = dict(base)
        for u in units:
            av[id(u)] = dict(cur)
            for b in u.boxes:
                inner = dict(cur)
                inner.setdefault(b.hyp, b.hid)
                rec(b.body, inner)
            cur.setdefault(u.f, u.id)

    base = {}
    for p in P.prem:
        base.setdefault(p.f, p.id)
    rec(P.body, base)
    return av


def ori_sites(P):
    av = available(P)
    out = []

    def rec(units):
        for u in units:
            if u.rule in ("ORI1", "ORI2"):
                other = u.f[2] if u.rule == "ORI1" else u.f[1]
                if other in av[id(u)] and u.f[1] != u.f[2]:
                    out.append((u, av[id(u)][other]))
            for b in u.boxes:
                rec(b.body)

    rec(P.body)
    return out


def prune(P):
    """drop units nothing (transitively) uses; returns the number of lines dropped."""
    before = n_lines(P)
    changed = True
    while changed:
        changed = False
        used = set()
        for units, _hid, _d, _owner in scopes(P):
            for u in units:
                used |= set(u.line_refs())
        for units, _hid, _d, _owner in scopes(P):
            for k in range(len(units) - 1):  # the last unit of a scope is its conclusion
                if units[k].id not in used:
                    units.pop(k)
                    changed = True
                    break
            if changed:
                break
    return before - n_lines(P)


def h_disjunct(P, text):
    """site = an ORI line A v B (A != B) where the other disjunct is already citable at that line.
    split: ORI1."""
    s = ori_sites(P)
    return {"room": bool(s), "sites": len(s), "ori1": sum(u.rule == "ORI1" for u, _ in s)}


def v_disjunct(P, text):
    """variant: every site switched to the other disjunct, then dead lines pruned."""
    Q = P.copy()
    s = ori_sites(Q)
    if not s:
        return {}
    for u, other in s:
        u.rule = "ORI2" if u.rule == "ORI1" else "ORI1"
        u.refs = [other]
    prune(Q)
    return {"swap": render(Q)}


# ============================================================================ habit 6: conjunction
# detour
def h_detour(P, text):
    """room: any proof that cites a premise (the detour X & X, X goes after the first cited
    premise).  Corpus: units
    whose ANDE cites an ANDI line (an existing detour through a conjunction)."""
    used = set()
    for units, _hid, _d, _owner in scopes(P):
        for u in units:
            used |= set(u.line_refs())
    ex = 0
    info, chain = all_lines(P)
    for units, _hid, _d, _owner in scopes(P):
        for u in units:
            if (
                u.rule in ("ANDE1", "ANDE2")
                and isinstance(info.get(u.refs[0]), Ln)
                and info[u.refs[0]].rule == "ANDI"
            ):
                ex += 1
    return {"room": any(p.id in used for p in P.prem), "existing": ex}


def v_detour(P, text):
    """variant: after the PR block, ( X & X ) : ANDI p p ; X : ANDE1 ; every citation of premise p
    redirected (+2)."""
    Q = P.copy()
    used = set()
    for units, _hid, _d, _owner in scopes(Q):
        for u in units:
            used |= set(u.line_refs())
    ps = [p for p in Q.prem if p.id in used]
    if not ps:
        return {}
    p = ps[0]
    a, e = Q.fresh(), Q.fresh()
    redirect(Q, p.id, e)
    Q.body[0:0] = [Ln(a, ("and", p.f, p.f), "ANDI", [p.id, p.id]), Ln(e, p.f, "ANDE1", [a])]
    return {"detour": render(Q)}


# ============================================================================ habit 7 (added): ex
# falso style
def vacuous_reductios(P):
    """DN units whose cited line is a NEGI unit (same scope) with hypothesis ~G whose box ends in F
    without using the
    hypothesis: explosion written as a reductio."""
    out = []
    for units, _hid, _d, _owner in scopes(P):
        byid = {u.id: (k, u) for k, u in enumerate(units)}
        for k, u in enumerate(units):
            if u.rule != "DN" or u.refs[0] not in byid:
                continue
            j, n = byid[u.refs[0]]
            if n.rule != "NEGI" or n.boxes[0].hyp != ("not", u.f):
                continue
            b = n.boxes[0]
            uses = any(b.hid in subtree_refs(v) for v in b.body)
            citers = sum(n.id in subtree_refs(w) for w in units)
            if b.body and not uses and citers == 1:
                out.append((units, j, k))
    return out


def h_exfalso(P, text):
    """site = a BOTE line (ex falso) or a vacuous reductio (~G assumed, F derived without it, ~~G,
    DN).  split: BOTE."""
    bote = sum(u.rule == "BOTE" for units, *_ in scopes(P) for u in units)
    vac = len(vacuous_reductios(P))
    return {"room": bool(bote + vac), "bote": bote, "vacuous": vac}


def v_exfalso(P, text):
    """variants: 'reductio' = every BOTE  G : BOTE f  becomes  | ~G : AS ; | F : R f ; ~~G : NEGI ;
    G : DN  (+3 lines
    each); 'bote' = the reverse, every vacuous reductio becomes a BOTE line."""
    Q = P.copy()
    n = 0
    for units, _hid, _d, _owner in scopes(Q):
        for u in list(units):
            if u.rule == "BOTE":
                f = u.refs[0]
                h, r, nn = Q.fresh(), Q.fresh(), Q.fresh()
                box = Box(h, ("not", u.f), [Ln(r, BOT, "R", [f])])
                k2 = units.index(u)
                units.insert(k2, Ln(nn, ("not", ("not", u.f)), "NEGI", [h, r], [box]))
                u.rule, u.refs = "DN", [nn]
                n += 1
    out = {"reductio": render(Q)} if n else {}
    # reverse: every vacuous reductio becomes ex falso (its hypothesis-free body moves out of the
    # box; a closing R is
    # dropped and the BOTE cites its source)
    Q = P.copy()
    n = 0
    while True:
        vr = vacuous_reductios(Q)
        if not vr:
            break
        units, j, k = vr[0]
        neg, dn = units[j], units[k]
        body = neg.boxes[0].body
        if body[-1].rule == "R":
            f, moved = body[-1].refs[0], body[:-1]
        else:
            f, moved = body[-1].id, body
        units[j : j + 1] = moved
        dn.rule, dn.refs = "BOTE", [f]
        n += 1
    if n:
        prune(Q)
        out["bote"] = render(Q)
    return out


# ============================================================================ habit 8 (added):
# redundant case split
def redundant_ores(P):
    """ORE units with a branch whose last line does not depend on the branch's assumption (the
    split can be deleted:
    the branch's derivation proves the goal outright)."""
    out = []

    def dep_on(box):
        bad = {box.hid}
        for v in box.body:
            if ext_refs(v) & bad:
                bad.add(v.id)
        return box.last() in bad

    for units, _hid, _d, _owner in scopes(P):
        for k, u in enumerate(units):
            if u.rule == "ORE":
                free = [i for i, b in enumerate(u.boxes) if b.body and not dep_on(b)]
                if free:
                    out.append((units, k, free[0]))
    return out


def h_redsplit(P, text):
    r = redundant_ores(P)
    ores = sum(u.rule == "ORE" for units, *_ in scopes(P) for u in units)
    return {"room": bool(r), "redundant": len(r), "ores": ores}


def v_redsplit(P, text):
    """variant: every redundant split replaced by the hypothesis-free branch's derivation (its
    lines move to the ORE's
    scope; citations of the ORE line go to the branch's last line); dead lines pruned."""
    Q = P.copy()
    n = 0
    while True:
        r = redundant_ores(Q)
        if not r:
            break
        units, k, i = r[0]
        u = units[k]
        b = u.boxes[i]
        bad = {b.hid}
        for v in b.body:
            if ext_refs(v) & bad:
                bad.add(v.id)
        keep = [v for v in b.body if v.id not in bad]
        last = b.body[-1]
        if last not in keep:
            break
        units[k : k + 1] = keep
        redirect(Q, u.id, last.id)
        n += 1
    if not n:
        return {}
    prune(Q)
    # the top-level conclusion must stay last; if the ORE was the root, `last` is now the root
    return {"nosplit": render(Q)}


H = {
    "order": (h_order, v_order),
    "conjunct": (h_conjunct, v_conjunct),
    "reit": (h_reit, v_reit),
    "reductio": (h_reductio, v_reductio),
    "disjunct": (h_disjunct, v_disjunct),
    "detour": (h_detour, v_detour),
    "exfalso": (h_exfalso, v_exfalso),
    "redsplit": (h_redsplit, v_redsplit),
}


# ============================================================================ stage: scan
def truth(f, v):
    t = f[0]
    if t == "atom":
        return v[f[1]]
    if t == "bot":
        return False
    if t == "not":
        return not truth(f[1], v)
    a, b = truth(f[1], v), truth(f[2], v)
    return a and b if t == "and" else (a or b if t == "or" else (not a) or b)


def inconsistent(prem):
    """no assignment of the four atoms satisfies every premise (ex falso then proves any
    conclusion)."""
    for bits in itertools.product((False, True), repeat=4):
        v = dict(zip("PQRS", bits, strict=True))
        if all(truth(p, v) for p in prem):
            return False
    return True


def scan_one(rec):
    text = rec["proof"]
    out = {"i": rec["_i"], "n_lines": rec["n_lines"]}
    try:
        P = build(text)
    except Unsupported as e:
        out["unsupported"] = str(e)
        return out
    out["roundtrip"] = render(P) == text
    out["depth"] = max_depth(P)
    out["inconsistent"] = inconsistent([p.f for p in P.prem])
    for h, (fh, _fv) in H.items():
        try:
            out[h] = fh(P, text)
        except Unsupported as e:
            out[h] = {"room": False, "err": str(e)}
    return out


def read_corpus(limit=None):
    recs = []
    with gzip.open(CORPUS, "rt") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            recs.append(
                {
                    "_i": i,
                    "prompt": r["prompt"],
                    "proof": r["proof"],
                    "n_lines": r["n_lines"],
                    "key": r.get("key"),
                    "thm": r["thm"],
                }
            )
    return recs


def stage_scan(a):
    import multiprocessing as mp

    recs = read_corpus(a.limit)
    t0 = time.time()
    with mp.Pool(a.procs) as pool, gzip.open(f"{a.out}/scan.jsonl.gz", "wt") as fo:
        rows = []
        for r in pool.imap(scan_one, recs, chunksize=200):
            rows.append(r)
            fo.write(json.dumps(r) + "\n")
    N = len(rows)
    ok = [r for r in rows if "unsupported" not in r]
    agg = {
        "N": N,
        "supported": len(ok),
        "unsupported": dict(
            collections.Counter(r["unsupported"] for r in rows if "unsupported" in r)
        ),
        "roundtrip_identical": sum(r["roundtrip"] for r in ok),
        "secs": round(time.time() - t0, 1),
        "keys_distinct": len({r["key"] for r in recs}),
        "thms_distinct": len({r["thm"] for r in recs}),
    }
    for h in HABITS:
        hs = [r[h] for r in ok]
        room = sum(x["room"] for x in hs)
        d = {"room": room, "room_pct": 100 * room / N, "errors": sum("err" in x for x in hs)}
        num = collections.Counter()
        for x in hs:
            for k, v in x.items():
                if isinstance(v, bool) or isinstance(v, (int, float)) and k != "n_orders":
                    num[k] += v
        d["sums"] = dict(num)
        agg[h] = d
    o = [r["order"] for r in ok]
    no = [x["n_orders"] for x in o]
    agg["order"]["n_orders_hist"] = {
        k: sum(1 for v in no if lo <= v < hi)
        for k, (lo, hi) in {
            "1": (1, 2),
            "2": (2, 3),
            "3-5": (3, 6),
            "6-23": (6, 24),
            "24-119": (24, 120),
            "120-5039": (120, 5040),
            ">=5040": (5040, 10**30),
        }.items()
    }
    agg["order"]["n_orders_median"] = sorted(no)[len(no) // 2]
    agg["order"]["room_perm"] = sum(x["room_perm"] for x in o)
    agg["order"]["movable"] = sum((x["hoistable"] + x["sinkable"]) > 0 for x in o)
    agg["order"]["is_pd"] = sum(x["is_pd"] for x in o if x["room"])
    agg["order"]["is_gd"] = sum(x["is_gd"] for x in o if x["room"])
    agg["order"]["room_by_len"] = {
        L: [
            sum(1 for r in ok if r["n_lines"] == L and r["order"]["room"]),
            sum(1 for r in ok if r["n_lines"] == L),
        ]
        for L in range(2, 13)
    }
    for h in HABITS:
        agg[h]["room_by_len"] = {
            L: [
                sum(1 for r in ok if r["n_lines"] == L and r[h]["room"]),
                sum(1 for r in ok if r["n_lines"] == L),
            ]
            for L in range(2, 13)
        }
    # proofs with ANY checker-neutral, zero-cost habit room (order, conjunct, disjunct)
    agg["any_free_room"] = sum(
        r["order"]["room"] or r["conjunct"]["room"] or r["disjunct"]["room"] for r in ok
    )
    agg["inconsistent"] = sum(r["inconsistent"] for r in ok)
    agg["max_depth_hist"] = dict(collections.Counter(r["depth"] for r in ok))
    # co-occurrence of the candidate habits an author pair could carry (zero-cost order + the R
    # habit, + conjunct)
    agg["room_order_and_reit"] = sum(r["order"]["room"] and r["reit"]["room"] for r in ok)
    agg["room_order_or_reit"] = sum(r["order"]["room"] or r["reit"]["room"] for r in ok)
    agg["room_order_reit_conj"] = sum(
        r["order"]["room"] and r["reit"]["room"] and r["conjunct"]["room"] for r in ok
    )
    agg["n_free_habits_hist"] = dict(
        collections.Counter(
            sum(bool(r[h]["room"]) for h in ("order", "conjunct", "reit", "disjunct")) for r in ok
        )
    )
    dump_json(agg, f"{a.out}/scan.json")
    print(
        json.dumps(
            {
                k: (
                    v
                    if not isinstance(v, dict)
                    else {kk: vv for kk, vv in v.items() if kk != "room_by_len"}
                )
                for k, v in agg.items()
            },
            indent=1,
        )
    )


# ============================================================================ stage: lean (+
# nd_verify + state replay)
NAME = re.compile(r"^[nh]\d+$")


def state_sigs(prompt, body):
    """the model-visible states of a proof (state_env.decompose, canonical names), names
    stripped."""
    from state_env import decompose

    steps, toks, env = decompose(prompt, body, canon=True)
    sigs = []
    for st, _act, _hs in steps:
        sigs.append(tuple("#" if NAME.match(t) else t for t in st))
    return sigs, len(steps)


def compare_states(sa, sb):
    """two proofs' model-visible state sequences.  k = first step whose state differs (every
    earlier action was the
    same, so every state from k on comes after the first differing action).  A state is *novel* if
    the other proof
    never shows it.  -> post-divergence lengths and novel counts, both directions."""
    k = 0
    while k < min(len(sa), len(sb)) and sa[k] == sb[k]:
        k += 1
    A, B = set(sa), set(sb)
    nb = sum(1 for s in sb[k:] if s not in A)
    na = sum(1 for s in sa[k:] if s not in B)
    return {
        "k": k,
        "post_a": len(sa) - k,
        "post_b": len(sb) - k,
        "novel_a": na,
        "novel_b": nb,
        "steps_a": len(sa),
        "steps_b": len(sb),
    }


def lean_many(srcs, workers):
    from nd2lean import lean_check

    chunks = [srcs[i : i + 40] for i in range(0, len(srcs), 40)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: lean_check(c, per_file=40), chunks))
    return [x for r in res for x in r]


def stage_lean(a):
    from nd2lean import TranslationError, translate

    scan = {}
    with gzip.open(f"{a.out}/scan.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            scan[r["i"]] = r
    recs = {r["_i"]: r for r in read_corpus(a.limit)}
    rng = random.Random(a.seed)
    jobs = []  # (habit, variant, i, prompt, original, rewritten)
    for h in HABITS:
        room = sorted(i for i, r in scan.items() if "unsupported" not in r and r[h]["room"])
        pick = rng.sample(room, min(a.lean_n, len(room)))
        for i in pick:
            rec = recs[i]
            P = build(rec["proof"])
            vs = H[h][1](P, rec["proof"])
            for vn, body in vs.items():
                jobs.append((h, vn, i, rec["prompt"], rec["proof"], body))
    rows, srcs, idx = [], [], []
    sigs = {}
    for h, vn, i, prompt, orig, body in jobs:
        row = {
            "habit": h,
            "variant": vn,
            "i": i,
            "body": body,
            "changed": body != orig,
            "dlines": sum(t == ";" for t in body.split()) - sum(t == ";" for t in orig.split()),
        }
        ok, why, _ = verify_text(prompt + " " + body)
        row["nd_ok"] = ok
        if not ok:
            row["nd_reason"] = why
        try:
            so, _ = state_sigs(prompt, orig)
            sv, nsteps = state_sigs(prompt, body)
            row["state_ok"] = True
            row["cmp"] = compare_states(so, sv)  # a = original, b = rewrite
            sigs[(h, i, vn)] = sv
        except Exception as e:
            row["state_ok"] = False
            row["state_err"] = f"{type(e).__name__}: {e}"[:100]
        try:
            srcs.append(translate(prompt, body))
            idx.append(len(rows))
        except TranslationError as e:
            row["lean_ok"] = False
            row["lean_reason"] = f"structural: {e}"
        rows.append(row)
    t0 = time.time()
    res = lean_many(srcs, a.workers)
    for k, (ok, msg) in zip(idx, res, strict=True):
        rows[k]["lean_ok"] = ok
        if not ok:
            rows[k]["lean_reason"] = msg[:200]
    with gzip.open(f"{a.out}/lean.jsonl.gz", "wt") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    agg = {}

    def vis(cmps):
        """visible = some post-divergence state of either proof is never shown by the other;
        persistence = the
        share of the rewrite's own steps whose state is novel.  A rewrite whose state sequence is
        identical to the
        other proof's (the habit changed only a term, which the state does not show) counts as not
        visible."""
        return {
            "compared": len(cmps),
            "visible": sum((c["novel_a"] + c["novel_b"]) > 0 for c in cmps),
            "novel_share_of_steps": sum(c["novel_b"] / max(1, c["steps_b"]) for c in cmps)
            / max(1, len(cmps)),
            "novel_share_post": sum(c["novel_b"] / max(1, c["post_b"]) for c in cmps)
            / max(1, len(cmps)),
            "post_steps_mean": sum(c["post_b"] for c in cmps) / max(1, len(cmps)),
            "steps_mean": sum(c["steps_b"] for c in cmps) / max(1, len(cmps)),
        }

    for (h, vn), g in itertools.groupby(
        sorted(rows, key=lambda r: (r["habit"], r["variant"])),
        key=lambda r: (r["habit"], r["variant"]),
    ):
        g = list(g)
        st = [r for r in g if r["state_ok"]]
        agg[f"{h}/{vn}"] = {
            "n": len(g),
            "changed": sum(r["changed"] for r in g),
            "nd_ok": sum(r["nd_ok"] for r in g),
            "lean_ok": sum(r["lean_ok"] for r in g),
            "state_ok": len(st),
            **vis([r["cmp"] for r in st if r["changed"]]),
            "dlines_mean": sum(r["dlines"] for r in g) / len(g),
            "dlines_hist": dict(sorted(collections.Counter(r["dlines"] for r in g).items())),
            "lean_reasons": dict(
                collections.Counter(
                    (r.get("lean_reason") or "")[:60] for r in g if not r["lean_ok"]
                ).most_common(3)
            ),
        }
    # the two authors' forms against each other (not against the corpus proof)
    for h, v1, v2 in (
        ("order", "pd", "gd"),
        ("reit", "rfree", "reit"),
        ("reductio", "outside", "inside"),
    ):
        cm = [
            compare_states(sigs[(hh, i, v1)], sigs[(hh, i, v2)])
            for (hh, i, vn) in list(sigs)
            if hh == h and vn == v1 and (h, i, v2) in sigs and sigs[(h, i, v1)] != sigs[(h, i, v2)]
        ]
        agg[f"{h}/{v1}~{v2}"] = vis(cm)
    agg["_lean_secs"] = round(time.time() - t0, 1)
    dump_json(agg, f"{a.out}/lean.json")
    print(json.dumps(agg, indent=1))


# ============================================================================ stage: search
# (genuinely different routes)
def _alt_search(S, G, A, b, d, H):
    """minlen.Search._search with the banned root intros removed (kept in step with minlen.py; the
    only edits are the
    ban checks marked BAN)."""

    bestc, bestt = b + 1, None
    can_box = S.max_depth is None or d < S.max_depth

    def consider(c, t):
        nonlocal bestc, bestt
        if c < bestc:
            bestc, bestt = c, t

    if b >= 1:
        for C in S.conj:
            if C[1] == G or C[2] == G:
                r = S.inp(C, A, min(b, bestc) - 1, d, H)
                if r:
                    consider(r[0] + 1, ("ANDE1" if C[1] == G else "ANDE2", G, r[1]))
    if b >= 1:
        for imp in S.imps:
            if imp[2] == G:
                r = S.inp(imp, A, min(b, bestc) - 1, d, H)
                if r:
                    r2 = S.inp(imp[1], A | {imp}, min(b, bestc) - 1 - r[0], d, H)
                    if r2:
                        consider(r[0] + r2[0] + 1, ("IMPE", G, r[1], r2[1]))
    if b >= 1 and not S.forbid_dn:
        nn = ("not", ("not", G))
        r = S.inp(nn, A, min(b, bestc) - 1, d, H)
        if r:
            consider(r[0] + 1, ("DN", G, r[1]))
    if G != BOT and b >= 1:
        r = S.inp(BOT, A, min(b, bestc) - 1, d, H)
        if r:
            consider(r[0] + 1, ("BOTE", G, r[1]))
    if G[0] == "and" and b >= 1:
        r = S.inp(G[1], A, min(b, bestc) - 1, d, H)
        if r:
            r2 = S.inp(G[2], A | {G[1]}, min(b, bestc) - 1 - r[0], d, H)
            if r2:
                consider(r[0] + r2[0] + 1, ("ANDI", G, r[1], r2[1]))
    if G[0] == "or" and b >= 1:
        for side, rule in ((1, "ORI1"), (2, "ORI2")):
            r = S.inp(G[side], A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, (rule, G, r[1]))
    if G[0] == "imp" and b >= 2 and "IMPI" not in S.ban:  # BAN
        a_, c_ = G[1], G[2]
        if a_ == c_ and can_box:
            consider(2, ("IMPI", G, a_, None))
        elif b >= 3 and can_box:
            r = S.best(c_, A | {a_}, min(b, bestc) - 2, d + 1, H | {a_})
            if r:
                consider(r[0] + 2, ("IMPI", G, a_, r[1]))
    if G[0] == "not" and b >= 3 and can_box and "NEGI" not in S.ban:  # BAN
        r = S.best(BOT, A | {G[1]}, min(b, bestc) - 2, d + 1, H | {G[1]})
        if r:
            consider(r[0] + 2, ("NEGI", G, G[1], r[1]))
    if G == BOT and b >= 1:
        for X in S.S:
            NX = ("not", X)
            if NX not in S.S and NX not in A:
                continue
            if NX in A:
                r = S.inp(X, A, min(b, bestc) - 1, d, H)
                if r:
                    consider(r[0] + 1, ("NEGE", BOT, r[1], ("R0", NX)))
            elif X in A:
                r = S.inp(NX, A, min(b, bestc) - 1, d, H)
                if r:
                    consider(r[0] + 1, ("NEGE", BOT, ("R0", X), r[1]))
            else:
                r = S.inp(X, A, min(b, bestc) - 2, d, H)
                if r:
                    r2 = S.inp(NX, A | {X}, min(b, bestc) - 1 - r[0], d, H)
                    if r2:
                        consider(r[0] + r2[0] + 1, ("NEGE", BOT, r[1], r2[1]))
    if (
        G != BOT and G[0] != "not" and b >= 4 and can_box and not S.forbid_dn and not S._raa_off
    ):  # BAN
        r = S.best(BOT, A | {("not", G)}, min(b, bestc) - 3, d + 1, H | {("not", G)})
        if r:
            consider(r[0] + 3, ("RAA", G, r[1]))
    if b >= 3 and can_box and "ORE" not in S.ban:  # BAN
        for D in S.disj:
            if D == G:
                continue
            r = S.inp(D, A, min(b, bestc) - 3, d, H)
            if not r:
                continue
            a_, c_ = D[1], D[2]
            A2 = A | {D}
            ca = (1, None) if a_ == G else None
            if ca is None:
                ra = S.best(G, A2 | {a_}, min(b, bestc) - r[0] - 3, d + 1, H | {a_})
                ca = (ra[0] + 1, ra[1]) if ra else None
            if ca is None:
                continue
            cb = (1, None) if c_ == G else None
            if cb is None:
                rb = S.best(G, A2 | {c_}, min(b, bestc) - r[0] - ca[0] - 1, d + 1, H | {c_})
                cb = (rb[0] + 1, rb[1]) if rb else None
            if cb is None:
                continue
            consider(r[0] + ca[0] + cb[0] + 1, ("ORE", G, r[1], a_, ca[1], c_, cb[1]))
    if S.lemmas and b >= 2:
        for L in S.S:
            if L in A or L == G:
                continue
            r = S.best(L, A, min(b, bestc) - 1, d, H)
            if not r:
                continue
            r2 = S.best(G, A | {L}, min(b, bestc) - r[0], d, H)
            if r2:
                consider(r[0] + r2[0], ("LEMMA", G, L, r[1], r2[1]))
    return (bestc, bestt) if bestt is not None else None


class BanSearch:
    """thin factory: minlen.Search whose _search is _alt_search (all options, minus the banned
    ones)."""

    _cls = None

    @classmethod
    def make(cls, prem, concl, ban, deadline):
        import minlen

        if cls._cls is None:

            class S(minlen.Search):
                def _search(self, G, A, b, d=0, H=frozenset()):
                    return _alt_search(self, G, A, b, d, H)

            cls._cls = S
        s = cls._cls(prem, concl, deadline=deadline)
        s.ban = set(ban)
        s._raa_off = "RAA" in s.ban or "DN" in s.ban
        s.forbid_dn = (
            s._raa_off
        )  # banning the classical reductio bans DN (else a lemma ~~G + DN rebuilds it)
        return s


def boxes_of_proof(text):
    """multiset of box-opening rules of an ND proof: IMPI, NEGI, ORE; a NEGI whose box assumes ~G
    cited by a DN that
    yields G is counted as RAA (classical reductio)."""
    lines = parse_proof_tokens(text.split())
    by = {x["idx"]: x for x in lines}
    raa = set()
    for x in lines:
        if x["rule"] == "DN" and by.get(x["refs"][0], {}).get("rule") == "NEGI":
            n = by[x["refs"][0]]
            if by[n["refs"][0]]["formula"] == ("not", x["formula"]):
                raa.add(n["idx"])
    c = collections.Counter()
    for x in lines:
        if x["rule"] in BOXRULES:
            c["RAA" if x["idx"] in raa else x["rule"]] += 1
    return dict(sorted(c.items()))


def search_one(args):
    """shortest proof (minlen, unrestricted); then for every box-opening rule it uses, the shortest
    proof with that
    rule banned.  An alternative within +slack lines has a different box multiset by
    construction."""
    rec, bound, tl, slack = args
    import minlen
    from minlen import parse_thm

    prem, concl = parse_thm(rec["prompt"])
    out = {"i": rec["_i"], "n_lines": rec["n_lines"], "inconsistent": inconsistent(prem)}
    t0 = time.time()
    s = BanSearch.make(prem, concl, (), time.time() + tl)
    try:
        r = s.run(bound)
    except minlen.Timeout:
        out["timeout"] = True
        return out
    out["timeout"] = False
    if r is None:
        out["L0"] = None
        return out
    L0, text = r
    ok = verify_text(rec["prompt"] + " " + text)[0]
    out.update({"L0": L0, "proof0": text, "ok0": ok, "boxes0": boxes_of_proof(text)})
    alts = {}
    for rule in out["boxes0"]:
        s2 = BanSearch.make(prem, concl, (rule,), time.time() + tl)
        try:
            r2 = s2.run(min(bound, L0 + slack))
        except minlen.Timeout:
            alts[rule] = "timeout"
            continue
        if r2 is None:
            alts[rule] = None
            continue
        L2, t2 = r2
        bx = boxes_of_proof(t2)
        alts[rule] = {
            "L": L2,
            "proof": t2,
            "ok": verify_text(rec["prompt"] + " " + t2)[0],
            "boxes": bx,
            "differs": bx != out["boxes0"],
        }
    out["alts"] = alts
    out["alt"] = any(
        isinstance(x, dict) and x["differs"] and x["ok"] and x["L"] <= L0 + slack
        for x in alts.values()
    )
    out["alt_same_len"] = any(
        isinstance(x, dict) and x["differs"] and x["ok"] and x["L"] == L0 for x in alts.values()
    )
    # the corpus proof itself as a witness: within +slack of the shortest and a different box
    # multiset
    cb = boxes_of_proof(rec["proof"])
    out["corpus_boxes"] = cb
    out["corpus_alt"] = rec["n_lines"] <= L0 + slack and cb != out["boxes0"]
    out["secs"] = round(time.time() - t0, 2)
    return out


def stage_search(a):
    import multiprocessing as mp

    recs = read_corpus(a.limit)
    rng = random.Random(a.seed + 7)
    sample = rng.sample(recs, min(a.search_n, len(recs)))
    t0 = time.time()
    rows = []
    with mp.Pool(a.procs) as pool:
        for r in pool.imap_unordered(
            search_one, [(x, a.bound, a.time, a.slack) for x in sample], chunksize=4
        ):
            rows.append(r)
    rows.sort(key=lambda r: r["i"])
    with gzip.open(f"{a.out}/search.jsonl.gz", "wt") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    # Lean check the alternatives (and the shortest proofs) of a subsample
    from nd2lean import translate

    byi = {x["_i"]: x for x in sample}
    srcs, meta = [], []
    for r in rows:
        if r.get("L0") is None:
            continue
        for rule, alt in r["alts"].items():
            if (
                isinstance(alt, dict)
                and alt["differs"]
                and alt["L"] <= r["L0"] + a.slack
                and len(srcs) < 2 * a.lean_n
            ):
                srcs.append(translate(byi[r["i"]]["prompt"], alt["proof"]))
                meta.append((r["i"], rule))
    res = lean_many(srcs, a.workers) if srcs else []
    lean_ok = sum(ok for ok, _ in res)
    n = len(rows)
    found = [r for r in rows if r.get("L0") is not None]
    with_box = [r for r in found if r["boxes0"]]
    either = [r for r in found if r["alt"] or r["corpus_alt"]]
    agg = {
        "n": n,
        "bound": a.bound,
        "slack": a.slack,
        "time_limit": a.time,
        "secs": round(time.time() - t0, 1),
        "timeouts": sum(r["timeout"] for r in rows),
        "found": len(found),
        "not_found_in_bound": sum(1 for r in rows if not r["timeout"] and r.get("L0") is None),
        "shortest_has_box": len(with_box),
        "alt_within_slack": sum(r["alt"] for r in found),
        "alt_same_length": sum(r["alt_same_len"] for r in found),
        "corpus_alt": sum(r["corpus_alt"] for r in found),
        "alt_or_corpus": len(either),
        "sample_inconsistent": sum(r["inconsistent"] for r in rows),
        "alt_inconsistent": sum(r["inconsistent"] for r in either),
        "alt_pairs": dict(
            collections.Counter(
                (
                    json.dumps(r["boxes0"])
                    + " -> "
                    + (
                        next(
                            (
                                json.dumps(x["boxes"])
                                for x in r["alts"].values()
                                if isinstance(x, dict)
                                and x["differs"]
                                and x["ok"]
                                and x["L"] <= r["L0"] + a.slack
                            ),
                            None,
                        )
                        or "corpus " + json.dumps(r["corpus_boxes"])
                    )
                )
                for r in either
            ).most_common(15)
        ),
        "alt_timeouts": sum(1 for r in found for x in r["alts"].values() if x == "timeout"),
        "corpus_minus_shortest_hist": dict(
            sorted(collections.Counter(r["n_lines"] - r["L0"] for r in found).items())
        ),
        "shortest_len_hist": dict(sorted(collections.Counter(r["L0"] for r in found).items())),
        "shortest_box_multisets": dict(
            collections.Counter(json.dumps(r["boxes0"]) for r in found).most_common(12)
        ),
        "alt_banned_rule": dict(
            collections.Counter(
                rule
                for r in found
                for rule, x in r["alts"].items()
                if isinstance(x, dict) and x["differs"] and x["L"] <= r["L0"] + a.slack
            )
        ),
        "alt_rows_by_shortest_len": {
            L: [
                sum(1 for r in found if r["L0"] == L and (r["alt"] or r["corpus_alt"])),
                sum(1 for r in found if r["L0"] == L),
            ]
            for L in range(2, 15)
        },
        "lean_checked": len(srcs),
        "lean_ok": lean_ok,
        "ok0_all": all(r["ok0"] for r in found),
    }
    dump_json(agg, f"{a.out}/search.json")
    print(json.dumps(agg, indent=1))


# ============================================================================ stage: disj (other
# disjunct derivable?)
def find_unit(P, uid):
    for units, _hid, d, _owner in scopes(P):
        for k, u in enumerate(units):
            if u.id == uid:
                return units, k, d
    raise KeyError(uid)


def lines_text(lines):
    toks = []
    for ln in lines:
        toks += (
            [f"N{ln['idx']}"]
            + ["|"] * ln["depth"]
            + [fstr(ln["formula"]), ":", ln["rule"]]
            + [f"N{r}" for r in ln["refs"]]
            + [";"]
        )
    return " ".join(toks + ["QED"])


def splice_derivation(P, uid, template, search, avail, new_rule):
    """insert minlen's derivation `template` (of a formula X) right before unit `uid`, switch that
    unit to `new_rule`
    citing X, prune, and return the new ND text."""
    text, idx = render(P, want_idx=True)
    lines = parse_proof_tokens(text.split())
    pos = idx[uid] - 1
    d = lines[pos]["depth"]
    forms = list(avail)
    el = [(None, f, None, None) for f in forms]
    gi = search.emit(template, {f: k for k, f in enumerate(forms)}, d, el)
    m = len(forms)
    new = el[m:]
    nn = len(new)

    def ni(r):
        return idx[avail[forms[r]]] if r < m else pos + 1 + (r - m)

    out = []
    for ln in lines[:pos]:
        out.append(ln)
    for j, (dep, f, rule, refs) in enumerate(new):
        out.append(
            {
                "idx": pos + 1 + j,
                "depth": dep,
                "formula": f,
                "rule": rule,
                "refs": [ni(r) for r in refs],
            }
        )
    for ln in lines[pos:]:
        x = dict(ln)
        x["idx"] = ln["idx"] + nn
        x["refs"] = [r if r < pos + 1 else r + nn for r in ln["refs"]]
        out.append(x)
    o = out[pos + nn]
    o["rule"] = new_rule
    o["refs"] = [ni(gi)]
    Q = build(lines_text(out))
    prune(Q)
    return render(Q)


def disj_one(args):
    """every ORI line A v B of one proof: is the other disjunct citable (cost 0), or derivable by
    minlen's DP from the
    formulas citable at that line within (lines the current side uses exclusively) + slack new
    lines?"""
    rec, slack, tl = args
    import minlen

    prem, concl = minlen.parse_thm(rec["prompt"])
    P = build(rec["proof"])
    av = available(P)
    sites, variant = [], None
    for units, _hid, d, _owner in scopes(P):
        for u in units:
            if u.rule not in ("ORI1", "ORI2") or u.f[1] == u.f[2]:
                continue
            other = u.f[2] if u.rule == "ORI1" else u.f[1]
            A = av[id(u)]
            Q = P.copy()
            qu, qk, _ = find_unit(Q, u.id)
            qu[qk].refs = [-1]
            excl = prune(Q)
            site = {"excl": excl, "available": other in A}
            if other in A:
                site["cost"] = 0
            else:
                s = minlen.Search(prem, concl, deadline=time.time() + tl)
                for f in minlen.subformulas(u.f, set()):
                    s.S |= {f, ("not", f), ("not", ("not", f))}
                s.conj = [f for f in s.S if f[0] == "and"]
                s.imps = [f for f in s.S if f[0] == "imp"]
                s.disj = [f for f in s.S if f[0] == "or"]
                try:
                    r = s.best(other, frozenset(A), excl + slack, d)
                except minlen.Timeout:
                    site["timeout"] = True
                    r = None
                site["cost"] = r[0] if r else None
                if r and variant is None:
                    try:
                        variant = splice_derivation(
                            P, u.id, r[1], s, A, "ORI2" if u.rule == "ORI1" else "ORI1"
                        )
                        site["variant_dlines"] = sum(t == ";" for t in variant.split()) - sum(
                            t == ";" for t in rec["proof"].split()
                        )
                    except Exception as e:
                        site["splice_err"] = f"{type(e).__name__}: {e}"[:80]
            site["delta"] = None if site["cost"] is None else site["cost"] - excl
            sites.append(site)
    return {"i": rec["_i"], "sites": sites, "variant": variant}


def stage_disj(a):
    import multiprocessing as mp

    from nd2lean import translate

    recs = read_corpus(a.limit)
    rng = random.Random(a.seed + 11)
    sample = rng.sample(recs, min(a.search_n, len(recs)))
    byi = {x["_i"]: x for x in sample}
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        rows = sorted(
            pool.imap_unordered(disj_one, [(x, a.slack, a.time) for x in sample], chunksize=8),
            key=lambda r: r["i"],
        )
    with gzip.open(f"{a.out}/disj.jsonl.gz", "wt") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    vs = [(r["i"], r["variant"]) for r in rows if r["variant"]][: a.lean_n]
    nd_ok = sum(verify_text(byi[i]["prompt"] + " " + v)[0] for i, v in vs)
    vis = 0
    for i, v in vs:
        c = compare_states(
            state_sigs(byi[i]["prompt"], byi[i]["proof"])[0], state_sigs(byi[i]["prompt"], v)[0]
        )
        vis += (c["novel_a"] + c["novel_b"]) > 0
    res = lean_many([translate(byi[i]["prompt"], v) for i, v in vs], a.workers) if vs else []
    sites = [s for r in rows for s in r["sites"]]

    def room(r, k):
        return any(s["delta"] is not None and s["delta"] <= k for s in r["sites"])

    agg = {
        "n": len(rows),
        "slack": a.slack,
        "secs": round(time.time() - t0, 1),
        "proofs_with_ori": sum(bool(r["sites"]) for r in rows),
        "sites": len(sites),
        "available": sum(s["available"] for s in sites),
        "derivable": sum(s["cost"] is not None for s in sites),
        "timeouts": sum(s.get("timeout", False) for s in sites),
        "splice_errors": sum("splice_err" in s for s in sites),
        "delta_hist": dict(
            sorted(collections.Counter(s["delta"] for s in sites if s["delta"] is not None).items())
        ),
        "room_delta_le0": sum(room(r, 0) for r in rows),
        "room_delta_le1": sum(room(r, 1) for r in rows),
        "room_delta_le2": sum(room(r, 2) for r in rows),
        "variants_checked": len(vs),
        "variants_nd_ok": nd_ok,
        "variants_lean_ok": sum(ok for ok, _ in res),
        "variants_state_visible": vis,
        "variant_dlines_hist": dict(
            sorted(
                collections.Counter(
                    s["variant_dlines"] for s in sites if "variant_dlines" in s
                ).items()
            )
        ),
    }
    dump_json(agg, f"{a.out}/disj.json")
    print(json.dumps(agg, indent=1))


# ============================================================================ stage: dan (reductio
# counts)
def dan_one(rec):
    from proof_strategies.strategies import analyse

    r = analyse(rec["proof"], rec["prompt"])
    sub = set(r["sub"]) if r else set()
    return {
        "i": rec["_i"],
        "contradiction": r is not None and "contradiction" in r["u"],
        "working": "contradiction.working" in sub,
        "template": "contradiction.template" in sub,
        "kinds": sorted(
            s.split(".", 1)[1]
            for s in sub
            if s.split(".", 1)[1] in ("vacuous", "pointless", "scaffolded", "neg_goal", "free")
            and s.startswith("contradiction.")
        ),
        "root": "contradiction.root" in sub,
        "p2": bool(r and r["p2"]),
    }


def stage_dan(a):
    import multiprocessing as mp

    sys.path.insert(1, PSFORK)
    recs = read_corpus(a.limit)
    with mp.Pool(a.procs) as pool:
        rows = list(pool.imap(dan_one, recs, chunksize=500))
    c = collections.Counter()
    kinds = collections.Counter()
    for r in rows:
        c["contradiction"] += r["contradiction"]
        c["working"] += r["working"]
        c["template_only"] += r["contradiction"] and not r["working"]
        c["patterns_reductio"] += r["p2"]
        c["root"] += r["root"]
        for k in r["kinds"]:
            kinds[k] += 1
    agg = {"N": len(rows), **c, "kinds_any": dict(kinds)}
    dump_json(agg, f"{a.out}/dan.json")
    print(json.dumps(agg, indent=1))


# ============================================================================ stage: table
DEFS = {
    "order": (
        "premise-driven (hoist hypothesis-free lines out of boxes; flat steps before boxes) vs "
        "goal-driven (sink lines into the one box that uses them; boxes first, "
        "flat steps just in time)"
    ),
    "conjunct": (
        "which of two independent conjunct derivations (ANDI) or extractions (ANDE1/ANDE2 of one "
        "conjunction) comes first"
    ),
    "reit": (
        "an outer line used inside a box: restated once by R after the assumption, "
        "or cited directly"
    ),
    "reductio": (
        "wrap the conclusion G: assume ~G, F by NEGE, ~~G by NEGI, G by DN "
        "(inside: whole derivation in the ~G box)"
    ),
    "disjunct": (
        "ORI1 vs ORI2 on A v B when the other disjunct is already citable (a: scan) "
        "or derivable within the same lines + 2 (b: sample)"
    ),
    "detour": "route a premise X through X & X (ANDI X X ; ANDE1) before use",
    "exfalso": (
        "from F conclude G by BOTE, or by a vacuous reductio "
        "(~G : AS ; F : R ; ~~G : NEGI ; G : DN)"
    ),
    "redsplit": (
        "keep a case split whose branch proves the goal without its case, "
        "or replace it by that branch"
    ),
}


def stage_table(a):
    S = load_json(f"{a.out}/scan.json")
    L = load_json(f"{a.out}/lean.json")
    D = load_json(f"{a.out}/dan.json") if os.path.exists(f"{a.out}/dan.json") else {}
    R = load_json(f"{a.out}/search.json") if os.path.exists(f"{a.out}/search.json") else {}
    J = load_json(f"{a.out}/disj.json") if os.path.exists(f"{a.out}/disj.json") else {}
    N = S["N"]

    def pc(x, n=N):
        return f"{100 * x / max(1, n):.1f} %"

    def vis(keys):
        out = []
        for k in keys:
            x = L.get(k)
            if x and x.get("compared") is not None:
                lab = k.split("/")[1]
                out.append(
                    f"{lab}: {x['visible']}/{x['compared']} differ, novel state in "
                    f"{100 * x['novel_share_of_steps']:.0f} % of steps"
                )
        return "; ".join(out)

    def cost(keys):
        return "; ".join(f"{k.split('/')[1]} {L[k]['dlines_mean']:+.1f}" for k in keys if k in L)

    def lean(keys):
        out = []
        for k in keys:
            x = L.get(k)
            if x:
                out.append(
                    f"{k.split('/')[1]}: Lean {x['lean_ok']}/{x['n']}, nd "
                    f"{x['nd_ok']}/{x['n']}, state {x['state_ok']}/{x['n']}"
                )
        return "; ".join(out)

    rows = []
    o = S["order"]
    s = o["sums"]
    rows.append(
        (
            "1 step order",
            DEFS["order"],
            pc(o["room"]),
            f"of room proofs: corpus = premise-driven form {pc(o['is_pd'], o['room'])}, = "
            f"goal-driven {pc(o['is_gd'], o['room'])}; "
            f"flat-before-box in {pc(s['pd_pairs'], s['pairs'])} of {s['pairs']:,} free pairs",
            "yes; " + vis(["order/pd~gd"]),
            "0",
            lean(["order/pd", "order/gd"]),
        )
    )
    c = S["conjunct"]
    s = c["sums"]
    rows.append(
        (
            "2 conjunct order",
            DEFS["conjunct"],
            pc(c["room"]),
            f"left first at {pc(s['left_first'], s['sites'])} of {s['sites']:,} sites (ANDE "
            f"pairs: {s['ande_sites']})",
            "yes, while both are in scope; " + vis(["conjunct/swap"]),
            "0",
            lean(["conjunct/swap"]),
        )
    )
    r = S["reit"]
    s = r["sums"]
    rows.append(
        (
            "3 reiteration",
            DEFS["reit"],
            pc(r["room"]),
            f"restated at {pc(s['reiterated'], s['sites'])} of {s['sites']:,} sites; of "
            f"{s['r_required'] + s['r_optional']:,} R lines {s['r_required']:,} required "
            f"(box ends on an outer line), {s['r_optional']:,} optional",
            "yes, inside that box; " + vis(["reit/rfree~reit"]),
            cost(["reit/reit", "reit/rfree"]),
            lean(["reit/reit", "reit/rfree"]),
        )
    )
    rows.append(
        (
            "4 idle reductio",
            DEFS["reductio"],
            pc(S["reductio"]["room"]),
            f"{D.get('contradiction', 0):,} proofs ({pc(D.get('contradiction', 0))}) hold a "
            f"reductio, {D.get('working', '?')} working; kinds: "
            + ", ".join(f"{k} {v:,}" for k, v in D.get("kinds_any", {}).items()),
            "inside: yes; outside: last 4 steps only; "
            + vis(["reductio/inside", "reductio/outside"]),
            cost(["reductio/inside", "reductio/outside"]),
            lean(["reductio/inside", "reductio/outside"]),
        )
    )
    d = S["disjunct"]
    s = d["sums"]
    jd = (
        f"; (b) derivable at <= 0 extra lines {pc(J['room_delta_le0'], J['n'])}, <= +2 "
        f"{pc(J['room_delta_le2'], J['n'])} (sample n = {J['n']:,})"
        if J
        else ""
    )
    jl = (
        (
            f"; (b) derivable splice: Lean {J['variants_lean_ok']}/{J['variants_checked']}, nd "
            f"{J['variants_nd_ok']}/{J['variants_checked']}, "
            f"state differs {J.get('variants_state_visible', '?')}/{J['variants_checked']}"
        )
        if J
        else ""
    )
    hist = J.get("variant_dlines_hist", {})
    splice_mean = sum(int(k) * v for k, v in hist.items()) / max(1, sum(hist.values()))
    rows.append(
        (
            "5 disjunct choice",
            DEFS["disjunct"],
            f"(a) {pc(d['room'])}{jd}",
            f"(a) ORI1 at {pc(s['ori1'], s['sites'])} of {s['sites']:,} sites",
            "(a) no when nothing else changes; "
            + vis(["disjunct/swap"])
            + "; (b) yes (the new derivation is in the state)",
            cost(["disjunct/swap"]) + (f"; (b) splice {splice_mean:+.1f}" if J else ""),
            lean(["disjunct/swap"]) + jl,
        )
    )
    t = S["detour"]
    s = t["sums"]
    rows.append(
        (
            "6 ∧ detour",
            DEFS["detour"],
            pc(t["room"]),
            f"{s['existing']} existing (ANDE of an ANDI line); the ¬¬ detour is the outside "
            "reductio (row 4)",
            "yes; " + vis(["detour/detour"]),
            cost(["detour/detour"]),
            lean(["detour/detour"]),
        )
    )
    e = S["exfalso"]
    s = e["sums"]
    rows.append(
        (
            "8 ex falso style (added)",
            DEFS["exfalso"],
            pc(e["room"]),
            f"BOTE {pc(s['bote'], s['bote'] + s['vacuous'])} of {s['bote'] + s['vacuous']:,} "
            f"sites (BOTE {s['bote']:,}, vacuous reductio {s['vacuous']:,})",
            "yes (¬¬G stays in scope); " + vis(["exfalso/reductio", "exfalso/bote"]),
            cost(["exfalso/reductio", "exfalso/bote"]),
            lean(["exfalso/reductio", "exfalso/bote"]),
        )
    )
    x = S["redsplit"]
    s = x["sums"]
    rows.append(
        (
            "9 redundant case split (added)",
            DEFS["redsplit"],
            pc(x["room"]),
            f"{s['redundant']:,} of {s['ores']:,} ORE lines ({pc(s['redundant'], s['ores'])}) "
            "are redundant",
            "partly; " + vis(["redsplit/nosplit"]),
            cost(["redsplit/nosplit"]),
            lean(["redsplit/nosplit"]),
        )
    )
    md = [
        f"Corpus: {N:,} K12 proofs; every proof parsed and re-rendered byte-identically "
        f"({S['roundtrip_identical']:,}). "
        f"Room = % of the {N:,} theorems. Checks on the rewrite: a seeded sample of room "
        "proofs per habit, each rewritten proof "
        "checked by Lean 4 core through nd2lean (Dan's check), by nd_verify, and replayed "
        "through state_env (canonical names).",
        "",
        "| habit | definition | % with room | current split | visible in later states | Δ "
        "lines per rewritten proof (mean) | checks on the rewrite |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    o = S["order"]
    md += [
        "",
        "Step order detail: within-scope reorderings alone (box placement fixed) give more "
        f"than one order in {pc(o['room_perm'])} of proofs "
        f"(median 1; "
        + ", ".join(f"{k}: {v:,}" for k, v in o["n_orders_hist"].items())
        + f"); a line can sit inside or outside a box in {pc(o['movable'])}.",
    ]
    md.append(
        "Room by proof length (lines 2..12), %: "
        + "; ".join(
            f"{h} "
            + " ".join(
                f"{100 * v[0] / max(1, v[1]):.0f}"
                for L_, v in sorted(S[h]["room_by_len"].items(), key=lambda kv: int(kv[0]))
            )
            for h in ("order", "conjunct", "reit", "disjunct", "exfalso", "redsplit")
        )
    )
    md.append(
        "No room for any of order / conjunct / reiteration / disjunct (a): "
        f"{pc(S['n_free_habits_hist'].get('0', 0))}; "
        f"room for order and reiteration both: {pc(S['room_order_and_reit'])}; premises "
        f"inconsistent (ex falso proves anything): {pc(S['inconsistent'])}."
    )
    if R:
        md += [
            "",
            f"7 different routes (minlen DP, random sample n = {R['n']:,}, bound {R['bound']} "
            f"lines, {R['time_limit']:.0f} s): "
            f"shortest proof found for {R['found']:,} ({R['timeouts']} timeouts); it opens a "
            f"box in {pc(R['shortest_has_box'], R['found'])}. "
            f"A proof within +{R['slack']} lines of the shortest with a different box multiset "
            "(each box rule of the shortest banned in turn, "
            f"or the corpus proof itself): {R['alt_or_corpus']:,} = "
            f"{pc(R['alt_or_corpus'], R['n'])} of the sample "
            f"(search {R['alt_within_slack']:,}, of which same length "
            f"{R['alt_same_length']:,}; corpus proof {R['corpus_alt']:,}); "
            f"Lean accepts {R['lean_ok']}/{R['lean_checked']} of the search's alternatives. "
            "Corpus proof minus shortest, lines: "
            + ", ".join(f"{k}: {v}" for k, v in R["corpus_minus_shortest_hist"].items())
            + ".",
        ]
        if "alt_inconsistent" in R:
            md[-1] += (
                f" Of the theorems with an alternative, {R['alt_inconsistent']:,} have "
                "inconsistent premises."
            )
    write_text(f"{a.out}/table.md", "\n".join(md) + "\n")
    print("\n".join(md))


def main():
    # minlen iterates over sets of formula tuples, so which of several equally short proofs it
    # returns depends on Python's per-process hash seed; pin it (re-exec once) so that every stage
    # is reproducible from run to run
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "scan", "lean", "search", "disj", "dan", "table"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=None, help="first N corpus records only (tests)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LEAN_GATE_WORKERS", 8)))
    ap.add_argument(
        "--lean_n",
        type=int,
        default=300,
        help="rewritten proofs per habit sent to nd_verify / Lean / state replay",
    )
    ap.add_argument("--search_n", type=int, default=5000)
    ap.add_argument("--bound", type=int, default=14)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stages = ["scan", "dan", "lean", "search", "disj", "table"] if a.stage == "all" else [a.stage]
    for st in stages:
        print(f"== {st}", flush=True)
        globals()[f"stage_{st}"](a)


if __name__ == "__main__":
    main()
