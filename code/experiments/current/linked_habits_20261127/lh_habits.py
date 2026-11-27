"""The natural habits of linked habits version 2 (SPEC2.md "The 24 linked habits"): readings and rewrites on
habits.py trees (CPU only). Used by lh_profile (the detector), lh_twin (minimal pairs) and lh_corpus (the builder).

Every reading works on a habits.py tree whose line ids are the written proof's N numbers, so a site's position is its
line in the written proof. Reading order (`read_tree`):

  1. the header notes (notes.note_reading) are removed from the tree;
  2. habit 8 (`halves`) is read on that tree, dead lines included (its form-1 line is an unused half);
  3. dead lines are dropped (habits.prune) -> Q;
  4. habit 9 (`restate`) is read on Q; its form-1 copies are collapsed (the user cites the premise again) -> S;
  5. habits 2-4 are read on S exactly as authors_corpus.detect reads them (`detect_tree`, the same functions), habit
     7 (`disjunct`) and the two habits left out of the linked set (`split`, `exfalso`) on S, and habit 1 (`route`)
     on the written proof (authors_profile.route_flags).

Habit definitions (form 1 / form 2; which author writes which form is LH/data/habit_table.json):

  route     sigma+ following R lines (authors_profile.route_flags); one site when present, form 1 only (A's).
  order     premise-driven / goal-driven normal form (authors_corpus); one site per proof with room.
  reit      restate an outer line inside the box / cite it directly; per (box, outer line) (habits.reit_sites).
  conj      right conjunct first / left first, free sites only (authors_corpus.detect).
  disjunct  an ORI on A v B (A != B) where the other disjunct is citable at that line, the line the ORI cites stays
            used without it (a premise, an assumption, or a line cited elsewhere), so a swap neither kills nor needs
            a line, and the swap alone leaves every other habit's reading unchanged (a free site): ORI2 (prove the
            right disjunct) / ORI1 (the left).
  halves    an ANDE e on a conjunction X & Y (X != Y), the only live ANDE of that conjunction in its scope: the other
            half is also taken, by an unused ANDE citing what e cites, right after e (right before e when e is its
            scope's last live line) / only e. (Conjunctions are matched through R copies.)
  restate   for each premise p, its first top-level use u (a top-level line citing p, or citing a copy of p that sits
            in the run of premise copies right before u and is cited by u only): u cites such a copy `X : R p` / u
            cites p directly.
  split     (not linked) a redundant case split (habits.redundant_ores): read as form-1 sites only; the form without
            the split leaves no trace in the proof, so form-2 sites cannot be read.
  exfalso   (not linked) a vacuous reductio (form 1, habits.vacuous_reductios) / a BOTE line (form 2).
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
W2 = HERE.parent / "two_authors_20261111"
AT = HERE.parent / "author_tags_20261110"
for _p in (str(AT), str(W2), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import authors_corpus as AC  # noqa: E402
import authors_profile as AP  # noqa: E402
import habits as H  # noqa: E402
import notes as N  # noqa: E402

NATURAL = ("route", "order", "reit", "conj", "split", "exfalso", "disjunct", "halves", "restate")
NOTE_HABITS = tuple(f"note_{c}" for c in N.CLASSES)
HABITS = NATURAL + NOTE_HABITS
REWRITABLE = (
    "disjunct",
    "halves",
    "restate",
)  # the new habits this module rewrites (1-4: W2's corpora)
NOT_LINKED = {
    "split": "form 2 (no split) leaves no trace: a proof that skipped the split cannot be told from one that never "
    "had room, so form-2 sites cannot be read on a proof (detector round trip and model proofs)",
    "exfalso": "form 1 (the vacuous reductio, DN of a NEGI) is Dan's sigma / sigma+ signature, the route of habit 1 "
    "(coe_sigma.flags 'sigma_chain'); writing it would make non-route records route records",
}
ORI_FORM = {"ORI2": 1, "ORI1": 2}  # disjunct: form 1 proves the right disjunct


# ============================================================================ tree helpers
def tree(text: str) -> H.Proof:
    return H.build(text)


def remove_ids(P: H.Proof, ids: set[int]) -> H.Proof:
    """A copy of P without the top-level units in `ids` (the header notes)."""
    Q = P.copy()
    Q.body = [u for u in Q.body if u.id not in ids]
    return Q


def unit_ids(P: H.Proof) -> set[int]:
    return {u.id for units, *_ in H.scopes(P) for u in units}


def dead_ids(P: H.Proof) -> set[int]:
    """Ids habits.prune would drop."""
    Q = P.copy()
    H.prune(Q)
    return unit_ids(P) - unit_ids(Q)


def citers(P: H.Proof) -> dict[int, list[H.Ln]]:
    """line id -> the units citing it (line citations, box citations excluded)."""
    out: dict[int, list[H.Ln]] = collections.defaultdict(list)
    for units, *_ in H.scopes(P):
        for u in units:
            for r in u.line_refs():
                out[r].append(u)
    return out


def scope_index(P: H.Proof) -> dict[int, tuple[list, int]]:
    """unit id -> (its scope's units list, index)."""
    return {u.id: (units, k) for units, *_ in H.scopes(P) for k, u in enumerate(units)}


def root(info: dict, i: int) -> int:
    """Follow R lines back to their source."""
    seen = set()
    while isinstance(info.get(i), H.Ln) and info[i].rule == "R" and i not in seen:
        seen.add(i)
        i = info[i].refs[0]
    return i


def set_refs(u: H.Ln, old: int, new: int) -> None:
    """Replace citations of `old` by `new` in one unit's own line citations."""
    if u.rule in ("IMPI", "NEGI"):
        return
    if u.rule == "ORE":
        if u.refs[0] == old:
            u.refs[0] = new
        return
    u.refs = [new if r == old else r for r in u.refs]


# ============================================================================ habit 8: halves
OTHER_HALF = {"ANDE1": "ANDE2", "ANDE2": "ANDE1"}


def halves_sites(B: H.Proof, dead: set[int]) -> list[dict[str, Any]]:
    """Habit 8 sites on a tree with dead lines (notes removed). -> [{e, sib (id or None), form 1 | 2 | None, rule,
    last}]; form None is a near miss: the unused other half is present but not in the builder's place."""
    info, _chain = H.all_lines(B)
    out = []
    for units, *_ in H.scopes(B):
        live_units = [u for u in units if u.id not in dead]
        last_live = live_units[-1].id if live_units else None
        groups: dict[int, list[int]] = collections.defaultdict(list)
        for k, u in enumerate(units):
            if u.rule in OTHER_HALF:
                groups[root(info, u.refs[0])].append(k)
        for src, ks in groups.items():
            f = H.formula_of(info, src) if src in info else None
            if not (isinstance(f, tuple) and f[0] == "and" and f[1] != f[2]):
                continue
            live = [k for k in ks if units[k].id not in dead]
            dd = [k for k in ks if units[k].id in dead]
            if len(live) != 1:
                continue
            k = live[0]
            e = units[k]
            j = (
                k - 1 if e.id == last_live else k + 1
            )  # the other half goes right after e, or right before a last line
            if not dd:
                out.append(
                    {"e": e.id, "sib": None, "form": 2, "rule": e.rule, "last": e.id == last_live}
                )
            elif (
                len(dd) == 1
                and dd[0] == j
                and units[j].rule == OTHER_HALF[e.rule]
                and units[j].refs == e.refs
            ):
                out.append(
                    {
                        "e": e.id,
                        "sib": units[j].id,
                        "form": 1,
                        "rule": e.rule,
                        "last": e.id == last_live,
                    }
                )
            elif any(units[j2].rule == OTHER_HALF[e.rule] for j2 in dd):
                # a near miss (audit F9a): the unused other half is there, but not where the builder puts it (or
                # citing another line): a site in neither form ("other"), not a missing site
                out.append(
                    {
                        "e": e.id,
                        "sib": None,
                        "form": None,
                        "rule": e.rule,
                        "last": e.id == last_live,
                    }
                )
    return out


# ============================================================================ habit 9: restate
def restate_sites(Q: H.Proof) -> list[dict[str, Any]]:
    """Habit 9 sites on a pruned tree (notes removed). -> [{p, u, copy (id or None), form 1|2}]."""
    prem = [p.id for p in Q.prem]
    pset = set(prem)
    cit = citers(Q)
    body = Q.body

    def is_copy(v: H.Ln) -> bool:
        return v.rule == "R" and v.refs[0] in pset

    out = []
    for p in prem:
        for k, u in enumerate(body):
            if is_copy(u):
                continue
            refs = u.line_refs()
            run = []
            j = k - 1
            while j >= 0 and is_copy(body[j]):
                run.append(body[j])
                j -= 1
            pat = [c for c in run if c.refs[0] == p and {w.id for w in cit.get(c.id, [])} == {u.id}]
            direct = p in refs
            via = [c for c in pat if c.id in refs]
            if not direct and not via:
                continue
            if direct and not via:
                out.append({"p": p, "u": u.id, "copy": None, "form": 2})
            elif via and not direct and len(via) == 1:
                out.append({"p": p, "u": u.id, "copy": via[0].id, "form": 1})
            break
    return out


def collapse_restate(Q: H.Proof, sites: list[dict[str, Any]]) -> H.Proof:
    """A copy of Q with every form-1 restate copy removed and its citers citing the premise."""
    S = Q.copy()
    for s in sites:
        if s["form"] != 1:
            continue
        for u in [w for units, *_ in H.scopes(S) for w in units]:
            set_refs(u, s["copy"], s["p"])
        S.body = [u for u in S.body if u.id != s["copy"]]
    return S


# ============================================================================ habits 2-4: W2's detectors on a tree
def detect_tree(S: H.Proof) -> dict[str, Any]:
    """authors_corpus.detect on a tree, with the sites' ids: the body of authors_corpus.detect (W2 commit c88db2d,
    lines 238-264), returning the site lists as well as the counts. -> {order, conjunct [n, a, nf, af], reit [n, a],
    reit_sites [(hid, outer id, reiterated)], conj_sites [(left id, right id, right_first, free)]}."""
    rs = H.reit_sites(S)
    T = S.copy()
    AC.strip_reit(T)
    of = AC._order_form_stripped(T)
    lasts = AC.scope_lasts(T)
    sites = []
    for left, right, kind in AC.conj_site_ids(T):
        pos = {u.id: k for k, u in enumerate(AC._scope_of(T, left))}
        right_first = pos[right] < pos[left]
        Q = AC.apply_conj(T, [(left, right, kind)], "B" if right_first else "A")
        free = Q is not None and AC.scope_lasts(Q) == lasts and AC._order_form_stripped(Q) == of
        sites.append((left, right, right_first, free))
    n = len(sites)
    a = sum(s[2] for s in sites)
    nf = sum(s[3] for s in sites)
    af = sum(s[2] and s[3] for s in sites)
    return {
        "order": of,
        "conjunct": [n, a, nf, af],
        "reit": [len(rs), sum(bool(x[3]) for x in rs)],
        "reit_sites": [(x[1], x[2], bool(x[3])) for x in rs],
        "conj_sites": sites,
    }


# ============================================================================ habit 7: disjunct
def available_latest(P: H.Proof) -> dict[int, dict]:
    """habits.available, but each formula maps to the innermost / latest line holding it (an R copy inside a box
    wins over the outer line, so a swap inside a reiterating box cites the copy)."""
    av = {}

    def rec(units, base):
        cur = dict(base)
        for u in units:
            av[u.id] = dict(cur)
            for b in u.boxes:
                inner = dict(cur)
                inner[b.hyp] = b.hid
                rec(b.body, inner)
            cur[u.f] = u.id

    base = {}
    for p in P.prem:
        base.setdefault(p.f, p.id)
    rec(P.body, base)
    return av


def disjunct_sites(S: H.Proof) -> list[dict[str, Any]]:
    """Habit 7 sites on S. -> [{u, cur, other, form 1|2}]."""
    av = available_latest(S)
    info, _ = H.all_lines(S)
    cit = citers(S)
    out = []
    for units, *_ in H.scopes(S):
        for u in units:
            if u.rule not in ORI_FORM or u.f[0] != "or" or u.f[1] == u.f[2]:
                continue
            other_f = u.f[2] if u.rule == "ORI1" else u.f[1]
            other = av[u.id].get(other_f)
            if other is None:
                continue
            cur = u.refs[0]
            x = info.get(cur)
            stays = (
                not isinstance(x, H.Ln)
                or x.rule == "PR"
                or any(w.id != u.id for w in cit.get(cur, []))
            )
            if stays:
                out.append({"u": u.id, "cur": cur, "other": other, "form": ORI_FORM[u.rule]})
    return out


# ============================================================================ the whole reading
def _put(out: dict[str, Any], h: str, pos: list[tuple[int, int | None]]) -> None:
    out[h] = {
        "sites": len(pos),
        "f1": sum(f == 1 for _, f in pos),
        "f2": sum(f == 2 for _, f in pos),
        "pos": pos,
    }


def _read_core(T: H.Proof, note_ids: set[int], text: str | None) -> tuple[dict[str, Any], H.Proof]:
    """Every natural habit but `disjunct`; -> (reading, S)."""
    B = remove_ids(T, note_ids)
    dead = dead_ids(B)
    hv = halves_sites(B, dead)
    near = {
        "ande_apart": sum(x["form"] is None for x in hv),
        "dead_r_in_box": sum(
            u.rule == "R" and u.id in dead
            for units, hid, *_ in H.scopes(B)
            if hid is not None
            for u in units
        ),
    }
    Q = B.copy()
    H.prune(Q)
    rt = restate_sites(Q)
    S = collapse_restate(Q, rt)
    det = detect_tree(S)
    out: dict[str, Any] = {
        "dead": len(dead),
        "detect": {k: det[k] for k in ("order", "conjunct", "reit")},
        "near": near,
    }
    ids = [
        i
        for units, hid, *_ in H.scopes(S)
        for i in ([hid] if hid is not None else []) + [u.id for u in units]
    ]
    first = min(ids) if ids else 0
    o = det["order"]
    _put(out, "order", [] if o == "none" else [(first, {"A": 1, "B": 2}.get(o))])
    _put(out, "reit", [(hid, 1 if r else 2) for hid, _r, r in det["reit_sites"]])
    _put(out, "conj", [(min(a, b), 1 if rf else 2) for a, b, rf, free in det["conj_sites"] if free])
    _put(out, "halves", [(s["e"], s["form"]) for s in hv])
    _put(out, "restate", [(s["u"], s["form"]) for s in rt])
    _put(out, "split", [(units[k].id, 1) for units, k, _i in H.redundant_ores(S)])
    vac = [(units[k].id, 1) for units, _j, k in H.vacuous_reductios(S)]
    bote = [(u.id, 2) for units, *_ in H.scopes(S) for u in units if u.rule == "BOTE"]
    _put(out, "exfalso", sorted(vac + bote))
    rf = AP.route_flags(text if text is not None else H.render(T))
    dn = [u.id for units, *_ in H.scopes(S) for u in units if u.rule == "DN"]
    _put(out, "route", [(dn[0] if dn else first, 1)] if rf["route"] else [])
    out["route_flags"] = rf
    return out, S


def _sig(rd: dict[str, Any]) -> tuple:
    return tuple(
        (h, rd[h]["sites"], rd[h]["f1"], rd[h]["f2"]) for h in NATURAL if h != "disjunct"
    ) + (tuple(sorted(rd["route_flags"].items())),)


def swap_ori(T: H.Proof, s: dict[str, Any]) -> None:
    """In place: the ORI of disjunct site `s` switched to its other disjunct."""
    units, k = scope_index(T)[s["u"]]
    u = units[k]
    u.rule = "ORI1" if u.rule == "ORI2" else "ORI2"
    u.refs = [s["other"]]


def read_tree(T: H.Proof, note_ids: set[int], text: str | None = None) -> dict[str, Any]:
    """The natural habits of a tree (module docstring). -> {habit: {"sites", "f1", "f2", "pos": [(position, form)]}}
    for the 9 natural habits, plus "dead", "detect" (W2's reading) and "route_flags". `text` (the written proof) is
    used for the route; without it the route is read on the rendered tree.

    A disjunct candidate (disjunct_sites) is a site only if swapping it alone leaves every other natural habit's
    reading and the route flags unchanged (a "free" site, as W2's free conjunct sites), so a swap never moves another
    habit."""
    out, S = _read_core(T, note_ids, text)
    cand = disjunct_sites(S)
    free = []
    if cand:
        ref = _sig(_read_core(T, note_ids, None)[0]) if text is not None else _sig(out)
        for s in cand:
            T2 = T.copy()
            swap_ori(T2, s)
            try:
                ok = _sig(_read_core(T2, note_ids, None)[0]) == ref
            except Exception:  # noqa: BLE001  (a swap the tree or a detector cannot read)
                ok = False
            if ok:
                free.append(s)
    _put(out, "disjunct", [(s["u"], s["form"]) for s in free])
    out["disjunct"]["detail"] = free
    return out


# ============================================================================ rewrites
def _note_free(T: H.Proof, note_ids: set[int]) -> tuple[H.Proof, set[int]]:
    B = remove_ids(T, note_ids)
    return B, dead_ids(B)


def set_disjunct(
    T: H.Proof, form: int, note_ids: set[int] = frozenset(), flip: bool = False
) -> int:
    """In place: every disjunct site in `form` (1: ORI2 on the right disjunct, 2: ORI1 on the left), or with
    `flip` every site in its other form. -> sites changed."""
    n = 0
    for s in read_tree(T, note_ids)["disjunct"]["detail"]:
        if flip or s["form"] != form:
            swap_ori(T, s)
            n += 1
    return n


def set_halves(T: H.Proof, form: int, note_ids: set[int] = frozenset(), flip: bool = False) -> int:
    """In place: every halves site in `form` (1: the other half taken too, unused; 2: removed), or with `flip`
    every site in its other form."""
    B, dead = _note_free(T, note_ids)
    info, _ = H.all_lines(T)
    n = 0
    for s in halves_sites(B, dead):
        if s["form"] is None:  # a near miss is left as written
            continue
        target = 3 - s["form"] if flip else form
        if s["form"] == target:
            continue
        units, k = scope_index(T)[s["e"]]
        e = units[k]
        if target == 1:
            src = root(info, e.refs[0])
            f = H.formula_of(info, src)
            rule = OTHER_HALF[e.rule]
            units.insert(
                k if s["last"] else k + 1,
                H.Ln(T.fresh(), f[2] if rule == "ANDE2" else f[1], rule, list(e.refs)),
            )
        else:
            j = next(i for i, w in enumerate(units) if w.id == s["sib"])
            units.pop(j)
        n += 1
    return n


def _unused_half(w: H.Ln | None, u: H.Ln, p: int, dead: set[int]) -> bool:
    """w is u's unused other half (habit 8), still citing premise p."""
    return (
        w is not None
        and w.id in dead
        and u.rule in OTHER_HALF
        and w.rule == OTHER_HALF[u.rule]
        and w.refs == [p]
    )


def set_restate(T: H.Proof, form: int, note_ids: set[int] = frozenset(), flip: bool = False) -> int:
    """In place: every restate site in `form` (1: `X : R p` right before the first top-level use, which cites it;
    2: the use cites p, the copy removed), or with `flip` every site in its other form."""
    B, dead = _note_free(T, note_ids)
    Q = B.copy()
    H.prune(Q)
    prem = {p.id: p.f for p in T.prem}
    n = 0
    for s in restate_sites(Q):
        target = 3 - s["form"] if flip else form
        if s["form"] == target:
            continue
        k = next(i for i, u in enumerate(T.body) if u.id == s["u"])
        u = T.body[k]
        if target == 1:
            c = H.Ln(T.fresh(), prem[s["p"]], "R", [s["p"]])
            set_refs(u, s["p"], c.id)
            before = T.body[k - 1] if k > 0 else None
            after = T.body[k + 1] if k + 1 < len(T.body) else None
            # u's unused other half (habit 8) cites what u cites; right before u (u is last) the copy goes first
            if _unused_half(before, u, s["p"], dead):
                before.refs = [c.id]
                T.body.insert(k - 1, c)
            else:
                T.body.insert(k, c)
                if _unused_half(after, u, s["p"], dead):
                    after.refs = [c.id]
        else:
            for w in [w for units, *_ in H.scopes(T) for w in units]:
                set_refs(w, s["copy"], s["p"])
            T.body = [w for w in T.body if w.id != s["copy"]]
        n += 1
    return n


SETTERS = {"disjunct": set_disjunct, "halves": set_halves, "restate": set_restate}
