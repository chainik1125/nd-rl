"""ND proofs as line lists: parse / render, atom renaming, cut composition, redundancy removal.

A proof is (prem, lines): `prem` the premise formulas (tuples, the fork's `nd_verify` encoding),
`lines` a list of Line(depth, f, rule, refs) with 0-based refs into `lines` (premise lines included,
rule 'PR'). Rendering gives the fork's ND text (`N1 f : RULE N.. ; ... QED`), which
`nd_verify.verify_text` and `state_env.decompose` read.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

sys.dont_write_bytecode = True
FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
if FORK not in sys.path:
    sys.path.insert(0, FORK)

from gen import fstr  # noqa: E402  (fork, read-only)
from nd_verify import verify_text  # noqa: E402
from nd_verify.verify import parse_formula, parse_proof_tokens  # noqa: E402

ATOMS = ("P", "Q", "R", "S")
BOT = ("bot",)


@dataclass
class Line:
    depth: int
    f: tuple
    rule: str
    refs: list = field(default_factory=list)


@dataclass
class Proof:
    prem: list
    concl: tuple
    lines: list

    def prompt(self) -> str:
        ps = " , ".join(fstr(p) for p in self.prem)
        return (
            f"THM {ps} SEQ {fstr(self.concl)} PRF"
            if self.prem
            else f"THM SEQ {fstr(self.concl)} PRF"
        )

    def thm(self) -> str:
        return " , ".join(fstr(p) for p in self.prem) + " |- " + fstr(self.concl)

    def body(self) -> str:
        toks = []
        for i, ln in enumerate(self.lines):
            toks += [f"N{i + 1}"] + ["|"] * ln.depth + [fstr(ln.f), ":", ln.rule]
            toks += [f"N{r + 1}" for r in ln.refs] + [";"]
        return " ".join(toks + ["QED"])

    def verify(self) -> tuple[bool, str]:
        ok, why, n = verify_text(self.prompt() + " " + self.body())
        return ok and n == len(self.lines), why

    def __len__(self) -> int:
        return len(self.lines)


def parse_prompt(prompt: str) -> tuple[list, tuple]:
    toks = prompt.split()
    assert toks[0] == "THM"
    i = 1
    prem = []
    if toks[i] != "SEQ":
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ",":
                i += 1
                continue
            break
    assert toks[i] == "SEQ"
    concl, i = parse_formula(toks, i + 1)
    return prem, concl


def parse(prompt: str, body: str) -> Proof:
    prem, concl = parse_prompt(prompt)
    raw = parse_proof_tokens(body.split())
    idx0 = raw[0]["idx"]
    lines = [Line(r["depth"], r["formula"], r["rule"], [x - idx0 for x in r["refs"]]) for r in raw]
    return Proof(prem, concl, lines)


# ----------------------------------------------------------------------------------------- formulas
def atoms_of(f, acc=None):
    acc = [] if acc is None else acc
    if f[0] == "atom":
        if f[1] not in acc:
            acc.append(f[1])
    elif f[0] == "not":
        atoms_of(f[1], acc)
    elif f[0] in ("and", "or", "imp"):
        atoms_of(f[1], acc)
        atoms_of(f[2], acc)
    return acc


def rename(f, m):
    if f[0] == "atom":
        return ("atom", m.get(f[1], f[1]))
    if f[0] == "bot":
        return f
    if f[0] == "not":
        return ("not", rename(f[1], m))
    return (f[0], rename(f[1], m), rename(f[2], m))


def fclass(f) -> tuple[str, list]:
    """(class string of f under atom renaming, atoms of f in first-appearance order)."""
    at = atoms_of(f)
    m = {a: ATOMS[i] for i, a in enumerate(at)}
    return fstr(rename(f, m)), at


def fdepth(f) -> int:
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + fdepth(f[1])
    return 1 + max(fdepth(f[1]), fdepth(f[2]))


def rename_proof(p: Proof, m: dict) -> Proof:
    return Proof(
        [rename(x, m) for x in p.prem],
        rename(p.concl, m),
        [Line(ln.depth, rename(ln.f, m), ln.rule, list(ln.refs)) for ln in p.lines],
    )


# -------------------------------------------------------------------------------------- composition
def cut(lemma: Proof, main: Proof, j: int, order: list | None = None) -> Proof:
    """Replace premise j of `main` (which must equal lemma.concl) by lemma's derivation. Premises of
    the result: lemma.prem + main.prem without j, de-duplicated (first occurrence kept), or permuted
    by `order` (a list of indices into that de-duplicated list). Body: lemma's body, then main's
    body. No simplification here."""
    assert main.prem[j] == lemma.concl
    prem = []
    for x in lemma.prem + [q for i, q in enumerate(main.prem) if i != j]:
        if x not in prem:
            prem.append(x)
    if order is not None:
        prem = [prem[i] for i in order]
    pos = {x: i for i, x in enumerate(prem)}
    lines = [Line(0, x, "PR", []) for x in prem]
    # lemma body
    lmap = {}
    for i, ln in enumerate(lemma.lines):
        if ln.rule == "PR":
            lmap[i] = pos[ln.f]
            continue
        lmap[i] = len(lines)
        lines.append(Line(ln.depth, ln.f, ln.rule, [lmap[r] for r in ln.refs]))
    p_line = lmap[len(lemma.lines) - 1]
    assert lines[p_line].depth == 0 and lines[p_line].f == lemma.concl
    mmap = {}
    for i, ln in enumerate(main.lines):
        if ln.rule == "PR":
            mmap[i] = p_line if ln.f == main.prem[j] else pos[ln.f]
            continue
        mmap[i] = len(lines)
        lines.append(Line(ln.depth, ln.f, ln.rule, [mmap[r] for r in ln.refs]))
    return Proof(prem, main.concl, lines)


# ----------------------------------------------------------------------------------- simplification
def _ctx(lines):
    """Per line: the tuple of open box ids after the line (as nd_verify computes it), and box id ->
    end line."""
    stack, ctx, nb = [], [], 0
    box_end, as_box = {}, {}
    for i, ln in enumerate(lines):
        if ln.rule == "AS":
            del stack[ln.depth - 1 :]
            nb += 1
            stack.append(nb)
            as_box[i] = nb
        else:
            del stack[ln.depth :]
        for b in stack:
            box_end[b] = i
        ctx.append(tuple(stack))
    return ctx, box_end, as_box


def _box_refs(ln):
    if ln.rule in ("IMPI", "NEGI"):
        return set(ln.refs)
    if ln.rule == "ORE":
        return set(ln.refs[1:])
    return set()


def simplify(p: Proof) -> Proof:
    """Remove redundant lines until nothing changes: (1) a non-PR / non-AS line whose formula is
    already on an earlier line citable from it -- its citations are redirected to that line (never a
    box-boundary line, i.e. one cited as a box start / end); (2) lines the conclusion does not
    depend on (premise lines always stay)."""
    lines = [Line(x.depth, x.f, x.rule, list(x.refs)) for x in p.lines]
    changed = True
    while changed:
        changed = False
        ctx, box_end, as_box = _ctx(lines)
        boundary = set()
        for ln in lines:
            boundary |= _box_refs(ln)
        # (1) duplicates
        first = {}
        redirect = {}
        for i, ln in enumerate(lines):
            if ln.rule in ("PR", "AS") or i in boundary or i == len(lines) - 1:
                first.setdefault(ln.f, []).append(i)
                continue
            cands = [j for j in first.get(ln.f, []) if ctx[j] == ctx[i][: len(ctx[j])]]
            if cands:
                redirect[i] = cands[0]
            first.setdefault(ln.f, []).append(i)
        if redirect:
            for ln in lines:
                ln.refs = [redirect.get(r, r) for r in ln.refs]
            changed = True
        # (2) dependency pruning
        keep = set(i for i, ln in enumerate(lines) if ln.rule == "PR")
        stack = [len(lines) - 1]
        while stack:
            i = stack.pop()
            if i in keep and lines[i].rule != "PR":
                continue
            keep.add(i)
            stack.extend(lines[i].refs)
        if len(keep) < len(lines):
            new_idx = {}
            out = []
            for i, ln in enumerate(lines):
                if i in keep:
                    new_idx[i] = len(out)
                    out.append(ln)
            for ln in out:
                ln.refs = [new_idx[r] for r in ln.refs]
            lines = out
            changed = True
    return Proof(list(p.prem), p.concl, lines)


def used_premises(p: Proof) -> set:
    """Indices of premise lines the conclusion depends on."""
    keep = set()
    stack = [len(p.lines) - 1]
    while stack:
        i = stack.pop()
        if i in keep:
            continue
        keep.add(i)
        stack.extend(p.lines[i].refs)
    return {i for i in keep if p.lines[i].rule == "PR"}


def max_box_depth(p: Proof) -> int:
    return max((ln.depth for ln in p.lines), default=0)


# ---------------------------------------------------------------------------------------- semantics
_ROWS = [dict(zip(ATOMS, ((k >> i) & 1 == 1 for i in range(4)), strict=True)) for k in range(16)]


def ev(f, v) -> bool:
    t = f[0]
    if t == "atom":
        return v[f[1]]
    if t == "bot":
        return False
    if t == "not":
        return not ev(f[1], v)
    if t == "and":
        return ev(f[1], v) and ev(f[2], v)
    if t == "or":
        return ev(f[1], v) or ev(f[2], v)
    return (not ev(f[1], v)) or ev(f[2], v)


def entails(prem, concl) -> bool:
    """Classical consequence over the 16 valuations of P, Q, R, S."""
    return all(ev(concl, v) for v in _ROWS if all(ev(p, v) for p in prem))


def satisfiable(prem) -> bool:
    return any(all(ev(p, v) for p in prem) for v in _ROWS)


def semantic_flags(prem, concl) -> dict:
    """consistent premises; every premise necessary (the conclusion does not follow from the
    others); the conclusion is not a tautology unless there are no premises."""
    return dict(
        consistent=satisfiable(prem),
        all_necessary=all(not entails(prem[:i] + prem[i + 1 :], concl) for i in range(len(prem))),
        concl_tautology=entails([], concl),
    )


# ------------------------------------------------------------------------ sinking reiterated lemmas
def _users(lines):
    u = [[] for _ in lines]
    for i, ln in enumerate(lines):
        for r in ln.refs:
            u[r].append(i)
    return u


def sink_once(p: Proof) -> Proof | None:
    """One rewrite, or None: a reiteration line i (`| X : R x`, a box boundary after `simplify`)
    whose source x is used only by i is replaced by x's private derivation moved into the box
    (depths shifted), saving the R line."""
    lines = p.lines
    users = _users(lines)
    for i, ln in enumerate(lines):
        if ln.rule != "R":
            continue
        x = ln.refs[0]
        if users[x] != [i] or lines[x].rule in ("PR", "AS"):
            continue
        # private cone of x: x plus dependencies all of whose users are in the cone
        cone = {x}
        changed = True
        while changed:
            changed = False
            for c in list(cone):
                for r in lines[c].refs:
                    if r in cone or lines[r].rule == "PR":
                        continue
                    if all(u in cone for u in users[r]):
                        cone.add(r)
                        changed = True
        # every cone line must sit at x's depth or deeper, and boxes opened in the cone stay whole:
        # an AS line in the cone must have its whole box in the cone (true when its closing rule is
        # in the cone)
        dx = lines[x].depth
        if any(lines[c].depth < dx for c in cone):
            continue
        shift = ln.depth - dx
        order = sorted(cone)
        # lines of boxes opened inside the cone that are not in the cone would be orphaned: reject
        ctx, _, as_box = _ctx(lines)
        boxes_in = {as_box[c] for c in order if c in as_box}
        if any(set(ctx[j]) & boxes_in for j in range(len(lines)) if j not in cone):
            continue
        new_pos = {}
        out = []
        for j, l2 in enumerate(lines):
            if j in cone:
                continue
            if j == i:
                for c in order:
                    new_pos[c] = len(out)
                    out.append(
                        Line(lines[c].depth + shift, lines[c].f, lines[c].rule, list(lines[c].refs))
                    )
                new_pos[i] = new_pos[x]
                continue
            new_pos[j] = len(out)
            out.append(Line(l2.depth, l2.f, l2.rule, list(l2.refs)))
        for l2 in out:
            l2.refs = [new_pos[r] for r in l2.refs]
        q = Proof(list(p.prem), p.concl, out)
        if q.verify()[0]:
            return q
    return None


def sink_all(p: Proof) -> Proof:
    while True:
        q = sink_once(p)
        if q is None:
            return p
        p = simplify(q)


def drop_unused_premises(p: Proof) -> Proof:
    """The proof without premise lines the conclusion does not depend on (and those premises dropped
    from the theorem). If p is minimal for its theorem, the result is minimal for the reduced
    theorem: adding the dropped premise lines back to any shorter proof of it would give a shorter
    proof of the original."""
    used = used_premises(p)
    keep = [i for i, ln in enumerate(p.lines) if ln.rule != "PR" or i in used]
    pos = {i: k for k, i in enumerate(keep)}
    lines = [
        Line(p.lines[i].depth, p.lines[i].f, p.lines[i].rule, [pos[r] for r in p.lines[i].refs])
        for i in keep
    ]
    prem = [p.lines[i].f for i in keep if p.lines[i].rule == "PR"]
    return Proof(prem, p.concl, lines)


# ---------------------------------------------------------------------------------------- discharge
def discharge(p: Proof, j: int) -> Proof | None:
    """Deduction: premise j (A) leaves the premise list and the conclusion becomes (A > C). Lines
    that do not depend on A (box segments as units) stay at their depth before a new box `| A : AS`;
    the A-dependent lines move into it (depth + 1); `( A > C ) : IMPI` closes it. Returns None if
    the conclusion does not depend on A."""
    lines = p.lines
    pr_idx = [i for i, ln in enumerate(lines) if ln.rule == "PR"]
    a_line = pr_idx[j]
    A = lines[a_line].f
    n = len(lines)
    # top-level segments: a depth-0 line is its own segment; a depth-1 AS starts a box segment
    # running to the line before the next depth-0 line or depth-1 AS
    seg = [None] * n
    s = -1
    for i, ln in enumerate(lines):
        if ln.depth == 0 or (ln.depth == 1 and ln.rule == "AS"):
            s += 1
        seg[i] = s
    dep = [False] * n
    dep[a_line] = True
    changed = True
    while changed:
        changed = False
        for i, ln in enumerate(lines):
            if not dep[i] and any(dep[r] for r in ln.refs):
                dep[i] = True
                changed = True
        dseg = {seg[i] for i in range(n) if dep[i] and lines[i].depth >= 1}
        for i in range(n):
            if not dep[i] and seg[i] in dseg:
                dep[i] = True
                changed = True
    if not dep[n - 1]:
        return None
    order_ind = [i for i in range(n) if not dep[i]]
    order_dep = [i for i in range(n) if dep[i] and i != a_line]
    new_pos = {}
    out = []
    for i in order_ind:
        new_pos[i] = len(out)
        out.append(Line(lines[i].depth, lines[i].f, lines[i].rule, list(lines[i].refs)))
    new_pos[a_line] = len(out)
    out.append(Line(1, A, "AS", []))
    for i in order_dep:
        new_pos[i] = len(out)
        out.append(Line(lines[i].depth + 1, lines[i].f, lines[i].rule, list(lines[i].refs)))
    for ln in out:
        ln.refs = [new_pos[r] for r in ln.refs]
    out.append(Line(0, ("imp", A, p.concl), "IMPI", [new_pos[a_line], new_pos[n - 1]]))
    prem = [q for k, q in enumerate(p.prem) if k != j]
    q = Proof(prem, ("imp", A, p.concl), out)
    q = simplify(q)
    return q if q.verify()[0] else None


def unfold_discharged(prem: list, concl, k: int) -> tuple[list, tuple]:
    """Undo k discharges: (prem, A1 > (A2 > ... C)) -> (prem + [A1..Ak], C)."""
    ants = []
    for _ in range(k):
        assert concl[0] == "imp"
        ants.append(concl[1])
        concl = concl[2]
    return prem + ants, concl
