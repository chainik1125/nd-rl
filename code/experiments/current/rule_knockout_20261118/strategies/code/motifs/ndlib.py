# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Parse Fitch-style ND proofs (spec: `N7 | ( P > Q ) : IMPE N1 N3 ; ... ; QED`) into a dependency graph."""

import gzip, json

D = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014/data/"
LIVE = "/Users/robbie/repos/nd-rl/artifacts/rule_knockout_20261118/live/"


def pf(toks, i=0):
    """formula tokens -> (tree, next index). tree: 'P' | 'F' | ('~',A) | (op,A,B)"""
    t = toks[i]
    if t != "(":
        return t, i + 1
    if toks[i + 1] == "~":
        a, j = pf(toks, i + 2)
        assert toks[j] == ")"
        return ("~", a), j + 1
    a, j = pf(toks, i + 1)
    op = toks[j]
    b, k = pf(toks, j + 1)
    assert toks[k] == ")"
    return (op, a, b), k + 1


def conn(f):
    return f[0] if isinstance(f, tuple) else ("F" if f == "F" else "atom")


class Line:
    __slots__ = ("i", "d", "f", "rule", "c", "box", "users", "live", "txt")


class Proof:
    """lines: dict idx->Line. boxes: list of dicts {id, start, end, depth, parent, closer}"""

    def __init__(self, text):
        self.L = {}
        self.order = []
        for s in text.split(" ; "):
            s = s.strip()
            if s == "QED" or not s:
                continue
            toks = s.split()
            ln = Line()
            ln.txt = s
            ln.i = int(toks[0][1:])
            j = 1
            while toks[j] == "|":
                j += 1
            ln.d = j - 1
            k = toks.index(":", j)
            ln.f, e = pf(toks, j)
            assert e == k, s
            ln.rule = toks[k + 1]
            ln.c = [int(t[1:]) for t in toks[k + 2 :]]
            ln.users = []
            ln.live = False
            self.L[ln.i] = ln
            self.order.append(ln.i)
        # boxes
        self.boxes = []
        stack = []  # open box ids
        for i in self.order:
            ln = self.L[i]
            if ln.rule == "AS":
                while len(stack) >= ln.d:
                    stack.pop()
                b = dict(
                    id=len(self.boxes),
                    start=i,
                    end=i,
                    depth=ln.d,
                    parent=stack[-1] if stack else None,
                    closer=None,
                    lines=[],
                )
                self.boxes.append(b)
                stack.append(b["id"])
            else:
                while len(stack) > ln.d:
                    stack.pop()
            ln.box = stack[-1] if stack else None
            if stack:
                for bid in stack:
                    self.boxes[bid]["end"] = i
                self.boxes[stack[-1]]["lines"].append(i)
        self.bystart = {b["start"]: b for b in self.boxes}
        for i in self.order:
            ln = self.L[i]
            for c in ln.c:
                if c in self.L:
                    self.L[c].users.append(i)
            if ln.rule in ("IMPI", "NEGI"):
                b = self.bystart.get(ln.c[0])
                if b is not None:
                    b["closer"] = i
            elif ln.rule == "ORE":
                for s in (ln.c[1], ln.c[3]):
                    b = self.bystart.get(s)
                    if b is not None:
                        b["closer"] = i
        self.last = self.order[-1]
        self.goal = self.L[self.last].f
        # live = backward reachable from last line
        st = [self.last]
        while st:
            i = st.pop()
            ln = self.L.get(i)
            if ln is None or ln.live:
                continue
            ln.live = True
            st.extend(ln.c)
        self.nprem = sum(1 for i in self.order if self.L[i].rule == "PR")
        self.body = len(self.order) - self.nprem

    def box_of_closer(self, i):
        """boxes closed by line i (IMPI/NEGI: 1, ORE: 2)"""
        ln = self.L[i]
        if ln.rule in ("IMPI", "NEGI"):
            return [self.bystart[ln.c[0]]]
        if ln.rule == "ORE":
            return [self.bystart[ln.c[1]], self.bystart[ln.c[3]]]
        return []

    def in_box(self, i, b):
        return b["start"] <= i <= b["end"]

    def anc_boxes(self, i):
        out = []
        bid = self.L[i].box
        while bid is not None:
            out.append(self.boxes[bid])
            bid = self.boxes[bid]["parent"]
        return out


def pretrain():
    for l in gzip.open(D + "pretrain.jsonl.gz", "rt"):
        yield json.loads(l)


def jl(path):
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            yield json.loads(l)
        except Exception:
            pass  # file may be mid-write


def fstr(f):
    if not isinstance(f, tuple):
        return f
    if f[0] == "~":
        return f"(~{fstr(f[1])})"
    return f"({fstr(f[1])}{f[0]}{fstr(f[2])})"
