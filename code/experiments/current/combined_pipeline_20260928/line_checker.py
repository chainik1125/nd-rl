# ruff: noqa
# pyright: reportMissingImports=false
"""Leon's incremental ND line checker, verbatim: origin/leon/home-rolled-rl @ 0f749e2,
code/experiments/current/rl_hillclimb_run2_20260924/recipes/rl_champion2_n060_fullpool.py lines
166-322 (the imports, _GARITY, _prompt_sequent and LineChecker). Tokenizer-independent: it checks ND
proof lines (spec.md format), which guided.py gets from the fork's lean_seq parser.
Lint is off for this file so it stays byte-comparable with the source.
"""

# ---------------------------------------------------------------- verifier-guided (line-level rejection) sampling
from nd_verify import parse_formula as _pf, ParseError as _PE, RULE_NAMES as _RN
from nd_verify.verify import BOT as _BOT

_GARITY = {
    "ANDI": 2,
    "ANDE1": 1,
    "ANDE2": 1,
    "IMPE": 2,
    "IMPI": 2,
    "ORI1": 1,
    "ORI2": 1,
    "ORE": 5,
    "NEGE": 2,
    "NEGI": 2,
    "BOTE": 1,
    "DN": 1,
    "R": 1,
}


def _prompt_sequent(prompt):
    toks = prompt.split()
    prem, i = [], 1
    if toks[i] != "SEQ":
        while True:
            f, i = _pf(toks, i)
            prem.append(f)
            if toks[i] == ",":
                i += 1
                continue
            break
    concl, _ = _pf(toks, i + 1)
    return prem, concl


class LineChecker:
    """Incremental version of nd_verify's checks: add_line() accepts a proof line only if nd_verify would accept
    it given the lines before it (index order, premise block, box structure, citations, rule). done_ok() says
    whether QED here would give a valid proof. A full proof accepted line by line passes nd_verify."""

    def __init__(self, prem, concl):
        self.prem, self.concl = prem, concl
        self.lines, self.stack, self.nb = [], [], 0
        self.ctx, self.box_lines, self.box_depth, self.as_lines, self.fml = {}, {}, {}, {}, {}
        self.n_pr, self.in_pr = 0, True

    def add_line(self, toks):
        try:
            t = toks[0]
            if not (t.startswith("N") and t[1:].isdigit()):
                return False
            idx, i, d = int(t[1:]), 1, 0
            while i < len(toks) and toks[i] == "|":
                d += 1
                i += 1
            G, i = _pf(toks, i)
            if toks[i] != ":" or toks[i + 1] not in _RN:
                return False
            rule, i = toks[i + 1], i + 2
            refs = []
            while i < len(toks) and toks[i].startswith("N") and toks[i][1:].isdigit():
                refs.append(int(toks[i][1:]))
                i += 1
            if i != len(toks) - 1 or toks[i] != ";":
                return False
        except (_PE, IndexError, KeyError):
            return False
        L = self.lines
        idx0 = L[0]["idx"] if L else idx
        if idx < 1 or idx != idx0 + len(L):
            return False
        stack = list(self.stack)
        if rule == "PR":
            if (
                not self.in_pr
                or d != 0
                or self.n_pr >= len(self.prem)
                or G != self.prem[self.n_pr]
                or refs
            ):
                return False
        elif self.n_pr != len(self.prem):
            return False
        new_box = None
        if rule == "AS":
            if d < 1 or d > len(stack) + 1 or refs:
                return False
            del stack[d - 1 :]
            new_box = self.nb + 1
            stack.append(new_box)
        else:
            if d > len(stack):
                return False
            del stack[d:]
        ci = tuple(stack)
        if rule not in ("PR", "AS"):
            if len(refs) != _GARITY[rule] or any(not (idx0 <= r < idx) for r in refs):
                return False
            ctx, F = self.ctx, self.fml

            def lc(j):
                return ctx[j] == ci[: len(ctx[j])]

            def bc(s, e):
                if s not in self.as_lines:
                    return False
                b = self.as_lines[s]
                if (
                    e != self.box_lines.get(b)
                    or L[e - idx0]["depth"] != self.box_depth[b]
                    or e < s
                    or b in ci
                ):
                    return False
                par = ctx[s][:-1]
                return par == ci[: len(par)]

            if rule in ("IMPI", "NEGI"):
                if not bc(refs[0], refs[1]):
                    return False
            elif rule == "ORE":
                if not lc(refs[0]) or not bc(refs[1], refs[2]) or not bc(refs[3], refs[4]):
                    return False
            elif not all(lc(r) for r in refs):
                return False
            f = lambda k: F[refs[k]]
            try:
                if rule == "R":
                    ok = G == f(0)
                elif rule == "ANDI":
                    ok = G == ("and", f(0), f(1))
                elif rule == "ANDE1":
                    ok = f(0)[0] == "and" and f(0)[1] == G
                elif rule == "ANDE2":
                    ok = f(0)[0] == "and" and f(0)[2] == G
                elif rule == "IMPE":
                    ok = f(0) == ("imp", f(1), G)
                elif rule == "IMPI":
                    ok = G == ("imp", f(0), f(1))
                elif rule == "ORI1":
                    ok = G[0] == "or" and G[1] == f(0)
                elif rule == "ORI2":
                    ok = G[0] == "or" and G[2] == f(0)
                elif rule == "ORE":
                    ok = f(0) == ("or", f(1), f(3)) and f(2) == G and f(4) == G
                elif rule == "NEGE":
                    ok = f(1) == ("not", f(0)) and G == _BOT
                elif rule == "NEGI":
                    ok = f(1) == _BOT and G == ("not", f(0))
                elif rule == "BOTE":
                    ok = f(0) == _BOT
                else:  # DN
                    ok = f(0) == ("not", ("not", G))
            except (TypeError, IndexError):
                ok = False
            if not ok:
                return False
        # commit
        if rule == "PR":
            self.n_pr += 1
        else:
            self.in_pr = False
        if new_box is not None:
            self.nb = new_box
            self.box_depth[new_box] = d
            self.as_lines[idx] = new_box
        self.stack = stack
        for b in stack:
            self.box_lines[b] = idx
        self.ctx[idx] = ci
        self.fml[idx] = G
        L.append({"idx": idx, "depth": d, "rule": rule})
        return True

    def done_ok(self):
        if not self.lines or self.n_pr != len(self.prem):
            return False
        last = self.lines[-1]
        if last["depth"] != 0 or self.fml[last["idx"]] != self.concl or last["rule"] == "AS":
            return False
        return not (last["rule"] == "PR" and self.concl not in self.prem)
