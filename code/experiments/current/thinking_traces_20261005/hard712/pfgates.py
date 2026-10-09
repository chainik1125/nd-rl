"""Gates and features for hard712 candidates (CPU only).

    from code.experiments.current.thinking_traces_20261005.hard712 import pfgates as GT
    reason, info = GT.shape_gate(prompt, gen_proof)       # T2 / T7 statement limits
    res = GT.certify(cand)                                # T1 + necessity + family

Gates, in the order `certify` applies them (cheap first):

* **shape (T2, T7)** on the statement: <= 2 premises, <= 7 connectives, 2-3 atoms, formula depth
  <= 3 (connectives on the deepest path, ~ included), no ~~ and no F anywhere, premises jointly
  satisfiable, no premise redundant (the goal does not follow classically without it), the goal
  is not a premise, no subformula X o X; at most `MAX_IRRELEVANT` irrelevant atom occurrences
  (`irrelevant`: replacing that one occurrence by a fresh atom leaves the theorem valid, e.g. an
  unused conjunct of an assumption or the unused half of a biconditional premise); not a
  substitution instance of a textbook72 theorem and no textbook72 theorem an instance of it
  (`textbook_match`; renaming classes are excluded separately); the constructed proof has 7-13
  lines and passes nd_verify.
* **shortest proof (T1)**: `fastlen.fast_minlen` (the fork's `minlen` search with a memo) with
  the constructed proof's formulas added to the search space, iterative deepening up to the
  constructed length. Keep when the shortest length L is 7-12 and the constructed proof has at
  most L + 1 lines. The proof found is the row's `reference_proof`.
* **target structure, and its necessity**: the shortest proof is classified into the target
  families below; for each structure it uses, `gap_analysis/necessity.restricted_min` searches
  again with the structure forbidden from L to L + `NEC_EXTRA` lines. A family counts only if its
  structure is *needed* (no proof without it within L + 2 lines in the search space).

  - `nested_neg` (T3): a negation or reductio box inside another box, whose body has >= 2
    elimination lines and uses the outer box's hypothesis; needed: no nested negation box.
  - `reductio` (T4): a reductio box (assume ~G, F, NEGI, DN) whose body holds a box or >= 3
    eliminations; needed: no DN.
  - `case_split` (T5): a case split inside an implication / negation / reductio box, both
    branches >= 2 lines; needed: no ORE.
  - `depth2` (T6): box depth >= 2 with >= 2 elimination lines at depth >= 2; needed: box depth
    <= 1.

Every feature of the gap analysis (`gap_analysis/common.thm_features` and `proof_features` of the
shortest proof, the latter prefixed `p_`) is recorded, plus the hard12 pool's `f_*` features.
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Any

from . import ext

FL = ext.fastlen()
GC = ext.gap_common()
from nd_verify import verify_text  # noqa: E402  (fork, on sys.path via fastlen)

MAX_PREM = 2
MAX_CONN = 7
MIN_ATOMS, MAX_ATOMS = 2, 3
MAX_FDEPTH = 3
MIN_L, MAX_L = 7, 12
MAX_GAP = 1
NEC_EXTRA = 2
SEARCH_SECS = 20.0
NEC_SECS = 20.0
MAX_IRRELEVANT = 2
TEXTBOOK = "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72"
ELIM = {"ANDE1", "ANDE2", "IMPE", "NEGE", "BOTE", "DN", "ORE"}
FAMILIES = ("nested_neg", "reductio", "case_split", "depth2")
NEC_KIND = {
    "nested_neg": "nested_neg",
    "reductio": "no_dn",
    "case_split": "no_ore",
    "depth2": "flat",
}


# ---------------------------------------------------------------------- statement shape
def statement(prompt: str) -> tuple[list[Any], Any]:
    return GC.parse_prompt(prompt)


_ATOMS5 = ("P", "Q", "R", "S", "T")
_VALS5 = [dict(zip(_ATOMS5, bits, strict=True)) for bits in itertools.product([0, 1], repeat=5)]


def _valid5(prem: list[Any], goal: Any) -> bool:
    return all(GC.ev(goal, v) for v in _VALS5 if all(GC.ev(p, v) for p in prem))


def _occ(f: Any, path: tuple[int, ...] = ()):
    if f[0] == "atom":
        yield path
    elif f[0] == "not":
        yield from _occ(f[1], (*path, 1))
    elif f[0] in ("and", "or", "imp"):
        yield from _occ(f[1], (*path, 1))
        yield from _occ(f[2], (*path, 2))


def _repl(f: Any, path: tuple[int, ...], g: Any) -> Any:
    if not path:
        return g
    out = list(f)
    out[path[0]] = _repl(f[path[0]], path[1:], g)
    return tuple(out)


def irrelevant(prem: list[Any], goal: Any) -> int:
    """Atom occurrences of the statement that do not matter: replacing that single occurrence by
    a fresh atom T leaves the theorem classically valid."""
    allf = [*prem, goal]
    n = 0
    for i, f in enumerate(allf):
        for p in _occ(f):
            new = [*allf[:i], _repl(f, p, ("atom", "T")), *allf[i + 1 :]]
            n += _valid5(new[:-1], new[-1])
    return n


def _match(pat: Any, f: Any, sig: dict[str, Any]) -> bool:
    if pat[0] == "atom":
        if pat[1] in sig:
            return sig[pat[1]] == f
        sig[pat[1]] = f
        return True
    if pat[0] != f[0]:
        return False
    if pat[0] == "bot":
        return True
    if pat[0] == "not":
        return _match(pat[1], f[1], sig)
    return _match(pat[1], f[1], sig) and _match(pat[2], f[2], sig)


def instance_of(pat: tuple[list[Any], Any], thm: tuple[list[Any], Any]) -> bool:
    """thm is a substitution instance of pat (atoms -> formulas), premises in any order."""
    (pp, pg), (tp, tg) = pat, thm
    if len(pp) != len(tp):
        return False
    for perm in itertools.permutations(tp):
        sig: dict[str, Any] = {}
        if _match(pg, tg, sig) and all(_match(a, b, sig) for a, b in zip(pp, perm, strict=True)):
            return True
    return False


_TB: list[tuple[list[Any], Any]] = []


def textbook_theorems() -> list[tuple[list[Any], Any]]:
    """textbook72 statements, parsed (used only to exclude candidates, never to build them)."""
    if not _TB:
        for fn in ("textbook_dev.jsonl", "textbook_train.jsonl"):
            with open(f"{TEXTBOOK}/{fn}") as f:
                for line in f:
                    if line.strip():
                        _TB.append(statement(json.loads(line)["prompt"]))
    return _TB


def textbook_match(prem: list[Any], goal: Any) -> str | None:
    """'instance' if the theorem is a substitution instance of a textbook72 theorem, 'general'
    if a textbook72 theorem is an instance of it, else None."""
    for t in textbook_theorems():
        if instance_of(t, (prem, goal)):
            return "instance"
        if instance_of((prem, goal), t):
            return "general"
    return None


def shape_gate(prompt: str, gen_proof: str) -> tuple[str | None, dict[str, Any]]:
    """(rejection reason or None, statement facts)."""
    prem, goal = statement(prompt)
    allf = [*prem, goal]
    info: dict[str, Any] = {
        "n_prem": len(prem),
        "n_conn": sum(_conn(f) for f in allf),
        "n_atoms": len(set().union(*(GC.atoms(f) for f in allf))),
        "fdepth": max(GC.fdepth(f) for f in allf),
    }
    nl = gen_proof.split().count(";")
    info["gen_lines"] = nl
    if len(prem) > MAX_PREM:
        return "prem", info
    if info["n_conn"] > MAX_CONN:
        return "conn", info
    if not MIN_ATOMS <= info["n_atoms"] <= MAX_ATOMS:
        return "atoms", info
    if info["fdepth"] > MAX_FDEPTH:
        return "fdepth", info
    sub = [g for f in allf for g in GC.subs(f)]
    if any(g[0] == "not" and g[1][0] == "not" for g in sub):
        return "dneg", info
    if any(g == GC.BOT for g in sub):
        return "bot", info
    if any(g[0] in ("and", "or", "imp") and g[1] == g[2] for g in sub):
        return "degenerate", info
    if goal in prem:
        return "goal_is_prem", info
    if len(set(prem)) < len(prem):
        return "dup_prem", info
    if prem and not any(all(GC.ev(p, v) for p in prem) for v in GC.VALS):
        return "inconsistent", info
    if any(GC.entails(prem[:i] + prem[i + 1 :], goal) for i in range(len(prem))):
        return "redundant_prem", info
    if not MIN_L <= nl <= MAX_L + MAX_GAP:
        return "gen_len", info
    info["n_irrelevant"] = irrelevant(prem, goal)
    if info["n_irrelevant"] > MAX_IRRELEVANT:
        return "irrelevant", info
    tm = textbook_match(prem, goal)
    if tm:
        return "textbook_" + tm, info
    ok, _, n = verify_text(prompt + " " + gen_proof)
    if not ok or n != nl:
        return "gen_invalid", info
    return None, info


def _conn(f: Any) -> int:
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + _conn(f[1])
    return 1 + _conn(f[1]) + _conn(f[2])


# ---------------------------------------------------------------------- proof structure
def line_deps(lines: list[dict[str, Any]]) -> dict[int, frozenset[int]]:
    """Per line number: the AS lines it depends on (transitively, discharged boxes removed)."""
    out: dict[int, frozenset[int]] = {}
    for ln in lines:
        r, n = ln["rule"], ln["n"]
        if r == "PR":
            out[n] = frozenset()
        elif r == "AS":
            out[n] = frozenset([n])
        elif r in ("IMPI", "NEGI"):
            s, e = ln["refs"]
            out[n] = out[e] - {s}
        elif r == "ORE":
            j, s1, e1, s2, e2 = ln["refs"]
            out[n] = out[j] | (out[e1] - {s1}) | (out[e2] - {s2})
        else:
            out[n] = frozenset().union(*(out[x] for x in ln["refs"])) if ln["refs"] else frozenset()
    return out


def structure(prompt: str, proof: str) -> dict[str, Any]:
    """Target-family structure of one proof (the shortest proof): which families' structures it
    has (before necessity), and the T3 / T4 / T5 detail flags."""
    lines = GC.parse_proof(proof)
    by_n = {ln["n"]: ln for ln in lines}
    boxes = GC.boxes_of(lines)
    dep = line_deps(lines)
    out: dict[str, Any] = {}

    def inside(b: dict[str, Any]) -> list[dict[str, Any]]:
        return [ln for ln in lines if b["s"] < ln["n"] <= b["e"]]

    # T3: nested negation / reductio box with work that uses the outer hypothesis
    t3 = t3_reuse = 0
    t3_any = 0
    for b in boxes:
        if b["kind"] not in ("negi", "raa"):
            continue
        anc = GC.ancestors(b, boxes)
        if not anc:
            continue
        t3_any = 1
        body = inside(b)
        n_el = sum(ln["rule"] in ELIM for ln in body)
        outer_as = {a["s"] for a in anc if a["kind"] in ("impi", "negi", "raa")}
        uses_outer = any(dep[ln["n"]] & outer_as for ln in body)
        if n_el >= 2 and uses_outer:
            t3 = 1
            if any(ln["rule"] in ("IMPE", "NEGE") and b["close"] in ln["refs"] for ln in lines):
                t3_reuse = 1
    out["s_nested_neg_any"] = t3_any
    out["s_nested_neg_work"] = t3
    out["s_nested_neg_reused"] = t3_reuse
    # T4: reductio box with work inside (a nested box or >= 3 eliminations)
    t4 = 0
    for b in boxes:
        if b["kind"] != "raa":
            continue
        body = inside(b)
        has_box = any(x is not b and b["s"] < x["s"] and x["e"] <= b["e"] for x in boxes)
        if has_box or sum(ln["rule"] in ELIM for ln in body) >= 3:
            t4 = 1
    out["s_raa_work"] = t4
    # T5: case split inside an I / N / C box with both branches >= 2 lines
    t5 = t5_hyp = 0
    for ln in lines:
        if ln["rule"] != "ORE":
            continue
        j, s1, e1, s2, e2 = ln["refs"]
        br = [b for b in boxes if b.get("ore") == ln["n"]]
        anc = GC.ancestors(br[0], boxes) if br else []
        in_box = any(a["kind"] in ("impi", "negi", "raa") for a in anc)
        if in_box and e1 - s1 >= 1 and e2 - s2 >= 1:
            t5 = 1
            if by_n[j]["rule"] == "AS":
                t5_hyp = 1
    out["s_split_in_box_work"] = t5
    out["s_split_on_hyp"] = t5_hyp
    # T6: depth >= 2 with >= 2 eliminations there
    d2 = sum(ln["depth"] >= 2 and ln["rule"] in ELIM for ln in lines)
    out["s_elim_depth2"] = d2
    out["s_depth2_work"] = int(max(ln["depth"] for ln in lines) >= 2 and d2 >= 2)
    out["has"] = {
        "nested_neg": t3,
        "reductio": t4,
        "case_split": t5,
        "depth2": out["s_depth2_work"],
    }
    return out


# ---------------------------------------------------------------------- search gates
def shortest(prompt: str, gen_proof: str, bound: int, secs: float = SEARCH_SECS) -> dict[str, Any]:
    extra = tuple(ln["f"] for ln in GC.parse_proof(gen_proof))
    return FL.fast_minlen(prompt, bound=bound, time_limit=secs, extra=extra)


def necessity(
    prompt: str, kind: str, L: int, extra: tuple[Any, ...], secs: float = NEC_SECS
) -> dict[str, Any]:
    """Restricted re-search; adds `verdict`: needed / longer / same / timeout."""
    NE = ext.gap_necessity()
    r = NE.restricted_min(prompt, kind, L, NEC_EXTRA, secs, extra=extra)
    if r.get("timeout"):
        v = "timeout"
    elif r.get("len") is None:
        v = "needed"
    elif r["len"] == L:
        v = "same"
    else:
        v = "longer"
    r["verdict"] = v
    return r


def certify(cand: dict[str, Any], secs: float = SEARCH_SECS) -> dict[str, Any]:
    """T1 and the necessity gate for one shape-passed candidate. -> cand plus `status` ('ok' or a
    rejection reason) and, when ok, `min_lines`, `reference_proof`, `families`, `necessity`, the
    structure flags and timings."""
    t0 = time.time()
    out = dict(cand)
    C = cand["gen_lines"]
    r = shortest(cand["prompt"], cand["proof"], C, secs)
    out["t_search"] = round(time.time() - t0, 3)
    if r.get("timeout"):
        out["status"] = "search_timeout"
        return out
    if r.get("error") or r.get("min_lines_ub") is None:
        out["status"] = "search_none"
        return out
    L = r["min_lines_ub"]
    out["min_lines"] = L
    if L < MIN_L:
        out["status"] = "short"
        return out
    if L > MAX_L:
        out["status"] = "long"
        return out
    if C - L > MAX_GAP:
        out["status"] = "gap"
        return out
    proof = r["proof"]
    out["reference_proof"] = proof
    st = structure(cand["prompt"], proof)
    out["structure"] = {k: v for k, v in st.items() if k != "has"}
    present = [f for f in FAMILIES if st["has"][f]]
    if not present:
        out["status"] = "no_structure"
        return out
    extra = tuple(ln["f"] for ln in GC.parse_proof(cand["proof"])) + tuple(
        ln["f"] for ln in GC.parse_proof(proof)
    )
    nec: dict[str, Any] = {}
    fams = []
    for f in present:
        t1 = time.time()
        n = necessity(cand["prompt"], NEC_KIND[f], L, extra)
        nec[f] = {"verdict": n["verdict"], "len": n.get("len"), "secs": round(time.time() - t1, 3)}
        if n["verdict"] == "needed":
            fams.append(f)
    out["necessity"] = nec
    out["families"] = fams
    out["t_total"] = round(time.time() - t0, 3)
    out["status"] = "ok" if fams else "not_needed"
    return out
