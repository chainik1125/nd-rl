"""Shared parsing and feature code for the textbook-vs-generator gap analysis (analysis only).

    from common import thm_features, proof_features, load_sets

Pure CPU. The fork (`nd-takehome`) is imported read-only, with no bytecode written:
`nd_verify.verify.parse_formula` parses formulas, `intuit.intuit_provable` (G4ip) decides
intuitionistic validity.

Formula tuples are the fork's: ('atom', a) ('bot',) ('not', A) ('and', A, B) ('or', A, B)
('imp', A, B).

* `thm_features(prompt)`: features of the theorem statement only (comparable across every set,
  including textbook problems, which have no reference proof).
* `proof_features(prompt, proof)`: features of one proof in the fork's ND text format
  (`N1 F : PR ; N2 | G : AS ; ... ; QED`). Boxes are recovered from the closing rules' citations
  (IMPI s e, NEGI s e, ORE d s1 e1 s2 e2), so nesting is exact.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
if FORK not in sys.path:
    sys.path.insert(0, FORK)

from intuit import intuit_provable  # noqa: E402  (fork, read-only)
from nd_verify.verify import parse_formula  # noqa: E402  (fork, read-only)

BOT = ("bot",)
ATOMS = ["P", "Q", "R", "S"]
ROOT = Path("/scratch/cowifornia/nd-rl/think")
TB = ROOT / "nd-takehome/data/eval_only/textbook72"
K12 = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
POOL = ROOT / "runs/pools/rl_hard12.jsonl"
H12TEST = ROOT / "runs/restart2_hard12/hard12_test_1k.jsonl"
T1320 = ROOT / "runs/pools/test_13_20/test_13_20.jsonl"
T2130 = ROOT / "runs/pools/test_21_30/test_21_30.jsonl"
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/gap_analysis"


# ---------------------------------------------------------------- parsing
def parse_prompt(prompt: str) -> tuple[list[Any], Any]:
    toks = prompt.split()
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
    concl, _ = parse_formula(toks, i + 1)
    return prem, concl


def parse_proof(proof: str) -> list[dict[str, Any]]:
    """-> [{n, depth, f, rule, refs}] in order."""
    out = []
    for seg in proof.split(" ; "):
        toks = seg.split()
        if not toks or toks[0] == "QED":
            continue
        n = int(toks[0][1:])
        i = 1
        d = 0
        while toks[i] == "|":
            d += 1
            i += 1
        f, i = parse_formula(toks, i)
        assert toks[i] == ":", seg
        rule = toks[i + 1]
        refs = [int(t[1:]) for t in toks[i + 2 :] if t.startswith("N")]
        out.append({"n": n, "depth": d, "f": f, "rule": rule, "refs": refs})
    return out


def fstr(f: Any) -> str:
    if f[0] == "atom":
        return f[1]
    if f[0] == "bot":
        return "F"
    if f[0] == "not":
        return f"( ~ {fstr(f[1])} )"
    op = {"and": "&", "or": "v", "imp": ">"}[f[0]]
    return f"( {fstr(f[1])} {op} {fstr(f[2])} )"


# ---------------------------------------------------------------- formula helpers
def fdepth(f: Any) -> int:
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + fdepth(f[1])
    return 1 + max(fdepth(f[1]), fdepth(f[2]))


def subs(f: Any):
    yield f
    if f[0] == "not":
        yield from subs(f[1])
    elif f[0] in ("and", "or", "imp"):
        yield from subs(f[1])
        yield from subs(f[2])


def atoms(f: Any) -> set[str]:
    return {g[1] for g in subs(f) if g[0] == "atom"}


def imp_nest(f: Any) -> int:
    """Largest number of `>` on one root-to-leaf path."""
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return imp_nest(f[1])
    return (f[0] == "imp") + max(imp_nest(f[1]), imp_nest(f[2]))


def right_chain(f: Any) -> int:
    """Curried-implication chain length: A > (B > (C > D)) -> 3."""
    k = 0
    while f[0] == "imp":
        k += 1
        f = f[2]
    return k


def is_bicond(f: Any) -> bool:
    return (
        f[0] == "and"
        and f[1][0] == "imp"
        and f[2][0] == "imp"
        and f[1][1] == f[2][2]
        and f[1][2] == f[2][1]
    )


def ev(f: Any, v: dict[str, bool]) -> bool:
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


VALS = [dict(zip(ATOMS, bits, strict=True)) for bits in itertools.product([False, True], repeat=4)]


def entails(prem: list[Any], c: Any) -> bool:
    return all(ev(c, v) for v in VALS if all(ev(p, v) for p in prem))


# ---------------------------------------------------------------- theorem features
def thm_features(prompt: str) -> dict[str, Any]:
    prem, goal = parse_prompt(prompt)
    allf = [*prem, goal]
    sub_all = [g for f in allf for g in subs(f)]
    out: dict[str, Any] = {}
    out["n_prem"] = len(prem)
    out["no_prem"] = int(not prem)
    out["n_atoms"] = len(set().union(*(atoms(f) for f in allf)))
    out["fdepth"] = max(fdepth(f) for f in allf)
    out["goal_depth"] = fdepth(goal)
    out["prem_depth"] = max((fdepth(p) for p in prem), default=0)
    out["prompt_tokens"] = len(prompt.split()) - 3
    for op, name in (("not", "neg"), ("and", "and"), ("or", "or"), ("imp", "imp")):
        out[f"n_{name}"] = sum(g[0] == op for g in sub_all)
    out["n_conn"] = out["n_neg"] + out["n_and"] + out["n_or"] + out["n_imp"]
    out["n_bicond"] = sum(is_bicond(g) for g in sub_all)
    out["has_bicond"] = int(out["n_bicond"] > 0)
    out["has_bot"] = int(any(g == BOT for g in sub_all))
    # goal shape
    gt = goal[0] if goal[0] != "bot" else "bot"
    out["goal_type"] = gt
    for t in ("atom", "not", "and", "or", "imp"):
        out[f"goal_{t}"] = int(gt == t)
    out["goal_bicond"] = int(is_bicond(goal))
    out["goal_neg_compound"] = int(goal[0] == "not" and goal[1][0] not in ("atom", "bot"))
    out["goal_imp_compound_ante"] = int(goal[0] == "imp" and goal[1][0] not in ("atom", "bot"))
    out["goal_imp_or_ante"] = int(goal[0] == "imp" and goal[1][0] == "or")
    out["goal_imp_imp_ante"] = int(goal[0] == "imp" and goal[1][0] == "imp")
    out["goal_chain"] = right_chain(goal)
    out["goal_new_atoms"] = (
        len(atoms(goal) - set().union(*(atoms(p) for p in prem))) if prem else len(atoms(goal))
    )
    # negation shapes (De Morgan-like)
    out["n_neg_and"] = sum(g[0] == "not" and g[1][0] == "and" for g in sub_all)
    out["n_neg_or"] = sum(g[0] == "not" and g[1][0] == "or" for g in sub_all)
    out["n_neg_imp"] = sum(g[0] == "not" and g[1][0] == "imp" for g in sub_all)
    out["n_dneg"] = sum(g[0] == "not" and g[1][0] == "not" for g in sub_all)
    out["has_neg_compound"] = int(out["n_neg_and"] + out["n_neg_or"] + out["n_neg_imp"] > 0)
    out["has_dneg"] = int(out["n_dneg"] > 0)
    out["n_neg_atom"] = sum(g[0] == "not" and g[1][0] == "atom" for g in sub_all)
    # implication shapes
    out["imp_nest"] = max(imp_nest(f) for f in allf)
    out["n_imp_imp_ante"] = sum(g[0] == "imp" and g[1][0] == "imp" for g in sub_all)
    out["has_higher_order_imp"] = int(out["n_imp_imp_ante"] > 0)
    out["n_imp_compound_ante"] = sum(
        g[0] == "imp" and g[1][0] not in ("atom", "bot") for g in sub_all
    )
    # premise shapes
    for op, name in (
        ("not", "neg"),
        ("and", "and"),
        ("or", "or"),
        ("imp", "imp"),
        ("atom", "atom"),
    ):
        out[f"n_prem_{name}"] = sum(p[0] == op for p in prem)
    out["prem_neg_compound"] = sum(p[0] == "not" and p[1][0] not in ("atom", "bot") for p in prem)
    # logic
    out["classical_only"] = int(not intuit_provable(prompt))
    out["prem_inconsistent"] = int(
        bool(prem) and not any(all(ev(p, v) for p in prem) for v in VALS)
    )
    out["n_redundant_prem"] = sum(entails(prem[:i] + prem[i + 1 :], goal) for i in range(len(prem)))
    out["goal_tautology"] = int(entails([], goal))
    # classical sub-shapes (only meaningful when classical_only)
    co = out["classical_only"]
    out["cl_goal_or"] = int(co and goal[0] == "or")
    out["cl_goal_atom"] = int(co and goal[0] == "atom")
    out["cl_higher_order"] = int(co and out["has_higher_order_imp"])
    out["cl_neg_compound_prem"] = int(co and out["prem_neg_compound"] > 0)
    return out


# ---------------------------------------------------------------- proof features
INTRO = {"ANDI", "ORI1", "ORI2", "IMPI", "NEGI"}


def boxes_of(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every box: {s, e, kind, close} with kind in impi / negi / raa / ore1 / ore2. A NEGI box is
    'raa' when its conclusion ~~G is used by a DN (reductio)."""
    by_n = {ln["n"]: ln for ln in lines}
    dn_src = {ln["refs"][0] for ln in lines if ln["rule"] == "DN" and ln["refs"]}
    out = []
    for ln in lines:
        r = ln["rule"]
        if r == "IMPI":
            out.append({"s": ln["refs"][0], "e": ln["refs"][1], "kind": "impi", "close": ln["n"]})
        elif r == "NEGI":
            raa = ln["n"] in dn_src and by_n[ln["refs"][0]]["f"][0] == "not"
            out.append(
                {
                    "s": ln["refs"][0],
                    "e": ln["refs"][1],
                    "kind": "raa" if raa else "negi",
                    "close": ln["n"],
                }
            )
        elif r == "ORE":
            _, s1, e1, s2, e2 = ln["refs"]
            out.append({"s": s1, "e": e1, "kind": "ore1", "close": ln["n"], "ore": ln["n"]})
            out.append({"s": s2, "e": e2, "kind": "ore2", "close": ln["n"], "ore": ln["n"]})
    return out


def ancestors(box: dict[str, Any], boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        b
        for b in boxes
        if b is not box
        and b["s"] <= box["s"]
        and box["e"] <= b["e"]
        and (b["s"], b["e"]) != (box["s"], box["e"])
    ]


def proof_features(prompt: str, proof: str) -> dict[str, Any]:
    prem, goal = parse_prompt(prompt)
    lines = parse_proof(proof)
    by_n = {ln["n"]: ln for ln in lines}
    body = [ln for ln in lines if ln["rule"] != "PR"]
    boxes = boxes_of(lines)
    out: dict[str, Any] = {"p_lines": len(lines), "p_box_depth": max(ln["depth"] for ln in lines)}
    rules = [
        "AS",
        "R",
        "ANDI",
        "ANDE1",
        "ANDE2",
        "ORI1",
        "ORI2",
        "ORE",
        "IMPI",
        "IMPE",
        "NEGI",
        "NEGE",
        "DN",
        "BOTE",
    ]
    for r in rules:
        out[f"p_n_{r}"] = sum(ln["rule"] == r for ln in body)
    out["p_n_ANDE"] = out["p_n_ANDE1"] + out["p_n_ANDE2"]
    out["p_n_ORI"] = out["p_n_ORI1"] + out["p_n_ORI2"]
    for r in ["ORE", "IMPI", "IMPE", "NEGI", "NEGE", "DN", "BOTE", "ANDE", "ANDI", "ORI", "R"]:
        out[f"p_has_{r}"] = int(out[f"p_n_{r}"] > 0)
    out["p_n_boxes"] = out["p_n_AS"]
    kinds = [b["kind"] for b in boxes]
    out["p_n_raa"] = kinds.count("raa")
    out["p_has_raa"] = int(out["p_n_raa"] > 0)
    out["p_n_negi_plain"] = kinds.count("negi")
    out["p_has_negi_plain"] = int(out["p_n_negi_plain"] > 0)
    out["p_dn_not_raa"] = int(
        any(ln["rule"] == "DN" and by_n[ln["refs"][0]]["rule"] != "NEGI" for ln in body)
    )

    # nesting
    def inside(child_kinds: set[str], parent_kinds: set[str]) -> int:
        n = 0
        for b in boxes:
            if b["kind"] not in child_kinds:
                continue
            anc = ancestors(b, boxes)
            if any(a["kind"] in parent_kinds for a in anc):
                n += 1
        return n

    ORE = {"ore1", "ore2"}
    NEG = {"negi", "raa"}
    out["p_box_in_ore"] = int(inside({"impi", "negi", "raa", "ore1", "ore2"}, ORE) > 0)
    out["p_raa_in_ore"] = int(inside({"raa"}, ORE) > 0)
    out["p_negbox_in_ore"] = int(inside(NEG, ORE) > 0)
    out["p_impi_in_ore"] = int(inside({"impi"}, ORE) > 0)
    out["p_ore_in_ore"] = int(inside(ORE, ORE) > 0)
    out["p_ore_in_negbox"] = int(inside(ORE, NEG) > 0)
    out["p_ore_in_raa"] = int(inside(ORE, {"raa"}) > 0)
    out["p_ore_in_impi"] = int(inside(ORE, {"impi"}) > 0)
    out["p_negbox_in_negbox"] = int(inside(NEG, NEG) > 0)
    out["p_negbox_in_impi"] = int(inside(NEG, {"impi"}) > 0)
    out["p_impi_in_negbox"] = int(inside({"impi"}, NEG) > 0)
    out["p_impi_in_impi"] = int(inside({"impi"}, {"impi"}) > 0)
    out["p_two_negboxes"] = int(sum(k in NEG for k in kinds) >= 2)
    # ORE detail
    ores = [ln for ln in body if ln["rule"] == "ORE"]
    hyp_ns = {ln["n"] for ln in body if ln["rule"] == "AS"}
    prem_ns = {ln["n"] for ln in lines if ln["rule"] == "PR"}
    out["p_ore_on_prem"] = int(any(o["refs"][0] in prem_ns for o in ores))
    out["p_ore_on_hyp"] = int(any(o["refs"][0] in hyp_ns for o in ores))
    out["p_ore_on_derived"] = int(any(o["refs"][0] not in prem_ns | hyp_ns for o in ores))
    out["p_ore_goal_bot"] = int(any(o["f"] == BOT for o in ores))
    out["p_ore_goal_compound"] = int(any(o["f"][0] not in ("atom", "bot") for o in ores))
    # a branch that ends by ex falso (disjunctive-syllogism pattern)
    ds = 0
    for o in ores:
        for e in (o["refs"][2], o["refs"][4]):
            if by_n[e]["rule"] == "BOTE":
                ds = 1
    out["p_ore_branch_bote"] = ds
    # both branches non-trivial (>= 2 lines each)
    nt = 0
    for o in ores:
        _, s1, e1, s2, e2 = o["refs"]
        if e1 - s1 >= 1 and e2 - s2 >= 1:
            nt = 1
    out["p_ore_both_branches_work"] = nt
    out["p_ore_branch_intro_end"] = int(
        any(by_n[o["refs"][k]]["rule"] in INTRO for o in ores for k in (2, 4))
    )
    # NEGE detail: the positive half built by an intro rule (build X to clash with ~X)
    neges = [ln for ln in body if ln["rule"] == "NEGE"]

    def src(n: int) -> str:
        r = by_n[n]["rule"]
        return "prem" if r == "PR" else ("hyp" if r == "AS" else r)

    pos_src = [src(x["refs"][0]) for x in neges]
    neg_src = [src(x["refs"][1]) for x in neges]
    out["p_nege_pos_intro"] = int(any(s in INTRO for s in pos_src))
    out["p_nege_pos_elim"] = int(
        any(s in {"ANDE1", "ANDE2", "IMPE", "DN", "ORE", "BOTE"} for s in pos_src)
    )
    out["p_nege_neg_hyp"] = int(any(s == "hyp" for s in neg_src))
    out["p_nege_neg_prem"] = int(any(s == "prem" for s in neg_src))
    out["p_nege_neg_derived"] = int(any(s not in ("hyp", "prem") for s in neg_src))
    # IMPE on a hypothesis implication (higher-order use)
    out["p_impe_on_hyp"] = int(any(src(x["refs"][0]) == "hyp" for x in body if x["rule"] == "IMPE"))
    out["p_impe_ante_derived"] = int(
        any(src(x["refs"][1]) not in ("hyp", "prem", "R") for x in body if x["rule"] == "IMPE")
    )
    # raa hypothesis shape: assumed ~G for compound vs atomic G
    raa_h = [by_n[b["s"]]["f"] for b in boxes if b["kind"] == "raa"]
    out["p_raa_compound"] = int(any(h[1][0] not in ("atom", "bot") for h in raa_h))
    out["p_raa_atom"] = int(any(h[1][0] == "atom" for h in raa_h))
    out["p_raa_top"] = int(
        bool(body) and body[-1]["rule"] == "DN" and by_n[body[-1]["refs"][0]]["rule"] == "NEGI"
    )
    out["p_raa_inner"] = int(any(b["kind"] == "raa" and ancestors(b, boxes) for b in boxes))
    # direction of the first step and of the top rule
    out["p_top_rule"] = body[-1]["rule"] if body else "PR"
    out["p_first_rule"] = body[0]["rule"] if body else "PR"
    out["p_first_forward"] = int(bool(body) and body[0]["rule"] != "AS")
    # box sizes
    sizes = [b["e"] - b["s"] + 1 for b in boxes]
    out["p_max_box_lines"] = max(sizes, default=0)
    out["p_lines_depth2plus"] = sum(ln["depth"] >= 2 for ln in lines)
    out["p_lines_depth3plus"] = sum(ln["depth"] >= 3 for ln in lines)
    # contradiction count
    out["p_n_bot_lines"] = sum(ln["f"] == BOT for ln in body)
    # box results used as lemmas (cited by a later non-closing rule)
    closers = {b["close"]: b for b in boxes}
    ELIM = {"IMPE", "NEGE", "ANDE1", "ANDE2", "BOTE"}
    use_elim = {"impi": 0, "negi": 0, "raa": 0, "ore": 0}
    use_intro = 0
    for ln in body:
        if ln["rule"] in ("IMPI", "NEGI", "DN") or ln["rule"] == "ORE":
            cited = ln["refs"][:1] if ln["rule"] == "ORE" else []  # ORE's disjunction is a real use
        else:
            cited = ln["refs"]
        for r in cited:
            b = closers.get(r)
            if b is None:
                continue
            k = "ore" if b["kind"] in ("ore1", "ore2") else b["kind"]
            if ln["rule"] in ELIM or ln["rule"] == "ORE":
                use_elim[k] = 1
            elif ln["rule"] in ("ANDI", "ORI1", "ORI2"):
                use_intro = 1
    # a DN applied to a NEGI result counts as reductio, not a lemma; DN of the DN result used later:
    raa_close = {
        by_n[ln["n"]]["n"]
        for ln in body
        if ln["rule"] == "DN" and by_n[ln["refs"][0]]["rule"] == "NEGI"
    }
    raa_lemma = int(any(r in raa_close for ln in body for r in ln["refs"] if ln["rule"] != "DN"))
    out["p_impi_lemma"] = use_elim[
        "impi"
    ]  # an implication proved in a box, then applied (higher-order)
    out["p_negi_lemma"] = use_elim["negi"]  # a negation proved in a box, then fed to IMPE / NEGE
    out["p_raa_lemma"] = raa_lemma  # a reductio result used as a lemma
    out["p_ore_lemma"] = use_elim["ore"]
    out["p_box_lemma_elim"] = int(any(use_elim.values()) or raa_lemma)
    out["p_box_lemma_intro"] = use_intro
    out["p_negbox_nested"] = int(any(b["kind"] in NEG and ancestors(b, boxes) for b in boxes))
    out["p_elim_depth2plus"] = sum(
        ln["depth"] >= 2 and ln["rule"] in ELIM | {"ORE", "DN"} for ln in body
    )
    # ORE whose two branches end in different intro sides (the case split must route the goal)
    crossed = 0
    for o in ores:
        r1, r2 = by_n[o["refs"][2]]["rule"], by_n[o["refs"][4]]["rule"]
        if {r1, r2} == {"ORI1", "ORI2"}:
            crossed = 1
    out["p_ore_crossed_ori"] = crossed
    out["p_skeleton"] = skeleton(lines, boxes)
    return out


KIND_CH = {"impi": "I", "negi": "N", "raa": "C"}


def skeleton(lines: list[dict[str, Any]], boxes: list[dict[str, Any]]) -> str:
    """Box-nesting skeleton: I(..) implication box, N(..) negation box, C(..) reductio box,
    O(..|..) case split (its two branches); children in proof order. A proof without boxes is
    '-'."""
    top = [b for b in boxes if not ancestors(b, boxes)]

    def render(bs: list[dict[str, Any]]) -> str:
        bs = sorted(bs, key=lambda b: b["s"])
        out = []
        seen_ore = set()
        for b in bs:
            if b["kind"] in ("ore1", "ore2"):
                if b["ore"] in seen_ore:
                    continue
                seen_ore.add(b["ore"])
                br = sorted([x for x in bs if x.get("ore") == b["ore"]], key=lambda x: x["s"])
                parts = [render(children(x)) for x in br]
                out.append("O(" + "|".join(parts) + ")")
            else:
                out.append(KIND_CH[b["kind"]] + "(" + render(children(b)) + ")")
        return "".join(out)

    def children(b: dict[str, Any]) -> list[dict[str, Any]]:
        anc_free = []
        inner = [
            x
            for x in boxes
            if x is not b
            and b["s"] <= x["s"]
            and x["e"] <= b["e"]
            and (x["s"], x["e"]) != (b["s"], b["e"])
        ]
        for x in inner:
            # direct children only: no other inner box strictly between
            if not any(
                y is not x
                and y["s"] <= x["s"]
                and x["e"] <= y["e"]
                and (y["s"], y["e"]) != (x["s"], x["e"])
                for y in inner
            ):
                anc_free.append(x)
        return anc_free

    s = render(top)
    return s or "-"


# ---------------------------------------------------------------- data loading
def jl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def k12_sample(n: int = 20000, seed: int = 20261008) -> list[dict[str, Any]]:
    rows = jl(K12)
    rng = random.Random(seed)
    return rng.sample(rows, n)
