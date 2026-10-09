"""Why are the hardest textbook problems' reference proofs so long, and which small pieces do they need?

For every evaluated textbook problem (scraped 269 + derived 132, from per_problem.csv) that the RL
model never solved on either seed or whose reference proof has >= 14 lines, this script:

  1. parses the reference proof and attributes every line to what it pays for: premise unpacking
     (ANDE on premises), case splits (ORI/ORE and the AS lines of case boxes), the content inside
     case boxes, classical detours (NEGI boxes on a negated goal followed by DN, NEGE/BOTE/DN),
     conditional boxes (AS + IMPI), chain steps (IMPE), construction (ANDI/ORI), reiteration;
  2. measures the structure: box depth, number of boxes by closing rule, nested ORE, longest IMPE
     chain, lines at depth 0 vs inside boxes, number of distinct 'case leaves';
  3. extracts every short sub-theorem (2-6 lines) the proof contains (same extractor as
     subproblems.py), names its curry-aware class, and checks whether Dan's K12 pool contains that
     class and how many K12 rows have the same rule multiset ('rule signature');
  4. reports how much of each hard proof is made of pieces K12 already knows, and the most frequent
     unknown pieces across all hard proofs: the candidate 'small problems that carry the hard rules'.

    cd code && uv run python pretrain_data/hard_problems.py [--out ../docs/pretrain-data/hard_problems.md]

The sealed 72 are not read. Candidate-set problems are inspected per problem (they are not sealed).
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../textbook_problem"), os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import eval_analysis as EA  # noqa: E402
import prove  # noqa: E402
from nd_data import canonical_theorem  # noqa: E402
from patterns import fstr, parse, prune_lines  # noqa: E402
from release_1002 import BR  # noqa: E402
from subproblems import subproblem  # noqa: E402

L = prove.L
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(ROOT, "experiments/textbook-problem/eval_dan_best_state/analysis/per_problem.csv")
K12 = os.path.join("/private/tmp/claude-501/-Users-cyu-Desktop-github-nd-rl/0e8fdc82-572d-4fd1-ba25-ff48dbf29acf/scratchpad",
                   "dan_data", "train_k12.jsonl.gz")


# ---------------------------------------------------------------- structure
def boxes_of(lines):
    """Box tree. Each box: dict(as=pos, end=pos (last line inside), depth, close=rule of the line that
    discharges it (IMPI/NEGI/ORE) or None, parent=box index or None)."""
    boxes = []
    stack = []  # open boxes (indices into boxes)
    for p, ln in enumerate(lines):
        d = ln["depth"]
        while stack and lines[boxes[stack[-1]]["as_"]]["depth"] > d:
            boxes[stack.pop()]["end"] = p - 1
        if ln["rule"] == "AS":
            # an AS at the same depth as the open box closes it (sibling box)
            if stack and lines[boxes[stack[-1]]["as_"]]["depth"] == d:
                boxes[stack.pop()]["end"] = p - 1
            boxes.append(dict(as_=p, depth=d, end=None, close=None, parent=stack[-1] if stack else None))
            stack.append(len(boxes) - 1)
    while stack:
        boxes[stack.pop()]["end"] = len(lines) - 1
    byidx = {ln["idx"]: p for p, ln in enumerate(lines)}
    for b in boxes:
        as_idx = lines[b["as_"]]["idx"]
        for p in range(b["end"] + 1, len(lines)):
            ln = lines[p]
            if as_idx in ln["refs"] and ln["depth"] < lines[b["as_"]]["depth"]:
                b["close"] = ln["rule"]
                b["close_pos"] = p
                break
    return boxes


def box_of(boxes, p):
    """innermost box containing position p, or None"""
    best = None
    for i, b in enumerate(boxes):
        if b["as_"] <= p <= b["end"] and (best is None or b["depth"] > boxes[best]["depth"]):
            best = i
    return best


def attribute(lines, goal):
    """-> Counter of line categories, plus structure dict."""
    boxes = boxes_of(lines)
    byidx = {ln["idx"]: ln for ln in lines}
    pos = {ln["idx"]: p for p, ln in enumerate(lines)}
    cat = collections.Counter()
    prem_idx = {ln["idx"] for ln in lines if ln["rule"] == "PR"}
    # reductio: NEGI box whose AS is ( ~ G' ) and a later DN cites the NEGI line
    dn_from = {byidx[ln["refs"][0]]["idx"] for ln in lines if ln["rule"] == "DN" and ln["refs"] and ln["refs"][0] in byidx}
    reductio_boxes = set()
    for i, b in enumerate(boxes):
        if b["close"] == "NEGI" and lines[b["close_pos"]]["idx"] in dn_from:
            reductio_boxes.add(i)
    ore_boxes = {i for i, b in enumerate(boxes) if b["close"] == "ORE"}
    impi_boxes = {i for i, b in enumerate(boxes) if b["close"] == "IMPI"}
    negi_boxes = {i for i, b in enumerate(boxes) if b["close"] == "NEGI"}
    for p, ln in enumerate(lines):
        r = ln["rule"]
        if r == "PR":
            continue
        if r == "AS":
            bi = box_of(boxes, p)
            c = boxes[bi]["close"]
            cat["AS:" + ("reductio" if bi in reductio_boxes else (c or "unclosed"))] += 1
        elif r == "ORE":
            cat["ORE"] += 1
        elif r == "ORI":
            # a manufactured split: the ORI line is the disjunction of a later ORE
            if any(x["rule"] == "ORE" and x["refs"] and x["refs"][0] == ln["idx"] for x in lines):
                cat["ORI:manufactured"] += 1
            else:
                cat["ORI"] += 1
        elif r in ("ANDE1", "ANDE2"):
            cat["ANDE:prem" if ln["refs"] and ln["refs"][0] in prem_idx else "ANDE"] += 1
        elif r == "DN":
            cat["DN:reductio" if ln["refs"] and ln["refs"][0] in {lines[boxes[i]["close_pos"]]["idx"] for i in reductio_boxes} else "DN"] += 1
        elif r == "NEGI":
            bi = next((i for i, b in enumerate(boxes) if b.get("close_pos") == p), None)
            cat["NEGI:reductio" if bi in reductio_boxes else "NEGI"] += 1
        else:
            cat[r] += 1
    # lines inside boxes by closing rule of the innermost box
    inside = collections.Counter()
    for p, ln in enumerate(lines):
        if ln["rule"] == "PR":
            continue
        bi = box_of(boxes, p)
        inside["depth0" if bi is None else ("reductio" if bi in reductio_boxes else (boxes[bi]["close"] or "unclosed"))] += 1
    # structure
    nested_ore = sum(1 for i in ore_boxes if boxes[i]["parent"] is not None and boxes[i]["parent"] in ore_boxes)
    ore_in_negi = sum(1 for i in ore_boxes if boxes[i]["parent"] is not None and boxes[i]["parent"] in negi_boxes)
    negi_in_ore = sum(1 for i in negi_boxes if boxes[i]["parent"] is not None and boxes[i]["parent"] in ore_boxes)
    # longest IMPE chain
    chain = {}
    for ln in lines:
        if ln["rule"] == "IMPE":
            prev = max((chain.get(r, 0) for r in ln["refs"]), default=0)
            chain[ln["idx"]] = prev + 1
    rules = collections.Counter(ln["rule"] for ln in lines if ln["rule"] not in ("PR", "AS"))
    struct = dict(
        n_lines=len(lines), n_prem=len(prem_idx), max_depth=max(ln["depth"] for ln in lines),
        n_boxes=len(boxes), boxes_ore=len(ore_boxes), boxes_impi=len(impi_boxes), boxes_negi=len(negi_boxes),
        boxes_reductio=len(reductio_boxes), nested_ore=nested_ore, ore_in_negi=ore_in_negi, negi_in_ore=negi_in_ore,
        impe_chain=max(chain.values(), default=0), ore_lines=rules["ORE"], dn_lines=rules["DN"],
        and_lines=rules["ANDE1"] + rules["ANDE2"] + rules["ANDI"], inside=dict(inside), rules=dict(rules),
    )
    return cat, struct


def dominant_cause(cat, struct):
    """one label per proof: what most of the non-premise lines pay for."""
    n = struct["n_lines"] - struct["n_prem"]
    case = cat["ORE"] + cat["AS:ORE"] + cat["ORI:manufactured"] + struct["inside"].get("ORE", 0) - cat["AS:ORE"]
    classical = cat["AS:reductio"] + cat["NEGI:reductio"] + cat["DN:reductio"] + cat["DN"] + cat["NEGE"] + cat["BOTE"] + struct["inside"].get("reductio", 0) - cat["AS:reductio"]
    cond = cat["AS:IMPI"] + cat["IMPI"] + struct["inside"].get("IMPI", 0) - cat["AS:IMPI"]
    shuffle = cat["ANDE:prem"] + cat["ANDE"] + cat["ANDI"] + cat["ORI"]
    chain = cat["IMPE"]
    parts = {"case split": case, "classical": classical, "conditional": cond, "∧/∨ shuffle": shuffle, "→E chain": chain}
    top = max(parts, key=parts.get)
    return top, {k: round(v / max(1, n), 2) for k, v in parts.items()}


# ---------------------------------------------------------------- skeletons
def subformulas(f):
    yield f
    if f[0] == "not":
        yield from subformulas(f[1])
    elif f[0] in ("and", "or", "imp"):
        yield from subformulas(f[1])
        yield from subformulas(f[2])


def n_bicond(f):
    """count of subformulas of the shape ( ( a > b ) & ( b > a ) ) (how textbooks' <-> is transcribed)"""
    return sum(1 for s in subformulas(f) if s[0] == "and" and s[1][0] == "imp" and s[2][0] == "imp"
               and s[1][1] == s[2][2] and s[1][2] == s[2][1])


SKEL_ATOMS = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def skeleton(lines):
    """Rule skeleton of a proof. A compound subformula is *transparent* when the proof uses its
    structure: it is a line formula and one of its immediate subformulas is also a line formula
    (for ¬¬t, t counts). Every other maximal compound subformula becomes an atom, the same atom for
    the same formula. -> (prompt over the new atoms, abstracted lines)."""
    F = {ln["formula"] for ln in lines}
    used_atoms = {t[1] for f in F for t in subformulas(f) if t[0] == "atom"}
    fresh = [x for x in SKEL_ATOMS if x not in used_atoms]
    opaque = {}

    def transparent(f):
        if f not in F:
            return False
        if f[0] == "not":
            return f[1] in F or (f[1][0] == "not" and f[1][1] in F)
        return f[1] in F or f[2] in F

    def abstract(f):
        if f[0] == "atom" or f[0] == "bot":
            return f
        if f[0] == "not" and f[1] not in F and f[1][0] == "not" and f[1][1] in F:
            return ("not", ("not", abstract(f[1][1])))   # ¬¬t used by DN: keep both negations
        if not transparent(f):
            if f not in opaque:
                opaque[f] = ("atom", fresh[len(opaque)] if len(opaque) < len(fresh) else f"Z{len(opaque)}")
            return opaque[f]
        if f[0] == "not":
            return ("not", abstract(f[1]))
        return (f[0], abstract(f[1]), abstract(f[2]))

    new = [dict(ln, formula=abstract(ln["formula"])) for ln in lines]
    # rename atoms by first appearance (premises first, then goal, then the rest) so that
    # canonical_theorem, which permutes only P Q R S, sees a canonical input
    order = []
    for ln in [x for x in new if x["rule"] == "PR"] + [new[-1]] + new:
        for t in subformulas(ln["formula"]):
            if t[0] == "atom" and t[1] not in order:
                order.append(t[1])
    ren = {a: SKEL_ATOMS[i] if i < len(SKEL_ATOMS) else a for i, a in enumerate(order)}

    def rn(f):
        if f[0] == "atom":
            return ("atom", ren[f[1]])
        if f[0] == "bot":
            return f
        if f[0] == "not":
            return ("not", rn(f[1]))
        return (f[0], rn(f[1]), rn(f[2]))

    new = [dict(ln, formula=rn(ln["formula"])) for ln in new]
    prem = [ln["formula"] for ln in new if ln["rule"] == "PR"]
    goal = new[-1]["formula"]
    import transcribe as T
    return T.make_prompt(prem, goal), new


def skeleton_key(lines):
    prompt, _ = skeleton(lines)
    return canonical_theorem(prompt)


# ---------------------------------------------------------------- K12 index
def load_k12():
    """-> (class index: key -> count, signature index, skeleton index, n rows)"""
    cls = collections.Counter()
    sig = collections.Counter()
    skel = collections.Counter()
    bg = collections.Counter()
    n = 0
    with gzip.open(K12, "rt") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            for k in BR.keys(r["prompt"]):
                cls[k] += 1
            lines = parse(r["proof"])
            if lines:
                pr = prune_lines(lines)
                sig[signature(pr)] += 1
                bg.update(bigrams(pr))
                try:
                    skel[skeleton_key(pr)] += 1
                except Exception:
                    pass
    return cls, sig, skel, bg, n


def bigrams(lines):
    """(rule of a line, rule of a line it cites) pairs; AS and PR kept as cited rules."""
    byidx = {ln["idx"]: ln for ln in lines}
    out = collections.Counter()
    for ln in lines:
        if ln["rule"] in ("PR", "AS"):
            continue
        for r in ln["refs"]:
            if r in byidx:
                out[(ln["rule"], byidx[r]["rule"])] += 1
    return out


def signature(lines):
    return tuple(sorted(collections.Counter(ln["rule"] for ln in lines if ln["rule"] not in ("PR",)).items()))


def rule_sig_str(sig):
    return " ".join(f"{r}×{c}" if c > 1 else r for r, c in sig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/pretrain-data/hard_problems.md"))
    ap.add_argument("--json", default=os.path.join(ROOT, "experiments/textbook-problem/eval_dan_best_state/analysis/hard_problems.json"))
    ap.add_argument("--max_piece", type=int, default=6)
    ap.add_argument("--min_lines", type=int, default=14)
    a = ap.parse_args()

    refs = EA.load_refs()
    rows = list(csv.DictReader(open(CSV)))
    fl = lambda x: float(x) if x not in ("", None) else None
    hard = []
    for r in rows:
        n = int(r["ref_lines"]) if r["ref_lines"] else None
        never = fl(r["rl_s0"]) == 0 and fl(r["rl_s1"]) == 0
        if n is None:
            continue
        if never or n >= a.min_lines:
            r = dict(r, ref_lines=n, never=never, long=n >= a.min_lines)
            hard.append(r)
    print(f"hard problems: {len(hard)} (never {sum(r['never'] for r in hard)}, long {sum(r['long'] for r in hard)}, "
          f"both {sum(r['never'] and r['long'] for r in hard)}); scraped {sum(r['set'] != 'derived' for r in hard)}")

    print("loading K12 …", flush=True)
    k12_cls, k12_sig, k12_skel, k12_bg, k12_n = load_k12()
    print("K12 rows", k12_n, "classes", len(k12_cls), "signatures", len(k12_sig), "skeletons", len(k12_skel))
    # the hand14 set repeats problems of batch3 / v1: count each renaming class once
    seen_cls = set()
    uniq = []
    for r in hard:
        ks = BR.keys(refs[r["name"]]["prompt"])
        if ks & seen_cls:
            r["twin"] = True
        else:
            seen_cls |= ks
        uniq.append(r)
    hard = uniq
    print("hard distinct classes", sum(not r.get("twin") for r in hard))

    per = []
    hard_bg = collections.Counter()
    hard_bg_lines = 0
    piece_count = collections.Counter()     # class -> number of hard proofs containing it
    piece_info = {}                          # class -> dict(thm, n_lines, sig, in_k12, k12_sig_rows, example)
    for r in hard:
        ref = refs[r["name"]]
        lines = parse(ref["reference_proof"])
        if lines is None:
            continue
        lines = prune_lines(lines)
        prem, goal = L.parse_prompt(ref["prompt"])
        cat, struct = attribute(lines, goal)
        cause, shares = dominant_cause(cat, struct)
        if not r.get("twin"):
            hard_bg.update(bigrams(lines))
            hard_bg_lines += sum(1 for ln in lines if ln["rule"] not in ("PR", "AS"))
        bic = n_bicond(goal) + sum(n_bicond(x) for x in prem)
        whole_skel = skeleton_key(lines)
        # pieces: premise-pruned sub-theorems of 2..max_piece lines
        pieces = {}
        uncovered_first = None
        for i in range(len(lines) - 1):
            sp = subproblem(lines, i, prune_prem=True)
            if sp is None:
                continue
            prompt, text = sp
            plines = parse(text)
            if plines is None:
                continue
            plines = prune_lines(plines)
            n_l = len(plines)
            if n_l < 2 or n_l > a.max_piece:
                continue
            keys = BR.keys(prompt)
            thm = canonical_theorem(prompt)
            key = ("ct", thm)
            in_k12 = any(k in k12_cls for k in keys)
            sk = skeleton_key(plines)
            sk_rows = k12_skel.get(sk, 0)
            sg = signature(plines)
            if key not in pieces:
                pieces[key] = dict(thm=thm, n_lines=n_l, n_prem=len(L.parse_prompt(prompt)[0]), sig=sg, in_k12=in_k12,
                                   skel=sk, skel_in_k12=sk_rows > 0, k12_skel_rows=sk_rows, k12_sig_rows=k12_sig.get(sg, 0))
            if not in_k12 and not sk_rows and (uncovered_first is None or n_l < uncovered_first[0]):
                uncovered_first = (n_l, sk)
        if not r.get("twin"):
            for key, info in pieces.items():
                piece_count[key] += 1
                piece_info.setdefault(key, dict(info, example=f"{r['set']}:{r['source']} {r['number']}{(' '+r['nuance']) if r['nuance'] else ''}"))
        n_pieces = len(pieces)
        n_known = sum(1 for p in pieces.values() if p["in_k12"])
        n_skel_known = sum(1 for p in pieces.values() if p["skel_in_k12"])
        per.append(dict(
            name=r["name"], set=r["set"], source=r["source"], number=r["number"], nuance=r["nuance"], sequent=r["sequent"],
            never=r["never"], long=r["long"], rl=(fl(r["rl_s0"]), fl(r["rl_s1"])), pt=fl(r["pt_d384"]),
            n_lines=struct["n_lines"], n_prem=struct["n_prem"], depth=struct["max_depth"], boxes=struct["n_boxes"],
            ore=struct["ore_lines"], nested_ore=struct["nested_ore"], ore_in_negi=struct["ore_in_negi"], negi_in_ore=struct["negi_in_ore"],
            reductio=struct["boxes_reductio"], dn=struct["dn_lines"], impe_chain=struct["impe_chain"], and_lines=struct["and_lines"],
            cause=cause, shares=shares, cat=dict(cat), inside=struct["inside"], rules=struct["rules"],
            pieces=n_pieces, pieces_known=n_known, pieces_skel_known=n_skel_known, first_unknown=uncovered_first,
            bicond=bic, goal_op=goal[0], twin=bool(r.get("twin")), depth0_share=round(struct["inside"].get("depth0", 0) / max(1, struct["n_lines"] - struct["n_prem"]), 2),
            whole_skel=whole_skel, whole_skel_in_k12=k12_skel.get(whole_skel, 0),
        ))

    # ------------------------------------------------------------ aggregates
    def agg(sel, label):
        if not sel:
            return {}
        n = len(sel)
        m = lambda k: round(sum(p[k] for p in sel) / n, 2)
        causes = collections.Counter(p["cause"] for p in sel)
        share = collections.defaultdict(float)
        for p in sel:
            for k, v in p["shares"].items():
                share[k] += v / n
        return dict(label=label, n=n, mean_lines=m("n_lines"), mean_prem=m("n_prem"), mean_depth=m("depth"), mean_boxes=m("boxes"),
                    mean_ore=m("ore"), with_nested_ore=sum(p["nested_ore"] > 0 for p in sel), with_ore_in_negi=sum(p["ore_in_negi"] > 0 for p in sel),
                    with_negi_in_ore=sum(p["negi_in_ore"] > 0 for p in sel),
                    with_reductio=sum(p["reductio"] > 0 for p in sel), with_dn=sum(p["dn"] > 0 for p in sel), with_impe_chain3=sum(p["impe_chain"] >= 3 for p in sel),
                    depth3plus=sum(p["depth"] >= 3 for p in sel), prem4plus=sum(p["n_prem"] >= 4 for p in sel),
                    causes=dict(causes.most_common()), mean_share={k: round(v, 2) for k, v in share.items()},
                    mean_pieces=m("pieces"), mean_pieces_known=m("pieces_known"), mean_pieces_skel_known=m("pieces_skel_known"),
                    piece_known_share=round(sum(p["pieces_known"] for p in sel) / max(1, sum(p["pieces"] for p in sel)), 2),
                    piece_skel_known_share=round(sum(p["pieces_skel_known"] for p in sel) / max(1, sum(p["pieces"] for p in sel)), 2),
                    all_pieces_known=sum(p["pieces"] > 0 and p["pieces_known"] == p["pieces"] for p in sel),
                    all_pieces_skel_known=sum(p["pieces"] > 0 and p["pieces_skel_known"] == p["pieces"] for p in sel),
                    with_bicond=sum(p["bicond"] > 0 for p in sel), mean_bicond=m("bicond"), mean_depth0_share=m("depth0_share"),
                    goal_ops=dict(collections.Counter(p["goal_op"] for p in sel).most_common()),
                    whole_skel_in_k12=sum(p["whole_skel_in_k12"] > 0 for p in sel))

    scraped = [p for p in per if p["set"] != "derived" and not p["twin"]]
    derived = [p for p in per if p["set"] == "derived" and not p["twin"]]
    groups = [agg(scraped, "scraped hard"), agg([p for p in scraped if p["never"]], "scraped never-solved"),
              agg([p for p in scraped if p["long"]], "scraped ≥14 lines"), agg(derived, "derived hard")]
    # pieces: unknown to K12, by frequency
    unknown = [(piece_count[k], piece_info[k]) for k in piece_count if not piece_info[k]["in_k12"] and not piece_info[k]["skel_in_k12"]]
    unknown.sort(key=lambda x: (-x[0], x[1]["n_lines"]))
    known = [(piece_count[k], piece_info[k]) for k in piece_count if piece_info[k]["in_k12"]]
    known.sort(key=lambda x: -x[0])
    skel_only = [(piece_count[k], piece_info[k]) for k in piece_count if not piece_info[k]["in_k12"] and piece_info[k]["skel_in_k12"]]
    skel_only.sort(key=lambda x: -x[0])
    # unknown pieces grouped by skeleton: the motifs
    skel_unknown = collections.Counter()
    skel_ex = {}
    for c, info in unknown:
        skel_unknown[info["skel"]] += c
        skel_ex.setdefault(info["skel"], info)
    sig_rows = [(c, s, rule_sig_str(skel_ex[s]["sig"]), skel_ex[s]["n_lines"], skel_ex[s]["example"]) for s, c in skel_unknown.most_common()]

    k12_lines = sum(k12_bg.values()) or 1
    hb = sum(hard_bg.values()) or 1
    bigram_rows = []
    for (a_, b_), c in hard_bg.most_common():
        k = k12_bg.get((a_, b_), 0)
        bigram_rows.append(dict(citing=a_, cited=b_, hard=c, hard_per_1k=round(1000 * c / hb, 1), k12=k, k12_per_1k=round(1000 * k / k12_lines, 2),
                                ratio=round((c / hb) / max(k / k12_lines, 1e-9), 1)))
    out = dict(hard=len(per), bigrams=bigram_rows, hard_distinct=sum(not p["twin"] for p in per), groups=groups, pieces_total=len(piece_count),
               pieces_unknown=len(unknown), pieces_known=len(known), pieces_skel_only=len(skel_only),
               unknown_pieces=[dict(count=c, **info) for c, info in unknown], known_pieces_top=[dict(count=c, **info) for c, info in known[:30]],
               skel_only_top=[dict(count=c, **info) for c, info in skel_only[:30]],
               unknown_skeletons=sig_rows, per_problem=per, k12_rows=k12_n)
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    write_md(a.out, out, scraped, derived, unknown, known, skel_only, sig_rows, a)
    print("wrote", a.json, "and", a.out)


def write_md(path, out, scraped, derived, unknown, known, skel_only, sig_rows, a):
    W = []
    w = W.append
    w("# Why the hardest textbook problems are hard, and which small pieces they need\n")
    w(f"Generated by `code/pretrain_data/hard_problems.py` from `per_problem.csv` (401 evaluated problems; scraped sets\n"
      f"v0, v1, batch3, hand14, lean_v2 plus derived), the release's reference proofs, and Dan's K12 pool ({out['k12_rows']:,} rows).\n"
      f"Hard = never solved by either RL seed (64 samples each) or reference proof ≥ {a.min_lines} lines: {out['hard']} problems, "
      f"{out['hard_distinct']} distinct renaming classes (hand14 repeats problems of batch3 and v1; twins are counted once below). "
      f"The sealed 72 are not read.\n")
    w("## 1. Structure of the hard proofs\n")
    w("| group | n | lines | prem | depth | boxes | ORE lines | nested ORE | ORE inside ¬I | ¬I inside ORE | reductio | DN | →E chain ≥3 | depth ≥3 | prem ≥4 | with ↔ | ↔ per problem | depth-0 share |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g in out["groups"]:
        if not g:
            continue
        w(f"| {g['label']} | {g['n']} | {g['mean_lines']} | {g['mean_prem']} | {g['mean_depth']} | {g['mean_boxes']} | {g['mean_ore']} | "
          f"{g['with_nested_ore']} | {g['with_ore_in_negi']} | {g['with_negi_in_ore']} | {g['with_reductio']} | {g['with_dn']} | {g['with_impe_chain3']} | {g['depth3plus']} | {g['prem4plus']} | "
          f"{g['with_bicond']} | {g['mean_bicond']} | {g['mean_depth0_share']} |")
    w("\nColumns from `nested ORE` to `with ↔` are counts of problems. `↔` = subformulas of the shape (a → b) ∧ (b → a), which is how the "
      "textbooks' biconditional is transcribed. `depth-0 share` = fraction of non-premise lines outside every box. Means over problems; proofs dependency-pruned.\n")
    w("Goal connective of the hard problems:\n")
    w("| group | goal connective (problems) |")
    w("| --- | --- |")
    for g in out["groups"]:
        if g:
            w(f"| {g['label']} | {', '.join(f'{k} {v}' for k, v in g['goal_ops'].items())} |")
    w("\n## 2. What the lines pay for\n")
    w("Each non-premise line is attributed to one of five costs. `case split` = ORI/ORE lines, the assumptions of case boxes and everything "
      "inside them; `classical` = reductio boxes (¬G assumed, ¬I, DN) and their contents plus other DN/¬E/⊥E; `conditional` = →I boxes and contents; "
      "`∧/∨ shuffle` = ∧E, ∧I and plain ∨I; `→E chain` = →E lines. Inner boxes are charged to the innermost box, so for a goal like "
      "A → (B → C) the outer →I box is charged only for what is not inside the inner one.\n")
    w("| group | dominant cause (problems) | mean share of lines |")
    w("| --- | --- | --- |")
    for g in out["groups"]:
        if not g:
            continue
        w(f"| {g['label']} | {', '.join(f'{k} {v}' for k, v in g['causes'].items())} | {', '.join(f'{k} {v}' for k, v in g['mean_share'].items())} |")
    w("\n## 3. How much of each hard proof is made of pieces K12 already contains\n")
    w(f"A piece is a sub-theorem of 2–{a.max_piece} lines: the theorem proved at a line, with only the premises and open assumptions its "
      "sub-proof cites. Two kinds of coverage: **class** = the piece's curry-aware renaming class occurs in K12 as a problem; **skeleton** = "
      "after replacing every subformula whose internal structure the sub-proof never uses by an atom, the resulting theorem occurs in K12 "
      "(so `((P→Q)∧(Q→P)) ⊢ (P→Q)` has skeleton `(A ∧ B) ⊢ A`). Skeleton coverage is what a model that generalises over formula shape would need.\n")
    w("| group | pieces per proof | class-known share | skeleton-known share | proofs with every piece skeleton-known | whole-proof skeleton in K12 |")
    w("| --- | --- | --- | --- | --- | --- |")
    for g in out["groups"]:
        if not g:
            continue
        w(f"| {g['label']} | {g['mean_pieces']} | {g['piece_known_share']} | {g['piece_skel_known_share']} | {g['all_pieces_skel_known']} / {g['n']} | {g['whole_skel_in_k12']} / {g['n']} |")
    w(f"\nDistinct pieces across all hard proofs: {out['pieces_total']} ({out['pieces_known']} class-known, {out['pieces_skel_only']} skeleton-known only, "
      f"{out['pieces_unknown']} unknown by both).\n")
    w("## 4. The motifs K12 lacks\n")
    w("Unknown pieces grouped by skeleton, ordered by how many distinct hard proofs contain one. These are the candidate small pretraining problems.\n")
    w("| hard proofs | lines | skeleton (canonical, premises ⊢ goal) | sub-proof rules | example source |")
    w("| --- | --- | --- | --- | --- |")
    for c, sk, rules, nl, ex in sig_rows[:50]:
        w(f"| {c} | {nl} | `{sk}` | {rules} | {ex} |")
    w("\n### 4.1 Unknown pieces as they occur (exact class), top 40\n")
    w("| hard proofs | lines | prem | piece (canonical) | sub-proof rules | example source |")
    w("| --- | --- | --- | --- | --- | --- |")
    for c, info in unknown[:40]:
        w(f"| {c} | {info['n_lines']} | {info['n_prem']} | `{info['thm']}` | {rule_sig_str(info['sig'])} | {info['example']} |")
    w("\n## 5. Pieces K12 knows only by skeleton (same shape, bigger formulas), top 25\n")
    w("| hard proofs | lines | piece (canonical) | skeleton | K12 rows with that skeleton |")
    w("| --- | --- | --- | --- | --- |")
    for c, info in skel_only[:25]:
        w(f"| {c} | {info['n_lines']} | `{info['thm']}` | `{info['skel']}` | {info['k12_skel_rows']:,} |")
    w("\n## 6. Pieces K12 contains as problems, top 20\n")
    w("| hard proofs | lines | piece (canonical) | sub-proof rules |")
    w("| --- | --- | --- | --- |")
    for c, info in known[:20]:
        w(f"| {c} | {info['n_lines']} | `{info['thm']}` | {rule_sig_str(info['sig'])} |")
    w("\n## 6.1 Rule bigrams: which rule feeds which, hard proofs vs K12\n")
    w("A bigram is (rule of a line, rule of a line it cites). Rates are per 1,000 citations. `ratio` = hard rate / K12 rate; "
      "large ratios are the rule combinations the hard proofs need far more often than K12 shows them.\n")
    w("| citing rule | cited rule | hard count | hard /1k | K12 count | K12 /1k | ratio |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for b in sorted(out["bigrams"], key=lambda b: -b["ratio"])[:40]:
        if b["hard"] >= 3:
            w(f"| {b['citing']} | {b['cited']} | {b['hard']} | {b['hard_per_1k']} | {b['k12']:,} | {b['k12_per_1k']} | {b['ratio']} |")
    w("\n## 7. Per problem (scraped sets, twins removed)\n")
    w("`rl` = solve rate seed 0 / seed 1 (64 samples); `pieces` = skeleton-known / total; `first unknown` = the smallest sub-theorem skeleton K12 lacks.\n")
    w("| set | source | sequent | lines | prem | depth | ↔ | ORE | reductio | DN | →E chain | cause | rl | pieces | first unknown |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in sorted(scraped, key=lambda p: -p["n_lines"]):
        fu = f"{p['first_unknown'][0]}L `{p['first_unknown'][1]}`" if p["first_unknown"] else "—"
        rl = "/".join("—" if x is None else f"{x:.2f}" for x in p["rl"])
        w(f"| {p['set']} | {p['source']} {p['number']} | `{p['sequent']}` | {p['n_lines']} | {p['n_prem']} | {p['depth']} | {p['bicond']} | {p['ore']} | {p['reductio']} | {p['dn']} | {p['impe_chain']} | {p['cause']} | {rl} | {p['pieces_skel_known']}/{p['pieces']} | {fu} |")
    w("\n## 8. Per problem (derived, twins removed)\n")
    w("| nuance | base sequent | lines | prem | depth | ↔ | ORE | reductio | cause | rl | pieces | first unknown |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in sorted(derived, key=lambda p: -p["n_lines"]):
        fu = f"{p['first_unknown'][0]}L `{p['first_unknown'][1]}`" if p["first_unknown"] else "—"
        rl = "/".join("—" if x is None else f"{x:.2f}" for x in p["rl"])
        w(f"| {p['nuance']} | `{p['sequent']}` | {p['n_lines']} | {p['n_prem']} | {p['depth']} | {p['bicond']} | {p['ore']} | {p['reductio']} | {p['cause']} | {rl} | {p['pieces_skel_known']}/{p['pieces']} | {fu} |")
    with open(path, "w") as f:
        f.write("\n".join(W) + "\n")


if __name__ == "__main__":
    main()
