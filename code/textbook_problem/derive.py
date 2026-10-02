"""Derived problems: scorable task nuances built from the pool of problems that have verified
reference proofs (candidate_v0, candidate_v1 and the hand proofs of the unsolved set). Never from
the sealed 72.

    cd code && uv run python textbook_problem/derive.py [--only nuance,...] [--no-lean] [--n 10]

Without --only, the KEPT nuances run (DROPPED lists the nine pruned for conceptual overlap and why).

Two scoring modes (docs/textbook-problem/log.md, 2026-10-01):
  verifier        the model's output placed after `prompt` must be an accepted proof; for prefix
                  tasks `task_input` already contains the given lines, so scoring is
                  nd_verify(prompt + " " + given + " " + output).
  verifier+check  as above, plus a structural check on the proof text (`check` field), decided by
                  `structural_ok` below: a line budget, a forbidden rule or a required rule.

Every row carries the base problem (name, source, licence), how it was derived, the text fed to the
model (`task_input`), the reference output, the full reference proof, token counts against the
evaluator's limits (input 624, output 400), the verifier verdicts, the Lean verdict and the audit.
Task-input conventions for whole-proof tasks (repair, cloze, gap, shorten): the sequent, then a
marker word, then the given material, then PRF; the model's output is a whole proof. These markers
are new to any model trained on the plain format, which the README says.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import prove  # noqa: E402  (sets up the fork paths)

from data import canon_key, load_pool, load_textbook72, thm_of_prompt  # noqa: E402
from nd_data import canonical_theorem  # noqa: E402
from patterns import fstr, lines_to_text  # noqa: E402
from patterns import parse as parse_proof  # noqa: E402

L = prove.L
BOT = L.BOT
DATA = os.path.join(HERE, "../../data/textbook_problem")
OUT = os.path.join(DATA, "candidates_derived")
INPUT_CAP, OUTPUT_CAP = 1024 - 400, 400
CLOSING = {"IMPI", "NEGI", "ORE", "ANDI"}
ELIM = {"ANDE1", "ANDE2", "IMPE", "NEGE", "BOTE", "DN", "ORI1", "ORI2", "ANDI", "R"}
LETTERS = ["P", "Q", "R", "S"]


def A(x):
    return ("atom", x)


def NOT(a):
    return ("not", a)


def AND(a, b):
    return ("and", a, b)


def OR(a, b):
    return ("or", a, b)


def IMP(a, b):
    return ("imp", a, b)


def prompt_of(prem, goal):
    head = "THM " + (" , ".join(fstr(p) for p in prem) + " " if prem else "")
    return f"{head}SEQ {fstr(goal)} PRF"


def ntok(s):
    return len(s.split())


def subformulas(f, acc=None):
    acc = set() if acc is None else acc
    acc.add(f)
    if f[0] == "not":
        subformulas(f[1], acc)
    elif f[0] in ("and", "or", "imp"):
        subformulas(f[1], acc)
        subformulas(f[2], acc)
    return acc


def substitute(f, atom, by):
    if f[0] == "atom":
        return by if f[1] == atom else f
    if f[0] == "not":
        return ("not", substitute(f[1], atom, by))
    if f[0] in ("and", "or", "imp"):
        return (f[0], substitute(f[1], atom, by), substitute(f[2], atom, by))
    return f


def rename(f, perm):
    if f[0] == "atom":
        return ("atom", perm.get(f[1], f[1]))
    if f[0] == "not":
        return ("not", rename(f[1], perm))
    if f[0] in ("and", "or", "imp"):
        return (f[0], rename(f[1], perm), rename(f[2], perm))
    return f


# ------------------------------------------------------------------ proofs
class Base:
    def __init__(self, row, src_id=None):
        self.name = row["name"]
        self.prompt = row["prompt"]
        self.prem, self.goal = L.parse_prompt(self.prompt)
        self.proof = row["reference_proof"]
        self.lines = parse_proof(self.proof)
        self.n = len(self.lines)
        feats = row.get("proof_features") or prove.proof_features(self.proof)
        self.rule_set = tuple(feats["rule_set"])
        self.n_boxes, self.depth = feats["n_boxes"], feats["max_box_depth"]
        s = row.get("source") or {}
        self.src = src_id or s.get("id")
        self.where = s.get("where") or f"hand proof of {row.get('source_number')}"
        self.number = s.get("number") or row.get("source_number")
        self.licence = s.get("licence") or {"roy": "CC BY-NC-SA 4.0", "pelletier": "BSD-3 transcription (Isabelle), Pelletier 1986"}.get(self.src)
        self.min_lines = row.get("min_lines")
        self.min_method = row.get("min_lines_method")
        self.classical_only = row.get("classical_only", not L.intuitionistically_valid(self.prem, self.goal))
        self.hand = "proof_features" not in row
        self.ptok, self.itok = ntok(self.proof), ntok(self.prompt)

    def fits(self):
        """Whole proof within the output cap, prompt within the input cap."""
        return self.ptok <= OUTPUT_CAP and self.itok <= INPUT_CAP

    @property
    def body(self):
        return [ln for ln in self.lines if ln["rule"] != "PR"]

    @property
    def atoms(self):
        return sorted({a for f in self.prem + [self.goal] for a in L.atoms(f)})

    def info(self):
        return dict(name=self.name, source=self.src, where=self.where, number=self.number, licence=self.licence)


class PB:
    """Proof builder: lines are appended with fresh numbers; `splice` copies a base body in."""

    def __init__(self):
        self.lines = []

    @property
    def n(self):
        return len(self.lines)

    def add(self, depth, formula, rule, refs=()):
        self.lines.append(dict(idx=self.n + 1, depth=depth, formula=formula, rule=rule, refs=list(refs)))
        return self.n

    def premises(self, prem):
        return {i + 1: self.add(0, p, "PR") for i, p in enumerate(prem)}

    def splice(self, body, depth_add, ref_map):
        m = dict(ref_map)
        last = None
        for ln in body:
            new = self.add(ln["depth"] + depth_add, ln["formula"], ln["rule"], [m[r] for r in ln["refs"]])
            m[ln["idx"]] = new
            last = new
        return m, last

    def text(self):
        return lines_to_text(self.lines)


def text_lines(proof):
    return [s.strip(" ;") for s in proof.replace(" QED", "").split(" ; ") if s.strip(" ;")]


def join(lines, qed):
    if not lines:
        return "QED" if qed else ""
    return " ; ".join(lines) + (" ; QED" if qed else " ;")


def verify(prompt, proof):
    ok, why, n = prove.verify_text(prompt + " " + proof)
    sok = prove.safe_verify(prompt, proof)
    return dict(nd_verify=ok, nd_reason=why, strict_verify=bool(sok[0]), lines=n)


def structural_ok(proof, spec):
    lines = parse_proof(proof)
    if spec["type"] == "max_lines":
        return len(lines) <= spec["n"]
    if spec["type"] == "forbid_rule":
        return all(ln["rule"] != spec["rule"] for ln in lines)
    if spec["type"] == "require_rule":
        return any(ln["rule"] == spec["rule"] for ln in lines)
    raise ValueError(spec)


# ------------------------------------------------------------------ records
def rec(nuance, base, prompt, full_proof, *, given=None, k=None, task_input=None, reference_output=None,
        scoring="verifier", check=None, derivation="", extra=None):
    """A row. For prefix tasks pass k (lines given); task_input/reference_output are then derived."""
    lines = text_lines(full_proof)
    if k is not None:
        given = join(lines[:k], False)
        reference_output = join(lines[k:], True)
        task_input = (prompt + " " + given).rstrip()
    v = verify(prompt, full_proof)
    checks = dict(full=v)
    if k is not None:
        checks["prefix_alone_rejected"] = not prove.verify_text(prompt + " " + given + " QED")[0]
        goal = prompt.split(" SEQ ")[1].replace(" PRF", "")
        bogus = given + f" N{k + 1} {goal} : R N{k} ; QED"
        checks["bogus_completion_rejected"] = not prove.verify_text(prompt + " " + bogus)[0]
        checks["open_boxes_at_cut"] = lines[k - 1].count("|")
        checks["remaining_lines"] = len(lines) - k
    if check is not None:
        checks["reference_meets_check"] = structural_ok(full_proof, check)
    r = dict(
        name=f"derived_{nuance}_" + hashlib.sha256((prompt + "|" + (task_input or "")).encode()).hexdigest()[:16],
        nuance=nuance, scoring=scoring, check=check, derivation=derivation,
        prompt=prompt, task_input=task_input, given=given, reference_output=reference_output,
        reference_proof=full_proof, reference_lines=len(lines),
        tokens=dict(task_input=ntok(task_input), reference_output=ntok(reference_output),
                    input_within_cap=ntok(task_input) <= INPUT_CAP, output_within_cap=ntok(reference_output) <= OUTPUT_CAP),
        base=base.info() if base is not None else None, checks=checks,
    )
    if extra:
        r.update(extra)
    return r


def ok(r):
    c = r["checks"]
    return c["full"]["nd_verify"] and c["full"]["strict_verify"] and all(
        c.get(k, True) for k in ("prefix_alone_rejected", "bogus_completion_rejected", "reference_meets_check")
    ) and r["tokens"]["output_within_cap"] and r["tokens"]["input_within_cap"]


# ------------------------------------------------------------------ selection
class Picker:
    def __init__(self, bases, seed=20261001):
        self.bases, self.used = bases, collections.Counter()
        self.rng = random.Random(seed)

    def pick(self, pred, n, key=None):
        """n bases satisfying pred, preferring unused bases, unseen rule sets and sources."""
        pool = [b for b in self.bases if b.body and pred(b)]
        chosen, rules, srcs = [], set(), set()
        while pool and len(chosen) < n:
            best = min(pool, key=lambda b: (self.used[b.name], b.rule_set in rules, b.src in srcs, (key or (lambda x: 0))(b), b.name))
            chosen.append(best)
            pool.remove(best)
            rules.add(best.rule_set)
            srcs.add(best.src)
        for b in chosen:
            self.used[b.name] += 1
        return chosen


# ------------------------------------------------------------------ nuances
def g_completion_closing(P, n):
    out = []
    for b in P.bases:
        i = b.n
        while i > 0 and b.lines[i - 1]["rule"] in CLOSING:
            i -= 1
        b._cut = i
    def pred(b):
        return b.n_boxes >= 1 and 2 <= b.n - b._cut <= 25 and b._cut > len(b.prem) and b.itok + b.ptok <= INPUT_CAP
    for b in P.pick(pred, n, key=lambda b: -len({ln["rule"] for ln in b.lines[b._cut:]})):
        out.append(rec("completion_closing", b, b.prompt, b.proof, k=b._cut,
                       derivation="cut before the suffix of box-closing rules; every leaf is already derived"))
    return out


def g_completion_contradiction(P, n):
    def cut(b):
        idx = [i for i, ln in enumerate(b.lines) if ln["formula"] == BOT and ln["rule"] == "NEGE"]
        return idx[-1] if idx else None
    def pred(b):
        i = cut(b)
        return i is not None and 2 <= b.n - i <= 25 and i > len(b.prem) + 1 and b.itok + b.ptok <= INPUT_CAP
    for b in P.pick(pred, n):
        yield rec("completion_contradiction", b, b.prompt, b.proof, k=cut(b),
                  derivation="cut just before the last F line; the solver must find the two clashing lines, then close")


def g_completion_midleaf(P, n):
    def cut(b):
        d = max(ln["depth"] for ln in b.lines)
        if d == 0:
            return None
        starts = [i for i, ln in enumerate(b.lines) if ln["rule"] == "AS" and ln["depth"] == d]
        s = starts[-1]
        e = s
        while e + 1 < b.n and b.lines[e + 1]["depth"] >= d:
            e += 1
        box = e - s + 1
        if box < 3:
            return None
        return s + 1 + box // 2
    def pred(b):
        k = cut(b)
        return k is not None and 3 <= b.n - k <= 25 and b.itok + b.ptok <= INPUT_CAP
    for b in P.pick(pred, n):
        yield rec("completion_midleaf", b, b.prompt, b.proof, k=cut(b),
                  derivation="cut halfway through the deepest box; the leaf is half done")


def g_completion_almost_nothing(P, n):
    def cut(b):
        i = [i for i, ln in enumerate(b.lines) if ln["rule"] == "AS"]
        return i[0] + 1 if i else None
    def pred(b):
        k = cut(b)
        return k is not None and 4 <= b.n - k <= 25 and b.itok + b.ptok <= INPUT_CAP
    for b in P.pick(pred, n):
        yield rec("completion_almost_nothing", b, b.prompt, b.proof, k=cut(b),
                  derivation="only the premises and the first assumption are given")


def harmless_box(pb, h, depth=0):
    a = pb.add(depth + 1, h, "AS")
    r = pb.add(depth + 1, h, "R", [a])
    pb.add(depth, IMP(h, h), "IMPI", [a, r])
    return a


def g_open_dead_end(P, n):
    def pred(b):
        return len(b.body) + 2 <= 25 and len(b.body) >= 4
    for b in P.pick(pred, n):
        first = b.body[0]["formula"] if b.body[0]["rule"] == "AS" else None
        if b.goal != BOT and first != NOT(b.goal):
            h = NOT(b.goal)  # a tempting reductio start that the reference proof does not take
        elif b.prem:
            h = b.prem[0]
        else:
            h = b.goal[1] if b.goal[0] in ("imp", "not", "and", "or") else b.goal
        pb = PB()
        pm = pb.premises(b.prem)
        harmless_box(pb, h)
        pb.splice(b.body, 0, pm)
        yield rec("open_dead_end", b, b.prompt, pb.text(), k=len(b.prem) + 1,
                  derivation=f"the prefix opens a box on {fstr(h)} that leads nowhere; the solver must close it harmlessly (R, →I) and then prove the goal")


def g_wrong_turn(P, n):
    def pred(b):
        return b.prem and 4 <= len(b.body) <= 20
    for b in P.pick(pred, n):
        pb = PB()
        pm = pb.premises(b.prem)
        p1 = b.prem[0]
        pb.add(0, OR(p1, b.goal), "ORI1", [pm[1]])
        pb.add(0, AND(p1, p1), "ANDI", [pm[1], pm[1]])
        harmless_box(pb, NOT(b.goal))
        k = pb.n + max(1, len(b.body) // 3)
        pb.splice(b.body, 0, pm)
        yield rec("wrong_turn", b, b.prompt, pb.text(), k=k,
                  derivation="the prefix contains three useless but valid derivations and a closed dead-end box, then the first third of the real proof")


def g_extend(P, n):
    forms = ["and_premise", "imp", "double_neg", "imp_and"]
    def pred(b):
        return len(b.body) + 5 <= 25 and len(b.body) >= 3
    for i, b in enumerate(P.pick(pred, n)):
        form = forms[i % len(forms)]
        if form == "and_premise" and not b.prem:
            form = "imp"
        pb = PB()
        pm = pb.premises(b.prem)
        _, g = pb.splice(b.body, 0, pm)
        k = pb.n
        X = A(b.atoms[0])
        if form == "and_premise":
            goal = AND(b.goal, b.prem[0])
            pb.add(0, goal, "ANDI", [g, pm[1]])
        elif form == "imp":
            goal = IMP(X, b.goal)
            a = pb.add(1, X, "AS")
            r = pb.add(1, b.goal, "R", [g])
            pb.add(0, goal, "IMPI", [a, r])
        elif form == "double_neg":
            goal = NOT(NOT(b.goal))
            a = pb.add(1, NOT(b.goal), "AS")
            f = pb.add(1, BOT, "NEGE", [g, a])
            pb.add(0, goal, "NEGI", [a, f])
        else:
            goal = AND(IMP(X, b.goal), b.goal)
            a = pb.add(1, X, "AS")
            r = pb.add(1, b.goal, "R", [g])
            c = pb.add(0, IMP(X, b.goal), "IMPI", [a, r])
            pb.add(0, goal, "ANDI", [c, g])
        prompt = prompt_of(b.prem, goal)
        yield rec("extend", b, prompt, pb.text(), k=k,
                  derivation=f"the prefix is a complete proof of the base goal; the prompt asks for the stronger goal ({form})")


def g_hint_pairs(P, n):
    def pred(b):
        body = b.body
        return body and body[0]["rule"] == "AS" and 4 <= len(body) <= 23
    for b in P.pick(pred, n):
        right = b.body[0]["formula"]
        S = sorted((subformulas(f) for f in b.prem + [b.goal]), key=str)
        S = sorted(set().union(*[subformulas(f) for f in b.prem + [b.goal]]), key=lambda f: (L.size(f), str(f)))
        wrong = next((f for f in S if f != right and f[0] != "atom" and f not in b.prem and f != b.goal and f != BOT), None)
        if wrong is None:
            wrong = next((f for f in S if f != right and f != BOT), None)
        pid = "hint_" + b.name[-8:]
        yield rec("hint_pairs", b, b.prompt, b.proof, k=len(b.prem) + 1,
                  derivation="right hint: the first assumption of the reference proof is given", extra=dict(pair_id=pid, variant="right"))
        pb = PB()
        pm = pb.premises(b.prem)
        harmless_box(pb, wrong)
        pb.splice(b.body, 0, pm)
        yield rec("hint_pairs", b, b.prompt, pb.text(), k=len(b.prem) + 1,
                  derivation=f"wrong hint: a box on {fstr(wrong)} is opened instead", extra=dict(pair_id=pid, variant="wrong"))
        yield rec("hint_pairs", b, b.prompt, b.proof, k=len(b.prem),
                  derivation="no hint: premises only", extra=dict(pair_id=pid, variant="none"))


def mutate(b, kind, rng):
    lines = [dict(ln, refs=list(ln["refs"])) for ln in b.lines]
    cands = [ln for ln in lines if ln["rule"] not in ("PR", "AS")]
    if kind == "wrong_cite":
        cands = [ln for ln in cands if ln["refs"]]
        ln = rng.choice(cands)
        j = rng.randrange(len(ln["refs"]))
        others = [i for i in range(1, ln["idx"]) if i != ln["refs"][j]]
        ln["refs"][j] = rng.choice(others)
    elif kind == "wrong_rule":
        swap = {"ANDE1": "ANDE2", "ANDE2": "ANDE1", "ORI1": "ORI2", "ORI2": "ORI1", "IMPE": "ANDE1", "NEGE": "IMPE",
                "IMPI": "NEGI", "NEGI": "IMPI", "DN": "R", "BOTE": "DN", "ORE": "ANDI", "ANDI": "IMPE", "R": "DN"}
        ln = rng.choice(cands)
        ln["rule"] = swap[ln["rule"]]
    else:  # wrong_formula
        cands = [ln for ln in cands if ln["formula"][0] in ("and", "or", "imp")]
        ln = rng.choice(cands)
        f = ln["formula"]
        ln["formula"] = (f[0], f[2], f[1])
    return lines_to_text(lines), ln["idx"]


def g_repair(P, n):
    rng = random.Random(7)
    kinds = ["wrong_cite", "wrong_rule", "wrong_formula"]
    def pred(b):
        return b.fits() and 5 <= b.n <= 25 and any(ln["formula"][0] in ("and", "or", "imp") for ln in b.body)
    for i, b in enumerate(P.pick(pred, n)):
        kind = kinds[i % 3]
        for _ in range(20):
            broken, at = mutate(b, kind, rng)
            if not prove.verify_text(b.prompt + " " + broken)[0]:
                break
        else:
            continue
        head = b.prompt.replace(" PRF", "")
        yield rec("repair", b, b.prompt, b.proof, given=broken, task_input=f"{head} BROKEN {broken} PRF",
                  reference_output=b.proof, derivation=f"one line corrupted ({kind}); output the whole corrected proof",
                  extra=dict(broken_line=at, mutation=kind, checks_broken_rejected=True))


def cloze(b, what):
    out = []
    for s in text_lines(b.proof):
        head, rule = s.split(" : ")
        num, *rest = head.split(" ")
        if rule == "PR" or (what == "formula" and rule == "AS"):
            out.append(s)
        elif what == "just":
            out.append(f"{head} : ?")
        else:
            bars = [t for t in rest if t == "|"]
            out.append(" ".join([num] + bars + ["?"]) + f" : {rule}")
    return join(out, True)


def g_cloze_just(P, n):
    for b in P.pick(lambda b: b.fits() and 5 <= b.n <= 25 and len(b.rule_set) >= 3, n):
        given = cloze(b, "just")
        yield rec("cloze_justifications", b, b.prompt, b.proof, given=given,
                  task_input=f"{b.prompt.replace(' PRF', '')} CLOZE {given} PRF", reference_output=b.proof,
                  derivation="every formula shown, every rule and citation blanked; output the whole proof")


def g_cloze_formula(P, n):
    for b in P.pick(lambda b: b.fits() and 5 <= b.n <= 25 and len(b.rule_set) >= 3, n):
        given = cloze(b, "formula")
        yield rec("cloze_formulas", b, b.prompt, b.proof, given=given,
                  task_input=f"{b.prompt.replace(' PRF', '')} CLOZE {given} PRF", reference_output=b.proof,
                  derivation="rules, citations, premises and assumptions shown, derived formulas blanked; output the whole proof")


def g_gap(P, n):
    for b in P.pick(lambda b: b.fits() and 8 <= b.n <= 25, n):
        lines = text_lines(b.proof)
        m = min(4, max(2, len(b.body) // 3))
        s = len(b.prem) + (len(b.body) - m) // 2
        given = join(lines[:s] + ["..."] + lines[s + m:], True)
        yield rec("gap", b, b.prompt, b.proof, given=given, task_input=f"{b.prompt.replace(' PRF', '')} GAP {given} PRF",
                  reference_output=b.proof, derivation=f"{m} consecutive lines removed from the middle (numbering kept); output the whole proof",
                  extra=dict(gap_lines=[s + 1, s + m]))


def g_curry(P, n):
    for i, b in enumerate(P.pick(lambda b: b.prem and b.n + len(b.prem) <= 30, n)):
        keep = 1 if (len(b.prem) >= 2 and i % 2) else 0
        curried = b.prem[keep:]
        goal = b.goal
        for p in reversed(curried):
            goal = IMP(p, goal)
        pb = PB()
        pm = pb.premises(b.prem[:keep])
        ass = []
        for d, p in enumerate(curried):
            idx = pb.add(d + 1, p, "AS")
            pm[keep + d + 1] = idx
            ass.append(idx)
        _, last = pb.splice(b.body, len(curried), pm)
        for d in range(len(curried) - 1, -1, -1):
            g = goal
            for _ in range(d):
                g = g[2]
            last = pb.add(d, g, "IMPI", [ass[d], last])
        prompt = prompt_of(b.prem[:keep], goal)
        yield rec("curry", b, prompt, pb.text(), k=None, task_input=prompt, reference_output=pb.text(),
                  derivation=f"{'all' if not keep else 'all but the first'} premises moved into the goal as nested conditionals")


def g_negated_conj(P, n):
    for i, b in enumerate(P.pick(lambda b: b.prem and b.n + 4 <= 28, n)):
        m = 2 if (len(b.prem) >= 2 and i % 2) else 1
        inside = b.prem[-m:]
        nested = [NOT(b.goal)]  # nested[j] = conjunction of inside[j:] and ¬goal
        for p in reversed(inside):
            nested.insert(0, AND(p, nested[0]))
        conj = nested[0]
        pb = PB()
        pm = pb.premises(b.prem[:-m])
        a = pb.add(1, conj, "AS")
        cur = a
        for j, p in enumerate(inside):
            pm[len(b.prem) - m + j + 1] = pb.add(1, p, "ANDE1", [cur])
            cur = pb.add(1, nested[j + 1], "ANDE2", [cur])
        ng = cur
        _, g = pb.splice(b.body, 1, pm)
        f = pb.add(1, BOT, "NEGE", [g, ng])
        pb.add(0, NOT(conj), "NEGI", [a, f])
        prompt = prompt_of(b.prem[:-m], NOT(conj))
        yield rec("negated_conjunction", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"the last {m} premise(s) and the negated goal folded into a negated conjunction goal")


def g_forward_control(P, n):
    for b in P.pick(lambda b: b.fits() and b.n_boxes == 0 and set(b.rule_set) <= ELIM and 4 <= b.n <= 20, n):
        yield rec("forward_control", b, b.prompt, b.proof, task_input=b.prompt, reference_output=b.proof,
                  derivation="pool problem whose reference proof uses elimination rules only, no boxes")


def g_wrappers(P, n):
    forms = ["dn_premise", "and_premise", "or_premise", "or_goal"]  # no X → g form: that is curry's shape
    for i, b in enumerate(P.pick(lambda b: b.n + 3 <= 28, n)):
        form = forms[i % 4]
        if form.endswith("premise") and not b.prem:
            form = "or_goal"
        pb = PB()
        prem, goal = list(b.prem), b.goal
        if form == "dn_premise":
            prem[0] = NOT(NOT(b.prem[0]))
            pm = pb.premises(prem)
            pm[1] = pb.add(0, b.prem[0], "DN", [pm[1]])
            pb.splice(b.body, 0, pm)
        elif form == "and_premise":
            prem[0] = AND(b.prem[0], b.prem[0])
            pm = pb.premises(prem)
            pm[1] = pb.add(0, b.prem[0], "ANDE1", [pm[1]])
            pb.splice(b.body, 0, pm)
        elif form == "or_premise":
            prem[0] = OR(b.prem[0], b.prem[0])
            pm = pb.premises(prem)
            a1 = pb.add(1, b.prem[0], "AS")
            a2 = pb.add(1, b.prem[0], "AS")
            pm[1] = pb.add(0, b.prem[0], "ORE", [pm[1], a1, a1, a2, a2])
            pb.splice(b.body, 0, pm)
        elif form == "or_goal":
            X = A(next((x for x in LETTERS if x not in b.atoms), b.atoms[-1]))
            goal = OR(b.goal, X)
            pm = pb.premises(prem)
            _, g = pb.splice(b.body, 0, pm)
            pb.add(0, goal, "ORI1", [g])
        else:
            X = A(next((x for x in LETTERS if x not in b.atoms), b.atoms[-1]))
            goal = IMP(X, b.goal)
            pm = pb.premises(prem)
            a = pb.add(1, X, "AS")
            _, g = pb.splice(b.body, 1, pm)
            pb.add(0, goal, "IMPI", [a, g])
        prompt = prompt_of(prem, goal)
        yield rec("wrappers", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"trivial wrapper: {form}")


def g_strengthened(P, n):
    def pred(b):
        return any(p[0] == "imp" for p in b.prem) and b.n + 1 <= 28
    for b in P.pick(pred, n):
        j = next(i for i, p in enumerate(b.prem) if p[0] == "imp")
        p = b.prem[j]
        strong = AND(p, IMP(p[2], p[1]))
        prem = list(b.prem)
        prem[j] = strong
        if not L.satisfiable(prem):
            continue
        pb = PB()
        pm = pb.premises(prem)
        pm[j + 1] = pb.add(0, p, "ANDE1", [pm[j + 1]])
        pb.splice(b.body, 0, pm)
        prompt = prompt_of(prem, b.goal)
        yield rec("strengthened_premise", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"premise {j + 1} replaced by its biconditional; the added direction is never used")


def g_near_miss(P, n):
    def pred(b):
        return any(p[0] == "imp" and IMP(p[2], p[1]) not in b.prem and L.satisfiable(b.prem + [IMP(p[2], p[1])]) for p in b.prem) and b.n + 1 <= 28
    for b in P.pick(pred, n):
        p = next(p for p in b.prem if p[0] == "imp" and IMP(p[2], p[1]) not in b.prem and L.satisfiable(b.prem + [IMP(p[2], p[1])]))
        prem = b.prem + [IMP(p[2], p[1])]
        pb = PB()
        pm = pb.premises(prem)
        pb.splice(b.body, 0, pm)
        prompt = prompt_of(prem, b.goal)
        yield rec("near_miss", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"the converse of premise {fstr(p)} added as a distractor; jointly satisfiable, never used")


def g_permutation(P, n):
    rng = random.Random(11)
    for b in P.pick(lambda b: len(b.atoms) >= 2 and 5 <= b.n <= 28, n):
        atoms = b.atoms
        while True:
            perm = atoms[:]
            rng.shuffle(perm)
            if perm != atoms:
                break
        perm = dict(zip(atoms, perm))
        prem = [rename(p, perm) for p in reversed(b.prem)]
        goal = rename(b.goal, perm)
        pb = PB()
        pm = pb.premises(prem)
        pmap = {i + 1: pm[len(b.prem) - i] for i in range(len(b.prem))}
        body = [dict(ln, formula=rename(ln["formula"], perm)) for ln in b.body]
        pb.splice(body, 0, pmap)
        prompt = prompt_of(prem, goal)
        yield rec("permutation", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"letters renamed {perm} and premises reversed; same renaming class as the base by design")


def g_distractor_scaling(P, n):
    bases = P.pick(lambda b: b.prem and b.n + 4 <= 24, n // 2)
    others = [p for b2 in P.bases for p in b2.prem]
    for b in bases:
        for k in (1, 2, 4):
            extra, seen = [], set(b.prem)
            for p in others:
                if len(extra) == k:
                    break
                if p in seen or not (L.atoms(p) & set(b.atoms)) or not L.satisfiable(b.prem + extra + [p]):
                    continue
                extra.append(p)
                seen.add(p)
            if len(extra) < k:
                continue
            prem = b.prem + extra
            pb = PB()
            pm = pb.premises(prem)
            pb.splice(b.body, 0, pm)
            prompt = prompt_of(prem, b.goal)
            yield rec("distractor_scaling", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                      derivation=f"{k} irrelevant premise(s) from other pool problems added, sharing letters, jointly satisfiable, never cited",
                      extra=dict(k_distractors=k))


def g_contrapositive(P, n):
    for i, b in enumerate(P.pick(lambda b: b.prem and b.n + 3 <= 28, n)):
        order = list(range(len(b.prem)))[::-1] if i % 2 == 0 else list(range(len(b.prem)))
        j = next((t for t in order if b.prem[t][0] != "not"), order[0])
        swapped = b.prem[j]
        prem = [p for t, p in enumerate(b.prem) if t != j] + [NOT(b.goal)]
        pb = PB()
        pm0 = pb.premises(prem)
        pm, c = {}, 0
        for t in range(len(b.prem)):
            if t != j:
                c += 1
                pm[t + 1] = pm0[c]
        ng = pm0[len(prem)]
        a = pb.add(1, swapped, "AS")
        pm[j + 1] = a
        _, g = pb.splice(b.body, 1, pm)
        f = pb.add(1, BOT, "NEGE", [g, ng])
        pb.add(0, NOT(swapped), "NEGI", [a, f])
        prompt = prompt_of(prem, NOT(swapped))
        yield rec("contrapositive", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"premise {j + 1} and the goal swapped with negations: p, ¬g ⊢ ¬p_{j + 1}")


def g_substitution(P, n):
    forms = [lambda a, b: OR(a, b), lambda a, b: AND(a, b), lambda a, b: IMP(b, a), lambda a, b: NOT(b)]
    for i, b in enumerate(P.pick(lambda b: len(b.atoms) >= 2 and 5 <= b.n <= 20, n)):
        x, y = b.atoms[0], b.atoms[1]
        by = forms[i % 4](A(x), A(y))
        prem = [substitute(p, x, by) for p in b.prem]
        goal = substitute(b.goal, x, by)
        lines = [dict(ln, formula=substitute(ln["formula"], x, by)) for ln in b.lines]
        proof = lines_to_text(lines)
        prompt = prompt_of(prem, goal)
        yield rec("substitution", b, prompt, proof, task_input=prompt, reference_output=proof,
                  derivation=f"atom {x} replaced by {fstr(by)} throughout; the proof skeleton is unchanged")


def g_composition(P, n):
    small = sorted([b for b in P.bases if b.body and len(b.atoms) <= 2 and 3 <= b.n <= 12], key=lambda b: (P.used[b.name], b.name))
    used = set()
    out = []
    for b1 in small:
        if len(out) == n:
            break
        for b2 in small:
            if b2 is b1 or b1.name in used or b2.name in used or b1.rule_set == b2.rule_set:
                continue
            free = [x for x in LETTERS if x not in b1.atoms]
            if len(free) < len(b2.atoms):
                continue
            perm = dict(zip(b2.atoms, free))
            prem = b1.prem + [rename(p, perm) for p in b2.prem]
            goal = AND(b1.goal, rename(b2.goal, perm))
            pb = PB()
            pm = pb.premises(prem)
            _, g1 = pb.splice(b1.body, 0, pm)
            pm2 = {i + 1: pm[len(b1.prem) + i + 1] for i in range(len(b2.prem))}
            _, g2 = pb.splice([dict(ln, formula=rename(ln["formula"], perm)) for ln in b2.body], 0, pm2)
            pb.add(0, goal, "ANDI", [g1, g2])
            prompt = prompt_of(prem, goal)
            r = rec("composition", b1, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                    derivation=f"two pool problems on disjoint letters joined by a conjunction goal; second problem {b2.name} renamed {perm}",
                    extra=dict(base2=b2.info()))
            out.append(r)
            used.update([b1.name, b2.name])
            P.used[b1.name] += 1
            P.used[b2.name] += 1
            break
    return out


def g_conjoined(P, n):
    by_prem = collections.defaultdict(list)
    for b in sorted(P.bases, key=lambda b: (P.used[b.name], b.name)):
        if b.body:
            for p in b.prem:
                by_prem[p].append(b)
    used, out = set(), []
    for p, bs in sorted(by_prem.items(), key=lambda kv: -len(kv[1])):
        for b1 in bs:
            for b2 in bs:
                if len(out) == n:
                    return out
                if b1 is b2 or b1.name in used or b2.name in used or b1.goal == b2.goal or b1.n + b2.n > 28:
                    continue
                prem = list(b1.prem) + [q for q in b2.prem if q not in b1.prem]
                if not L.satisfiable(prem):
                    continue
                pb = PB()
                pm = pb.premises(prem)
                _, g1 = pb.splice(b1.body, 0, pm)
                pm2 = {i + 1: pm[prem.index(q) + 1] for i, q in enumerate(b2.prem)}
                _, g2 = pb.splice(b2.body, 0, pm2)
                pb.add(0, AND(b1.goal, b2.goal), "ANDI", [g1, g2])
                prompt = prompt_of(prem, AND(b1.goal, b2.goal))
                out.append(rec("conjoined", b1, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                               derivation=f"two pool problems sharing the premise {fstr(p)} merged under a conjunction goal; second problem {b2.name}",
                               extra=dict(base2=b2.info())))
                used.update([b1.name, b2.name])
                P.used[b1.name] += 1
                P.used[b2.name] += 1
    return out


def g_demorgan(P, n, secs=120):
    """Classically equivalent surface forms; new proofs by the fork's bounded search."""
    forms = ["imp_premise_to_or", "neg_and_premise_to_or", "or_premise_to_neg_and", "imp_goal_to_or"]
    tried = 0
    out = []
    for b in sorted(P.bases, key=lambda b: (P.used[b.name], b.n, b.name)):
        if len(out) == n or tried >= 4 * n:
            break
        if b.n > 14:
            continue
        for form in forms[tried % 4:] + forms[: tried % 4]:  # rotate the first form tried, for variety
            prem, goal = list(b.prem), b.goal
            if form == "imp_premise_to_or" and any(p[0] == "imp" for p in prem):
                j = next(i for i, p in enumerate(prem) if p[0] == "imp")
                prem[j] = OR(NOT(prem[j][1]), prem[j][2])
            elif form == "neg_and_premise_to_or" and any(p[0] == "not" and p[1][0] == "and" for p in prem):
                j = next(i for i, p in enumerate(prem) if p[0] == "not" and p[1][0] == "and")
                prem[j] = OR(NOT(prem[j][1][1]), NOT(prem[j][1][2]))
            elif form == "or_premise_to_neg_and" and any(p[0] == "or" for p in prem):
                j = next(i for i, p in enumerate(prem) if p[0] == "or")
                prem[j] = NOT(AND(NOT(prem[j][1]), NOT(prem[j][2])))
            elif form == "imp_goal_to_or" and goal[0] == "imp":
                goal = OR(NOT(goal[1]), goal[2])
            else:
                continue
            prompt = prompt_of(prem, goal)
            tried += 1
            t0 = time.time()
            u = prove.minlen(prompt, 40, secs)
            if u["proof"] is None:
                print(f"   demorgan: no proof within {secs}s for {prompt}", flush=True)
                continue
            out.append(rec("demorgan", b, prompt, u["proof"], task_input=prompt, reference_output=u["proof"],
                           derivation=f"{form}: a classically equivalent surface form; new proof by minlen in {time.time() - t0:.0f}s",
                           extra=dict(search_secs=round(time.time() - t0, 1))))
            P.used[b.name] += 1
            break
    return out


def g_line_budget(P, n):
    for b in P.pick(lambda b: b.fits() and b.min_method == "whole search" and 5 <= b.n <= 20 and b.min_lines == b.n, n):
        yield rec("line_budget", b, b.prompt, b.proof, task_input=b.prompt, reference_output=b.proof, scoring="verifier+check",
                  check=dict(type="max_lines", n=b.min_lines),
                  derivation="prove within the proven minimum number of lines (minlen completed its search)")


def pad(b, rng):
    pb = PB()
    pm = pb.premises(b.prem)
    if b.prem:
        pb.add(0, OR(b.prem[0], b.goal), "ORI1", [pm[1]])
        pb.add(0, AND(b.prem[0], b.prem[0]), "ANDI", [pm[1], pm[1]])
    harmless_box(pb, NOT(b.goal) if rng.random() < 0.5 else (b.prem[0] if b.prem else b.goal))
    m, last = pb.splice(b.body, 0, pm)
    return pb


def g_shorten(P, n):
    rng = random.Random(3)
    for b in P.pick(lambda b: b.fits() and b.min_method == "whole search" and 5 <= b.n <= 20 and b.min_lines == b.n, n):
        padded = pad(b, rng).text()
        yield rec("shorten", b, b.prompt, b.proof, given=padded, task_input=f"{b.prompt.replace(' PRF', '')} SHORTEN {padded} PRF",
                  reference_output=b.proof, scoring="verifier+check", check=dict(type="max_lines", n=b.min_lines),
                  derivation=f"a valid padded proof ({len(text_lines(padded))} lines) is given; produce one within the proven minimum",
                  extra=dict(checks_padded_valid=prove.verify_text(b.prompt + " " + padded)[0]))


def g_no_dn(P, n):
    def pred(b):
        return b.fits() and not b.classical_only and "DN" not in b.rule_set and L.count(b.goal, "not") >= 1 and 5 <= b.n <= 25
    for b in P.pick(pred, n):
        yield rec("no_dn", b, b.prompt, b.proof, task_input=b.prompt, reference_output=b.proof, scoring="verifier+check",
                  check=dict(type="forbid_rule", rule="DN"),
                  derivation="intuitionistically valid, negation in the goal; the proof must not use DN")


def g_force_ore(P, n):
    for b in P.pick(lambda b: b.fits() and "ORE" in b.rule_set and 5 <= b.n <= 25, n):
        yield rec("force_ore", b, b.prompt, b.proof, task_input=b.prompt, reference_output=b.proof, scoring="verifier+check",
                  check=dict(type="require_rule", rule="ORE"),
                  derivation="the proof must use ∨E")


GENERATORS = collections.OrderedDict([
    ("completion_closing", g_completion_closing),
    ("completion_contradiction", g_completion_contradiction),
    ("completion_midleaf", g_completion_midleaf),
    ("completion_almost_nothing", g_completion_almost_nothing),
    ("open_dead_end", g_open_dead_end),
    ("wrong_turn", g_wrong_turn),
    ("extend", g_extend),
    ("hint_pairs", g_hint_pairs),
    ("repair", g_repair),
    ("cloze_justifications", g_cloze_just),
    ("cloze_formulas", g_cloze_formula),
    ("gap", g_gap),
    ("curry", g_curry),
    ("conjoined", g_conjoined),
    ("red_herring", None),
    ("negated_conjunction", g_negated_conj),
    ("forward_control", g_forward_control),
    ("wrappers", g_wrappers),
    ("strengthened_premise", g_strengthened),
    ("near_miss", g_near_miss),
    ("permutation", g_permutation),
    ("distractor_scaling", g_distractor_scaling),
    ("contrapositive", g_contrapositive),
    ("substitution", g_substitution),
    ("demorgan", g_demorgan),
    ("composition", g_composition),
    ("line_budget", g_line_budget),
    ("shorten", g_shorten),
    ("no_dn", g_no_dn),
    ("force_ore", g_force_ore),
])


def g_red_herring(P, n):
    others = [p for b2 in P.bases for p in b2.prem]
    for b in P.pick(lambda b: b.prem and b.n + 1 <= 26, n):
        extra = next((p for p in others if p not in b.prem and (L.atoms(p) & set(b.atoms)) and L.satisfiable(b.prem + [p])), None)
        if extra is None:
            continue
        prem = b.prem + [extra]
        pb = PB()
        pm = pb.premises(prem)
        pb.splice(b.body, 0, pm)
        prompt = prompt_of(prem, b.goal)
        yield rec("red_herring", b, prompt, pb.text(), task_input=prompt, reference_output=pb.text(),
                  derivation=f"one premise from another pool problem added ({fstr(extra)}); shares letters, jointly satisfiable, never cited")


GENERATORS["red_herring"] = g_red_herring

# Pruned 2026-10-01 for conceptual overlap (docs/textbook-problem/log.md): one representative per family.
DROPPED = collections.OrderedDict([
    ("completion_midleaf", "a middle cut between completion_closing and the hint pairs' right-hint cut"),
    ("completion_almost_nothing", "identical to the right-hint variant of hint_pairs"),
    ("open_dead_end", "identical mechanic to the wrong-hint variant of hint_pairs"),
    ("gap", "cloze with lines removed; the two cloze nuances cover it"),
    ("red_herring", "distractor_scaling at k = 1, now a level of that nuance"),
    ("strengthened_premise", "near_miss with the distractor inside a premise"),
    ("contrapositive", "folds the negated goal like negated_conjunction; both end in ¬I"),
    ("composition", "conjoined with disjoint letters"),
    ("shorten", "line_budget with a padded proof and a marker word; same efficiency question"),
])
KEPT = [n for n in GENERATORS if n not in DROPPED]
FAMILIES = collections.OrderedDict([
    ("continuing", ["completion_closing", "completion_contradiction", "wrong_turn", "extend", "hint_pairs"]),
    ("reading and repairing", ["repair", "cloze_justifications", "cloze_formulas"]),
    ("backward-easy and control", ["curry", "conjoined", "negated_conjunction", "forward_control"]),
    ("perturbation and transformation", ["wrappers", "near_miss", "permutation", "distractor_scaling", "substitution", "demorgan"]),
    ("constraints on the proof", ["line_budget", "no_dn", "force_ore"]),
])
assert sorted(n for f in FAMILIES.values() for n in f) == sorted(KEPT)
PREFIX_NUANCES = {"completion_closing", "completion_contradiction", "completion_midleaf", "completion_almost_nothing",
                  "open_dead_end", "wrong_turn", "extend", "hint_pairs"}
SAME_SEQUENT = PREFIX_NUANCES | {"repair", "cloze_justifications", "cloze_formulas", "gap", "forward_control",
                                 "line_budget", "shorten", "no_dn", "force_ore", "permutation"}


# ------------------------------------------------------------------ audit
def load_bases():
    rows = []
    for fn in ("candidate_v0", "candidate_v1"):
        rows += [Base(json.loads(l)) for l in open(os.path.join(DATA, fn, "problems.jsonl"))]
    hand = os.path.join(DATA, "candidates_v3", "hand_proofs.jsonl")
    if os.path.exists(hand):
        for l in open(hand):
            r = json.loads(l)
            rows.append(Base(r, src_id="roy" if r["source_number"][0] in "ET" else "pelletier"))
    return rows


def auditor(bases):
    def keys(rows):
        return {canonical_theorem(r["prompt"]) for r in rows}, {canon_key(thm_of_prompt(r["prompt"])) for r in rows}
    sealed = {"textbook72": keys(load_textbook72()), "test_files": keys(load_pool("test_short_prompts") + load_pool("test_long_prompts"))}
    named = {n: keys(load_pool(n)) for n in ("validation_36", "heldout", "transfer", "rl_targets", "train")}
    pool = {b.name: (canonical_theorem(b.prompt), canon_key(thm_of_prompt(b.prompt))) for b in bases}
    def audit(prompt, base_name):
        ct, ck = canonical_theorem(prompt), canon_key(thm_of_prompt(prompt))
        s = [n for n, (a, c) in sealed.items() if ct in a or ck in c]
        m = [n for n, (a, c) in named.items() if ct in a or ck in c]
        p = [n for n, (a, c) in pool.items() if (ct == a or ck == c) and n != base_name]
        return dict(sealed=s, named_pools=m, pool_other_than_base=p, same_class_as_base=(ct, ck) == pool.get(base_name)), ct
    return audit


def keep_rows(name, rows, audit, seen_ct, dropped):
    kept = []
    for r in rows:
        aud, ct = audit(r["prompt"], r["base"]["name"])
        r["audit"] = aud
        if aud["sealed"]:
            dropped["sealed class (dropped, not recorded)"] += 1
            continue
        if name not in SAME_SEQUENT and (aud["same_class_as_base"] or aud["pool_other_than_base"]):
            dropped["collapses onto a pool problem"] += 1
            continue
        if name not in SAME_SEQUENT and ct in seen_ct:
            dropped["duplicate within nuance"] += 1
            continue
        if not ok(r):
            dropped["failed a check"] += 1
            print("   FAILED", name, r["derivation"][:60], json.dumps(r["checks"])[:200], r["tokens"])
            continue
        seen_ct.add(ct)
        kept.append(r)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--lean-only", action="store_true", help="Lean-check the existing nuance files, generate nothing")
    a = ap.parse_args()
    if a.lean_only:
        return lean_pass()
    bases = load_bases()
    print(f"{len(bases)} bases with reference proofs")
    P = Picker(bases)
    audit = auditor(bases)
    os.makedirs(OUT, exist_ok=True)
    only = set(a.only.split(",")) if a.only else set(KEPT)
    summary = {}
    for name, gen in GENERATORS.items():
        if name not in only:
            continue
        t0 = time.time()
        per = 3 if name == "hint_pairs" else 1
        kept, seen_ct, dropped, names = [], set(), collections.Counter(), set()
        for _round in range(4):
            need = (a.n * per - len(kept) + per - 1) // per
            if need <= 0:
                break
            rows = [r for r in gen(P, need) if r["name"] not in names]
            if not rows:
                break
            names.update(r["name"] for r in rows)
            kept += keep_rows(name, rows, audit, seen_ct, dropped)
        if not a.no_lean and kept:
            import lean
            try:
                oks = lean.check([(r["prompt"], r["reference_proof"]) for r in kept])
            except FileNotFoundError:
                oks = [None] * len(kept)
            for r, o in zip(kept, oks):
                r["lean_ok"] = o
            os.makedirs(os.path.join(OUT, "lean"), exist_ok=True)
            with open(os.path.join(OUT, "lean", f"{name}.lean"), "w") as f:
                for r in kept:
                    f.write(f"-- {r['name']}\n{lean.lean_theorem(r['prompt'], r['reference_proof'])}\n\n")
        with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        summary[name] = dict(kept=len(kept), dropped=dict(dropped), secs=round(time.time() - t0, 1),
                             lean_ok=sum(1 for r in kept if r.get("lean_ok")), bases=len({r["base"]["name"] for r in kept}),
                             rule_sets=len({tuple(prove.proof_features(r["reference_proof"])["rule_set"]) for r in kept}))
        print(f"{name:26s} kept {len(kept):3d}  bases {summary[name]['bases']:3d}  rule sets {summary[name]['rule_sets']:2d}  "
              f"lean {summary[name]['lean_ok']:3d}  dropped {dict(dropped)}  {summary[name]['secs']}s", flush=True)
    sp = os.path.join(OUT, "summary.json")
    old = json.load(open(sp)) if os.path.exists(sp) else {}
    old.update(summary)
    with open(sp, "w") as f:
        json.dump(old, f, indent=1)


def lean_pass():
    """Fill lean_ok for every row of every nuance file and write lean/<nuance>.lean."""
    import lean
    os.makedirs(os.path.join(OUT, "lean"), exist_ok=True)
    sp = os.path.join(OUT, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    for name in KEPT:
        fn = os.path.join(OUT, f"{name}.jsonl")
        if not os.path.exists(fn):
            continue
        rows = [json.loads(l) for l in open(fn)]
        if not rows:
            continue
        t0 = time.time()
        oks = lean.check([(r["prompt"], r["reference_proof"]) for r in rows])
        for r, o in zip(rows, oks):
            r["lean_ok"] = o
        with open(os.path.join(OUT, "lean", f"{name}.lean"), "w") as f:
            for r in rows:
                f.write(f"-- {r['name']}\n{lean.lean_theorem(r['prompt'], r['reference_proof'])}\n\n")
        with open(fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary.setdefault(name, {})["lean_ok"] = sum(1 for r in rows if r["lean_ok"])
        print(f"{name:26s} lean {sum(1 for r in rows if r['lean_ok']):3d}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
