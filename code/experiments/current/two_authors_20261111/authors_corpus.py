#!/usr/bin/env python3
"""Two-author pretraining corpora and their controls (two_authors_20261111, CPU only).

    python authors_corpus.py [--limit N] [--out-dir DIR] [--procs 8] [--force]

Writes into TA/data (TA = /scratch/cowifornia/nd-rl/two_authors):

  authors_half.jsonl.gz   author by the existing hash draw (tag_core.corpus_tag): route records -> A, else parity
  authors_p10.jsonl.gz    author by the route-share rule (tagcorpus.route_share_tags, p 0.10): A = 1,550 route records
                          + 13,950 others
  control_half.jsonl.gz   the same records; each habit's form drawn per record by its own hash, at the per-site
  control_p10.jsonl.gz    A-form share the matching authors corpus has; the tag label drawn the same way
  <name>.manifest.json    one per corpus;  authors_build.json  the build summary (calibration, state comparison)

Source: Dan's 1 % dose corpus `train_k12dose1.jsonl.gz` (tag_core.SRC_CORPUS, 155,000 records, 1,550 route records
flagged `coe_dose`). Every statement is kept; only the written proof changes. See SPEC.md (the contract) and
CORPUS.md (what was built, the numbers, the pitfalls).

The habits (author A / author B), applied to every proof in this fixed order:
  1 strip      every reiteration written with R (an optional R line in a box that restates a line from outside
               the box) is removed and its citers cite the source directly. This is B's reiteration form and the
               neutral starting point for A (otherwise step 2 would hoist A's R lines out of their boxes).
  2 order      premise-driven (habits.normal_form 'pd') / goal-driven ('gd'), where the two differ (room).
  3 conjunct   right conjunct first / left conjunct first at every conjunct site (habits.conj_sites). A site
               whose swap would change the proof's step-order form (the order's normal form) is left as the
               order put it and counted as a conflict.
  4 reit       A only: every outer line a box uses at its own level is restated once by R after the box's
               assumption (habits.v_reit 'reit').
After every stage the proof must pass the local checks (limits: <= 12 lines, box depth <= 2, premises untouched;
nd_verify; Dan's state_env replay with canonical names and the lean_staten tokenizer (the name limits); Dan's route
detectors unchanged: sigma and sigma+). A stage that fails is skipped (the habit keeps the previous form) and counted.
Every final proof that differs from the source is then checked by Lean (nd2lean.translate + nd2lean.lean_check, as
habits.py does); a failure drops the last applied stage and the record is rebuilt and rechecked.

Detectors (used for every manifest number and in the tests): reiteration per site on the final proof
(habits.reit_sites); step order and conjunct sites on the proof with its reiteration stripped (R copies would be
hoisted by the order normal form and would add conjunct "sites" at the top of every box).

Record format: the source record (proof, text, n_lines, rules, pat recomputed when the proof changes) plus
  author       A | B  (control: a nominal label drawn independently of everything else; it only picks the tag)
  tag_shown    the tag token is in this record's states (a hash half of each author)
  author_tag   the token the states carry: "A" | "B" | "none" (tag_core.TagTokenizer reads "none" as untagged)
  forms        the form assigned to each habit: {"order", "conjunct", "reit"} -> "A" | "B"
  realized     what the detectors read on the final proof: order "A" | "B" | "none" (no room) | "neither";
               conjunct and reit [sites, sites in A-form]
  skipped      habit -> why its rewrite was not applied (only when one was not)
  route        sigma+ by Dan's coe_assemble.knock (equals coe_dose on this corpus)
  src_proof    the source proof;  ta_corpus  the corpus name
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ATDIR = HERE.parent / "author_tags_20261110"
sys.path.insert(0, str(ATDIR))
import habits as H  # noqa: E402  (puts Dan's tree on sys.path)
import tag_core as T  # noqa: E402
import tagcorpus as TC  # noqa: E402

TA = Path(os.environ.get("TA", "/scratch/cowifornia/nd-rl/two_authors"))
DATA = TA / "data"
LEAN_CACHE = TA / "cache" / "lean_verdicts.jsonl"
SRC = T.SRC_CORPUS
SEED = T.SEED
HABITS = ("order", "conjunct", "reit")
STAGES = ("strip", "order", "conjunct", "reit")  # the fixed order of the rewrites
STAGE_HABIT = {"strip": "reit", "order": "order", "conjunct": "conjunct", "reit": "reit"}
COMBOS = tuple(itertools.product("AB", repeat=3))  # (order, conjunct, reit) forms
ALL_A, ALL_B = ("A", "A", "A"), ("B", "B", "B")
MAX_LINES, MAX_DEPTH, MAX_PREM = 12, 2, 3
VERSIONS = {"half": None, "p10": 0.10}  # route share of A (None: the existing hash draw)
CORPORA = ("authors_half", "authors_p10", "control_half", "control_p10")
LEAN_PER_FILE = 200


# ============================================================================ habit rewrites (on habits.py trees)
def site_r_lines(P: H.Proof) -> list[tuple[list, int]]:
    """(box body, index) of every reiteration written with R: an R line inside a box, not the box's last line, whose
    source lies outside the box. These are exactly habits.r_lines' 'optional' R lines that make a habits.reit_sites
    site 'reiterated' (a last line restating an outer line is 'required' and stays)."""
    out = []
    for units, hid, _d, _owner in H.scopes(P):
        if hid is None:
            continue
        mine = {hid} | {u.id for u in units}
        for k, u in enumerate(units[:-1]):
            if u.rule == "R" and u.refs[0] not in mine:
                out.append((units, k))
    return out


def strip_reit(P: H.Proof) -> int:
    """In place: remove every reiteration written with R (site_r_lines); its citers cite the source. -> lines removed.
    Other R lines (top-level restatements, a box's required last line, copies of lines of the same box) are kept."""
    n = 0
    while True:
        s = site_r_lines(P)
        if not s:
            return n
        units, k = s[0]
        u = units.pop(k)
        H.redirect(P, u.id, u.refs[0])
        n += 1


def add_reit(P: H.Proof) -> H.Proof:
    """habits.v_reit's 'reit' variant: every outer line a box uses at its own level is restated once by R right after
    the box's assumption, and that level's citations go to the copy. -> a new tree."""
    return H.build(H.v_reit(P, "")["reit"])


def height(u: H.Ln) -> int:
    """box nesting inside one unit (0 for a unit without boxes)."""
    return max((1 + max((height(v) for v in b.body), default=0) for b in u.boxes), default=0)


def sink_all_capped(P: H.Proof, max_depth: int = MAX_DEPTH) -> int:
    """habits.sink_all (goal-driven placement), except that a unit is not moved into a box when its own boxes would
    then sit deeper than `max_depth` (habits.sink_all can put a box inside a box at depth 2: depth 3, past K12's cap).
    Where no move hits the cap the moves, and so the result, are habits.sink_all's. -> moves."""
    moves = 0
    changed = True
    while changed:
        changed = False
        for units, _hid, d, _owner in H.scopes(P):
            if len(units) <= 1:
                continue
            for k in range(len(units) - 2, -1, -1):
                u = units[k]
                if d + 1 + height(u) > max_depth:
                    continue
                target = None
                ok = True
                for w in units[k + 1 :]:
                    if u.id in w.line_refs():
                        ok = False
                        break
                    hit = [b for b in w.boxes if any(u.id in H.subtree_refs(v) for v in b.body)]
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


def normal_form(P0: H.Proof, mode: str) -> H.Proof:
    """habits.normal_form with the goal-driven sink capped at box depth 2 (sink_all_capped); 'pd' is habits' own
    (hoisting never deepens a proof)."""
    if mode == "pd":
        return H.normal_form(P0, "pd")
    P = P0.copy()
    sink_all_capped(P)
    return H.order_all(P, "gd")


def conj_site_ids(P: H.Proof) -> list[tuple[int, int, str]]:
    """habits.conj_sites as (left unit id, right unit id, kind): for an ANDI the units deriving its left and right
    conjunct, for an ANDE1/ANDE2 pair the ANDE1 and the ANDE2 line. Right first = A's form."""
    return [(units[a].id, units[b].id, kind) for units, a, b, kind in H.conj_sites(P)]


def _scope_of(P: H.Proof, uid: int) -> list:
    for units, _hid, _d, _owner in H.scopes(P):
        if any(u.id == uid for u in units):
            return units
    raise KeyError(uid)


def apply_conj(P: H.Proof, sites: list[tuple[int, int, str]], form: str) -> H.Proof | None:
    """A copy of P with every given site in `form`'s orientation (A: the right conjunct's unit first, B: the left's),
    each scope re-sorted stably (habits.topo with the sites as extra edges; the order is otherwise kept, as
    habits.v_conjunct does for one site). None if the sites' edges form a cycle."""
    Q = P.copy()
    edges: dict[int, tuple[list, list]] = {}
    for left, right, _k in sites:
        units = _scope_of(Q, left)
        first, second = (right, left) if form == "A" else (left, right)
        edges.setdefault(id(units), (units, []))[1].append((first, second))
    for units, es in edges.values():
        pos = {u.id: k for k, u in enumerate(units)}
        try:
            units[:] = H.topo(units, lambda k: k, extra=[(pos[f], pos[s]) for f, s in es])
        except H.Unsupported:
            return None
    return Q


def scope_lasts(P: H.Proof) -> list[int]:
    return [units[-1].id if units else None for units, _h, _d, _o in H.scopes(P)]


# ============================================================================ detectors
def _order_form_stripped(S: H.Proof) -> str:
    s = H.render(S)
    pd, gd = H.render(normal_form(S, "pd")), H.render(normal_form(S, "gd"))
    if pd == gd:
        return "none"
    return "A" if s == pd else "B" if s == gd else "neither"


def order_form(P: H.Proof) -> str:
    """Step-order form of a proof: 'A' (it is its own premise-driven normal form), 'B' (its own goal-driven normal
    form, depth-capped), 'none' (the two normal forms coincide: no room), 'neither'. Read on the proof with its
    reiteration stripped (an R copy does not depend on the box's assumption, so the premise-driven form would hoist
    it)."""
    S = P.copy()
    strip_reit(S)
    return _order_form_stripped(S)


def detect(text: str) -> dict[str, Any]:
    """What a proof's habits are, per site. -> {'order': form, 'conjunct': [sites, right first, free sites, free sites
    right first], 'reit': [sites, reiterated]}. Reiteration is read on the proof (habits.reit_sites); order and
    conjunct sites on the proof with its reiteration stripped (habits.conj_sites would count an R copy at the top of a
    box as a conjunct unit). A conjunct site is free when swapping it alone leaves the step-order form unchanged; at
    the other sites the order fixes which conjunct comes first."""
    P = H.build(text)
    rs = H.reit_sites(P)
    S = P.copy()
    strip_reit(S)
    of = _order_form_stripped(S)
    lasts = scope_lasts(S)
    n = a = nf = af = 0
    for left, right, kind in conj_site_ids(S):
        pos = {u.id: k for k, u in enumerate(_scope_of(S, left))}
        right_first = pos[right] < pos[left]
        Q = apply_conj(S, [(left, right, kind)], "B" if right_first else "A")
        free = Q is not None and scope_lasts(Q) == lasts and _order_form_stripped(Q) == of
        n += 1
        a += right_first
        nf += free
        af += free and right_first
    return {"order": of, "conjunct": [n, a, nf, af], "reit": [len(rs), sum(bool(x[3]) for x in rs)]}


# ============================================================================ local checks
_W: dict[str, Any] = {}


def _tools() -> dict[str, Any]:
    """Dan's tools, loaded once per process."""
    if not _W:
        import lean_tok
        from nd_verify import verify_text
        from patterns import PATTERNS, classify
        from state_env import decompose

        knock, _flags, has_sigma = TC.dan_detectors()
        _W.update(
            verify=verify_text, decompose=decompose, tok=lean_tok.LeanTokenizer("lean_staten"),
            knock=knock, has_sigma=has_sigma, classify=classify, patterns=PATTERNS,
        )  # fmt: skip
    return _W


def route_flags(proof: str) -> tuple[bool, bool]:
    """(sigma+ by coe_assemble.knock, sigma by coe_sigma.has_sigma)."""
    W = _tools()
    drop, fl = W["knock"](proof)
    if drop is None:
        raise ValueError("Dan's parser rejects the proof")
    return bool(drop), bool(W["has_sigma"](fl))


def n_lines(text: str) -> int:
    return text.split().count(";")


def check(prompt: str, text: str, src: dict[str, Any]) -> str | None:
    """None if `text` passes every local check, else the first failing one."""
    W = _tools()
    n = n_lines(text)
    if n > MAX_LINES:
        return f"lines>{MAX_LINES}"
    P = H.build(text)
    if H.max_depth(P) > MAX_DEPTH:
        return f"depth>{MAX_DEPTH}"
    if len(P.prem) > MAX_PREM or [(p.f) for p in P.prem] != src["prem"]:
        return "premises"
    ok, _why, _ = W["verify"](prompt + " " + text)
    if not ok:
        return "nd_verify"
    try:
        steps, _toks, _env = W["decompose"](prompt, text, canon=True)
        for st, act, _hs in steps:
            W["tok"].encode_toks(st)
            W["tok"].encode_toks(act)
    except Exception:  # noqa: BLE001  (ParseFail, a name past n64, a token outside lean_staten)
        return "state_env"
    if route_flags(text) != src["route_flags"]:
        return "route_detector"
    return None


# ============================================================================ the pipeline
def rewrite(prompt: str, proof: str, combo: tuple[str, str, str], banned: frozenset = frozenset(),
            memo: dict | None = None) -> dict[str, Any]:  # fmt: skip
    """One proof in the forms `combo` = (order, conjunct, reit), stages in STAGES order; `banned` stages are not
    applied (the Lean fallback). -> {'text', 'applied', 'skipped' {habit: why}, 'conj_conflicts', 'det'}."""
    memo = {} if memo is None else memo
    if "src" not in memo:
        P0 = H.build(proof)
        memo["src"] = {"prem": [p.f for p in P0.prem], "route_flags": route_flags(proof)}
        memo["chk"], memo["det"] = {}, {}
    src = memo["src"]

    def ok(text: str) -> str | None:
        if text not in memo["chk"]:
            memo["chk"][text] = check(prompt, text, src)
        return memo["chk"][text]

    form = dict(zip(HABITS, combo, strict=True))
    P, text = H.build(proof), proof
    applied: list[str] = []
    skipped: dict[str, str] = {}

    def attempt(stage: str, Q: H.Proof) -> None:
        nonlocal P, text
        t = H.render(Q)
        if t == text:
            return
        why = ok(t)
        if why is not None:
            skipped.setdefault(STAGE_HABIT[stage], why if stage != "strip" else f"strip: {why}")
            return
        P, text = H.build(t), t
        applied.append(stage)

    for stage in sorted(
        banned, key=STAGES.index
    ):  # (a frozenset of str: its own order depends on the hash seed)
        skipped.setdefault(STAGE_HABIT[stage], "lean" if stage != "strip" else "strip: lean")
    if "strip" not in banned:
        Q = P.copy()
        if strip_reit(Q):
            attempt("strip", Q)
    if "order" not in banned:
        pd, gd = normal_form(P, "pd"), normal_form(P, "gd")
        if H.render(pd) != H.render(gd):
            attempt("order", pd if form["order"] == "A" else gd)
    conflicts = 0
    if "conjunct" not in banned:
        sites = conj_site_ids(P)
        if sites:
            before, lasts = order_form(P), scope_lasts(P)
            acc: list = []
            cur = P
            for s in sites:
                Q = apply_conj(P, [*acc, s], form["conjunct"])
                if Q is None or scope_lasts(Q) != lasts or order_form(Q) != before:
                    conflicts += 1
                    continue
                acc.append(s)
                cur = Q
            attempt("conjunct", cur)
    if form["reit"] == "A" and "reit" not in banned and H.reit_sites(P):
        attempt("reit", add_reit(P))
    if text not in memo["det"]:
        memo["det"][text] = detect(text)
    _n, _a, nf, af = memo["det"][text]["conjunct"]
    if af != (nf if form["conjunct"] == "A" else 0) and "conjunct" not in skipped:
        # two free sites that need opposite orders (an ANDE1/ANDE2 pair whose conjuncts an ANDI puts back swapped)
        skipped["conjunct"] = "sites need opposite orders"
    return {"text": text, "applied": applied, "skipped": skipped, "conj_conflicts": conflicts,
            "det": memo["det"][text]}  # fmt: skip


# ============================================================================ state comparison (pitfall 3)
def step_sigs(prompt: str, text: str) -> list[tuple[tuple, tuple]]:
    """The model-visible steps of a proof (state_env.decompose, canonical names): (state, action) with the state's
    names stripped and every name the action cites replaced by its line's position in the state."""
    steps, _toks, _env = _tools()["decompose"](prompt, text, canon=True)
    out = []
    for st, act, _hs in steps:
        pos, k, sig = {}, 0, []
        for j, t in enumerate(st):
            if H.NAME.match(t):
                if j + 1 < len(st) and st[j + 1] == ":":
                    pos.setdefault(t, k)
                    k += 1
                sig.append("#")
            else:
                sig.append(t)
        a = tuple(f"@{pos[t]}" if t in pos else ("#" if H.NAME.match(t) else t) for t in act)
        out.append((tuple(sig), a))
    return out


def compare_steps(sa: list, sb: list) -> dict[str, int]:
    """Each step of one form against the other form of the same theorem: 'novel' (its state never occurs in the
    other), 'choice' (the state occurs there, but the other form takes another action from it), 'same'.
    -> counts for both directions, and the choice steps after the first one."""
    out: collections.Counter = collections.Counter()
    for tag, x, y in (("a", sa, sb), ("b", sb, sa)):
        m: dict[tuple, set] = collections.defaultdict(set)
        for s, a in y:
            m[s].add(a)
        n_choice = 0
        for s, a in x:
            if s not in m:
                out[f"{tag}_novel"] += 1
            elif a in m[s]:
                out[f"{tag}_same"] += 1
            else:
                out[f"{tag}_choice"] += 1
                n_choice += 1
        out[f"{tag}_steps"] += len(x)
        out[f"{tag}_pairs_2plus_choices"] += n_choice >= 2
    k = 0
    while k < min(len(sa), len(sb)) and sa[k] == sb[k]:
        k += 1
    out["pairs"] += 1
    out["first_diverge_same_state"] += k < len(sa) and k < len(sb) and sa[k][0] == sb[k][0]
    return dict(out)


STATE_PAIRS = {  # habit -> (A-form combo, B-form combo, label): the habit alone changes, the others held fixed
    "order": [(("A", "B", "B"), ALL_B, "others B"), (ALL_A, ("B", "A", "A"), "others A")],
    "conjunct": [(("B", "A", "B"), ALL_B, "others B"), (ALL_A, ("A", "B", "A"), "others A")],
    "reit": [(("B", "B", "A"), ALL_B, "others B"), (ALL_A, ("A", "A", "B"), "others A")],
    "author": [(ALL_A, ALL_B, "all habits")],
}


# ============================================================================ per record (worker)
def process(rec: dict[str, Any]) -> dict[str, Any]:
    """All eight form combinations of one record (local checks only; Lean runs later on the ones a corpus uses), and
    the state comparison of each habit's two forms."""
    prompt, proof = rec["prompt"], rec["proof"]
    memo: dict = {}
    texts = [proof]
    tix = {proof: 0}
    res = {}
    for c in COMBOS:
        r = rewrite(prompt, proof, c, memo=memo)
        if r["text"] not in tix:
            tix[r["text"]] = len(texts)
            texts.append(r["text"])
        res["".join(c)] = {"t": tix[r["text"]], "applied": r["applied"], "skipped": r["skipped"],
                           "conj_conflicts": r["conj_conflicts"], "det": r["det"]}  # fmt: skip
    n_r = [sum(ln["rule"] == "R" for ln in H.parse_proof_tokens(t.split())) for t in texts]
    st: dict[str, dict] = {}
    sigs: dict[int, list] = {}
    for h, pairs in STATE_PAIRS.items():
        for ca, cb, lab in pairs:
            ta, tb = res["".join(ca)]["t"], res["".join(cb)]["t"]
            if ta == tb:
                continue
            for t in (ta, tb):
                if t not in sigs:
                    sigs[t] = step_sigs(prompt, texts[t])
            st[f"{h}|{lab}"] = compare_steps(sigs[ta], sigs[tb])
    return {"i": rec["_i"], "texts": texts, "res": res, "route_flags": memo["src"]["route_flags"],
            "src_lines": n_lines(proof), "states": st, "n_r": n_r}  # fmt: skip


def combo_totals(ph1: list[dict]) -> dict[str, Any]:
    """Pitfall 1: every form combination (order, conjunct, reit) on the same records, local checks only (before Lean):
    order room and A-form, conjunct sites (all, free, right first), reiteration sites and reiterated sites, R lines,
    lines, and the skips by habit and reason."""
    out: dict[str, Any] = {}
    for c in ("".join(x) for x in COMBOS):
        t: collections.Counter = collections.Counter()
        sk: collections.Counter = collections.Counter()
        for p in ph1:
            r = p["res"][c]
            d = r["det"]
            t["records"] += 1
            t["order_room"] += d["order"] != "none"
            t["order_A"] += d["order"] == "A"
            t["conj_sites"] += d["conjunct"][0]
            t["conj_right_first"] += d["conjunct"][1]
            t["conj_free_sites"] += d["conjunct"][2]
            t["conj_free_right_first"] += d["conjunct"][3]
            t["conj_conflicts"] += r["conj_conflicts"]
            t["reit_records_with_room"] += d["reit"][0] > 0
            t["reit_sites"] += d["reit"][0]
            t["reit_reiterated"] += d["reit"][1]
            t["r_lines"] += p["n_r"][r["t"]]
            t["lines"] += n_lines(p["texts"][r["t"]])
            for h, why in r["skipped"].items():
                sk[f"{h}: {why}"] += 1
        out[c] = {**dict(t), "skipped": dict(sorted(sk.items()))}
    return out


# ============================================================================ draws
def u01(*parts: Any) -> float:
    """A uniform number in [0, 1) from sha256 of 'seed:part:part...' (the first 8 bytes)."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def tag_shown(key: str, seed: int = SEED) -> bool:
    """The tag is in the record's states iff the first byte of sha256('<seed>:show:<key>') is even (the same draw in
    all four corpora, so the untagged halves are the same records)."""
    return hashlib.sha256(f"{seed}:show:{key}".encode()).digest()[0] % 2 == 0


def author_draws(
    items: list[tuple[str, bool]], seed: int = SEED
) -> dict[str, tuple[list[str], dict | None]]:
    """The author of every source record in each version: 'half' the existing parity draw (tag_core.corpus_tag),
    'p10' the route-share rule (tagcorpus.route_share_tags). Drawn on the whole source, so --limit keeps them."""
    out: dict[str, tuple[list[str], dict | None]] = {}
    for v, p in VERSIONS.items():
        if p is None:
            out[v] = ([T.corpus_tag(k, d, seed) for k, d in items], None)
        else:
            out[v] = TC.route_share_tags(items, p, seed)
    return out


# ============================================================================ Lean
def _lean_key(prompt: str, body: str) -> str:
    return hashlib.sha256(f"{prompt}\n{body}".encode()).hexdigest()


def lean_verdicts(
    pairs: list[tuple[str, str]], workers: int, cache: Path | None
) -> dict[str, bool]:
    """(prompt, body) -> Lean verdict, keyed by _lean_key; Dan's nd2lean.translate + lean_check (Lean 4 core), batches
    of LEAN_PER_FILE theorems per Lean process, `workers` processes at a time; verdicts cached in `cache`."""
    from nd2lean import TranslationError, lean_check, translate

    known: dict[str, bool] = {}
    if cache is not None and cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                known[x["h"]] = x["ok"]
    todo, srcs, seen = [], [], set()
    for p, b in pairs:
        k = _lean_key(p, b)
        if k in known or k in seen:
            continue
        seen.add(k)
        try:
            srcs.append(translate(p, b))
            todo.append(k)
        except TranslationError:
            known[k] = False
    root = Path(tempfile.mkdtemp(prefix="ta_lean_", dir=os.environ.get("TMPDIR")))
    try:
        chunks = [srcs[i : i + LEAN_PER_FILE] for i in range(0, len(srcs), LEAN_PER_FILE)]

        def run(c: list[str]) -> list[tuple[bool, str]]:
            return lean_check(c, per_file=LEAN_PER_FILE, workdir=tempfile.mkdtemp(dir=root))

        with ThreadPoolExecutor(workers) as ex:
            res = [x for r in ex.map(run, chunks) for x in r]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    new = {k: bool(ok) for k, (ok, _msg) in zip(todo, res, strict=True)}
    known.update(new)
    if cache is not None and new:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a") as f:
            for k, v in new.items():
                f.write(json.dumps({"h": k, "ok": v}) + "\n")
    return {_lean_key(p, b): known[_lean_key(p, b)] for p, b in pairs}


# ============================================================================ build
def limit_cores(n: int = 8) -> list[int]:
    """Pin this process, and so every process it starts (the rewrite pool, the Lean processes, which are themselves
    multi-threaded), to at most n cores (serrano: CPU builds on at most 8). -> the cores in use."""
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) > n:
        cores = cores[-n:]
        os.sched_setaffinity(0, cores)
    return cores


def read_source(src: Path, limit: int | None) -> tuple[list[dict], list[tuple[str, bool]]]:
    """-> (the first `limit` records (all if None), (key, dose) of every source record)."""
    recs: list[dict] = []
    items: list[tuple[str, bool]] = []
    keys: set[str] = set()
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            assert r["key"] not in keys, f"duplicate key {r['key']}"
            assert T.TAG_FIELD not in r, "the source is already tagged"
            keys.add(r["key"])
            items.append((r["key"], bool(r.get("coe_dose"))))
            if limit is None or len(recs) < limit:
                recs.append(r)
    return recs, items


def run_records(recs: list[dict], procs: int) -> list[dict]:
    import multiprocessing as mp

    slim = [{"_i": i, "prompt": r["prompt"], "proof": r["proof"]} for i, r in enumerate(recs)]
    if procs <= 1:
        return [process(x) for x in slim]
    with mp.Pool(procs) as pool:
        return list(pool.imap(process, slim, chunksize=32))


def resolve(recs: list[dict], ph1: list[dict], needed: set[tuple[int, str]], workers: int,
            cache: Path | None, log: dict) -> dict[tuple[int, str], dict]:  # fmt: skip
    """The final form of every needed (record, combo): its local-check result, rechecked by Lean whenever it differs
    from the source proof; a Lean failure drops the last applied stage and the record is rebuilt (rewrite with that
    stage banned) and rechecked, until Lean accepts or nothing is left to drop. `log` collects the counts."""
    cur: dict[tuple[int, str], dict] = {}
    for i, c in needed:
        r = ph1[i]["res"][c]
        cur[(i, c)] = {**r, "text": ph1[i]["texts"][r["t"]], "banned": []}
    pending = sorted(needed)
    rounds = 0
    while pending:
        rounds += 1
        pairs = sorted({(recs[i]["prompt"], cur[(i, c)]["text"]) for i, c in pending
                        if cur[(i, c)]["text"] != recs[i]["proof"]})  # fmt: skip
        log["lean_checked"] = log.get("lean_checked", 0) + len(pairs)
        verdict = lean_verdicts(pairs, workers, cache) if pairs else {}
        nxt = []
        for i, c in pending:
            res, prompt, proof = cur[(i, c)], recs[i]["prompt"], recs[i]["proof"]
            if res["text"] == proof or verdict[_lean_key(prompt, res["text"])]:
                continue
            stage = res["applied"][-1]
            log.setdefault("lean_failed_stage", collections.Counter())[stage] += 1
            banned = frozenset([*res["banned"], stage])
            r2 = rewrite(prompt, proof, tuple(c), banned=banned)
            cur[(i, c)] = {**r2, "banned": sorted(banned)}
            nxt.append((i, c))
        pending = nxt
    log["lean_rounds"] = rounds
    return cur


def rules_of(text: str) -> list[str]:
    return sorted({ln["rule"] for ln in H.parse_proof_tokens(text.split())} - {"PR"})


def pat_of(text: str, old: dict) -> dict:
    """Dan's generator flags (patterns.classify on the pruned proof, which reproduces every source record's `pat`)."""
    cl = _tools()["classify"](text)
    return {k: bool(cl[k]) for k in old}


def out_line(
    src: dict, res: dict, author: str, shown: bool, forms: tuple, corpus: str, route: bool
) -> str:
    r = dict(src)
    text = res["text"]
    if text != src["proof"]:
        r["proof"] = text
        if "text" in r:
            r["text"] = r["prompt"] + " " + text
        r["n_lines"] = n_lines(text)
        r["rules"] = rules_of(text)
        if "pat" in r:
            r["pat"] = pat_of(text, r["pat"])
    d = res["det"]
    r.update({"author": author, "tag_shown": shown, T.TAG_FIELD: author if shown else "none",
              "forms": dict(zip(HABITS, forms, strict=True)),
              "realized": {"order": d["order"], "conjunct": d["conjunct"], "reit": d["reit"]}})  # fmt: skip
    if res["skipped"]:
        r["skipped"] = res["skipped"]
    r.update({"route": route, "src_proof": src["proof"], "ta_corpus": corpus})
    return json.dumps(r) + "\n"


def write_gz(path: Path, lines: list[str]) -> None:
    TC.write_gz(path, lines)


# ---------------------------------------------------------------------------- statistics
def binary_forms(det: dict) -> dict[str, int | None]:
    """Record-level form of each habit: 1 = A-form at every site, 0 = B-form at every site, None = no room or mixed.
    `conjunct` uses the free sites, `conjunct_all` every site."""
    o = {"A": 1, "B": 0}.get(det["order"])
    n, a, nf, af = det["conjunct"]
    s, ra = det["reit"]

    def whole(k: int, m: int) -> int | None:
        return None if k == 0 else 1 if m == k else 0 if m == 0 else None

    return {
        "order": o,
        "conjunct": whole(nf, af),
        "conjunct_all": whole(n, a),
        "reit": whole(s, ra),
    }


def assoc(xs: list, ys: list) -> dict[str, Any]:
    """Association of two 0/1 variables over the records where both are defined: n, the 2x2 table, phi, and the
    difference P(x = 1 | y = 1) - P(x = 1 | y = 0)."""
    t = collections.Counter(
        (x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None
    )
    n11, n10, n01, n00 = t[(1, 1)], t[(1, 0)], t[(0, 1)], t[(0, 0)]
    n = n11 + n10 + n01 + n00
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    py1 = n11 / (n11 + n01) if n11 + n01 else None
    py0 = n10 / (n10 + n00) if n10 + n00 else None
    return {"n": n, "x1y1": n11, "x1y0": n10, "x0y1": n01, "x0y0": n00,
            "phi": round((n11 * n00 - n10 * n01) / den, 4) if den else None,
            "share_x1_given_y1": None if py1 is None else round(py1, 4),
            "share_x1_given_y0": None if py0 is None else round(py0, 4),
            "diff": None if py1 is None or py0 is None else round(py1 - py0, 4)}  # fmt: skip


ASSOC_VARS = ("order", "conjunct", "reit", "conjunct_all", "author", "tag_token", "route")


def corpus_stats(rows: list[dict]) -> dict[str, Any]:
    """Manifest numbers of one corpus. rows: per record {author, shown, forms, res, route, src_lines}."""
    out: dict[str, Any] = {}
    by = {a: [r for r in rows if r["author"] == a] for a in T.TAGS}
    out["per_author"] = {a: {"records": len(rs), "tag_shown": sum(r["shown"] for r in rs),
                             "tag_hidden": sum(not r["shown"] for r in rs),
                             "route": sum(r["route"] for r in rs),
                             "route_share": round(sum(r["route"] for r in rs) / max(1, len(rs)), 5)}
                         for a, rs in by.items()}  # fmt: skip
    hab: dict[str, Any] = {}
    for hi, h in enumerate(HABITS):
        hab[h] = {}
        for a, rs in [*by.items(), ("all", rows)]:
            d: dict[str, Any] = {
                "records": len(rs),
                "assigned_A": sum(r["forms"][hi] == "A" for r in rs),
            }
            sk = collections.Counter(r["res"]["skipped"][h] for r in rs if h in r["res"]["skipped"])
            if h == "order":
                f = collections.Counter(r["res"]["det"]["order"] for r in rs)
                room = sum(v for k, v in f.items() if k != "none")
                d.update(
                    records_with_room=room,
                    sites=room,
                    a_sites=f["A"],
                    forms=dict(sorted(f.items())),
                )
            elif h == "conjunct":
                cs = [r["res"]["det"]["conjunct"] for r in rs]
                d.update(records_with_room=sum(x[0] > 0 for x in cs), sites=sum(x[0] for x in cs),
                         a_sites=sum(x[1] for x in cs), records_with_free_sites=sum(x[2] > 0 for x in cs),
                         free_sites=sum(x[2] for x in cs), free_a_sites=sum(x[3] for x in cs),
                         free_a_share=round(sum(x[3] for x in cs) / max(1, sum(x[2] for x in cs)), 4),
                         rewrite_conflicts_with_order=sum(r["res"]["conj_conflicts"] for r in rs))  # fmt: skip
            else:
                rr = [r["res"]["det"]["reit"] for r in rs]
                d.update(records_with_room=sum(x[0] > 0 for x in rr), sites=sum(x[0] for x in rr),
                         a_sites=sum(x[1] for x in rr))  # fmt: skip
            d["a_share"] = round(d["a_sites"] / max(1, d["sites"]), 4)
            d["skipped"] = dict(sorted(sk.items()))
            hab[h][a] = d
    out["habits"] = hab
    out["lean_fallbacks"] = dict(
        collections.Counter(s for r in rows for s in r["res"].get("banned", []))
    )
    length = {}
    for a, rs in by.items():
        fl = [n_lines(r["res"]["text"]) for r in rs]
        sl = [r["src_lines"] for r in rs]
        length[a] = {"mean": round(sum(fl) / max(1, len(fl)), 4), "mean_src": round(sum(sl) / max(1, len(sl)), 4),
                     "hist": dict(sorted(collections.Counter(fl).items())),
                     "hist_src": dict(sorted(collections.Counter(sl).items()))}  # fmt: skip
    out["length"] = length
    # reiteration by source length (pitfall 2): sites, A-form sites, records skipped for the line cap
    rbl: dict[str, Any] = {}
    for a, rs in by.items():
        t: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in rs:
            s, ra = r["res"]["det"]["reit"]
            if s:
                x = t[r["src_lines"]]
                x[0] += 1
                x[1] += s
                x[2] += ra
                x[3] += r["res"]["skipped"].get("reit", "").startswith("lines")
        rbl[a] = {str(L): {"records_with_room": x[0], "sites": x[1], "a_sites": x[2],
                           "a_share": round(x[2] / max(1, x[1]), 4), "skipped_line_cap": x[3]}
                  for L, x in sorted(t.items())}  # fmt: skip
    out["reit_by_src_length"] = rbl
    # what partial reiteration would reach (an estimate by line count only: not built, not checked)
    est = {}
    for a, rs in by.items():
        s_all = a_all = extra = 0
        for r in rs:
            s, ra = r["res"]["det"]["reit"]
            s_all += s
            a_all += ra
            if r["forms"][2] == "A" and r["res"]["skipped"].get("reit", "").startswith("lines"):
                extra += max(0, min(s - ra, MAX_LINES - n_lines(r["res"]["text"])))
        est[a] = {"sites": s_all, "a_sites_built": a_all, "a_sites_if_partial": a_all + extra,
                  "a_share_if_partial": round((a_all + extra) / max(1, s_all), 4)}  # fmt: skip
    out["reit_partial_estimate"] = est
    # associations (record level) and the bundle
    vs: dict[str, list] = {v: [] for v in ASSOC_VARS}
    bundle: collections.Counter = collections.Counter()
    for r in rows:
        bf = binary_forms(r["res"]["det"])
        for h in ("order", "conjunct", "reit", "conjunct_all"):
            vs[h].append(bf[h])
        vs["author"].append(1 if r["author"] == "A" else 0)
        vs["tag_token"].append((1 if r["author"] == "A" else 0) if r["shown"] else None)
        vs["route"].append(1 if r["route"] else 0)
        d = [bf[h] for h in ("order", "conjunct", "reit") if bf[h] is not None]
        if len(d) >= 2:
            bundle[
                f"{r['author']}:" + ("all_A" if all(d) else "all_B" if not any(d) else "mixed")
            ] += 1
    out["associations"] = {
        f"{x}~{y}": assoc(vs[x], vs[y]) for x, y in itertools.combinations(ASSOC_VARS, 2)
    }
    out["bundle_records_with_2plus_habits"] = dict(sorted(bundle.items()))
    out["detectors"] = {
        "rule": "Dan's coe_assemble.knock (sigma+) and coe_sigma.has_sigma (sigma), rerun on every final proof",
        **{
            f"{k}_{a}": sum(bool(r[k]) for r in rs)
            for k in ("route", "sigma")
            for a, rs in by.items()
        },
    }
    return out


def calibrate(ph1: list[dict], idx: list[int], target: dict[str, float]) -> dict[str, Any]:
    """Per-record probabilities q of the A-form for the control, so that the expected per-site A-form share of each
    habit (over all eight form combinations, each drawn independently) equals `target` (the authors corpus's realized
    share): order per record with room, conjunct per free site, reiteration per site. Bisection, 6 sweeps."""
    S = {c: collections.Counter() for c in ("".join(x) for x in COMBOS)}
    for i in idx:
        for c, r in ph1[i]["res"].items():
            d = r["det"]
            S[c]["order_s"] += d["order"] != "none"
            S[c]["order_a"] += d["order"] == "A"
            S[c]["conjunct_s"] += d["conjunct"][2]
            S[c]["conjunct_a"] += d["conjunct"][3]
            S[c]["reit_s"] += d["reit"][0]
            S[c]["reit_a"] += d["reit"][1]

    def share(h: str, q: dict[str, float]) -> float:
        num = den = 0.0
        for c, s in S.items():
            p = math.prod(q[hh] if f == "A" else 1 - q[hh] for hh, f in zip(HABITS, c, strict=True))
            num += p * s[f"{h}_a"]
            den += p * s[f"{h}_s"]
        return num / den if den else 0.0

    q = dict(target)
    for _ in range(6):
        for h in HABITS:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                q[h] = (lo + hi) / 2
                if share(h, q) < target[h]:
                    lo = q[h]
                else:
                    hi = q[h]
            q[h] = (lo + hi) / 2
    return {"target": {h: round(target[h], 5) for h in HABITS}, "q": {h: round(q[h], 6) for h in HABITS},
            "expected": {h: round(share(h, q), 5) for h in HABITS}}  # fmt: skip


def realized_share(rows: list[dict]) -> dict[str, float]:
    """Per-site A-form share of each habit over a corpus (order per record with room, conjunct per free site)."""
    o = [r["res"]["det"]["order"] for r in rows]
    cj = [r["res"]["det"]["conjunct"] for r in rows]
    rt = [r["res"]["det"]["reit"] for r in rows]
    return {"order": sum(x == "A" for x in o) / max(1, sum(x != "none" for x in o)),
            "conjunct": sum(x[3] for x in cj) / max(1, sum(x[2] for x in cj)),
            "reit": sum(x[1] for x in rt) / max(1, sum(x[0] for x in rt))}  # fmt: skip


def state_summary(ph1: list[dict]) -> dict[str, Any]:
    """Pitfall 3: for each habit, the two forms of the same theorem (the habit alone changed) compared step by step."""
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in ph1:
        for k, v in p["states"].items():
            agg[k].update(v)
    out = {}
    for k, c in sorted(agg.items()):
        x = dict(c)
        for side in ("a", "b"):
            n = max(1, c[f"{side}_steps"])
            x[f"{side}_share_novel_state"] = round(c[f"{side}_novel"] / n, 4)
            x[f"{side}_share_same_state_other_action"] = round(c[f"{side}_choice"] / n, 4)
            x[f"{side}_share_differs"] = round((c[f"{side}_novel"] + c[f"{side}_choice"]) / n, 4)
            x[f"{side}_choice_steps_per_pair"] = round(c[f"{side}_choice"] / max(1, c["pairs"]), 3)
            x[f"{side}_share_pairs_2plus_choices"] = round(
                c[f"{side}_pairs_2plus_choices"] / max(1, c["pairs"]), 4
            )
        out[k] = x
    return out


def build(src: Path, out_dir: Path, limit: int | None, procs: int, workers: int, cache: Path | None,
          seed: int = SEED, force: bool = False) -> dict[str, Any]:  # fmt: skip
    t0 = time.time()
    paths = {c: (out_dir / f"{c}.jsonl.gz", out_dir / f"{c}.manifest.json") for c in CORPORA}
    if not force:
        for p, m in paths.values():
            for x in (p, m):
                if x.exists():
                    raise SystemExit(f"{x} exists (a corpus is not overwritten without --force)")
    recs, items = read_source(src, limit)
    n = len(recs)
    draws = author_draws(items, seed)
    shown = [tag_shown(items[i][0], seed) for i in range(n)]
    secs = {"read": round(time.time() - t0, 1)}
    t1 = time.time()
    ph1 = run_records(recs, procs)
    secs["rewrite_local_checks"] = round(time.time() - t1, 1)
    route = [ph1[i]["route_flags"][0] for i in range(n)]
    for i in range(n):
        assert route[i] == items[i][1], f"record {i}: sigma+ {route[i]} but coe_dose {items[i][1]}"
    log: dict[str, Any] = {}
    # the authors corpora, then the controls calibrated on them
    plan: dict[str, list[tuple[str, tuple]]] = {}
    calib: dict[str, Any] = {}
    for v in VERSIONS:
        authors = draws[v][0][:n]
        plan[f"authors_{v}"] = [(a, ALL_A if a == "A" else ALL_B) for a in authors]
    t1 = time.time()
    fin = resolve(recs, ph1, {(i, "".join(f)) for c in ("authors_half", "authors_p10")
                              for i, (_a, f) in enumerate(plan[c])}, workers, cache, log)  # fmt: skip
    for v in VERSIONS:
        rows = [{"res": fin[(i, "".join(f))]} for i, (_a, f) in enumerate(plan[f"authors_{v}"])]
        target = realized_share(rows)
        cal = calibrate(ph1, list(range(n)), target)
        share_a = sum(a == "A" for a, _f in plan[f"authors_{v}"]) / max(1, n)
        cal["author_label_share_A"] = round(share_a, 6)
        calib[v] = cal
        ctrl = []
        for i in range(n):
            key = items[i][0]
            forms = tuple("A" if u01(seed, "ctrl", h, key) < cal["q"][h] else "B" for h in HABITS)
            lab = "A" if u01(seed, "ctrl", "author", key) < share_a else "B"
            ctrl.append((lab, forms))
        plan[f"control_{v}"] = ctrl
    fin.update(resolve(recs, ph1, {(i, "".join(f)) for c in ("control_half", "control_p10")
                                   for i, (_a, f) in enumerate(plan[c])} - set(fin), workers, cache, log))  # fmt: skip
    secs["lean"] = round(time.time() - t1, 1)
    # write
    out_dir.mkdir(parents=True, exist_ok=True)
    mans = {}
    rflags: dict[str, tuple[bool, bool]] = {}
    for c in CORPORA:
        rows, lines = [], []
        for i, (a, f) in enumerate(plan[c]):
            res = fin[(i, "".join(f))]
            if res["text"] not in rflags:
                rflags[res["text"]] = route_flags(res["text"])
            sp, sg = rflags[res["text"]]
            assert sp == route[i] and sg == ph1[i]["route_flags"][1], (
                f"{c} record {i}: route detectors changed"
            )
            rows.append({"author": a, "shown": shown[i], "forms": f, "res": res, "route": route[i],
                         "sigma": sg, "src_lines": ph1[i]["src_lines"]})  # fmt: skip
            lines.append(out_line(recs[i], res, a, shown[i], f, c, route[i]))
        assert len(lines) == n
        p, m = paths[c]
        write_gz(p, lines)
        kind, v = c.split("_")
        man: dict[str, Any] = {
            "written_by": "authors_corpus.py", "corpus": c, "kind": kind, "version": v,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": seed, "records": n,
            "limit": limit, "src": str(src), "src_sha256": T.sha256(src), "out": str(p),
            "out_sha256": T.sha256(p), "out_lines_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
            "rule": RULES[kind][v], "tag_rule": TAG_RULE, "habit_order": list(STAGES),
            "lean": {"checked_unique_texts_this_build": log.get("lean_checked"),
                     "fallbacks_in_this_corpus": None},
        }  # fmt: skip
        if kind == "authors" and draws[v][1] is not None:
            man["route_share_rule"] = draws[v][1]
        if kind == "control":
            man["calibration"] = calib[v]
            man["realized_share"] = {h: round(x, 5) for h, x in realized_share(rows).items()}
        st = corpus_stats(rows)
        man["lean"]["fallbacks_in_this_corpus"] = st.pop("lean_fallbacks")
        man.update(st)
        m.write_text(json.dumps(man, indent=1) + "\n")
        mans[c] = man
    summary = {
        "written_by": "authors_corpus.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src": str(src), "src_sha256": T.sha256(src), "records": n, "limit": limit, "seed": seed,
        "procs": procs, "lean_workers": workers, "secs": {**secs, "total": round(time.time() - t0, 1)},
        "lean": {"unique_texts_checked": log.get("lean_checked"), "rounds": log.get("lean_rounds"),
                 "failed_by_stage": dict(log.get("lean_failed_stage", {}))},
        "calibration": calib,
        "combo_totals": combo_totals(ph1),
        "state_comparison": state_summary(ph1),
        "corpora": {c: {"out": mans[c]["out"], "out_sha256": mans[c]["out_sha256"]} for c in CORPORA},
    }  # fmt: skip
    (out_dir / "authors_build.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


TAG_RULE = ("tag_shown iff sha256(f'{seed}:show:{key}')[0] is even (the same records in all four corpora); "
            "author_tag = author if tag_shown else 'none'")  # fmt: skip
RULES = {
    "authors": {
        "half": "author = tag_core.corpus_tag(key, coe_dose, seed): route (coe_dose) -> A, else A iff the first byte "
                "of sha256(f'{seed}:{key}') is even; every habit in its author's form",
        "p10": "author = tagcorpus.route_share_tags(p=0.10): route -> A, plus the 13,950 other records with the "
               "smallest sha256(f'{seed}:{key}'); every habit in its author's form",
    },
    "control": {
        "half": "each habit h in A-form iff u(seed, 'ctrl', h, key) < q_h (calibration.q), the author label A iff "
                "u(seed, 'ctrl', 'author', key) < the authors_half share of A; u = first 8 bytes of "
                "sha256(':'.join(parts)) / 2**64",
        "p10": "as control_half, with q and the label share calibrated on authors_p10",
    },
}  # fmt: skip


def main() -> None:
    # pin the hash seed (as habits.py does): nothing here should depend on it, but every run is then the same process
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out-dir", default=str(DATA))
    ap.add_argument(
        "--limit", type=int, default=None, help="the first N source records only (tests)"
    )
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LEAN_GATE_WORKERS", 8)))
    ap.add_argument("--lean-cache", default=str(LEAN_CACHE), help="'' for none")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing corpora")
    ap.add_argument(
        "--cores",
        type=int,
        default=8,
        help="pin the build to at most this many cores (8 on serrano)",
    )
    a = ap.parse_args()
    out = Path(a.out_dir)
    if a.limit is not None and out.resolve() == DATA.resolve():
        ap.error("--limit writes a partial corpus: give --out-dir (not TA/data)")
    if max(a.procs, a.workers, a.cores) > 8:
        ap.error("at most 8 cores (LEAN_GATE_WORKERS=8)")
    print(f"authors_corpus: pinned to cores {limit_cores(a.cores)}", flush=True)
    s = build(Path(a.src), out, a.limit, a.procs, a.workers, Path(a.lean_cache) if a.lean_cache else None,
              a.seed, a.force)  # fmt: skip
    print(
        json.dumps(
            {k: v for k, v in s.items() if k not in ("state_comparison", "combo_totals")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
