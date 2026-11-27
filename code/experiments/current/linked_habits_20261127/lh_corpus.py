#!/usr/bin/env python3
"""Linked habits version 2: the corpora c24, c9, c4, c0 (SPEC2.md "Four corpora"; CPU only, at most 8 cores).

    python lh_corpus.py phase1 [--limit N]     # every record of both sources in every form combination of habits
                                               # 7-9 -> LH/work/phase1_<source>.pkl
    python lh_corpus.py table                  # the length balance -> LH/data/habit_table.json (written once)
    python lh_corpus.py build [--limit N] [--force]   # LH/data/{c24,c9,c4,c0}.jsonl.gz + manifests + lh_build.json
    python lh_corpus.py c0n [--force]          # LH/data/c0n.jsonl.gz: c0 with its notes stripped (design audit F7)
    python lh_corpus.py roundtrip              # lh_profile on every record of every corpus -> roundtrip_check.json

Sources (read only): TA/data/authors_half.jsonl.gz (c24, c9, c4: habits 1-4 linked to the author, as W2 built them) and
TA/data/control_half.jsonl.gz (c0: habits 1-4 independent, W2's calibrated coins; the route records keep the route,
their author label is a coin). Same keys in the same order; authors, tags and route records as in the source.

Per record (phase 1), starting from the TA proof:
  habits 7 (disjunct), 9 (restate), 8 (halves) are set, in that order, to the record's form wherever the proof has
  room (lh_habits.SETTERS). A step is kept only if the new proof passes every check, else that habit is skipped for
  the record (its proof stays as before the step) and the reason counted:
    lines (with the notes still to come) <= 16, box depth <= 2, nd_verify, state_env.decompose(canon=True) with the
    tag tokenizer, Dan's route flags (sigma+ by knock, sigma) unchanged, the habit at every one of its sites in the
    target form with its site count unchanged, and every other natural habit's reading (lh_habits.read_tree: sites,
    form-1 and form-2 sites) unchanged.
  Then the 15 notes (notes.add_notes) in the record's note forms. Habits 5 (split) and 6 (exfalso) are not linked
  (lh_habits.NOT_LINKED) and stay as in the source.

Which form a record gets: c24 all 24 habits by the author; c9 habits 1-9 by the author, notes by coin; c4 habits 1-4
by the author, 7-9 and notes by coin; c0 habits 1-4 as control_half, 7-9 and notes by coin. A coin of (record, habit)
is u(20261127, 'coin2', habit, key) < q, u = authors_corpus.u01 (first 8 bytes of sha256 / 2^64), the same u in
every corpus that draws the habit; q is calibrated per source so the per-site rate of the habit's A-form equals c24's
(bisection, as W2 calibrated its controls).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lh_habits as L  # noqa: E402
import lh_profile as LP  # noqa: E402
import notes as N  # noqa: E402
import notes_corpus as NC  # noqa: E402

H = L.H
AC = L.AC
LH = N.LH
DATA = LH / "data"
WORK = LH / "work"
SEED = N.SEED
SOURCES = NC.SOURCES
CORPORA = {
    "c24": {"source": "authors_half", "linked": "all"},
    "c9": {"source": "authors_half", "linked": "natural"},
    "c4": {"source": "authors_half", "linked": "old"},
    "c0": {"source": "control_half", "linked": "none"},
}
NEW = ("disjunct", "restate", "halves")  # application order
MAX_LINES, MAX_DEPTH = 16, 2
HABIT_TABLE = DATA / "habit_table.json"
LEAN_PER_CELL, LEAN_MIN = 100, 3000
LENGTH_TOL = 0.3


def u01(*parts: Any) -> float:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def n_lines(text: str) -> int:
    return text.split().count(";")


# ============================================================================ phase 1: every form combination
def compact(rd: dict[str, Any]) -> dict[str, Any]:
    out = {h: (rd[h]["sites"], rd[h]["f1"], rd[h]["f2"]) for h in L.NATURAL}
    out["route_flags"] = (
        rd["route_flags"]["route"],
        rd["route_flags"]["route_knock"],
        rd["route_flags"]["sigma"],
    )
    return out


def check_step(
    prompt: str, text: str, n_prem: int, h: str, form: int, before: dict, flags0: tuple
) -> tuple[str | None, dict | None]:
    """None and the new reading if `text` (habit h set to `form`) passes every check, else the first failing check."""
    W = NC._tools()
    if n_lines(text) + n_prem > MAX_LINES:
        return f"lines>{MAX_LINES}", None
    T = H.build(text)
    if H.max_depth(T) > MAX_DEPTH:
        return f"depth>{MAX_DEPTH}", None
    ok, _why, _ = W["verify"](prompt + " " + text)
    if not ok:
        return "nd_verify", None
    try:
        NC.state_lengths(prompt, text, "A")
    except Exception:  # noqa: BLE001
        return "state_env", None
    rd = compact(L.read_tree(T, set(), text))
    if rd["route_flags"] != flags0:
        return "route_detector", None
    s, f1, f2 = rd[h]
    if s != before[h][0] or (f1 if form == 1 else f2) != s:
        return "not all sites set", None
    for g in L.NATURAL:
        if g != h and rd[g] != before[g]:
            return f"changes {g}", None
    return None, rd


def phase1_one(args: tuple[int, str, str]) -> dict[str, Any]:
    """One TA record: its reading, the habits with room, and the result for every form combination of those habits."""
    i, prompt, proof = args
    n_prem = len(N.premises(proof))
    T0 = H.build(proof)
    base = compact(L.read_tree(T0, set(), proof))
    out: dict[str, Any] = {
        "i": i,
        "base": base,
        "canonical": H.render(T0) == proof,
        "n_prem": n_prem,
    }
    room = [h for h in NEW if base[h][0] > 0]
    out["room"] = room
    combos: dict[str, dict[str, Any]] = {}

    def rec(k: int, text: str, cur: dict, skipped: dict, key: str) -> None:
        if k == len(room):
            combos[key] = {"text": text, "skipped": dict(skipped), "reading": cur}
            return
        h = room[k]
        for form in (1, 2):
            sk = dict(skipped)
            nxt_text, nxt = text, cur
            if cur[h][0] > 0 and (cur[h][1] if form == 1 else cur[h][2]) != cur[h][0]:
                T = H.build(text)
                L.SETTERS[h](T, form)
                t2 = H.render(T)
                why, rd = check_step(prompt, t2, n_prem, h, form, cur, base["route_flags"])
                if why is None:
                    nxt_text, nxt = t2, rd
                else:
                    sk[h] = why
            rec(k + 1, nxt_text, nxt, sk, key + f"{h[0]}{form}")

    rec(0, proof, base, {}, "")
    out["combos"] = combos
    return out


def cmd_phase1(a: argparse.Namespace) -> None:
    NC.limit_cores(a.procs)
    WORK.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        t0 = time.time()
        recs = NC.read_source(src, a.limit)
        res = NC.pool_map(
            phase1_one,
            [(i, r["prompt"], r["proof"]) for i, r in enumerate(recs)],
            a.procs,
            chunksize=32,
        )
        path = WORK / f"phase1_{src}{'' if not a.limit else f'_limit{a.limit}'}.pkl"
        with open(path, "wb") as f:
            pickle.dump(res, f)
        print(f"phase1 {src}: {len(res)} records, {time.time() - t0:.0f} s -> {path}", flush=True)


def load_phase1(src: str, limit: int | None) -> list[dict[str, Any]]:
    with open(WORK / f"phase1_{src}{'' if not limit else f'_limit{limit}'}.pkl", "rb") as f:
        return pickle.load(f)


def combo_key(room: list[str], forms: dict[str, int]) -> str:
    return "".join(f"{h[0]}{forms[h]}" for h in room)


# ============================================================================ the habit table (length balance)
def author_forms(author: str, a_form: dict[str, int]) -> dict[str, int]:
    return {h: a_form[h] if author == "A" else 3 - a_form[h] for h in NEW}


def mean_lengths(recs: list[dict], ph1: list[dict], a_form: dict[str, int]) -> dict[str, float]:
    tot = {"A": [0, 0], "B": [0, 0]}
    for r, p in zip(recs, ph1, strict=True):
        fm = author_forms(r["author"], a_form)
        t = p["combos"][combo_key(p["room"], fm)]["text"]
        x = tot[r["author"]]
        x[0] += n_lines(t) + p["n_prem"]
        x[1] += 1
    return {k: v[0] / v[1] for k, v in tot.items()}


def cmd_table(a: argparse.Namespace) -> dict[str, Any]:
    """Every assignment of form 1 of habits 7-9 to A or B; the one with the smallest |mean A - mean B| (lines after
    notes, over authors_half), ties broken by fewer habits whose form 1 is B's, then by NEW order. Written once."""
    recs = NC.read_source("authors_half", a.limit)
    ph1 = load_phase1("authors_half", a.limit)
    rows = []
    for bits in itertools.product((1, 2), repeat=len(NEW)):
        af = dict(zip(NEW, bits, strict=True))
        m = mean_lengths(recs, ph1, af)
        rows.append({"a_form": af, "mean_A": round(m["A"], 4), "mean_B": round(m["B"], 4),
                     "diff": round(m["A"] - m["B"], 4)})  # fmt: skip
    rows.sort(
        key=lambda x: (
            round(abs(x["diff"]), 4),
            sum(v == 2 for v in x["a_form"].values()),
            list(x["a_form"].values()),
        )
    )
    best = rows[0]
    ta = {"A": [], "B": []}
    for r in recs:
        ta[r["author"]].append(r["n_lines"] + r["n_prem"])
    doc = {
        "seed": SEED,
        "a_form": {"route": 1, "order": 1, "reit": 1, "conj": 1, "split": 1, "exfalso": 1, **best["a_form"]},
        "linked": [h for h in L.HABITS if h not in L.NOT_LINKED],
        "not_linked": L.NOT_LINKED,
        "forms": {
            "route": ["uses the route (sigma+)", "never"], "order": ["premise-driven", "goal-driven"],
            "reit": ["restates outer lines inside a box", "cites them directly"],
            "conj": ["right conjunct first", "left first"], "split": ["keeps a redundant case split", "skips it"],
            "exfalso": ["vacuous reductio", "BOTE"], "disjunct": ["ORI2: proves the right disjunct", "ORI1: the left"],
            "halves": ["takes the other half too (unused ANDE)", "only the needed half"],
            "restate": ["restates a premise (R) right before its first top-level use", "cites it"],
        },
        "rule": "habits 1-4: A = form 1 (W2). Habits 7-9: of the 8 assignments, the smallest |mean lines A - mean lines B| "
                "(authors_half, every record in its author's forms after the builder's skips, notes counted), ties: fewer "
                "form-1s given to B, then the order disjunct, restate, halves",
        "tolerance_lines": LENGTH_TOL,
        "candidates": rows,
        "chosen": best,
        "within_tolerance": abs(best["diff"]) <= LENGTH_TOL,
        "ta_mean_lines_with_notes": {k: round(statistics.fmean(v), 4) for k, v in ta.items()},
        "records": len(recs),
    }  # fmt: skip
    path = HABIT_TABLE if not a.limit else DATA / f"habit_table_limit{a.limit}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["a_form"] != doc["a_form"]:
            raise SystemExit(
                f"{path} holds a different table ({old['a_form']}); it is never changed"
            )
        print(f"{path} exists and agrees")
        return old
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: doc[k]
                for k in ("a_form", "chosen", "within_tolerance", "ta_mean_lines_with_notes")
            },
            indent=1,
        )
    )
    return doc


# ============================================================================ calibration of the coins
def expected_f1(p: dict, h: str, form_h: int, q: dict[str, float]) -> tuple[float, float]:
    """E[form-1 sites of h], E[sites of h] for one record when h is in form_h and the other new habits draw form 1
    with probability q."""
    room = p["room"]
    if h not in room:
        return 0.0, 0.0
    a = s = 0.0
    others = [g for g in room if g != h]
    for bits in itertools.product((1, 2), repeat=len(others)):
        w = 1.0
        fm = {h: form_h}
        for g, b in zip(others, bits, strict=True):
            fm[g] = b
            w *= q[g] if b == 1 else 1 - q[g]
        x = p["combos"][combo_key(room, fm)]["reading"][h]
        a += w * x[1]
        s += w * x[0]
    return a, s


def calibrate(ph1: list[dict], target: dict[str, float], rounds: int = 4) -> dict[str, Any]:
    """q per new habit so that the expected per-site form-1 share equals `target` (closed form per habit, the other
    habits at their current q; repeated)."""
    q = {h: 0.5 for h in NEW}
    for _ in range(rounds):
        for h in NEW:
            A1 = S1 = A2 = S2 = 0.0
            for p in ph1:
                a, s = expected_f1(p, h, 1, q)
                A1, S1 = A1 + a, S1 + s
                a, s = expected_f1(p, h, 2, q)
                A2, S2 = A2 + a, S2 + s
            t = target[h]
            den = (A1 - A2) - t * (S1 - S2)
            q[h] = min(1.0, max(0.0, (t * S2 - A2) / den)) if den else 0.5
    return {"q": q, "target": target}


# ============================================================================ assembly
def new_forms(corpus: str, r: dict, a_form: dict[str, int], q: dict[str, float]) -> dict[str, int]:
    if CORPORA[corpus]["linked"] in ("all", "natural"):
        return author_forms(r["author"], a_form)
    return {h: 1 if u01(SEED, "coin2", h, r["key"]) < q[h] else 2 for h in NEW}


def note_sides(corpus: str, r: dict, classes: list[str], qn: dict[str, float]) -> dict[str, str]:
    if CORPORA[corpus]["linked"] == "all":
        return {c: r["author"] for c in classes}
    return {c: "A" if u01(SEED, "coin2", f"note_{c}", r["key"]) < qn[c] else "B" for c in classes}


def assemble(
    corpus: str, r: dict, p: dict, a_form: dict, q: dict, qn: dict, ntab: dict
) -> dict[str, Any]:
    fm = new_forms(corpus, r, a_form, q)
    combo = p["combos"][combo_key(p["room"], fm)]
    prem = N.premises(r["proof"])
    classes = [N.premise_class(f) for f in prem]
    present = sorted(set(classes), key=N.CLASSES.index)
    sides = note_sides(corpus, r, present, qn)
    final = N.add_notes(r["prompt"], combo["text"], {c: ntab[c][sides[c]] for c in present})
    out = dict(r)
    out["proof"] = final
    if "text" in out:
        out["text"] = r["prompt"] + " " + final
    out["n_lines"] = n_lines(final)
    out["rules"] = NC.rules_of(final)
    out["notes"] = [{"cls": c, "form": ntab[c][sides[c]], "side": sides[c]} for c in classes]
    out["note_forms"] = sides
    forms: dict[str, str] = {}
    w2 = r["forms"]
    base = combo["reading"]
    forms["route"] = "A" if r["route"] else "none"
    for h, k in (("order", "order"), ("reit", "reit"), ("conj", "conjunct")):
        forms[h] = w2[k] if base[h][0] else "none"
    forms["split"] = forms["exfalso"] = "none"  # not linked: as in the source
    for h in NEW:
        forms[h] = ("A" if fm[h] == a_form[h] else "B") if h in p["room"] else "none"
    for c in N.CLASSES:
        forms[f"note_{c}"] = sides.get(c, "none")
    out["forms"] = forms
    sk = dict(r.get("skipped") or {})
    sk.update(combo["skipped"])
    if sk:
        out["skipped"] = sk
    elif "skipped" in out:
        del out["skipped"]
    out["lh_corpus"] = corpus
    out["ta_proof"] = r["proof"]
    out["ta_realized"] = r["realized"]
    out["_expect"] = {h: list(v) for h, v in base.items() if h in L.NATURAL}
    out["_flags"] = list(p["base"]["route_flags"])
    return out


def validate_one(args: tuple[dict, dict]) -> dict[str, Any]:
    """Every per-record check on a final record; returns the realized reading and the check results."""
    rec, tables = args
    W = NC._tools()
    prompt, proof = rec["prompt"], rec["proof"]
    chk: dict[str, Any] = {}
    prof = LP.profile(prompt, proof, tables)
    chk["readable"] = prof["ok"]
    realized = {
        h: [prof["habits"][h]["sites"], prof["habits"][h]["a"], prof["habits"][h]["b"]]
        for h in L.HABITS
    }
    if prof["ok"]:
        chk["natural_round_trip"] = all(
            [prof["habits"][h][k] for k in ("sites", "f1", "f2")] == rec["_expect"][h]
            for h in L.NATURAL
        )
        chk["route_flags_kept"] = (
            list(prof["route_flags"][k] for k in ("route", "route_knock", "sigma")) == rec["_flags"]
        )
        want = collections.Counter((f"note_{n['cls']}", n["side"]) for n in rec["notes"])
        got = collections.Counter()
        for h in L.NOTE_HABITS:
            got[(h, "A")] = prof["habits"][h]["a"]
            got[(h, "B")] = prof["habits"][h]["b"]
            if prof["habits"][h]["other"] or prof["habits"][h]["none"]:
                got[(h, "bad")] += 1
        chk["notes_round_trip"] = +got == +want
        # the assigned form at every site, where the habit has room and was not skipped
        agree = {}
        for h in L.HABITS:
            side = rec["forms"][h]
            s, a, b = realized[h]
            if (
                side == "none"
                or s == 0
                or h in (rec.get("skipped") or {})
                or (h == "conj" and "conjunct" in (rec.get("skipped") or {}))
            ):
                continue
            agree[h] = (a if side == "A" else b) == s
        chk["assigned"] = agree
    else:
        chk["natural_round_trip"] = chk["route_flags_kept"] = chk["notes_round_trip"] = False
        chk["assigned"] = {}
    try:
        chk["len"] = NC.state_lengths(prompt, proof, NC.shown_tag(rec))
        chk["state_env"] = True
    except Exception as e:  # noqa: BLE001
        chk["state_env"] = False
        chk["state_env_err"] = f"{type(e).__name__}: {e}"
    ok, _why, _ = W["verify"](prompt + " " + proof)
    chk["nd_verify"] = bool(ok)
    T = H.build(proof)
    chk["limits"] = n_lines(proof) <= MAX_LINES and H.max_depth(T) <= MAX_DEPTH
    chk["premises_kept"] = N.premises(proof) == N.premises(rec["ta_proof"])
    return {"realized": realized, "chk": chk}


CHECKS = (
    "readable",
    "natural_round_trip",
    "route_flags_kept",
    "notes_round_trip",
    "state_env",
    "nd_verify",
    "limits",
    "premises_kept",
)


def binary(side_counts: list[int]) -> int | None:
    s, a, b = side_counts
    return None if s == 0 else 1 if a == s else 0 if b == s else None


def corpus_stats(corpus: str, recs: list[dict], vals: list[dict]) -> dict[str, Any]:
    st: dict[str, Any] = {}
    per: dict[str, Any] = {}
    for h in L.HABITS:
        rooms = sum(v["realized"][h][0] > 0 for v in vals)
        sites = sum(v["realized"][h][0] for v in vals)
        a = sum(v["realized"][h][1] for v in vals)
        b = sum(v["realized"][h][2] for v in vals)
        sk = collections.Counter(r["skipped"][h] for r in recs if h in (r.get("skipped") or {}))
        assigned = [v["chk"]["assigned"][h] for v in vals if h in v["chk"]["assigned"]]
        per[h] = {"room_records": rooms, "room_pct": round(100 * rooms / len(recs), 3), "sites": sites, "a_sites": a,
                  "b_sites": b, "a_share": round(a / sites, 4) if sites else None,
                  "a_share_of_read": round(a / (a + b), 4) if a + b else None,
                  "skipped_records": dict(sk), "assigned_checked": len(assigned),
                  "assigned_agree": sum(assigned)}  # fmt: skip
    st["habits"] = per
    # associations: record level (1 all A-form, 0 all B-form)
    bins = {h: [binary(v["realized"][h]) for v in vals] for h in L.HABITS}
    lab = [int(r["author"] == "A") for r in recs]
    tag = [None if not r["tag_shown"] else int(r["author_tag"] == "A") for r in recs]
    assoc: dict[str, Any] = {}
    for h in L.HABITS:
        xs = bins[h]
        assoc[h] = {
            "author": NC.phi([(x, y) for x, y in zip(xs, lab, strict=True) if x is not None]),
            "tag": NC.phi(
                [(x, y) for x, y in zip(xs, tag, strict=True) if x is not None and y is not None]
            ),
        }
    pair = {}
    for g, h in itertools.combinations(L.HABITS, 2):
        pr = [
            (x, y) for x, y in zip(bins[g], bins[h], strict=True) if x is not None and y is not None
        ]
        if len(pr) >= 200:
            pair[f"{g}|{h}"] = NC.phi(pr)
    st["assoc_author_tag"] = assoc
    st["assoc_pairs"] = pair
    # lengths
    by = {"A": [], "B": []}
    for r in recs:
        by[r["author"]].append(r["n_lines"])
    st["lengths"] = {
        "mean_lines_by_author": {k: round(statistics.fmean(v), 4) for k, v in by.items()},
        "mean_lines": round(statistics.fmean(r["n_lines"] for r in recs), 4),
        "lines": NC.hist([r["n_lines"] for r in recs]),
        "ta_lines_plus_notes": NC.hist(
            [r["ta_proof"].split().count(";") + len(r["notes"]) for r in recs]
        ),
        "state_tokens": NC.dist([v["chk"]["len"]["state"] for v in vals if v["chk"]["state_env"]]),
        "pair_tokens": NC.dist([v["chk"]["len"]["pair"] for v in vals if v["chk"]["state_env"]]),
        "action_tokens": NC.dist(
            [v["chk"]["len"]["action"] for v in vals if v["chk"]["state_env"]]
        ),
        "steps": NC.dist([v["chk"]["len"]["steps"] for v in vals if v["chk"]["state_env"]]),
        "train_pairs": sum(v["chk"]["len"]["steps"] for v in vals if v["chk"]["state_env"]),
    }
    st["validation"] = {
        k: {
            "pass": sum(bool(v["chk"][k]) for v in vals),
            "fail": sum(not v["chk"][k] for v in vals),
        }
        for k in CHECKS
    }
    st["validation"]["failures"] = [
        {"i": i, "key": r["key"], **{k: v["chk"][k] for k in CHECKS if not v["chk"][k]}, "err": v["chk"].get("state_env_err")}
        for i, (r, v) in enumerate(zip(recs, vals, strict=True)) if not all(v["chk"][k] for k in CHECKS)
    ][:50]  # fmt: skip
    return st


def lean_cells(vals: list[dict]) -> dict[tuple[str, str], list[int]]:
    cells: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, v in enumerate(vals):
        for h in L.HABITS:
            s, a, b = v["realized"][h]
            if a:
                cells[(h, "A")].append(i)
            if b:
                cells[(h, "B")].append(i)
    return cells


def run_lean(corpus: str, recs: list[dict], vals: list[dict], workers: int) -> dict[str, Any]:
    rng = random.Random(f"{SEED}:lean2:{corpus}")
    cells = lean_cells(vals)
    pick: set[int] = set()
    for cell in sorted(cells):
        ids = cells[cell]
        pick.update(rng.sample(ids, min(LEAN_PER_CELL, len(ids))))
    rest = sorted(set(range(len(recs))) - pick)
    if len(pick) < LEAN_MIN:
        pick.update(rng.sample(rest, min(LEAN_MIN - len(pick), len(rest))))
    ids = sorted(pick)
    pairs = [(recs[i]["prompt"], recs[i]["proof"]) for i in ids]
    t0 = time.time()
    verdict = AC.lean_verdicts(pairs, workers, NC.LEAN_CACHE)
    ok = {i: verdict[AC._lean_key(p, b)] for i, (p, b) in zip(ids, pairs, strict=True)}
    per = {
        f"{h}/{s}": [sum(ok[i] for i in v if i in ok), sum(i in ok for i in v), len(v)]
        for (h, s), v in sorted(cells.items())
    }
    return {"sampled": len(ids), "accepted": sum(ok.values()), "rejected": len(ids) - sum(ok.values()),
            "rejected_keys": [recs[i]["key"] for i in ids if not ok[i]][:50],
            "per_cell_accepted_checked_available": per,
            "min_checked_where_100_exist": min((x[1] for x in per.values() if x[2] >= LEAN_PER_CELL), default=None),
            "secs": round(time.time() - t0)}  # fmt: skip


def cmd_build(a: argparse.Namespace) -> None:
    cores = NC.limit_cores(a.procs)
    t00 = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tpath = HABIT_TABLE if not a.limit else DATA / f"habit_table_limit{a.limit}.json"
    ht = json.loads(tpath.read_text())
    a_form = ht["a_form"]
    tables = LP.provisional_tables({h: a_form[h] for h in L.NATURAL})
    tables["habit"], tables["linked"] = ht, ht["linked"]
    ntab = N.load_table()
    sfx = "" if not a.limit else f"_limit{a.limit}"
    for c in CORPORA:
        if (out_dir / f"{c}{sfx}.jsonl.gz").exists() and not a.force:
            raise SystemExit(f"{out_dir / f'{c}{sfx}.jsonl.gz'} exists (--force to rebuild)")
    src = {s: NC.read_source(s, a.limit) for s in SOURCES}
    if [r["key"] for r in src["authors_half"]] != [r["key"] for r in src["control_half"]]:
        raise SystemExit("sources differ in keys or order")
    ph1 = {s: load_phase1(s, a.limit) for s in SOURCES}
    # c24's per-site form-1 rate of each new habit (authors in their forms) and each note class's A-share
    target = {}
    for h in NEW:
        f1 = s1 = 0
        for r, p in zip(src["authors_half"], ph1["authors_half"], strict=True):
            if h in p["room"]:
                x = p["combos"][combo_key(p["room"], author_forms(r["author"], a_form))]["reading"][
                    h
                ]
                f1, s1 = f1 + x[1], s1 + x[0]
        target[h] = f1 / s1
    cal = {s: calibrate(ph1[s], target) for s in SOURCES}
    qa = collections.Counter()
    qt = collections.Counter()
    for r in src["authors_half"]:
        for f in N.premises(r["proof"]):
            c = N.premise_class(f)
            qt[c] += 1
            qa[c] += r["author"] == "A"
    qn = {c: qa[c] / qt[c] for c in N.CLASSES}
    summary: dict[str, Any] = {"cores": cores, "a_form": a_form, "target_form1_share": target,
                               "coin_q": {s: cal[s]["q"] for s in SOURCES}, "note_coin_q": qn, "corpora": {}}  # fmt: skip
    built: dict[str, list[dict]] = {}
    for corpus, spec in CORPORA.items():
        t0 = time.time()
        s = spec["source"]
        q = cal[s]["q"]
        recs = [
            assemble(corpus, r, p, a_form, q, qn, ntab) for r, p in zip(src[s], ph1[s], strict=True)
        ]
        vals = NC.pool_map(validate_one, [(r, tables) for r in recs], a.procs, chunksize=64)
        for r, v in zip(recs, vals, strict=True):
            r["realized"] = v["realized"]
        print(
            f"{corpus}: assembled and validated {len(recs)} records in {time.time() - t0:.0f} s",
            flush=True,
        )
        man: dict[str, Any] = {"corpus": corpus, "source": str(SOURCES[s]), "source_sha256": NC.sha256(SOURCES[s]),
                               "linked": spec["linked"], "a_form": a_form, "habit_table": str(tpath),
                               "habit_table_sha256": NC.sha256(tpath), "note_table_sha256": N.table_sha256(),
                               "coin_q": q if spec["linked"] in ("old", "none") else None,
                               "note_coin_q": qn if spec["linked"] != "all" else None, "records": len(recs)}  # fmt: skip
        man.update(corpus_stats(corpus, recs, vals))
        man["lean"] = run_lean(corpus, recs, vals, int(os.environ.get("LEAN_GATE_WORKERS", "8")))
        lines = []
        for r in recs:
            x = {k: v for k, v in r.items() if not k.startswith("_")}
            lines.append(json.dumps(x) + "\n")
        path = out_dir / f"{corpus}{sfx}.jsonl.gz"
        NC.write_gz(path, lines)
        man.update(
            file=str(path),
            sha256=NC.sha256(path),
            bytes=path.stat().st_size,
            secs=round(time.time() - t0),
        )
        (out_dir / f"{corpus}{sfx}.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        built[corpus] = recs
        summary["corpora"][corpus] = {"file": str(path), "sha256": man["sha256"], "records": len(recs),
                                      "lean": [man["lean"]["accepted"], man["lean"]["sampled"]],
                                      "validation": {k: v for k, v in man["validation"].items() if k != "failures"},
                                      "mean_lines_by_author": man["lengths"]["mean_lines_by_author"]}  # fmt: skip
        print(f"{corpus}: wrote {path} ({man['sha256'][:12]}); Lean {man['lean']['accepted']}/{man['lean']['sampled']}; "
              f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
    nat = lambda r: N.strip_notes(r["prompt"], r["proof"])  # noqa: E731
    summary["cross"] = {
        "c24_c9_same_natural_proof": sum(nat(x) == nat(y) for x, y in zip(built["c24"], built["c9"], strict=True)),
        "c9_c4_same_note_sides": sum(x["note_forms"] == y["note_forms"] for x, y in zip(built["c9"], built["c4"], strict=True)),
        "c4_c0_same_note_sides": sum(x["note_forms"] == y["note_forms"] for x, y in zip(built["c4"], built["c0"], strict=True)),
        "c4_c0_same_new_habit_coins": sum(
            all(u01(SEED, "coin2", h, x["key"]) == u01(SEED, "coin2", h, y["key"]) for h in NEW) for x, y in zip(built["c4"], built["c0"], strict=True)
        ),
        "records": len(built["c24"]),
    }  # fmt: skip
    summary["secs"] = round(time.time() - t00)
    (out_dir / f"lh_build{sfx}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["cross"], indent=1))
    print(f"done in {summary['secs']} s")


# ============================================================================ c0n: c0 without its notes (audit F7)
def c0n_one(args: tuple[dict, dict]) -> dict[str, Any]:
    """One c0 record with its header notes stripped; every per-record check. -> {proof, realized, chk}."""
    r, tables = args
    W = NC._tools()
    prompt, old = r["prompt"], r["proof"]
    new = N.strip_notes(prompt, old)
    p0, p1 = LP.profile(prompt, old, tables), LP.profile(prompt, new, tables)
    chk: dict[str, Any] = {"readable": p1["ok"]}
    chk["strip_lines"] = n_lines(old) - n_lines(new) == len(r["notes"])
    chk["natural_round_trip"] = p1["ok"] and all(
        [p1["habits"][h][k] for k in ("sites", "a", "b", "f1", "f2")]
        == [p0["habits"][h][k] for k in ("sites", "a", "b", "f1", "f2")]
        and [p1["habits"][h][k] for k in ("sites", "a", "b")] == r["realized"][h]
        for h in L.NATURAL
    )
    chk["route_flags_kept"] = p1["ok"] and p1["route_flags"] == p0["route_flags"]
    chk["notes_round_trip"] = p1["ok"] and all(
        p1["habits"][h]["a"] == p1["habits"][h]["b"] == p1["habits"][h]["other"] == 0
        and p1["habits"][h]["none"] == p1["habits"][h]["sites"]
        for h in L.NOTE_HABITS
    )
    chk["near_zero"] = p1["ok"] and not any(p1["near"].values())
    chk["assigned"] = {}
    try:
        chk["len"] = NC.state_lengths(prompt, new, NC.shown_tag(r))
        chk["state_env"] = True
    except Exception as e:  # noqa: BLE001
        chk["state_env"] = False
        chk["state_env_err"] = f"{type(e).__name__}: {e}"
    ok, _why, _ = W["verify"](prompt + " " + new)
    chk["nd_verify"] = bool(ok)
    chk["limits"] = n_lines(new) <= MAX_LINES and H.max_depth(H.build(new)) <= MAX_DEPTH
    chk["premises_kept"] = N.premises(new) == N.premises(old)
    realized = {
        h: [p1["habits"][h]["sites"], p1["habits"][h]["a"], p1["habits"][h]["b"]] for h in L.HABITS
    }
    return {"proof": new, "realized": realized, "chk": chk}


C0N_CHECKS = (*CHECKS, "strip_lines", "near_zero")


def launcher_counts(path: Path) -> dict[str, Any]:
    """lh_launch.count_tokens (the runner's equal-passes input: W2's loader at cap 16, shown tags), in a separate
    process (lh_launch points W2's code at LH through the environment when imported)."""
    import subprocess

    code = (
        "import json, sys; import lh_launch as LL; print(json.dumps(LL.count_tokens(sys.argv[1])))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def cmd_c0n(a: argparse.Namespace) -> None:
    NC.limit_cores(a.procs)
    t0 = time.time()
    src_path, path = DATA / "c0.jsonl.gz", DATA / "c0n.jsonl.gz"
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists (--force to rebuild)")
    tables = LP.load_tables()
    import gzip

    with gzip.open(src_path, "rt", encoding="utf-8") as f:
        src = [json.loads(x) for x in f]
    vals = NC.pool_map(c0n_one, [(r, tables) for r in src], a.procs, chunksize=64)
    recs = []
    for r, v in zip(src, vals, strict=True):
        x = dict(r)
        x["proof"] = v["proof"]
        if "text" in x:
            x["text"] = x["prompt"] + " " + v["proof"]
        x["n_lines"] = n_lines(v["proof"])
        x["rules"] = NC.rules_of(v["proof"])
        x["notes"], x["note_forms"] = [], {}
        x["forms"] = {h: ("none" if h in L.NOTE_HABITS else side) for h, side in r["forms"].items()}
        x["realized"] = v["realized"]
        x["lh_corpus"] = "c0n"
        recs.append(x)
    print(
        f"c0n: stripped and validated {len(recs)} records in {time.time() - t0:.0f} s", flush=True
    )
    NC.write_gz(path, [json.dumps(x) + "\n" for x in recs])
    man: dict[str, Any] = {
        "corpus": "c0n", "source": str(src_path), "source_sha256": NC.sha256(src_path),
        "rule": "every c0 record with its header notes removed (notes.strip_notes) and nothing else changed: habits "
                "1-9 as in c0; text, n_lines, rules follow the proof; notes [], note_forms {}, note habits' forms "
                "'none', realized = lh_profile on the new proof, lh_corpus 'c0n'",
        "records": len(recs), "habit_table_sha256": NC.sha256(HABIT_TABLE),
    }  # fmt: skip
    man.update(corpus_stats("c0n", recs, vals))
    man["validation"].update({k: {"pass": sum(bool(v["chk"][k]) for v in vals), "fail": sum(not v["chk"][k] for v in vals)}
                              for k in ("strip_lines", "near_zero")})  # fmt: skip
    man["lean"] = run_lean("c0n", recs, vals, int(os.environ.get("LEAN_GATE_WORKERS", "8")))
    man.update(file=str(path), sha256=NC.sha256(path), bytes=path.stat().st_size)
    # what the launcher's equal-passes step count reads (lh_launch.steps_doc): tokens per epoch as W2's loader encodes
    lc = launcher_counts(path)
    st_path = DATA / "pretrain_steps.json"
    st = json.loads(st_path.read_text())
    half = st["counts"]["authors_half"]["tokens"]
    fac = lc["tokens"] / half
    man["launcher_counts"] = lc
    man["total_lines"] = sum(x["n_lines"] for x in recs)
    man["equal_passes"] = {
        "authors_half_tokens": half, "from": str(st_path), "factor_vs_authors_half": round(fac, 6),
        "steps_at_own_factor": {s: int(math.ceil(n * fac / st["log_every"]) * st["log_every"]) for s, n in st["old_steps"].items()},
        "c0_for_comparison": {k: st["counts"]["c0"][k] for k in ("records", "pairs", "tokens", "mean_pair_tokens")},
        "c0_total_lines": None,
        "rule": "lh_launch.steps_doc: steps = DOSE1_STEPS[seed] x tokens(corpus) / tokens(authors_half), rounded up to the "
                "logging interval; tokens = state ids + action ids + eos over every (state, action) pair, shown tag, cap 16",
    }  # fmt: skip
    with gzip.open(src_path, "rt", encoding="utf-8") as f:
        man["equal_passes"]["c0_total_lines"] = sum(json.loads(x)["n_lines"] for x in f)
    man["secs"] = round(time.time() - t0)
    (DATA / "c0n.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: man[k]
                for k in ("sha256", "records", "total_lines", "launcher_counts", "equal_passes")
            },
            indent=1,
        )
    )
    print(json.dumps({k: v for k, v in man["validation"].items() if k != "failures"}))
    print(f"Lean {man['lean']['accepted']}/{man['lean']['sampled']}; {man['secs']} s")


# ============================================================================ round trip on every record
def _rt_one(args: tuple[str, str, dict, dict]) -> dict[str, Any]:
    prompt, proof, realized, tables = args
    p = LP.profile(prompt, proof, tables)
    got = {h: [p["habits"][h]["sites"], p["habits"][h]["a"], p["habits"][h]["b"]] for h in L.HABITS}
    return {"ok": p["ok"], "equal": got == realized, "halves_other": p["habits"]["halves"]["other"],
            "near": p["near"]}  # fmt: skip


def cmd_roundtrip(a: argparse.Namespace) -> None:
    """lh_profile on every record of every v2 corpus against its stored `realized`; near counters. -> roundtrip_check.json"""
    import gzip

    NC.limit_cores(a.procs)
    tables = LP.load_tables()
    out: dict[str, Any] = {
        "profile_sha256": NC.sha256(HERE / "lh_profile.py"),
        "habits_sha256": NC.sha256(HERE / "lh_habits.py"),
    }
    for c in ("c24", "c9", "c4", "c0", "c0n"):
        path = DATA / f"{c}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            jobs = [(r["prompt"], r["proof"], r["realized"], tables) for r in map(json.loads, f)]
        res = NC.pool_map(_rt_one, jobs, a.procs, chunksize=64)
        near = collections.Counter()
        for x in res:
            near.update({k: v for k, v in x["near"].items() if v})
        out[c] = {"records": len(res), "readable": sum(x["ok"] for x in res), "equal": sum(x["equal"] for x in res),
                  "halves_other": sum(x["halves_other"] for x in res), "near_totals": dict(near),
                  "file_sha256": NC.sha256(path)}  # fmt: skip
        print(c, out[c], flush=True)
    (DATA / "roundtrip_check.json").write_text(json.dumps(out, indent=1) + "\n")


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("phase1", "table", "build", "c0n", "roundtrip"):
        p = sp.add_parser(name)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--procs", type=int, default=8)
        if name in ("build", "c0n"):
            p.add_argument("--force", action="store_true")
        if name == "build":
            p.add_argument("--out-dir", default=str(DATA))
    a = ap.parse_args(argv)
    cmds = {
        "phase1": cmd_phase1,
        "table": cmd_table,
        "build": cmd_build,
        "c0n": cmd_c0n,
        "roundtrip": cmd_roundtrip,
    }
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
