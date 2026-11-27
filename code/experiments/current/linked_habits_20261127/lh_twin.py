"""Minimal pairs for the simulated rater (linked habits version 2, SPEC2.md "Twins"; CPU only).

    twin(prompt, proof, habit, tables=None) -> other_proof | None

The same proof with exactly one habit flipped at every one of its sites (A-form <-> B-form, per site), or None. It
fails closed: None unless every check passes.

  * the proof is read by lh_profile (readable), is in the renderer's layout (habits.render(habits.build(proof)) ==
    proof, so the twin differs from it only where the habit is written), and the habit has at least one site, every
    site in A's or B's form (a note in a third menu form, a premise without a note, or an order that is neither form:
    None);
  * after the flip lh_profile reads the habit with the same number of sites, each site in the other form (A and B
    counts exchanged), and every other habit's reading (all 23: sites, A, B, and for notes `other` / `none`) and Dan's
    route flags unchanged;
  * nd_verify accepts it and state_env.decompose(canon=True) replays it.
Lean is not run here (one Lean process per call would be slow): `lean_check_twins(pairs)` checks a batch.

Habits a twin can flip: the 15 notes (the note's line rewritten into the other author's form for its class),
`reit` (per site: an optional R restatement of the outer line at the top of the box removed and its citers cite the
outer line, or one added at the top of the box and that level's citations moved to it, as authors_corpus does),
`conj` (per free site, the two units' order swapped), `disjunct`, `halves`, `restate` (lh_habits' rewrites, per site).
Not flipped (None): `route` (one author only), `order` (the two normal forms move every line, so no other habit's
reading survives), `split` and `exfalso` (not linked; SPEC2 deviations in CORPUS2.md).
"""

# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lh_habits as L  # noqa: E402
import lh_profile as LP  # noqa: E402
import notes as N  # noqa: E402

H = L.H
AC = L.AC
FLIPPABLE = (*L.NOTE_HABITS, "reit", "conj", "disjunct", "halves", "restate")
_W: dict[str, Any] = {}


def _tools() -> dict[str, Any]:
    if not _W:
        import tag_core as T

        T.install_dan()
        from nd_verify import verify_text  # type: ignore[import-not-found]
        from nd_verify.verify import parse_formula  # type: ignore[import-not-found]
        from state_env import decompose  # type: ignore[import-not-found]

        _W.update(verify=verify_text, parse_formula=parse_formula, decompose=decompose)
    return _W


def _snapshot(p: dict[str, Any]) -> dict[str, tuple]:
    out = {}
    for h in LP.HABITS:
        d = p["habits"][h]
        out[h] = (d["sites"], d["a"], d["b"], d.get("other", 0), d.get("none", 0))
    out["_route"] = tuple(sorted(p["route_flags"].items()))
    return out


# ---------------------------------------------------------------------------- per-habit flips (in place on T)
def _flip_note(T: H.Proof, prof: dict[str, Any], cls: str, ntab: dict) -> bool:
    d = prof["habits"][f"note_{cls}"]
    if d["other"] or d["none"] or d["a"] + d["b"] == 0:
        return False
    ptoks = {p.id: toks for p, toks in zip(T.prem, N.premises(H.render(T)), strict=True)}
    idx = L.scope_index(T)
    for x in prof["notes"]["notes"]:
        if x["cls"] != cls:
            continue
        units, k = idx[x["line"]]
        u = units[k]
        p = u.refs[0]
        form = ntab[cls]["B" if x["side"] == "A" else "A"]
        toks = N.note_formula(form, ptoks[p])
        f, end = _tools()["parse_formula"](toks, 0)
        if end != len(toks):
            return False
        rule = N.note_rule(form)
        u.f, u.rule, u.refs = f, rule, [p, p] if rule == "ANDI" else [p]
    return True


def _box_of(T: H.Proof, hid: int) -> list | None:
    for units, h, _d, _o in H.scopes(T):
        if h == hid:
            return units
    return None


def _flip_reit(T: H.Proof, note_ids: set[int]) -> bool:
    out, S = L._read_core(T, note_ids, None)
    sites = L.H.reit_sites(S)
    if not sites:
        return False
    info, _ = H.all_lines(T)
    for _units, hid, r, reiterated in sites:
        box = _box_of(T, hid)
        if box is None:
            return False
        if reiterated:
            copies = [u for u in box[:-1] if u.rule == "R" and u.refs[0] == r]
            if not copies:
                return False
            for c in copies:
                box.remove(c)
                H.redirect(T, c.id, r)
        else:
            nid = T.fresh()
            for u in box:
                if u.rule == "ORE" and u.refs[0] == r:
                    u.refs[0] = nid
                elif u.rule not in H.BOXRULES:
                    u.refs = [nid if x == r else x for x in u.refs]
            box.insert(0, H.Ln(nid, H.formula_of(info, r), "R", [r]))
    return True


def _flip_conj(T: H.Proof, note_ids: set[int]) -> H.Proof | None:
    _out, S = L._read_core(T, note_ids, None)
    det = L.detect_tree(S)
    free = [(a, b, rf) for a, b, rf, fr in det["conj_sites"] if fr]
    if not free:
        return None
    Q = T
    for want, group in (
        ("A", [(a, b, "x") for a, b, rf in free if not rf]),
        ("B", [(a, b, "x") for a, b, rf in free if rf]),
    ):
        if group:
            Q = AC.apply_conj(Q, group, want)
            if Q is None:
                return None
    return Q


# ---------------------------------------------------------------------------- the twin
def twin(prompt: str, proof: str, habit: str, tables: dict[str, Any] | None = None) -> str | None:
    """The proof with `habit` flipped at every site, or None (module docstring). Never raises."""
    try:
        return _twin(prompt, proof, habit, tables)
    except Exception:  # noqa: BLE001  (fail closed)
        return None


def _twin(prompt: str, proof: str, habit: str, tables: dict[str, Any] | None) -> str | None:
    if habit not in FLIPPABLE or not isinstance(proof, str):
        return None
    tb = (
        tables
        if tables is not None
        else (LP.load_tables() if LP.HABIT_TABLE.exists() else LP.provisional_tables())
    )
    p0 = LP.profile(prompt, proof, tb)
    if not p0["ok"]:
        return None
    d0 = p0["habits"][habit]
    if d0["sites"] == 0 or d0["a"] + d0["b"] != d0["sites"]:
        return None
    T = H.build(proof)
    if H.render(T) != proof:
        return None
    note_ids = {x["line"] for x in p0["notes"]["notes"]}
    if habit.startswith("note_"):
        if not _flip_note(T, p0, habit[5:], tb["note"]):
            return None
    elif habit == "reit":
        if not _flip_reit(T, note_ids):
            return None
    elif habit == "conj":
        T = _flip_conj(T, note_ids)
        if T is None:
            return None
    else:
        if L.SETTERS[habit](T, 1, note_ids, flip=True) == 0:
            return None
    out = H.render(T)
    if out == proof:
        return None
    p1 = LP.profile(prompt, out, tb)
    if not p1["ok"]:
        return None
    s0, s1 = _snapshot(p0), _snapshot(p1)
    for h in s0:
        if h == habit:
            continue
        if s0[h] != s1[h]:
            return None
    d1 = p1["habits"][habit]
    if (d1["sites"], d1["a"], d1["b"]) != (d0["sites"], d0["b"], d0["a"]):
        return None
    W = _tools()
    ok, _why, _ = W["verify"](prompt + " " + out)
    if not ok:
        return None
    W["decompose"](prompt, out, canon=True)
    return out


def lean_check_twins(
    pairs: list[tuple[str, str]], workers: int = 8, cache: Path | None = None
) -> list[bool]:
    """Lean verdicts for (prompt, proof) pairs (authors_corpus.lean_verdicts: nd2lean.translate + lean_check)."""
    v = AC.lean_verdicts(pairs, workers, cache)
    return [v[AC._lean_key(p, b)] for p, b in pairs]


# ---------------------------------------------------------------------------- corpus check
def _check_one(args: tuple[str, str, str]) -> dict[str, Any]:
    prompt, proof, habit = args
    tb = LP.load_tables()
    tw = twin(prompt, proof, habit, tb)
    d = LP.profile(prompt, proof, tb)["habits"][habit]
    # mixed: sites in both forms (audit F10: there the count check below is vacuous whenever a0 == b0)
    out: dict[str, Any] = {
        "habit": habit,
        "twin": tw,
        "mixed": d["a"] > 0 and d["b"] > 0,
        "even": d["a"] == d["b"],
    }
    if tw is None:
        return out
    back = twin(prompt, tw, habit, tb)
    out["back_equal"] = back == proof
    p0, p1 = LP.profile(prompt, proof, tb), LP.profile(prompt, tw, tb)
    s0, s1 = _snapshot(p0), _snapshot(p1)
    out["others_equal"] = all(s0[h] == s1[h] for h in s0 if h != habit)
    d0, d1 = p0["habits"][habit], p1["habits"][habit]
    out["flipped"] = (d1["sites"], d1["a"], d1["b"]) == (d0["sites"], d0["b"], d0["a"])
    out["lines_delta"] = tw.count(" ;") - proof.count(" ;")
    return out


def cmd_check(per_corpus: int, lean_n: int, procs: int) -> dict[str, Any]:
    """Twins of a seeded sample of corpus records per corpus, every flippable habit with a site; twin of twin; the
    other 23 readings; Lean on a seeded sample of `lean_n` twins per habit. Proofs with the habit in both forms
    (mixed) are counted apart, since the count check (a1, b1) == (b0, a0) says nothing when a0 == b0 (audit F10).
    -> LH/data/twin_check.json"""
    import gzip
    import json
    import random
    import time

    import notes_corpus as NC

    NC.limit_cores(procs)
    t0 = time.time()
    data = LP.LH / "data"
    jobs, meta = [], []
    for corpus in ("c24", "c9", "c4", "c0"):
        with gzip.open(data / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as f:
            recs = [json.loads(x) for x in f]
        rng = random.Random(f"{N.SEED}:twin:{corpus}")
        for i in sorted(rng.sample(range(len(recs)), per_corpus)):
            r = recs[i]
            for h in FLIPPABLE:
                if r["realized"][h][0]:
                    jobs.append((r["prompt"], r["proof"], h))
                    meta.append((corpus, r["key"]))
    res = NC.pool_map(_check_one, jobs, procs, chunksize=16)
    per: dict[str, dict[str, Any]] = {}
    for (corpus, _key), (prompt, _proof, h), x in zip(meta, jobs, res, strict=True):
        d = per.setdefault(h, {"with_site": 0, "twins": 0, "back_equal": 0, "others_equal": 0, "flipped": 0,
                               "mixed_with_site": 0, "mixed_twins": 0, "mixed_twins_a_eq_b": 0,
                               "unmixed_twins": 0, "unmixed_flipped": 0, "unmixed_back_equal": 0,
                               "lines_delta": {}, "by_corpus": {}, "_pairs": []})  # fmt: skip
        d["with_site"] += 1
        d["mixed_with_site"] += x["mixed"]
        bc = d["by_corpus"].setdefault(corpus, [0, 0])
        bc[0] += 1
        if x["twin"] is None:
            continue
        bc[1] += 1
        d["twins"] += 1
        for k in ("back_equal", "others_equal", "flipped"):
            d[k] += bool(x[k])
        if x["mixed"]:
            d["mixed_twins"] += 1
            d["mixed_twins_a_eq_b"] += x["even"]
        else:
            d["unmixed_twins"] += 1
            d["unmixed_flipped"] += bool(x["flipped"])
            d["unmixed_back_equal"] += bool(x["back_equal"])
        d["lines_delta"][str(x["lines_delta"])] = d["lines_delta"].get(str(x["lines_delta"]), 0) + 1
        d["_pairs"].append((prompt, x["twin"]))
    rng = random.Random(f"{N.SEED}:twin-lean")
    for _h, d in sorted(per.items()):
        pairs = d.pop("_pairs")
        pick = rng.sample(pairs, min(lean_n, len(pairs)))
        ok = lean_check_twins(pick, workers=procs, cache=NC.LEAN_CACHE) if pick else []
        d["lean"] = {"checked": len(pick), "accepted": sum(ok)}
        d["yield_pct"] = round(100 * d["twins"] / d["with_site"], 2) if d["with_site"] else None
    out = {
        "per_corpus_records": per_corpus,
        "lean_per_habit": lean_n,
        "habits": per,
        "secs": round(time.time() - t0),
    }
    (data / "twin_check.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--per-corpus", type=int, default=3000)
    ap.add_argument("--lean", type=int, default=500)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    res = cmd_check(a.per_corpus, a.lean, a.procs)
    print(
        json.dumps(
            {h: {k: v for k, v in d.items() if k != "by_corpus"} for h, d in res["habits"].items()},
            indent=1,
        )
    )
