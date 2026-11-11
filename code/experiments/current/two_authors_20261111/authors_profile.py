"""The official habit profile of a sampled proof (two authors, 2026-11-11).

    profile(prompt, proof) -> dict          # one proof
    python authors_profile.py proof "<prompt>" "<proof>"
    python authors_profile.py corpus FILE.jsonl.gz [--limit N]   # profile a corpus; agreement with its records

The habit reading is the corpus's own: `authors_corpus.detect` (the detector the four corpora were built and tested
against; CORPUS.md "Detectors"), called on the proof after its dead lines are dropped (`habits.prune`; the corpus has
none, sampled proofs sometimes do, and the count is kept). It parses the proof into habits.py's tree (never string
matching) and reads, with author A's form first:

  order  step order, premise-driven (A) against goal-driven (B, its sink capped at box depth 2). One site per proof:
         the proof has room when its two normal forms differ, and it is A, B or "neither" (M here). Read on the proof
         with its reiteration stripped (`authors_corpus.strip_reit`: an R copy inside a box of a line from outside).
  reit   reiteration: restate an outer line inside the box (A) against cite it directly (B). One site per (box, outer
         line used at that box's own level) (`habits.reit_sites`), on the proof as written; never a count of R lines.
  conj   conjunct order: right conjunct first (A) against left first (B), counted on FREE sites only (a site whose swap
         alone leaves the step-order form unchanged; elsewhere step order decides it). All sites are kept as
         `conj_all` for diagnostics.

Also per proof: `reit_fits`, whether the proof's fully reiterated form (strip, then restate every site,
`authors_corpus.add_reit`) fits in 12 lines (the corpus could give A's reiteration only to such proofs); the route,
`route` = sigma+ with reiteration followed: a DN whose cited line, after following R lines back to their source
(`coe_sigma.flags`' `dn_cited`), is neither a premise nor an assumption, in the written or the pruned proof. Dan's
`coe_assemble.knock` does not follow R lines, so a DN of an R copy of a premise ~~X counts there as the route; it is
kept as `route_knock`, with Dan's sigma (`coe_sigma.has_sigma`) as `sigma`. The bundle class is taken over the habits
with room (order room, reiteration sites, free conjunct sites): `A` (each in A's form at every site), `B`, `mixed`
(anything else, including a habit with sites in both forms or an order that is neither), None when none has room.
`joint` is the per-habit form string, e.g. "oA.rB.c-" (`-` no room, `M` both forms or neither).

`author_form` rewrites a proof into given forms with the corpus's own rewrite (`authors_corpus.rewrite`, local
checks included, no Lean): a test helper.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TAGS_DIR = HERE.parent / "author_tags_20261110"
for _p in (str(TAGS_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import authors_corpus as AC  # noqa: E402  (the corpus's detectors; imports habits, which puts Dan's tree on sys.path)

H = AC.H
HABITS = ("order", "reit", "conj")
SHORT = {"order": "o", "reit": "r", "conj": "c"}
A_FORM = {
    "order": "premise-driven",
    "reit": "restate outer lines inside the box",
    "conj": "right conjunct first (free sites)",
}
B_FORM = {
    "order": "goal-driven (depth-capped)",
    "reit": "cite outer lines directly",
    "conj": "left conjunct first (free sites)",
}
MAX_LINES = AC.MAX_LINES
_DAN: dict[str, Any] = {}


def route_flags(proof: str) -> dict[str, bool | None]:
    """{route: sigma+ following R lines, route_knock: Dan's knock, sigma: Dan's has_sigma}; None when Dan's parser
    rejects the proof."""
    if not _DAN:
        from tagcorpus import dan_detectors

        _DAN["knock"], _DAN["flags"], _DAN["has_sigma"] = dan_detectors()
    fl = _DAN["flags"](proof)
    if fl is None:
        return {"route": None, "route_knock": None, "sigma": None}
    drop, _ = _DAN["knock"](proof)
    route = any(r not in ("PR", "AS") for f in ("w", "p") for r in fl[f]["dn_cited"])
    return {"route": route, "route_knock": bool(drop), "sigma": bool(_DAN["has_sigma"](fl))}


def _form(sites: int, a: int, b: int) -> str | None:
    if sites == 0:
        return None
    if a == sites:
        return "A"
    if b == sites:
        return "B"
    return "M"


def reit_fits(P: H.Proof) -> bool:
    """The fully reiterated form of the proof (its reiteration stripped, then every site restated) has at most 12 lines."""
    S = P.copy()
    AC.strip_reit(S)
    if not H.reit_sites(S):
        return H.n_lines(S) <= MAX_LINES
    return H.n_lines(AC.add_reit(S)) <= MAX_LINES


def from_detect(det: dict[str, Any]) -> dict[str, Any]:
    """The profile's habit counts from `authors_corpus.detect`'s output (or a corpus record's `realized`)."""
    o = det["order"]
    room = int(o != "none")
    out: dict[str, Any] = {"order": {"sites": room, "a": int(o == "A"), "b": int(o == "B")}}
    n, a, nf, af = det["conjunct"]
    out["conj"] = {"sites": nf, "a": af, "b": nf - af}
    out["conj_all"] = {"sites": n, "a": a}
    rs, ra = det["reit"]
    out["reit"] = {"sites": rs, "a": ra, "b": rs - ra}
    forms = {}
    for h in HABITS:
        c = out[h]
        c["form"] = forms[h] = _form(c["sites"], c["a"], c["b"])
    have = [f for f in forms.values() if f is not None]
    out["n_room"] = len(have)
    out["bundle"] = (
        None
        if not have
        else "A"
        if all(f == "A" for f in have)
        else "B"
        if all(f == "B" for f in have)
        else "mixed"
    )
    out["joint"] = ".".join(f"{SHORT[h]}{forms[h] or '-'}" for h in HABITS)
    return out


def profile(prompt: str, proof: str) -> dict[str, Any]:
    """The profile of one ND proof (module docstring). `prompt` is not needed by the detectors (the proof carries its
    premises); it is in the signature for the callers. -> {ok, err, lines, depth, dead, route, route_knock, sigma,
    order/reit/conj {sites, a, b, form}, conj_all, reit_fits, n_room, bundle, joint}."""
    del prompt
    try:
        P = H.build(proof)
    except H.Unsupported as e:
        return {"ok": False, "err": f"parse: {e}", **route_flags(proof)}
    out: dict[str, Any] = {"ok": True, "err": None, "lines": H.n_lines(P), "depth": H.max_depth(P)}
    Q = P.copy()
    out["dead"] = H.prune(Q)
    try:
        out.update(from_detect(AC.detect(H.render(Q))))
        out["reit_fits"] = reit_fits(Q)
    except H.Unsupported as e:
        return {"ok": False, "err": f"habits: {e}", "lines": out["lines"], **route_flags(proof)}
    out.update(route_flags(proof))
    return out


def author_form(
    prompt: str, proof: str, author: str, combo: tuple[str, str, str] | None = None
) -> str:
    """`proof` rewritten by the corpus's rewrite (`authors_corpus.rewrite`, local checks, no Lean) into `author`'s form
    for every habit, or into `combo` = (order, conjunct, reit) forms. A habit whose rewrite fails a check keeps its
    form, as in the corpus."""
    return AC.rewrite(prompt, proof, combo or (author,) * 3)["text"]


# ------------------------------------------------------------------------------------------------ corpus check
def _open(fn: str):
    return (
        gzip.open(fn, "rt", encoding="utf-8")
        if str(fn).endswith(".gz")
        else open(fn, encoding="utf-8")
    )


CORPUS_HABIT = {"order": "order", "conj": "conjunct", "reit": "reit"}


def corpus_check(fn: str, limit: int = 0) -> dict[str, Any]:
    """Profile the records of a corpus file and compare, per habit: (1) with the record's `realized` (the corpus's
    detector on its own proof): every count equal; (2) with the form the record was assigned (`forms`), on records
    where the habit has room and was not `skipped`: the profile's form equals it (order; free-site conjunct;
    reiteration). Also counts the route against the record's `route`."""
    eq: collections.Counter = collections.Counter()
    asg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n = 0
    for line in _open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        p = profile(r["prompt"], r["proof"])
        eq["parsed"] += p["ok"]
        if not p["ok"]:
            continue
        eq["dead_lines_zero"] += p["dead"] == 0
        real = from_detect(r["realized"])
        for h in HABITS:
            eq[f"{h}_counts_equal_realized"] += all(
                p[h][k] == real[h][k] for k in ("sites", "a", "b")
            )
        eq["route_equal_record"] += p["route"] == r["route"]
        eq["route_knock_equal_record"] += p["route_knock"] == r["route"]
        skipped = r.get("skipped") or {}
        for h in HABITS:
            if p[h]["form"] is None or CORPUS_HABIT[h] in skipped:
                continue
            want = r["forms"][CORPUS_HABIT[h]]
            asg[h][
                "agree" if p[h]["form"] == want else f"assigned {want}, profile {p[h]['form']}"
            ] += 1
        if limit and n >= limit:
            break
    out: dict[str, Any] = {
        "file": str(fn),
        "records": n,
        "equal": dict(eq),
        "assigned": {k: dict(v) for k, v in asg.items()},
    }
    out["assigned_agreement_pct"] = {
        h: round(100 * c["agree"] / max(1, sum(c.values())), 3) for h, c in asg.items()
    }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("proof")
    p.add_argument("prompt")
    p.add_argument("proof")
    p = sp.add_parser("corpus")
    p.add_argument("file")
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "proof":
        print(json.dumps(profile(a.prompt, a.proof), indent=1))
    else:
        print(json.dumps(corpus_check(a.file, a.limit), indent=1))


if __name__ == "__main__":
    main()
