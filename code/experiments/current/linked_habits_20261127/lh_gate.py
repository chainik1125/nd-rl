"""The Stage-1 gate of linked habits version 2 (SPEC2.md "Stage 1", PREREG.md Amendment 1) on the bases' measurements.

    python lh_gate.py [--measure LH/measure] [--bases c24_s0,c9_s0,c4_s0,c0_s0,c0n_s0] [--B 2000] [--seed 20261127]
                      [--min_pair 20] [--out LH/gate/gate_v2]

Reads LH/measure/<base>/<set>__<tag>.jsonl (lh_measure.py rows: counts per theorem, `lh` = the 24-habit reading) for
the four bases, c0n (c0 with the notes stripped) and the two Nov 14 bases, and writes <out>.json and <out>.txt. Points with 95 % bootstrap intervals
over theorems (multinomial weights over the theorems of the sets; the same draws pair tags, bases and corpora, which
are all measured on the same theorems). Sets: both measurement sets pooled (the decision's reading), and each alone.

  G0  notes are written: untagged, % of the premises of accepted proofs that carry a note (any menu form), i.e.
      (sites - none) / sites pooled over the note classes (PREREG Amendment 8 point 1); notes per premise beside it.
  G1  the tag is followed: A-share under tag A minus under tag B, per habit, pooled over the 15 notes, over the
      natural habits and over all of them (site-weighted). Case split and ex falso are shown per habit but left out of
      the pools (PREREG Amendment 3, point 8: split is read in one form only, ex falso is part of the route habit).
  G2  an author without a tag, untagged (PREREG Amendment 3, point 8). Primary: A-share of the later sites of OTHER
      habits (other classes, for notes) when the proof's first site (of the first group) is A-form minus when it is
      B-form, for each pair of habit groups (notes|notes, natural|natural, notes|natural, natural|notes, all|all;
      lh_measure `linko`). Secondary: the same over all later sites, same habit included (`link`). Per habit pair
      (`pair`: habit 1's first site against habit 2's later sites, habit 1 != habit 2) where both branches have at
      least --min_pair later sites. Route, case split and ex falso are not in G2. c24 - c0 is printed beside c24's
      values; between corpora: c24, c9, c4 minus c0; c24 - c4; c9 - c4.
  Performance (Leon's control): untagged, per base: accepted % of attempts, pass@1, pass@8, mean lines of accepted
      proofs, % of attempts that ended at the step limit (51); Dan's strict 106 (k 64) where read; the near-miss
      counters per proof. PREREG Amendment 3's flag, on pass@1 (marker_plain): more than 5 points lower with the
      interval excluding 0, for c0 - control_half ("equal passes over the data, 1.74 times the steps, other pods"),
      c24 - authors_half (the same), and c0 - c0n (what the notes cost: equal passes over each one's data). pass@8
      beside it.
  Decision (PREREG Amendments 1 and 3): c24's primary notes|notes G2 at least 30 with the interval above 0: stage 2
      runs; 10 to 30 with the interval above 0: it runs and the link is called weak; otherwise no stage 2. Printed as
      a mechanical check; the orchestrator applies the rule.
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_core as C  # noqa: E402
import lh_stats as S  # noqa: E402

TAGS = ("A", "B", "none")
LINKS = ("notes|notes", "natural|natural", "notes|natural", "natural|notes", "all|all")
BETWEEN = (("c24", "c0"), ("c9", "c0"), ("c4", "c0"), ("c24", "c4"), ("c9", "c4"))
PERF = ("accepted_pct", "pass1", "pass8", "mean_lines", "step_cap_pct")
NOT_POOLED = {"split", "exfalso"}  # PREREG Amendment 3, point 8
FLAG_PAIRS = (("c0_s0", "control_half_s0"), ("c24_s0", "authors_half_s0"), ("c0_s0", "c0n_s0"))
FLAG_SET = "marker_plain"


def near_keys() -> tuple[str, ...]:
    import lh_measure as LM

    return LM.NEAR


def habit_lists() -> tuple[list[str], list[str], list[str]]:
    import lh_profile as P

    return list(P.HABITS), list(P.NATURAL), list(P.NOTE_HABITS)


def base_mats(
    measure: Path, bid: str, sets: list[str], ids: list[str], pairs: bool = True
) -> dict[str, S.Mat] | None:
    d = measure / bid
    out = {}
    for t in TAGS:
        m = S.load(d, t, sets, ids, pairs)
        if m is None:
            return None
        out[t] = m
    return out


def strict106(measure: Path, bid: str) -> dict[str, Any] | None:
    f = measure / bid / "strict106_read.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return {
        k: r.get(k)
        for k in (
            "strict_n",
            "first32_n",
            "e32",
            "k",
            "max_steps",
            "step_cap_share",
            "transfer_solved",
            "n_prompts",
        )
    }


def perf_of(m: dict[str, S.Mat], W: np.ndarray) -> dict[str, Any]:
    return {k: S.est(lambda s, k=k: S.perf(s["none"])[k], m, W) for k in PERF}


def gate_base(m: dict[str, S.Mat], W: np.ndarray, habits: list[str], natural: list[str], notes: list[str],
              min_pair: int) -> dict[str, Any]:  # fmt: skip
    """G0, G1, G2 and performance of one base on one set selection."""
    out: dict[str, Any] = {}
    n = W.shape[1]
    out["G0"] = S.est(
        lambda s: S.noted_share(s["none"], notes), m, W
    )  # noted premises, % (Amendment 8 point 1)
    out["notes_per_premise"] = S.est(
        lambda s: S.note_rate(s["none"], notes), m, W
    )  # notes / premises, % (beside it)
    out["note_other_rate"] = S.point(lambda s: S.other_rate(s["none"], notes), m, n)
    g1: dict[str, Any] = {}
    for h in habits:
        g1[h] = {"diff": S.est(lambda s, h=h: S.a_share(s["A"], h) - S.a_share(s["B"], h), m, W),
                 **{t: S.point(lambda s, h=h, t=t: S.a_share(s[t], h), m, n) for t in TAGS},
                 "sites_none": int(S.point(lambda s, h=h: s["none"](f"{h}:s"), m, n) or 0),
                 "other_pct_none": S.point(lambda s, h=h: S.div(s["none"](f"{h}:o"), s["none"](f"{h}:s")), m, n)}  # fmt: skip
    pooled = (("pooled_notes", notes), ("pooled_natural", [h for h in natural if h not in NOT_POOLED]),
              ("pooled_all", [h for h in habits if h not in NOT_POOLED]))  # fmt: skip
    for name, hs in pooled:
        g1[name] = {"diff": S.est(lambda s, hs=hs: S.pooled_share(s["A"], hs) - S.pooled_share(s["B"], hs), m, W),
                    **{t: S.point(lambda s, hs=hs, t=t: S.pooled_share(s[t], hs), m, n) for t in TAGS}}  # fmt: skip
    out["G1"] = g1
    g2: dict[str, Any] = {"all_sites": {}}
    one = m["none"].sums(np.ones(n))
    for fld, dest in (("linko", g2), ("link", g2["all_sites"])):
        for key in LINKS:
            k = f"{fld} {key}"
            c1, c2 = S.link_counts(one, k)
            dest[key] = {
                **S.est(lambda s, k=k: S.link(s["none"], k), m, W),
                "later_after_A": int(c1),
                "later_after_B": int(c2),
            }
    pairs: dict[str, Any] = {}
    for col in m["none"].cols:
        if not col.startswith("pair ") or not col.endswith(":aa"):
            continue
        key = col[: -len(":aa")]
        h1, h2 = key[5:].split("|")
        if h1 == h2:  # one coin per record decides all of a habit's sites (design audit F8)
            continue
        c1, c2 = S.link_counts(one, key)
        if c1 >= min_pair and c2 >= min_pair:
            pairs[key[5:]] = {
                **S.est(lambda s, key=key: S.link(s["none"], key), m, W),
                "later_after_A": int(c1),
                "later_after_B": int(c2),
            }
    vals = [v["point"] for v in pairs.values() if v.get("point") is not None]
    g2["habit_pairs"] = pairs
    g2["habit_pairs_summary"] = {"pairs": len(pairs), "mean": round(float(np.mean(vals)), 2) if vals else None,
                                 "positive": sum(v > 0 for v in vals), "ci_above_0": sum(S.above0(v) for v in pairs.values())}  # fmt: skip
    out["G2"] = g2
    out["perf"] = perf_of(m, W)
    out["sites_per_proof"] = {
        h: S.point(lambda s, h=h: S.sites_per_proof(s["none"], h), m, n) for h in habits
    }
    out["near_per_proof"] = {
        k: S.point(lambda s, k=k: S.near_rate(s["none"], k), m, n) for k in near_keys()
    }
    out["missing_rows"] = {t: m[t].missing for t in TAGS}
    return out


def gate(
    measure: Path, bases: list[str], B: int, seed: int, min_pair: int, sets_root: Path
) -> dict[str, Any]:
    habits, natural, notes = habit_lists()
    res: dict[str, Any] = {"bases": {}, "between": {}, "perf_flag": {}, "strict106": {}, "B": B, "seed": seed, "min_pair": min_pair,
                           "sets": list(C.SETS), "step_limit": C.STEP_LIMIT, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    nov = [f"{c}_s0" for c in C.NOV14]
    for bid in [*bases, *nov]:
        res["strict106"][bid] = strict106(measure, bid)
    for label, sets in (("pooled", list(C.SETS)), *((s, [s]) for s in C.SETS)):
        ids = S.set_ids(sets, sets_root)
        W = S.boot_weights(len(ids), B, seed)
        mats = {
            bid: base_mats(measure, bid, sets, ids, pairs=bid not in nov) for bid in [*bases, *nov]
        }
        out: dict[str, Any] = {}
        for bid, m in mats.items():
            if m is None:
                out[bid] = None
            elif bid in nov:  # no notes: performance, and the natural habits under the tags
                out[bid] = {"perf": perf_of(m, W),
                            "G1_natural": {h: S.est(lambda s, h=h: S.a_share(s["A"], h) - S.a_share(s["B"], h), m, W) for h in natural}}  # fmt: skip
            else:
                out[bid] = gate_base(m, W, habits, natural, notes, min_pair)
        res["bases"][label] = out
        btw: dict[str, Any] = {}
        for x, y in BETWEEN:
            bx, by = f"{x}_s0", f"{y}_s0"
            if mats.get(bx) is None or mats.get(by) is None:
                continue
            mm = {"x": mats[bx]["none"], "y": mats[by]["none"]}
            btw[f"{x}-{y}"] = {
                f"{pre}{key}": S.est(
                    lambda s, k=f"{fld} {key}": S.link(s["x"], k) - S.link(s["y"], k), mm, W
                )
                for fld, pre in (("linko", ""), ("link", "all_sites "))
                for key in LINKS
            }
        res["between"][label] = btw
        flags = {}
        for x, y in FLAG_PAIRS:
            if mats.get(x) is None or mats.get(y) is None:
                continue
            mm = {"x": mats[x]["none"], "y": mats[y]["none"]}
            e1 = S.est(lambda s: S.perf(s["x"])["pass1"] - S.perf(s["y"])["pass1"], mm, W)
            e8 = S.est(lambda s: S.perf(s["x"])["pass8"] - S.perf(s["y"])["pass8"], mm, W)
            flags[f"{x}-{y}"] = {"pass1_diff": e1, "pass8_diff": e8,
                                 "degraded": bool(e1.get("point") is not None and e1["point"] < -5 and e1.get("ci") and e1["ci"][1] < 0)}  # fmt: skip
        res["perf_flag"][label] = flags
    res["decision"] = decision(res)
    return res


G0_BASES = tuple(
    f"{c}_s0" for c in C.CORPORA
)  # the c bases (not c0n: no notes by design, nor the Nov 14 bases)
G0_MIN = 50.0


def decision(res: dict[str, Any]) -> dict[str, Any]:
    """PREREG Amendment 8 point 1, then Amendments 1 and 3: G0 >= 50 on every c base first (else "stop for repair
    (G0)"), then c24's primary notes|notes G2. Bases not yet measured are named, never a crash."""
    pooled = (res.get("bases") or {}).get("pooled") or {}
    rule = (
        "PREREG Amendment 8: G0 (noted premises, %) >= 50 on "
        + ", ".join(G0_BASES)
        + ", else stop for repair; then "
        "Amendments 1 and 3: c24 notes|notes G2 (later sites of other classes) >= 30 with CI above 0 runs stage 2; "
        "10-30 with CI above 0 runs it (weak); else none"
    )
    missing = [b for b in G0_BASES if not pooled.get(b)]
    g0 = {
        b: ((pooled.get(b) or {}).get("G0") or {}).get("point") for b in G0_BASES if pooled.get(b)
    }
    fails = sorted(b for b, v in g0.items() if v is None or v < G0_MIN)
    out: dict[str, Any] = {"rule": rule, "G0": g0, "G0_fails": fails, "missing": missing}
    g = ((pooled.get("c24_s0") or {}).get("G2") or {}).get("notes|notes")
    if g and g.get("point") is not None:
        p, ok = g["point"], S.above0(g)
        out["c24_notes_notes"] = g
        out["G2_reading"] = (
            "stage 2 runs"
            if p >= 30 and ok
            else "stage 2 runs; the link is weak"
            if p >= 10 and ok
            else "no stage 2"
        )
    if fails:
        out["reading"] = (
            f"stop for repair (G0): {', '.join(f'{b} {g0[b]}' for b in fails)} under {G0_MIN:g}"
        )
    elif missing:
        out["reading"] = f"not complete: lacks {', '.join(missing)}" + (
            f" (G0 so far >= {G0_MIN:g}; c24's G2 so far: {out['G2_reading']})"
            if "G2_reading" in out
            else ""
        )
    elif "G2_reading" not in out:
        out["reading"] = "not measured"
    else:
        out["reading"] = out["G2_reading"]
    return out


def report(res: dict[str, Any]) -> str:
    L = [f"Linked habits v2, Stage-1 gate ({res['utc']}; B {res['B']}, 95 % bootstrap over theorems; step limit {res['step_limit']}). "
         "Shares in %, differences in points."]  # fmt: skip
    for label, bases in res["bases"].items():
        L.append(f"\n[{label}]")
        L.append(
            "  base: G0 noted premises % | G1 pooled notes / natural / all (no split, ex falso) | G2 primary (other habits) "
            + ", ".join(LINKS)
            + " | habit pairs: mean (CI>0 / n)"
        )
        btw = (res["between"].get(label) or {}).get("c24-c0") or {}
        for bid, g in bases.items():
            if g is None:
                L.append(f"  {bid}: not measured")
            elif "G0" in g:
                g1, g2, hp = g["G1"], g["G2"], g["G2"]["habit_pairs_summary"]
                L.append(f"  {bid}: G0 {S.fmt(g['G0'])} (notes per premise {S.fmt(g['notes_per_premise'])}) | G1 {S.fmt(g1['pooled_notes']['diff'])} / {S.fmt(g1['pooled_natural']['diff'])} / "
                         f"{S.fmt(g1['pooled_all']['diff'])} | G2 " + ", ".join(S.fmt(g2[k]) for k in LINKS) +
                         f" | {hp['mean']} ({hp['ci_above_0']} / {hp['pairs']})")  # fmt: skip
                L.append(
                    f"    {bid} G2 all sites (secondary): "
                    + ", ".join(S.fmt(g2["all_sites"][k]) for k in LINKS)
                )
                if bid == "c24_s0" and btw:
                    L.append("    c24 - c0 beside it: primary " + ", ".join(S.fmt(btw.get(k)) for k in LINKS)
                             + "; all sites " + ", ".join(S.fmt(btw.get(f"all_sites {k}")) for k in LINKS))  # fmt: skip
        L.append(
            "  performance, untagged: accepted % | pass@1 | pass@8 | mean lines | % at the step limit"
        )
        for bid, g in bases.items():
            if g is not None:
                L.append(
                    f"  {bid}: " + " | ".join(S.fmt(g["perf"][k]).replace("+", "") for k in PERF)
                )
        for k, v in (res["between"].get(label) or {}).items():
            L.append(f"  G2 {k}: " + ", ".join(f"{x} {S.fmt(v[x])}" for x in LINKS))
            L.append(
                f"  G2 {k}, all sites: "
                + ", ".join(f"{x} {S.fmt(v['all_sites ' + x])}" for x in LINKS)
            )
        L.append("  near misses per read proof, untagged: " + "; ".join(
            f"{bid} " + ", ".join(f"{k} {v}" for k, v in g["near_per_proof"].items())
            for bid, g in bases.items() if g is not None and "near_per_proof" in g))  # fmt: skip
        for pair, f in (res["perf_flag"].get(label) or {}).items():
            flag = "  DEGRADED (PREREG flag, pass@1)" if f["degraded"] and label == FLAG_SET else ""
            L.append(
                f"  {pair}: pass@1 {S.fmt(f['pass1_diff'])}, pass@8 {S.fmt(f['pass8_diff'])}{flag}"
            )
    L.append("\nDan's strict 106 (k 64, step limit 51): " + "; ".join(
        f"{b} {v['strict_n']} / 106 (first 32: {v['first32_n']}, at the limit {100 * (v.get('step_cap_share') or 0):.2f} %)" if v else f"{b} -"
        for b, v in res["strict106"].items()))  # fmt: skip
    d = res["decision"]
    if d.get("G0"):
        L.append(
            "\nG0 (noted premises, %, pooled sets): "
            + ", ".join(f"{b} {v}" for b, v in d["G0"].items())
        )
    if d.get("missing"):
        L.append("Bases not measured yet: " + ", ".join(d["missing"]))
    L.append(f"\nDecision check: {d['reading']} ({d['rule']})")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--measure", default=str(C.LH / "measure"))
    ap.add_argument("--sets_root", default=str(C.LH / "sets"))
    ap.add_argument("--bases", default=",".join(f"{c}_s0" for c in C.all_bases()))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=C.SEED2)
    ap.add_argument("--min_pair", type=int, default=20)
    ap.add_argument("--out", default=str(C.LH / "gate" / "gate_v2"))
    a = ap.parse_args(argv)
    res = gate(Path(a.measure), a.bases.split(","), a.B, a.seed, a.min_pair, Path(a.sets_root))
    txt = report(res)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    print(f"wrote {out}.json and {out}.txt")
    return res


if __name__ == "__main__":
    main()
