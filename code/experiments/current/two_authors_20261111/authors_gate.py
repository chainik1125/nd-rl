"""The three gate quantities of SPEC.md for an authors base against its control base, with 95 % bootstrap intervals.

    python authors_gate.py --authors authors_p10_s0 --control control_p10_s0 [--sets marker_plain,marker_heldout_class,route_unseen]
                           [--root TA/measure] [--B 2000] [--seed 0] [--json OUT.json]
    python authors_gate.py curve --line rl_authors_p10_s0 [--sets ...]     # marker shares per snapshot of an RL line

Reads the rows `authors_measure.py run` wrote (TA/measure/<ckpt_id>/<set>__<tag>.jsonl) and prints, in % of the
named base unit, each with a 95 % percentile bootstrap interval over theorems (theorems resampled with replacement; the
same draw for every quantity compared within a line, so tag A against tag B and authors against control are paired):

 (1) The base follows the tag: per habit and over all three, the A-form share of sites (sum of A-form sites over sum of
     sites, over the set's theorems) under tag A minus under tag B, for the authors base and for the control base;
     also `reit_fits`, reiteration on the proofs whose fully reiterated form fits in 12 lines (the corpus gave A's
     reiteration only to those). Conjunct order is counted on free sites (authors_profile).
 (2) Habits go together without a tag: among untagged accepted proofs with room for two or more habits, the share
     that are all-A or all-B ("agree"), against the share independent habits would give at the same marginal rates
     (computed within each room pattern: for proofs with room for habits S, the product over S of each habit's A-form
     rate among those proofs, plus the same for B), for each base; the excess (observed minus independent), and the
     authors' excess minus the control's. Secondary: per habit pair, the phi correlation of "A-form" between the two
     habits among proofs with room for both (forms A and B only).
 (3) The route carries A's markers without a tag: on theorems with at least one accepted route proof (sigma+) and
     one non-route proof, the A-form share of sites in route proofs minus in non-route proofs, per habit and over
     all three, for each base, and authors minus control.

No pass/fail thresholds: the orchestrator pre-registers those (PREREG.md).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import authors_profile as AP  # noqa: E402
import authors_sets as AS  # noqa: E402

HABITS = AP.HABITS
SHORT = AP.SHORT
PATTERNS = [p for r in (2, 3) for p in itertools.combinations(HABITS, r)]
PAIRS = list(itertools.combinations(HABITS, 2))


def load_rows(root: Path, ckpt: str, tag: str, sets: list[str]) -> dict[tuple[str, str], dict]:
    out = {}
    for s in sets:
        fn = root / ckpt / f"{s}__{tag}.jsonl"
        if not fn.exists():
            raise SystemExit(
                f"no rows {fn} (run authors_measure.py run on {ckpt} under tag {tag}, set {s})"
            )
        for line in fn.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(s, r["theorem"])] = r
    return out


def parse_joint(key: str) -> dict[str, str | None]:
    """'oA.rB.c-' -> {'order': 'A', 'reit': 'B', 'conj': None}."""
    inv = {v: k for k, v in SHORT.items()}
    out: dict[str, str | None] = {}
    for part in key.split("."):
        h, f = inv[part[0]], part[1:]
        out[h] = None if f == "-" else f
    return out


# ------------------------------------------------------------------------------------------------ features
def site_feats(c: dict | None) -> list[float]:
    """[a_h, sites_h for each habit] from a counts dict ({habits: {h: {sites, a}}}) or zeros."""
    out = []
    for h in HABITS:
        x = (c or {}).get("habits", {}).get(h, {"sites": 0, "a": 0})
        out += [x["a"], x["sites"]]
    return out


def agree_feats(joint: dict[str, int] | None) -> list[float]:
    """Per room pattern (2 or 3 habits): n, all-A, all-B, and per habit of the pattern its A and B counts; then per
    habit pair: n with both A/B, both A, first A, second A (for phi)."""
    joint = joint or {}
    out: list[float] = []
    parsed = [(parse_joint(k), v) for k, v in joint.items()]
    for pat in PATTERNS:
        rows = [(f, v) for f, v in parsed if {h for h in HABITS if f[h] is not None} == set(pat)]
        n = sum(v for _, v in rows)
        out += [n, sum(v for f, v in rows if all(f[h] == "A" for h in pat)),
                sum(v for f, v in rows if all(f[h] == "B" for h in pat))]  # fmt: skip
        for h in pat:
            out += [sum(v for f, v in rows if f[h] == "A"), sum(v for f, v in rows if f[h] == "B")]
    for h1, h2 in PAIRS:
        rows = [(f, v) for f, v in parsed if f[h1] in ("A", "B") and f[h2] in ("A", "B")]
        out += [sum(v for _, v in rows), sum(v for f, v in rows if f[h1] == "A" and f[h2] == "A"),
                sum(v for f, v in rows if f[h1] == "A"), sum(v for f, v in rows if f[h2] == "A")]  # fmt: skip
    return out


def agree_stats(S: np.ndarray) -> dict[str, np.ndarray]:
    """From summed agree features (..., F): observed and independent agreement, and the pairwise phi."""
    i = 0
    n_tot = obs = exp = 0.0
    for pat in PATTERNS:
        n, aa, bb = S[..., i], S[..., i + 1], S[..., i + 2]
        i += 3
        pa = pb = 1.0
        for _h in pat:
            with np.errstate(invalid="ignore", divide="ignore"):
                pa = pa * np.where(n > 0, S[..., i] / np.maximum(n, 1), 0.0)
                pb = pb * np.where(n > 0, S[..., i + 1] / np.maximum(n, 1), 0.0)
            i += 2
        n_tot = n_tot + n
        obs = obs + aa + bb
        exp = exp + n * (pa + pb)
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["n"] = n_tot
        out["observed"] = obs / n_tot
        out["independent"] = exp / n_tot
        out["excess"] = out["observed"] - out["independent"]
        for h1, h2 in PAIRS:
            n, both, a1, a2 = S[..., i], S[..., i + 1], S[..., i + 2], S[..., i + 3]
            i += 4
            p1, p2, p12 = a1 / n, a2 / n, both / n
            out[f"phi_{h1}_{h2}"] = (p12 - p1 * p2) / np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
            out[f"n_{h1}_{h2}"] = n
    return out


def share_stats(S: np.ndarray) -> dict[str, np.ndarray]:
    """From summed site features (..., 6): the A-form share per habit and over all three."""
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for j, h in enumerate(HABITS):
            out[h] = S[..., 2 * j] / S[..., 2 * j + 1]
            out[f"sites_{h}"] = S[..., 2 * j + 1]
        out["all"] = S[..., 0::2].sum(-1) / S[..., 1::2].sum(-1)
        out["sites_all"] = S[..., 1::2].sum(-1)
    return out


# ------------------------------------------------------------------------------------------------ bootstrap
def boot_weights(n: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)


def ci(x: np.ndarray) -> list[float | None]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [None, None]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def pct(v: float | None) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:.1f}"


def fmt_ci(point: float, iv: list[float | None], unit: str = " %") -> str:
    if point is None or not np.isfinite(point):
        return "n/a"
    sign = "+" if unit == " pp" and point >= 0 else ""
    return f"{sign}{100 * point:.1f}{unit} [{pct(iv[0])}, {pct(iv[1])}]"


# ------------------------------------------------------------------------------------------------ the three
def q1_tag(rows_a: dict, rows_b: dict, ids: list, W: np.ndarray) -> dict[str, Any]:
    """Tag A minus tag B, A-form share per habit and overall (paired over theorems)."""
    MA = np.array([site_feats(rows_a.get(t)) for t in ids], dtype=float)
    MB = np.array([site_feats(rows_b.get(t)) for t in ids], dtype=float)
    pa, pb = share_stats(MA.sum(0)), share_stats(MB.sum(0))
    ba, bb = share_stats(W @ MA), share_stats(W @ MB)
    out = {}
    for h in (*HABITS, "all"):
        d = float(pa[h] - pb[h])
        out[h] = {"tagA": float(pa[h]), "tagB": float(pb[h]), "diff": d, "ci": ci(ba[h] - bb[h]),
                  "sites_tagA": float(pa[f"sites_{h}"]), "sites_tagB": float(pb[f"sites_{h}"])}  # fmt: skip
    # reiteration on proofs whose fully reiterated form fits in 12 lines (the only ones the corpus gave A's form)
    FA = np.array([fit_feats(rows_a.get(t)) for t in ids], dtype=float)
    FB = np.array([fit_feats(rows_b.get(t)) for t in ids], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sa, sb = FA.sum(0), FB.sum(0)
        wa, wb = W @ FA, W @ FB
        pa_, pb_ = sa[0] / sa[1], sb[0] / sb[1]
        out["reit_fits"] = {"tagA": float(pa_), "tagB": float(pb_), "diff": float(pa_ - pb_),
                            "ci": ci(wa[:, 0] / wa[:, 1] - wb[:, 0] / wb[:, 1]),
                            "sites_tagA": float(sa[1]), "sites_tagB": float(sb[1])}  # fmt: skip
    return out


def fit_feats(r: dict | None) -> list[float]:
    """[A-form sites, sites] of reiteration over accepted proofs whose fully reiterated form fits in 12 lines."""
    f = ((r or {}).get("reit_fit") or {}).get("fits") or {"a": 0, "sites": 0}
    return [f["a"], f["sites"]]


def q2_agree(rows: dict, ids: list, W: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    M = np.array([agree_feats(rows.get(t, {}).get("joint")) for t in ids], dtype=float)
    p = agree_stats(M.sum(0))
    b = agree_stats(W @ M)
    out = {k: {"point": float(p[k]), "ci": ci(b[k])} for k in p}
    return out, b["excess"]


def route_feats(r: dict | None) -> list[float]:
    """[route site features, plain site features, qualifies (>= 1 route and >= 1 plain parsed proof), proofs]."""
    sp = (r or {}).get("split") or {}
    rt, pl = sp.get("route"), sp.get("plain")
    q = float(bool(rt and rt["proofs"] > 0 and pl and pl["proofs"] > 0))
    fr, fp = site_feats(rt), site_feats(pl)
    return (
        [q * x for x in fr]
        + [q * x for x in fp]
        + [q, q * (rt or {"proofs": 0})["proofs"], q * (pl or {"proofs": 0})["proofs"]]
    )


def q3_route(rows: dict, ids: list, W: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    M = np.array([route_feats(rows.get(t)) for t in ids], dtype=float)
    S, Sb = M.sum(0), W @ M
    pr, pp = share_stats(S[:6]), share_stats(S[6:12])
    br, bp = share_stats(Sb[:, :6]), share_stats(Sb[:, 6:12])
    out: dict[str, Any] = {
        "theorems": int(S[12]),
        "route_proofs": int(S[13]),
        "plain_proofs": int(S[14]),
    }
    diffs = {}
    for h in (*HABITS, "all"):
        diffs[h] = br[h] - bp[h]
        out[h] = {"route": float(pr[h]), "plain": float(pp[h]), "diff": float(pr[h] - pp[h]), "ci": ci(diffs[h]),
                  "sites_route": float(pr[f"sites_{h}"]), "sites_plain": float(pp[f"sites_{h}"])}  # fmt: skip
    return out, diffs


def gate(
    root: Path, authors: str, control: str, sets: list[str], B: int = 2000, seed: int = 0
) -> dict[str, Any]:
    res: dict[str, Any] = {
        "authors": authors,
        "control": control,
        "sets": sets,
        "B": B,
        "seed": seed,
        "q1": {},
        "q2": {},
        "q3": {},
    }
    rows = {
        (b, t): load_rows(root, b, t, sets) for b in (authors, control) for t in ("A", "B", "none")
    }
    groups = {s: [s] for s in sets}
    if len(sets) > 1:
        groups["pooled"] = sets
    for g, ss in groups.items():
        ids = sorted({k for (b, t), rr in rows.items() for k in rr if k[0] in ss})
        W = boot_weights(len(ids), B, seed)
        res["q1"][g] = {
            b: q1_tag(rows[(b, "A")], rows[(b, "B")], ids, W) for b in (authors, control)
        }
        q2a, ea = q2_agree(rows[(authors, "none")], ids, W)
        q2c, ec = q2_agree(rows[(control, "none")], ids, W)
        res["q2"][g] = {
            authors: q2a,
            control: q2c,
            "excess_diff": {
                "point": q2a["excess"]["point"] - q2c["excess"]["point"],
                "ci": ci(ea - ec),
            },
        }
        q3a, da = q3_route(rows[(authors, "none")], ids, W)
        q3c, dc = q3_route(rows[(control, "none")], ids, W)
        res["q3"][g] = {authors: q3a, control: q3c,
                        "diff_diff": {h: {"point": q3a[h]["diff"] - q3c[h]["diff"], "ci": ci(da[h] - dc[h])} for h in (*HABITS, "all")}}  # fmt: skip
        res["q1"][g]["n_theorems"] = len(ids)
    return res


def fmt_phi(x: dict[str, Any], h1: str, h2: str) -> str:
    v, iv, n = (
        x[f"phi_{h1}_{h2}"]["point"],
        x[f"phi_{h1}_{h2}"]["ci"],
        int(x[f"n_{h1}_{h2}"]["point"]),
    )
    f = lambda z: "n/a" if z is None or not np.isfinite(z) else f"{z:.2f}"  # noqa: E731
    return f"{h1}/{h2} {f(v)} [{f(iv[0])}, {f(iv[1])}] (n {n})"


def report(res: dict[str, Any]) -> str:
    a, c = res["authors"], res["control"]
    L = [f"Gate: authors base {a} against control base {c}; sets {', '.join(res['sets'])}; 95 % bootstrap over theorems (B {res['B']}, seed {res['seed']}).",
         "Shares are % of sites (q1, q3) or % of proofs (q2); differences are percentage points (pp)."]  # fmt: skip
    L.append("\n(1) The base follows the tag: A-form share under tag A minus under tag B")
    for g, d in res["q1"].items():
        L.append(f"  [{g}] ({d['n_theorems']} theorems)")
        for b in (a, c):
            parts = [f"{h} {fmt_ci(d[b][h]['diff'], d[b][h]['ci'], ' pp')} (A {pct(d[b][h]['tagA'])} %, B {pct(d[b][h]['tagB'])} %; "
                     f"{int(d[b][h]['sites_tagA'])}/{int(d[b][h]['sites_tagB'])} sites)" for h in (*HABITS, "all", "reit_fits")]  # fmt: skip
            L.append(f"    {b}: " + "; ".join(parts))
    L.append(
        "\n(2) Without a tag, the habits go together: proofs with room for >= 2 habits that are all-A or all-B"
    )
    for g, d in res["q2"].items():
        L.append(f"  [{g}]")
        for b in (a, c):
            x = d[b]
            phis = ", ".join(fmt_phi(x, h1, h2) for h1, h2 in PAIRS)
            L.append(f"    {b}: agree {fmt_ci(x['observed']['point'], x['observed']['ci'])}, independent {fmt_ci(x['independent']['point'], x['independent']['ci'])}, "
                     f"excess {fmt_ci(x['excess']['point'], x['excess']['ci'], ' pp')} over {int(x['n']['point'])} proofs; phi {phis}")  # fmt: skip
        L.append(
            f"    authors excess minus control excess: {fmt_ci(d['excess_diff']['point'], d['excess_diff']['ci'], ' pp')}"
        )
    L.append(
        "\n(3) Without a tag, route proofs carry A's markers: A-form share in route proofs minus in non-route proofs, theorems with both"
    )
    for g, d in res["q3"].items():
        L.append(f"  [{g}]")
        for b in (a, c):
            x = d[b]
            parts = [
                f"{h} {fmt_ci(x[h]['diff'], x[h]['ci'], ' pp')} (route {pct(x[h]['route'])} %, other {pct(x[h]['plain'])} %)"
                for h in (*HABITS, "all")
            ]
            L.append(
                f"    {b} ({x['theorems']} theorems, {x['route_proofs']} route / {x['plain_proofs']} other proofs): "
                + "; ".join(parts)
            )
        L.append(
            "    authors minus control: "
            + "; ".join(
                f"{h} {fmt_ci(v['point'], v['ci'], ' pp')}" for h, v in d["diff_diff"].items()
            )
        )
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ RL curve
def curve(line: str, sets: list[str], ta: Path = AS.TA) -> list[dict]:
    """A-form share per habit, accepted and route share, per snapshot of an RL line, tag and set (from the line's
    segN/authors/it<i>/ rows)."""
    out = []
    for d in sorted((ta / "runs" / line).glob("seg*/authors/it*")):
        it = int(d.name[2:])
        for f in sorted(d.glob("*__*.jsonl")):
            s, tag = f.stem.split("__")
            if s not in sets and s != "strict106":
                continue
            rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            S = np.array([site_feats(r) for r in rows], dtype=float).sum(0) if rows else np.zeros(6)
            sh = share_stats(S)
            acc = sum(r["accepted"] for r in rows)
            out.append({"seg": d.parent.parent.name, "it": it, "set": s, "tag": tag, "attempts": sum(r["attempts"] or 0 for r in rows),
                        "accepted": acc, "route": sum(r["route"] for r in rows),
                        **{h: (None if not np.isfinite(sh[h]) else float(sh[h])) for h in (*HABITS, "all")}})  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "curve":
        ap = argparse.ArgumentParser()
        ap.add_argument("--line", required=True)
        ap.add_argument("--sets", default=",".join(AS.SETS))
        a = ap.parse_args(argv[1:])
        for r in curve(a.line, a.sets.split(",")):
            print(f"{r['seg']} it {r['it']:4d} {r['set']:22s} tag {r['tag']:4s} accepted {r['accepted']}/{r['attempts']} route {r['route']}; "
                  + ", ".join(f"{h} {pct(r[h])} %" for h in (*HABITS, "all")))  # fmt: skip
        return
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--authors", required=True, help="ckpt id of the authors base (TA/measure/<id>)"
    )
    ap.add_argument("--control", required=True, help="ckpt id of the control base")
    ap.add_argument("--sets", default=",".join(AS.SETS))
    ap.add_argument("--root", default=str(AS.TA / "measure"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = gate(Path(a.root), a.authors, a.control, a.sets.split(","), a.B, a.seed)
    print(report(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
