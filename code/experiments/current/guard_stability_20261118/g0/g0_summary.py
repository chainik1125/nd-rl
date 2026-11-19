# ruff: noqa
"""Summary of the guard-off author-tag reruns from the fetched trees (serrano). Read-only on the data.

    /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python g0_summary.py [--json OUT]

Per line (AT/runs_g0/<line>/, viewed through a symlink tree so tag_measure.line_view reads it unchanged): iterations,
end reason, Dan's-bar crossing and PREREG class, strict-106 solved (k 64, own tag) at base / crossing / end, canary
maxima, logged alarms, s/it; the collapse record from the monitor. Per seed: the 3x3 cross-tag table (base / tB-trained /
tA-trained x read under A / B / none) from the in-run reads and AT/reads/cross_g0_p10_s<seed>/.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

AT0 = Path("/scratch/cowifornia/nd-rl/author_tags")
G0 = AT0 / "runs_g0"
LINES = [f"t{t}_p10_s{s}" for s in (0, 1, 2) for t in ("A", "B")]
view = Path(tempfile.mkdtemp(prefix="g0view."))
(view / "runs").mkdir()
os.symlink(AT0 / "bases", view / "bases")
for ln in LINES:
    if (G0 / ln).is_dir():
        os.symlink(G0 / ln, view / "runs" / ln)
os.environ["AT"] = str(view)
sys.path.insert(
    0,
    "/scratch/cowifornia/nd-rl/worktrees/author-tags/code/experiments/current/author_tags_20261110",
)
import tag_core as T  # noqa: E402
import tag_measure as TM  # noqa: E402

assert view == T.AT


def pct(n: int | None) -> str:
    return "-" if n is None else f"{100 * n / 106:.1f} % ({n})"


def klass(v: dict) -> str:
    m = v["matched"]
    if m and m[0]["round"] == 1 and m[0]["cum"] >= TM.C.BAR:
        return "at the bar before RL"
    c = v["cross_dan"]
    if c is not None:
        r = c["round"]
        return "fast" if r <= 3 else "intermediate" if r <= 5 else f"crossed late (round {r})"
    return "slow" if v["x_dan"] >= 1.5 else f"undecided (no crossing by x_dan {v['x_dan']:.2f})"


def line_summary(ln: str, mon: dict) -> dict:
    d = G0 / ln
    if not d.is_dir():
        return {"line": ln, "missing": True}
    segs = json.loads((d / "segs.json").read_text())
    ended = all(s.get("end") for s in segs)
    v = TM.line_view(ln, final=ended)
    ends = []
    for s in segs:
        e = d / f"seg{s['seg']}" / "end.json"
        ends.append(json.loads(e.read_text()) if e.exists() else None)
    last = ends[-1] or {}
    cv = {c["it"]: c for c in v["curve"]}
    base = cv.get(0, {}).get("solved64")
    cross = v["cross_dan"]
    at_cross = cv.get(cross["it"], {}).get("solved64") if cross else None
    end_it = max(cv) if cv else None
    secs, colon, wrong, alarms = [], [], [], 0
    for s in segs:
        for l in open(d / f"seg{s['seg']}" / "metrics.jsonl"):
            r = json.loads(l)
            if r.get("iter", 0) > 1:
                secs.append(r["secs"])
            c = r.get("canary") or {}
            if "colon" in c:
                colon.append(c["colon"])
                wrong.append(c["wrong_max"])
    for f in (d / "logs").glob(f"{ln}_seg*.log"):
        alarms += sum(1 for x in open(f, errors="replace") if x.startswith("ALARM it"))
    m = (mon.get("tag") or {}).get(ln, {})
    return {"line": ln, "segs": len(segs), "host": m.get("host"), "start": segs[0]["start"], "end": segs[-1].get("end"),
            "iters": v["last_iter"], "x_dan": round(v["x_dan"], 3), "gpuh": round(v["gpuh"], 2),
            "how": last.get("how"), "reason": last.get("reason"), "lost": v["lost"],
            "matched_cum": [x["cum"] for x in v["matched"]],
            "cross": None if cross is None else {k: (round(x, 3) if isinstance(x, float) else x) for k, x in cross.items()},
            "class": klass(v), "solved_base": base, "solved_cross": at_cross, "solved_end": cv.get(end_it, {}).get("solved64"),
            "end_read_it": end_it, "max_colon": max(colon, default=None), "max_wrong": max(wrong, default=None),
            "alarms_logged": alarms, "s_per_it_median": round(statistics.median(secs), 1) if secs else None,
            "s_per_it_mean": round(statistics.mean(secs), 1) if secs else None,
            "collapsed": m.get("collapsed"), "pre_reads": m.get("pre_reads"),
            "curve": [[c["it"], round(c["x_dan"], 3), c["solved64"], c["first32"]] for c in v["curve"]]}  # fmt: skip


def strict(p: Path) -> int | None:
    return json.loads(p.read_text())["strict_n"] if p.exists() else None


def cross_table(seed: int, S: dict) -> dict:
    R = AT0 / "reads" / f"cross_g0_p10_s{seed}"
    rows = {}
    a, b = S.get(f"tA_p10_s{seed}", {}), S.get(f"tB_p10_s{seed}", {})
    rows["base"] = {
        "A": a.get("solved_base"),
        "B": b.get("solved_base"),
        "none": strict(R / "base_none.json"),
    }
    for t, s in (("B", b), ("A", a)):
        pre = R / f"precollapse_t{t}_p10_s{seed}.json"
        fin = R / f"final_t{t}_p10_s{seed}.json"
        src = pre if pre.exists() else fin
        if not src.exists():
            rows[f"t{t}-trained"] = {"note": "no final snapshot read yet"}
            continue
        it = json.loads(src.read_text())["it"]
        ln = f"t{t}_p10_s{seed}"
        own = None
        for f in sorted((G0 / ln).glob(f"seg*/reads/it{it:04d}.json")):
            own = strict(f)
        r = {t: own, "it": it}
        for o in ("A", "B", "none"):
            if o != t:
                r[o] = strict(R / f"t{t}{it}_{o}.json")
        rows[f"t{t}-trained"] = r
    return rows


def matched_table(seed: int, S: dict, N: int) -> dict:
    """The same 3x3 at the matched iteration N: own tag from the in-run read of it N, the others from matched_N<N>/."""
    R = AT0 / "reads" / f"cross_g0_p10_s{seed}"
    a, b = S.get(f"tA_p10_s{seed}", {}), S.get(f"tB_p10_s{seed}", {})
    rows = {
        "base": {
            "A": a.get("solved_base"),
            "B": b.get("solved_base"),
            "none": strict(R / "base_none.json"),
        }
    }
    for t in ("B", "A"):
        ln = f"t{t}_p10_s{seed}"
        own = None
        for f in sorted((G0 / ln).glob(f"seg*/reads/it{N:04d}.json")):
            own = strict(f)
        r = {t: own, "it": N}
        for o in ("A", "B", "none"):
            if o != t:
                r[o] = strict(R / f"matched_N{N}" / f"t{t}{N}_{o}.json")
        rows[f"t{t}-trained"] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--N", type=int, default=120, help="matched iteration for the second cross-tag table"
    )
    a = ap.parse_args()
    ms = G0 / "monitor_state.json"
    mon = json.loads(ms.read_text()) if ms.exists() else {}
    S = {ln: line_summary(ln, mon) for ln in LINES}
    tabs = {s: cross_table(s, S) for s in (0, 1, 2)}
    mtabs = {s: matched_table(s, S, a.N) for s in (0, 1, 2)}
    mtabs60 = {s: matched_table(s, S, 60) for s in (0, 1, 2)}
    for ln, s in S.items():
        if s.get("missing"):
            print(f"{ln}: not fetched")
            continue
        print(f"{ln} [{s['host']}, {s['segs']} seg]: it {s['iters']}, x_dan {s['x_dan']}, end {s['how']} ({s['reason']}); "
              f"cross {s['cross']}; class {s['class']}; strict-106 base {pct(s['solved_base'])}, cross "
              f"{pct(s['solved_cross'])}, end (it {s['end_read_it']}) {pct(s['solved_end'])}; max P(:) {s['max_colon']}, "
              f"max wrong {s['max_wrong']}; alarms {s['alarms_logged']}; s/it median {s['s_per_it_median']} mean "
              f"{s['s_per_it_mean']}; collapsed {s['collapsed']}; lost {s['lost']}")  # fmt: skip
    print("own-tag strict-106 curve per line (it: solved of 106 at k 64):")
    for ln, s in S.items():
        if not s.get("missing"):
            print(f"  {ln}: " + " ".join(f"{c[0]}:{c[2]}" for c in s["curve"]))
    for seed, t in tabs.items():
        print(f"seed {seed} cross-tag (strict 106, k 64):")
        for k, r in t.items():
            print(
                f"  {k:10s} "
                + "  ".join(f"{o}: {pct(r.get(o))}" for o in ("A", "B", "none"))
                + (f"  (it {r['it']})" if "it" in r else "")
                + (f"  {r['note']}" if "note" in r else "")
            )
    for seed, t in mtabs60.items():
        print(f"seed {seed} cross-tag at matched it 60 (strict 106, k 64):")
        for k, r in t.items():
            print(f"  {k:10s} " + "  ".join(f"{o}: {pct(r.get(o))}" for o in ("A", "B", "none")))
    for seed, t in mtabs.items():
        print(f"seed {seed} cross-tag at matched it {a.N} (strict 106, k 64):")
        for k, r in t.items():
            print(f"  {k:10s} " + "  ".join(f"{o}: {pct(r.get(o))}" for o in ("A", "B", "none")))
    if a.json:
        Path(a.json).write_text(
            json.dumps(
                {
                    "lines": S,
                    "cross_end": tabs,
                    "cross_matched": mtabs,
                    "cross_matched_60": mtabs60,
                    "N": a.N,
                },
                indent=1,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
