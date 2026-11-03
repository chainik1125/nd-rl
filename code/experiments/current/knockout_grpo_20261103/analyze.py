"""Numbers for knockout_grpo: bar, per-run crossings, Dan-matched cumulative y, Dan on our axis.

    python3 analyze.py            # reads OUT (collected from every machine) -> numbers.json / .md

Stdlib only. Definitions in PREREG.md ("Measurement"). Counts are of the strict 106 and also
given as % of 106.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
from pathlib import Path

import measure as M

OUT = Path(os.environ.get("KO_OUT", "/scratch/cowifornia/nd-rl/knockout_grpo"))
HERE = Path(__file__).resolve().parent
ARMS = ("full", "d1", "d01", "ko")
SHARE = {"full": 2.48, "d1": 1.0, "d01": 0.1, "ko": 0.0}
N106 = 106


def pct(x: float) -> float:
    return 100.0 * x / N106


def base_read(b: str) -> dict | None:
    d = OUT / "evals" / f"base_{b}_k256"
    if not (d / "rows_plain_B64.jsonl").exists():
        return None
    rows = M.read_rows(d)
    e = M.e32(rows)
    return {
        "e32": e["e32"],
        "first32": e["first32"],
        "solved_any256": e["solved_any"],
        "solved_first32": sorted(n for n, v in e["per"].items() if v[2] > 0),
    }


def curve(tag: str) -> list[dict]:
    p = OUT / "evals" / tag / "curve.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows = {r["c"]: r for r in rows}  # last row per c
    return [rows[c] for c in sorted(rows)]


def _interp(prev: dict | None, r: dict, bar: float) -> dict:
    """The crossing between `prev` (below the bar, or None = the base at R 0) and `r` (at or
    above it): linear in log10 R; bounded by r when it is the first checkpoint."""
    if prev is None:
        return {"R": r["R"], "R_upper_only": True, "c": r["c"], "gpu_h": r["secs"] / 3600}
    lo, hi = math.log10(prev["R"]), math.log10(r["R"])
    f = (bar - prev["e32"]) / (r["e32"] - prev["e32"])
    return {"R": 10 ** (lo + f * (hi - lo)), "c_lo": prev["c"], "c_hi": r["c"],
            "R_lo": prev["R"], "R_hi": r["R"],
            "gpu_h": (prev["secs"] + f * (r["secs"] - prev["secs"])) / 3600}  # fmt: skip


def crossing(base_e32: float, rows: list[dict], bar: float) -> dict:
    """Primary (pre-registered): the first checkpoint with E32 >= bar, interpolated in log10 R.
    Held (Amendment 1): the first streak of >= 3 consecutive checkpoints >= bar whose last is at
    >= 2x its first's R; `holds` says whether the first crossing's own streak is that one."""
    if base_e32 >= bar:
        return {"status": "at_bar_before_rl", "R": 0.0}
    if not rows:
        return {"status": "no_run"}
    first = held = None
    prev = None
    streak: list[dict] = []
    streak_prev = None
    for r in rows:
        if r["e32"] >= bar:
            if not streak:
                streak_prev = prev
            streak.append(r)
            if first is None:
                first = _interp(prev, r, bar)
            if held is None and len(streak) >= 3 and streak[-1]["R"] >= 2.0 * streak[0]["R"]:
                held = {
                    **_interp(streak_prev, streak[0], bar),
                    "c_held": r["c"],
                    "R_held_at": r["R"],
                }
        else:
            streak = []
        prev = r
    last = rows[-1]
    if first is None:
        return {"status": "not_reached", "R_max": last["R"], "c_max": last["c"],
                "best_e32": max(x["e32"] for x in rows), "gpu_h": last["secs"] / 3600}  # fmt: skip
    out = {"status": "reached", **first, "held": held,
           "holds": held is not None and abs(held["R"] - first["R"]) < 1e-12}  # fmt: skip
    if held is None:
        out["held_status"] = f"not held by R {last['R']:.3g} (c {last['c']})"
    return out


def dan_cumulative(tag: str, base: dict, rows: list[dict], bar_d: float) -> dict:
    """Dan-matched cumulative y: union of first-32 solves over rounds (base, c = 35, 70, ...)."""
    solved = set(base["solved_first32"])
    out = [{"round": 1, "c": 0, "R": 0.0, "y": len(solved)}]
    for r in rows:
        if r["c"] % 35 != 0:
            continue
        e = M.e32(M.read_rows(OUT / "evals" / tag / f"c{r['c']:04d}"))
        solved |= {n for n, v in e["per"].items() if v[2] > 0}
        out.append({"round": len(out) + 1, "c": r["c"], "R": r["R"], "y": len(solved)})
    hit = next((x for x in out if x["y"] >= bar_d), None)
    return {"rounds": out, "cross": hit}


def main() -> None:
    bases = {f"{a}_s{s}": base_read(f"{a}_s{s}") for a in ARMS for s in range(3)}
    full = [bases[f"full_s{s}"] for s in range(3)]
    if any(b is None for b in full):
        raise SystemExit("full base reads missing")
    bar = st.median(b["e32"] for b in full)
    bar_d = st.median(b["first32"] for b in full)
    res: dict = {
        "bar_e32": bar,
        "bar_e32_pct": pct(bar),
        "bar_dan_first32": bar_d,
        "bases": bases,
        "runs": {},
    }
    for a in ARMS:
        for s in range(3):
            tag = f"{a}_s{s}"
            b = bases[tag]
            if b is None:
                continue
            rows = curve(tag)
            x = {
                "base_e32": b["e32"],
                "base_e32_pct": pct(b["e32"]),
                "base_first32": b["first32"],
                "n_checkpoints": len(rows),
                "cross": crossing(b["e32"], rows, bar),
            }
            if rows:
                x["last"] = rows[-1]
                x["curve"] = [{k: r[k] for k in ("c", "R", "e32", "first32", "secs")} for r in rows]
                x["dan_cumulative"] = dan_cumulative(tag, b, rows, bar_d)
            res["runs"][tag] = x
    dan = OUT / "dan_axis.json"
    if dan.exists():
        res["dan_axis"] = json.loads(dan.read_text())
    (HERE / "numbers.json").write_text(json.dumps(res, indent=1))
    lines = ["# knockout_grpo numbers\n", f"Bar (median E32 of the full bases, n 256): **{bar:.2f}** of 106 "  # noqa: E501
             f"({pct(bar):.1f} %); Dan-matched single-draw bar {bar_d:g}.\n",
             "| run | base E32 (% of 106) | base first-32 | crossing |", "|---|---|---|---|"]  # fmt: skip  # noqa: E501
    for tag, x in res["runs"].items():
        c = x["cross"]
        if c["status"] == "reached":
            cs = f"R {c['R']:.3g} ({c['gpu_h']:.2f} GPU-h)"
        elif c["status"] == "not_reached":
            cs = f"not reached by R {c['R_max']:.3g} (best E32 {c['best_e32']:.1f})"
        else:
            cs = c["status"]
        lines.append(
            f"| {tag} | {x['base_e32']:.2f} ({x['base_e32_pct']:.1f} %) | {x['base_first32']} | {cs} |"  # noqa: E501
        )
    (HERE / "numbers.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
