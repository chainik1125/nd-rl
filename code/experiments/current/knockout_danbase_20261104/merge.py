"""One line's merged view across its segments, and the stop rule (PREREG.md).

    python3 merge.py [LINE ...]      # print each line's merged view (serrano; KD/sup/state.json)

A segment's files are read from serrano's live tree (`KD/runs`), else from the one-way collected
copy (`KD/collected/<host>/`), else from serrano's backup copy (`KD/backup/<line>/seg<k>/`).
Segment k+1 resumed from a checkpoint of iteration i0: segment k's metrics rows and reads after
i0 are superseded (their iterations were lost) and dropped, and listed as `lost`.
Compute: every metrics row counts (rolled-back iterations too), in order; x(i) is the cumulative
compute after the last row of iteration i. GPU-h: wall time of the segments (start to end).
Dan's measure (primary): 16 matched draws, one per round of his ladder (the line's `ladder`):
round 1 the base read (iteration 0), round r ≥ 2 the latest read with x_dan ≤ his x_r; each draw
is the read's first 32 attempts; the cumulative count is the union of the strict-106 members
solved, against his bar 14.
"""

# ruff: noqa: E501, SIM105, SIM102
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GPUH_CAP = 15.0
HOST = __import__("socket").gethostname().split(".")[0]


def seg_root(line: str, seg: dict) -> Path | None:
    k = seg["seg"]
    for p in (C.KD / "runs" / line / f"seg{k}" if seg["host"] == HOST else None,
              C.KD / "collected" / seg["host"] / line / f"seg{k}",
              C.KD / "backup" / line / f"seg{k}"):  # fmt: skip
        if p is not None and (p / "metrics.jsonl").exists():
            return p
    return None


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for x in p.read_text().splitlines():
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            pass  # a row being written
    return out


def _utc(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone


def line_view(line: str, segs: list[dict], final: bool = False) -> dict:
    L = C.LINES[line]
    lad = C.dan_ladder(L["ladder"])
    D = lad["D"]
    rows: list[dict] = []
    reads: dict[int, dict] = {}
    lost: list[list[int]] = []
    gpuh = 0.0
    for j, seg in enumerate(segs):
        root = seg_root(line, seg)
        nxt = segs[j + 1] if j + 1 < len(segs) else None
        i0 = None
        if nxt is not None:
            m = re.search(r"resume_it(\d+)\.pt$", nxt.get("resume") or "")
            i0 = int(m.group(1)) if m else 0
        if root is not None:
            rs = _rows(root / "metrics.jsonl")
            rs = [r for r in rs if "sampled_gen_tokens" in r]
            if i0 is not None:
                past = [r["iter"] for r in rs if r["iter"] > i0]
                if past:
                    lost.append([i0 + 1, max(past)])
                rs = [r for r in rs if r["iter"] <= i0]
            rows += [{**r, "seg": seg["seg"]} for r in rs]
            for f in sorted((root / "reads").glob("it*.json")):
                it = int(f.stem[2:])
                if i0 is None or it <= i0:
                    try:
                        reads[it] = json.loads(f.read_text())
                    except json.JSONDecodeError:
                        pass
            end = root / "end.json"
            t1 = _utc(json.loads(end.read_text())["utc"]) if end.exists() else time.time()
        else:
            t1 = time.time() if nxt is None else _utc(nxt["start"])
        if nxt is not None:
            t1 = min(t1, _utc(nxt["start"]))
        gpuh += max(0.0, t1 - _utc(seg["start"])) / 3600
    xd = xo = 0.0
    x_by_it: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for r in rows:
        xd += C.x_dan_row(r, D)
        xo += C.x_ours_row(r, D)
        x_by_it[r["iter"]] = (xd, xo)
    curve = []
    for it in sorted(reads):
        if it in x_by_it:
            rd = reads[it]
            curve.append({"it": it, "x_dan": x_by_it[it][0], "x_ours": x_by_it[it][1],
                          "first32": rd.get("first32_n"), "e32": rd.get("e32"),
                          "solved64": rd.get("strict_n"),
                          "first32_solved": rd.get("first32_solved", [])})  # fmt: skip
    # Dan's measure: matched draws
    matched, cum, union, cross = [], [], set(), None
    for r, xr in enumerate(lad["x"], start=1):
        if r == 1:
            pick = next((c for c in curve if c["it"] == 0), None)
        else:  # drawn once a later read exists (reads are made in order) or the line ended
            ok = [c for c in curve if c["x_dan"] <= xr]
            pick = ok[-1] if ok and ok[-1]["it"] > 0 else None
            if not (final or any(c["x_dan"] > xr for c in curve)):
                pick = None
        if pick is None:
            break
        union |= set(pick["first32_solved"])
        matched.append({"round": r, "it": pick["it"], "x_dan": pick["x_dan"],
                        "x_ours": pick["x_ours"], "his_x": xr, "draw": len(pick["first32_solved"]),
                        "cum": len(union)})  # fmt: skip
        cum.append(len(union))
        if cross is None and len(union) >= C.BAR:
            cross = matched[-1]
    return {"line": line, "rows": len(rows), "last_iter": rows[-1]["iter"] if rows else 0,
            "x_dan": xd, "x_ours": xo, "gpuh": gpuh, "curve": curve, "matched": matched,
            "cross_dan": cross, "lost": lost, "x_cap": lad["x"][15]}  # fmt: skip


def held(curve: list[dict]) -> dict | None:
    """Snapshot first-32 count ≥ 14 at ≥ 3 consecutive reads, the last at ≥ 2 × the first's x."""
    for i, c in enumerate(curve):
        if c["it"] == 0 or (c["first32"] or 0) < C.BAR:
            continue
        j = i
        while j + 1 < len(curve) and (curve[j + 1]["first32"] or 0) >= C.BAR:
            j += 1
            if j - i >= 2 and curve[j]["x_dan"] >= 2 * c["x_dan"]:
                return {"from_it": c["it"], "to_it": curve[j]["it"], "x_dan": curve[j]["x_dan"]}
    return None


def stop_reason(line: str, v: dict) -> str | None:
    if v["x_dan"] >= v["x_cap"]:
        return f"compute cap: x_dan {v['x_dan']:.2f} ≥ his round-16 x {v['x_cap']:.2f}"
    if v["gpuh"] >= GPUH_CAP:
        return f"GPU-h cap: {v['gpuh']:.1f} ≥ {GPUH_CAP:g}"
    h = held(v["curve"])
    if v["cross_dan"] is not None and h is not None:
        return f"reached (Dan's measure at x_dan {v['cross_dan']['x_dan']:.3f}) and held ({h})"
    return None


def main() -> None:
    st = json.loads((C.KD / "sup" / "state.json").read_text())
    for line in sys.argv[1:] or sorted(C.LINES):
        if line not in st or not st[line]["segs"]:
            print(line, "no segments")
            continue
        v = line_view(line, st[line]["segs"])
        c = v["curve"][-1] if v["curve"] else {}
        print(f"{line}: it {v['last_iter']} x_dan {v['x_dan']:.3f} (cap {v['x_cap']:.2f}) "
              f"x_ours {v['x_ours']:.3f} GPU-h {v['gpuh']:.2f}; reads {len(v['curve'])}, last "
              f"first32 {c.get('first32')} e32 {c.get('e32')}; Dan's measure "
              f"{[m['cum'] for m in v['matched']]} cross {v['cross_dan']}; lost {v['lost']}; "
              f"stop {stop_reason(line, v)}")  # fmt: skip


if __name__ == "__main__":
    main()
