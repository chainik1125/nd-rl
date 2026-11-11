"""Dan's measure for a tagged RL line, and its stop rule (adapted from knockout_danbase_20261104/merge.py at d78d12a).

    python tag_measure.py [LINE ...]      # print each line's view (default: every line with segments)

Same definitions as merge.line_view: every metrics row counts toward x (rolled-back iterations too); a resumed
segment supersedes the earlier one's rows and reads past its resume point (`lost`); Dan's measure is the union of the
strict-106 members solved by the first 32 attempts of 16 matched draws on his ladder's x grid (round 1 the base read,
round r >= 2 the latest read with x_dan <= his x_r), against his bar 14. Two differences, both because the base is
ours: D is the tagged base's own Stage-1 tokens (`<base>.tag.json`, written by tag_pretrain.py), and the x grid is
his `la_dose1_s<seed>` ladder's (the dd1 lines' grid), read in units of each base's own D.

Stop rule (TAGS.md): the first of x_dan >= tag_core.X_CAP (1.5), GPU-h >= tag_core.GPUH_CAP, or reached and held
(his measure has crossed 14, and the snapshot first-32 count is >= 14 at 3 consecutive reads, the last at >= 2 x the
first's x_dan: merge.held, unchanged).

Segments: `AT/runs/<line>/segs.json`, a list of {seg, host, gpu, start, resume, end}, written by tag_line.py.
"""

# ruff: noqa: E501, SIM105
from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402


def _ko(name: str):
    """A knockout script loaded by path under a private name (that folder is not on sys.path)."""
    key = f"ko_{name}"
    if key in sys.modules:
        return sys.modules[key]
    if str(T.KO) not in sys.path:
        sys.path.append(str(T.KO))  # merge.py imports `common` by bare name
    spec = importlib.util.spec_from_file_location(key, T.KO / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


C = _ko("common")
M = _ko("merge")


def base_D(base: str) -> float:
    """The tagged base's Stage-1 tokens (its sidecar from tag_pretrain.py)."""
    p = T.AT / "bases" / f"{base}.tag.json"
    return float(json.loads(p.read_text())["train_tokens"])


def segs_of(line: str) -> list[dict]:
    p = T.AT / "runs" / line / "segs.json"
    return json.loads(p.read_text()) if p.exists() else []


def line_view(
    line: str, segs: list[dict] | None = None, final: bool = False, D: float | None = None
) -> dict:
    L = T.LINES[line]
    segs = segs_of(line) if segs is None else segs
    lad = C.dan_ladder(L["ladder"])
    D = base_D(L["base"].removesuffix(".pt")) if D is None else D
    rows: list[dict] = []
    reads: dict[int, dict] = {}
    lost: list[list[int]] = []
    gpuh = 0.0
    for j, seg in enumerate(segs):
        root = T.AT / "runs" / line / f"seg{seg['seg']}"
        nxt = segs[j + 1] if j + 1 < len(segs) else None
        i0 = None
        if nxt is not None:
            m = re.search(r"_it(\d+)_(?:rolling|alarm)\.pt$", nxt.get("resume") or "")
            i0 = int(m.group(1)) if m else 0
        rs = [r for r in M._rows(root / "metrics.jsonl") if "sampled_gen_tokens" in r]
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
        t1 = M._utc(seg["end"]) if seg.get("end") else time.time()
        if nxt is not None:
            t1 = min(t1, M._utc(nxt["start"]))
        gpuh += max(0.0, t1 - M._utc(seg["start"])) / 3600
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
                          "first32": rd.get("first32_n"), "e32": rd.get("e32"), "solved64": rd.get("strict_n"),
                          "first32_solved": rd.get("first32_solved", []), "tag": rd.get("author_tag")})  # fmt: skip
    matched, union, cross = [], set(), None
    for r, xr in enumerate(lad["x"], start=1):
        if r == 1:
            pick = next((c for c in curve if c["it"] == 0), None)
        else:
            ok = [c for c in curve if c["x_dan"] <= xr]
            pick = ok[-1] if ok and ok[-1]["it"] > 0 else None
            if not (any(c["x_dan"] > xr for c in curve) or (final and xd >= xr)):
                pick = None
        if pick is None:
            break
        union |= set(pick["first32_solved"])
        matched.append({"round": r, "it": pick["it"], "x_dan": pick["x_dan"], "x_ours": pick["x_ours"],
                        "his_x": xr, "draw": len(pick["first32_solved"]), "cum": len(union)})  # fmt: skip
        if cross is None and len(union) >= C.BAR:
            cross = matched[-1]
    return {"line": line, "tag": L["tag"], "seed": L["seed"], "D": D, "rows": len(rows),
            "last_iter": rows[-1]["iter"] if rows else 0, "x_dan": xd, "x_ours": xo, "gpuh": gpuh, "curve": curve,
            "matched": matched, "cross_dan": cross, "lost": lost, "x_cap": T.X_CAP}  # fmt: skip


def stop_reason(v: dict) -> str | None:
    if v["x_dan"] >= T.X_CAP:
        return f"compute cap: x_dan {v['x_dan']:.3f} >= {T.X_CAP:g}"
    if v["gpuh"] >= T.GPUH_CAP:
        return f"GPU-h cap: {v['gpuh']:.2f} >= {T.GPUH_CAP:g}"
    h = M.held(v["curve"])
    if v["cross_dan"] is not None and h is not None:
        return f"reached (Dan's measure at x_dan {v['cross_dan']['x_dan']:.3f}) and held ({h})"
    return None


def main() -> None:
    lines = sys.argv[1:] or [x for x in T.LINES if segs_of(x)]
    for line in lines:
        v = line_view(line)
        c = v["curve"][-1] if v["curve"] else {}
        print(f"{line} (tag {v['tag']}): it {v['last_iter']} x_dan {v['x_dan']:.3f} (cap {T.X_CAP}) "
              f"x_ours {v['x_ours']:.3f} GPU-h {v['gpuh']:.2f}; reads {len(v['curve'])}, last first32 "
              f"{c.get('first32')} e32 {c.get('e32')}; Dan's measure {[m['cum'] for m in v['matched']]} "
              f"cross {v['cross_dan']}; lost {v['lost']}; stop {stop_reason(v)}")  # fmt: skip


if __name__ == "__main__":
    main()
