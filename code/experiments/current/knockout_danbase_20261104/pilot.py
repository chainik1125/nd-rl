"""The guard-off pilot (PREREG.md, Amendment 2) against dd01_s2 at the same x_dan (read-only).

    python3 pilot.py            # the table on his round grid up to x 3.0, and the collapse check
    python3 pilot.py --no-sigma # the same without σ (its Lean dumps take a minute or two to read)

Each row is one of his rounds: his measure's cumulative count and the draw behind it (matched draws,
merge.line_view), σ's share of the accepted training proofs since the previous round (Dan's
coe_sigma on the LEAN_GATE_DUMP, tables.sigma_by_iter), pass@1's mean over the last 10 iterations
at or below that x, the lr of the last such iteration as a multiple of --lr, and the iterations with
an alarm so far, by type. Writes pilot.md and prints it.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import guard as GD  # noqa: E402
import merge as M  # noqa: E402

PILOT, REF = "dd01g0_s2", "dd01_s2"
X_MAX = 3.0
LR = 1.5e-4
COLLAPSE_AFTER_X, COLLAPSE_N = 0.5, 20  # for 20 consecutive iterations after x 0.5 ...
COLLAPSE_PASS, COLLAPSE_NLL = 0.70, 2.0  # ... pass@1's 10-iteration mean, canary action NLL


def x_rows(line: str, segs: list[dict]) -> list[dict]:
    """The merged rows with their cumulative x_dan."""
    D = C.dan_ladder(C.LINES[line]["ladder"])["D"]
    x, out = 0.0, []
    for r in GD.merged_rows(line, segs):
        x += C.x_dan_row(r, D)
        out.append({**r, "x": x})
    return out


def sigma_rows(line: str, segs: list[dict]) -> dict[int, dict]:
    import tables as T

    keep = {}
    for j, seg in enumerate(segs[:-1]):
        m = re.search(r"resume_it(\d+)\.pt$", segs[j + 1].get("resume") or "")
        keep[seg["seg"]] = int(m.group(1)) if m else 0
    return T.sigma_by_iter(line, segs, keep)


def at_round(rows: list[dict], v: dict, r: int, xr: float, xp: float, sig: dict | None) -> dict:
    # a line that has not run to his x_r has no value there
    upto = [q for q in rows if q["x"] <= xr] if rows and rows[-1]["x"] >= xr else []
    m = next((q for q in v["matched"] if q.get("his_x") == xr), None)
    alarms: dict[str, int] = {}
    for q in upto:
        for k in {GD._kind(a) for a in q.get("alarms") or []}:
            alarms[k] = alarms.get(k, 0) + 1
    out = {
        "round": r,
        "x": xr,
        "cum": m["cum"] if m else None,
        "draw": m["draw"] if m else None,
        "pass1": statistics.mean(q["reward_pass1"] for q in upto[-10:]) if upto else None,
        "lr": upto[-1]["lr"] / LR if upto else None,
        "alarms": alarms,
        "it": upto[-1]["iter"] if upto else None,
    }
    if sig is not None:
        win = [sig[q["iter"]] for q in upto if q["x"] > xp and q["iter"] in sig]
        n = sum(w["accepted"] for w in win)
        out["sigma"] = 100 * sum(w["sigma"] for w in win) / n if n else None
    return out


def collapse(rows: list[dict]) -> str | None:
    """The first time the pre-registered collapse holds (PREREG.md, Amendment 2), None if not."""
    low = high = 0
    for i, q in enumerate(rows):
        if q["x"] <= COLLAPSE_AFTER_X:
            continue
        p = statistics.mean(w["reward_pass1"] for w in rows[max(0, i - 9) : i + 1])
        low = low + 1 if p < COLLAPSE_PASS else 0
        nll = (q.get("canary") or {}).get("nll")
        high = high + 1 if nll is not None and nll > COLLAPSE_NLL else 0
        if low >= COLLAPSE_N:
            return (
                f"pass@1 10-it mean under 70 % for 20 iterations, by it {q['iter']}, x {q['x']:.2f}"
            )
        if high >= COLLAPSE_N:
            return (
                f"canary action NLL above 2.0 for 20 iterations, by it {q['iter']}, x {q['x']:.2f}"
            )
    return None


def fmt(v, f: str = "{:.0f}") -> str:
    return "—" if v is None else f.format(v)


def table(a: list[dict], b: list[dict], with_sigma: bool) -> str:
    s_h = " σ %, pilot / dd01_s2 |" if with_sigma else ""
    s_s = "---|" if with_sigma else ""
    out = [f"| round (his x) | his measure, pilot / dd01_s2 | draw (first 32) |{s_h} pass@1 %, pilot / "
           f"dd01_s2 | lr ×, pilot / dd01_s2 | iterations with an alarm by type, pilot / dd01_s2 |",
           f"|---|---|---|{s_s}---|---|---|"]  # fmt: skip
    for p, r in zip(a, b, strict=True):
        if p["it"] is None or p["pass1"] is None:
            continue

        def al(d: dict) -> str:
            return ", ".join(f"{k} {n}" for k, n in sorted(d.items())) or "none"

        sg = ""
        if with_sigma:
            sg = f" {fmt(p.get('sigma'), '{:.2f}')} / {fmt(r.get('sigma'), '{:.2f}')} |"
        out.append(f"| {p['round']} ({p['x']:.2f}) | {fmt(p['cum'])} / {fmt(r['cum'])} | "
                   f"{fmt(p['draw'])} / {fmt(r['draw'])} |{sg} "
                   f"{fmt(100 * p['pass1'])} / {fmt(100 * r['pass1'])} | "
                   f"{fmt(p['lr'], '{:g}')} / {fmt(r['lr'], '{:g}')} | "
                   f"{al(p['alarms'])} / {al(r['alarms'])} |")  # fmt: skip
    return "\n".join(out)


def main() -> None:
    with_sigma = "--no-sigma" not in sys.argv
    st = json.loads((C.KD / "sup" / "state.json").read_text())
    final = st[PILOT]["status"] == "done"
    xs = [x for x in C.dan_ladder(C.LINES[PILOT]["ladder"])["x"] if x <= X_MAX]
    res = {}
    for line in (PILOT, REF):
        segs = st[line]["segs"]
        rows = x_rows(line, segs)
        v = M.line_view(line, segs, final=(line == REF or final))
        sig = sigma_rows(line, segs) if with_sigma else None
        res[line] = {
            "rows": rows,
            "v": v,
            "at": [
                at_round(rows, v, r + 1, xr, xs[r - 1] if r else 0.0, sig)
                for r, xr in enumerate(xs)
            ],
            "tail": sig,
        }
    rows, v = res[PILOT]["rows"], res[PILOT]["v"]
    lines = [f"Pilot {PILOT}: status {st[PILOT]['status']}, x_dan {v['x_dan']:.2f}, GPU-h "
             f"{v['gpuh']:.1f}, it {v['last_iter']}, crossing {v['cross_dan']}, "
             f"stop {st[PILOT].get('stop_reason')}.",
             "", table(res[PILOT]["at"], res[REF]["at"], with_sigma), ""]  # fmt: skip
    if with_sigma and res[PILOT]["tail"]:
        tail = [
            res[PILOT]["tail"][q["iter"]] for q in rows[-50:] if q["iter"] in res[PILOT]["tail"]
        ]
        n = sum(w["accepted"] for w in tail)
        lines.append(f"σ over the pilot's last 50 iterations: "
                     f"{100 * sum(w['sigma'] for w in tail) / max(1, n):.2f} %.")  # fmt: skip
    lines.append(f"Collapse (as pre-registered): {collapse(rows) or 'no'}.")
    text = "\n".join(lines) + "\n"
    (C.HERE / "pilot.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
