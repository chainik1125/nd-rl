# ruff: noqa: E501
"""summarize.py RUN_DIR [RUN_DIR ...] [--upto N] [--json OUT]: per-run summary of GRPO metrics.jsonl rows.

For each run (a seg dir holding metrics.jsonl): iterations completed; at it 10/20/40/60 (the closest logged at or
before): pass@1 on the iteration's training prompts (reward_pass1), canary NLL, mean P(':'), mean log P(':'),
top wrong-token P (and each vs the canary base); alarms by type and rollbacks; the effective lr over time; the
largest P(':') / base and the lowest pass@1 after it 10; median seconds per iteration. Read-only.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

POINTS = (1, 5, 10, 20, 30, 40, 50, 60)


def kind(msg: str) -> str:
    if msg.startswith("canary"):
        return "canary " + re.match(r"canary (.+?) [0-9.e+-]+ >", msg).group(1)
    if "running median" in msg:
        return "clip-low rel"
    if msg.startswith("clip-low"):
        return "clip-low abs"
    for k in ("grad-norm", "capped", "overlong", "think banned", "banned in their phase"):
        if k in msg:
            return k
    return msg[:30]


def load(d: Path, upto: int | None) -> list[dict]:
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
    return [r for r in rows if upto is None or r["iter"] <= upto]


def at(rows: list[dict], it: int) -> dict | None:
    c = [r for r in rows if r["iter"] <= it]
    return c[-1] if c else None


def would(rows: list[dict], mult: float, rel: float) -> dict:
    """Iterations whose canary / relative clip-low values exceed the thresholds (mult x base; rel x the row's
    running median and 1e-3, on the accepted tokens' clip-low as grpo.clip_low_basis): the alarms a guard at
    these settings would have raised on this trajectory (it would have diverged at its first rollback)."""
    out: dict = collections.defaultdict(list)
    for r in rows:
        c, b = r["canary"], r["canary"]["base"]
        for k in ("nll", "wrong_max", "colon"):
            if c[k] > mult * b[k]:
                out[f"canary {k}"].append(r["iter"])
        cl = r.get("clip_low_accepted", r["clip_low"])
        med = r.get("clip_low_median")
        if med is not None and cl > max(rel * med, 1e-3):
            out["clip-low rel"].append(r["iter"])
    return dict(out)


def summary(d: Path, upto: int | None) -> dict:
    rows = load(d, upto)
    base = rows[0]["canary"]["base"]
    pts = {}
    for p in POINTS:
        r = at(rows, p)
        if r is None or any(v["iter"] == r["iter"] for v in pts.values()):
            continue
        c = r["canary"]
        pts[p] = {
            "iter": r["iter"],
            "pass1": r["reward_pass1"],
            "nll": c["nll"],
            "colon": c["colon"],
            "colon_x": c["colon"] / base["colon"],
            "colon_logp": c["colon_logp"],
            "wrong": c["wrong_max"],
            "lr": r["lr"],
            "lr_scale": r.get("lr_scale", 1.0),
            "lr_mult": r.get("lr_mult"),
        }
    alarms = collections.Counter(kind(m) for r in rows for m in r["alarms"])
    alarm_iters = collections.defaultdict(list)
    for r in rows:
        for m in r["alarms"]:
            alarm_iters[kind(m)].append(r["iter"])
    rb = [
        (r["iter"], r["rollback"]["to_iter"], r["rollback"]["lr_scale"])
        for r in rows
        if r.get("rollback")
    ]
    later = [r for r in rows if r["iter"] >= 10]
    colon_max = max(rows, key=lambda r: r["canary"]["colon"])
    lr_changes = []
    for r in rows:
        if not lr_changes or abs(r["lr"] - lr_changes[-1][1]) > 1e-12:
            lr_changes.append((r["iter"], r["lr"]))
    return {
        "dir": str(d),
        "iters": rows[-1]["iter"],
        "n_rows": len(rows),
        "base": {k: base[k] for k in ("nll", "colon", "colon_logp", "wrong_max")},
        "points": pts,
        "alarms": dict(alarms),
        "alarm_iters": dict(alarm_iters),
        "rollbacks": rb,
        "lr_changes": lr_changes,
        "colon_max": (
            colon_max["iter"],
            colon_max["canary"]["colon"],
            colon_max["canary"]["colon"] / base["colon"],
        ),
        "colon_logp_max": max(r["canary"]["colon_logp"] for r in rows),
        "nll_max": max(r["canary"]["nll"] for r in rows),
        "wrong_max_max": max(r["canary"]["wrong_max"] for r in rows),
        "pass1_last5": statistics.mean(r["reward_pass1"] for r in rows[-5:]),
        "pass1_min_after10": min((r["reward_pass1"] for r in later), default=None),
        "pass1_max": max(r["reward_pass1"] for r in rows),
        "secs_median": statistics.median(r["secs"] for r in rows),
        "secs_sum": sum(r["secs"] for r in rows),
        "grad_norm_max_max": max(r["grad_norm_max"] for r in rows),
        "clip_low_max": max(r["clip_low"] for r in rows),
        "entropy_last": rows[-1].get("entropy"),
        "would_default_5_2": would(rows, 5.0, 2.0),
        "would_relaxed_20_3": would(rows, 20.0, 3.0),
    }


def show(s: dict, name: str) -> None:
    b = s["base"]
    print(
        f"== {name}: {s['iters']} iterations ({s['n_rows']} rows), {s['secs_median']:.0f} s/it median, "
        f"{s['secs_sum'] / 60:.0f} min of iterations"
    )
    print(
        f"   base: nll {b['nll']:.4f} P(:) {b['colon']:.3g} logP(:) {b['colon_logp']:.2f} wrong {b['wrong_max']:.4f}"
    )
    for v in s["points"].values():
        m = f" m {v['lr_mult']:.3f}" if v["lr_mult"] is not None else ""
        print(
            f"   it {v['iter']:>3}: pass@1 {100 * v['pass1']:5.1f}%  nll {v['nll']:.4f}  P(:) {v['colon']:.3g} "
            f"({v['colon_x']:.1f}x)  logP(:) {v['colon_logp']:.2f}  wrong {100 * v['wrong']:.3f}%  "
            f"lr {v['lr']:.3g} (scale {v['lr_scale']:g}{m})"
        )
    print(f"   alarms: {s['alarms']}")
    for k, its in s["alarm_iters"].items():
        print(f"     {k}: its {its}")
    print(f"   rollbacks (it, to it, lr scale): {s['rollbacks']}")
    print(f"   would alarm at default (5x, clip-low 2x): {s['would_default_5_2']}")
    print(f"   would alarm at relaxed (20x, clip-low 3x): {s['would_relaxed_20_3']}")
    print(f"   lr changes (it, lr): {[(i, f'{x:.3g}') for i, x in s['lr_changes']]}")
    cm = s["colon_max"]
    print(
        f"   max P(:) {cm[1]:.3g} at it {cm[0]} ({cm[2]:.1f}x base); max logP(:) {s['colon_logp_max']:.2f}; "
        f"max nll {s['nll_max']:.4f}; max wrong {100 * s['wrong_max_max']:.3f}%"
    )
    print(
        f"   pass@1: max {100 * s['pass1_max']:.1f}%, min after it 10 "
        f"{100 * (s['pass1_min_after10'] or 0):.1f}%, mean of last 5 {100 * s['pass1_last5']:.1f}%; "
        f"max grad-norm {s['grad_norm_max_max']:.3g}; max clip-low {s['clip_low_max']:.4f}"
    )


def series(d: Path, upto: int | None) -> None:
    """Per iteration: pass@1, canary NLL / P(':') / log P(':') / wrong-token P (and x base), anchor KL, lr, secs."""
    rows = load(d, upto)
    b = rows[0]["canary"]["base"]
    print(f"-- {d}: it pass1 | nll (x) | P(:) (x) logP | wrong (x) | kl_anchor | lr | secs")
    for r in rows:
        c = r["canary"]
        kl = r.get("kl_anchor")
        print(
            f"{r['iter']:>3} {100 * r['reward_pass1']:5.1f} | {c['nll']:.4f} ({c['nll'] / b['nll']:.2f}) | "
            f"{c['colon']:.2e} ({c['colon'] / b['colon']:5.2f}) {c['colon_logp']:6.2f} | "
            f"{100 * c['wrong_max']:.3f}% ({c['wrong_max'] / b['wrong_max']:.2f}) | "
            f"{'-' if kl is None else f'{kl:.2e}'} | {r['lr']:.3g} | {r['secs']:.0f}"
            f"{' ALARM' if r['alarms'] else ''}{' ROLLBACK' if r.get('rollback') else ''}"
        )


def timing(d: Path, split_utc: str | None) -> None:
    """Seconds per iteration before / after a UTC time (an iteration's time: its lean_gate.jsonl row)."""
    rows = load(d, None)
    gate = [json.loads(x) for x in (d / "lean_gate.jsonl").read_text().splitlines() if x.strip()]
    n = min(len(rows), len(gate))
    pairs = [(gate[i]["utc"], rows[i]["secs"], rows[i]["iter"]) for i in range(n)]
    if split_utc is None:
        print(
            f"   timing {d.parent.name}: {statistics.mean(x[1] for x in pairs):.1f} s/it mean over {n}"
        )
        return
    pre = [x for x in pairs if x[0] < split_utc][1:]  # it 1 holds start-up
    post = [x for x in pairs if x[0] >= split_utc][1:]  # the first one straddles the split

    def f(xs: list) -> str:
        if not xs:
            return "none"
        return f"{statistics.mean(x[1] for x in xs):.1f} s/it over its {xs[0][2]}-{xs[-1][2]}"

    print(f"   timing {d.parent.name}: before {split_utc}: {f(pre)}; after: {f(post)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--upto", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--series", action="store_true")
    ap.add_argument(
        "--split_utc", default=None, help="timing before / after this UTC (a GPU neighbour started)"
    )
    a = ap.parse_args()
    if a.series:
        for d in a.dirs:
            series(Path(d), a.upto)
        return
    out = {}
    for d in a.dirs:
        p = Path(d)
        name = p.parent.name if p.name.startswith("seg") else p.name
        s = summary(p, a.upto)
        show(s, name)
        if (p / "lean_gate.jsonl").exists():
            timing(p, a.split_utc)
        out[name] = s
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
