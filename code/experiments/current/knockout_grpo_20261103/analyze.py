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
ARMS = ("full", "d1", "d01", "ko", "kos")
SHARE = {"full": 2.48, "d1": 1.0, "d01": 0.1, "ko": 0.0, "kos": 0.0}
VARIANTS = ("", "_dp", "_gr", "_dp_gr")  # our pool; Dan's pool (Am. 2); guard relaxed (Am. 4)
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


def train_sigma(tag: str) -> dict | None:
    """σ share of distinct Lean-accepted training proofs per iteration (train_sigma.py output)."""
    p = OUT / "train_sigma" / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows = [{**r, "stage": k} for k in ("L2", "L3") for r in d.get(k, [])]
    if not rows:
        return None
    sh = [100.0 * r["sigma"] / max(1, r["accepted_distinct"]) for r in rows]
    dn = [100.0 * r["dn_any"] / max(1, r["accepted_distinct"]) for r in rows]
    return {"first_pct": sh[0], "last_pct": sh[-1], "max_pct": max(sh),
            "last_at": f"{rows[-1]['stage']} it {rows[-1]['iter']}",
            "dn_any_pct_range": [min(dn), max(dn)], "n_iters": len(rows)}  # fmt: skip


def facts(dan_axis: list[dict]) -> dict:
    """Corpus, pool and accounting numbers the figure caption quotes (from their own files)."""
    sg = {k: json.loads((OUT / "sigma" / f).read_text())["share_pct"]
          for k, f in (("full", "s1g3.json"), ("d1", "corpus_d1.json"), ("d01", "corpus_d01.json"),
                       ("ko", "corpus_ko.json"), ("kos", "corpus_ko_strict.json"),
                       ("k12", "k12.json"))}  # fmt: skip
    pool = json.loads((OUT / "pool_sigma.json").read_text())
    plan = json.loads((OUT / "corpora" / "ko_strict" / "plan.json").read_text())
    n_ko = json.loads((OUT / "sigma" / "corpus_ko.json").read_text())["counts"]["records"]
    leak = json.loads((OUT / "trace_leak4.json").read_text())["ko"]
    r1, later = [], []
    for o in dan_axis:
        for r in o["rounds"]:
            if r.get("x_ours"):
                (r1 if r["round"] == 1 else later).append(r["x_dan"] / r["x_ours"])
    return {
        "sigma_plus_share_pct": {k: v["sigma_plus"] for k, v in sg.items()},
        "pool_strict_pct": {k: pool[k]["strict_pct"] for k in pool},
        "dan_pool_n": pool["dan_rl_targets"]["n"],
        "ko_records": n_ko,
        # the detour plan in the narration at the effort level each record was trained on
        "ko_plan_trained": leak["detour_goal_trained"],
        "ko_plan_trained_pct": 100.0 * leak["detour_goal_trained"] / n_ko,
        "ko_plan_negi_trained": leak["reductio_plan_trained"],
        # at either non-empty effort level (e64 or e256): the records ko_strict removed
        "ko_plan_any_level_removed": len(plan["bad"]),
        "ko_plan_any_level_pct": 100.0 * len(plan["bad"]) / n_ko,
        # Dan's x / ours over his ladders' points: round 1 is sampling only (his x counts the
        # re-read state prefill, ours does not); later rounds add fine-tuning and replay
        "dan_x_over_ours_round1": [min(r1), max(r1)],
        "dan_x_over_ours_later": [min(later), max(later)],
    }


def line_state(tag: str) -> dict:
    """Whether the line has ended, and how (line_done.json; think_ban_stop.json for kos_s1)."""
    ev = OUT / "evals" / tag
    done = (
        json.loads((ev / "line_done.json").read_text())
        if (ev / "line_done.json").exists()
        else None
    )
    tb = done.get("think_ban_stop") if done else None
    if (ev / "think_ban_stop.json").exists():
        tb = json.loads((ev / "think_ban_stop.json").read_text())["stage"]
    tbj = OUT / "think_ban.json"  # think_ban.py: a STOP line in the stage logs
    if done and not tb and tbj.exists():
        st_ = json.loads(tbj.read_text()).get(tag, {}).get("stages", {})
        tb = next((k for k, v in st_.items() if v.get("stop")), None)
    return {"line_done": done is not None, "think_ban_stop": tb}


def _row(tag: str, x: dict) -> str:
    c, last, ts = x["cross"], x.get("last"), x.get("train_sigma")
    base = f"{x['base_e32']:.2f} ({x['base_e32_pct']:.1f} %)"
    ls = f"c {last['c']}, R {last['R']:.3g}, {last['e32']:.1f}" if last else "—"
    sg = (f"{ts['first_pct']:.1f} → {ts['last_pct']:.1f} % at {ts['last_at']} "
          f"(max {ts['max_pct']:.1f})" if ts else "")  # fmt: skip
    if c["status"] == "reached":
        if c.get("R_upper_only"):
            r = f"≤ {c['R']:.3g} (first checkpoint, c {c['c']})"
        else:
            r = f"{c['R']:.3g} (c {c['c_lo']}–c {c['c_hi']})"
        h = c.get("held")
        if h is None:
            hd = "not held: " + c.get("held_status", "")
        else:
            hd = f"c {h.get('c_hi', h.get('c'))}–c {h['c_held']}, done at R {h['R_held_at']:.3g}"
            if not c.get("holds"):
                hd = f"later streak from R {h['R']:.3g}: " + hd
        cells = [base, r, f"{c['gpu_h']:.2f}", hd, ls, sg]
    elif c["status"] == "not_reached":
        nr = f"not reached by {c['R_max']:.3g} (best {c['best_e32']:.1f})"
        if x.get("think_ban_stop"):
            nr += f"; think-ban stop in {x['think_ban_stop']}"
        elif not x.get("line_done"):
            nr += "; running"
        cells = [base, nr, f"{c['gpu_h']:.2f}", "", ls, sg]
    else:
        cells = [base, c["status"], "", "", ls, sg]
    return "| " + " | ".join([tag, *cells]) + " |"


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
            for v in VARIANTS:
                tag = f"{a}_s{s}{v}"
                b = bases[f"{a}_s{s}"]
                if b is None or (v and not (OUT / "evals" / tag).exists()):
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
                    x["curve"] = [
                        {k: r[k] for k in ("c", "R", "e32", "first32", "secs")} for r in rows
                    ]
                    x["dan_cumulative"] = dan_cumulative(tag, b, rows, bar_d)
                x["train_sigma"] = train_sigma(tag)
                x.update(line_state(tag))
                res["runs"][tag] = x
    res["facts"] = facts(res.get("dan_axis") or json.loads((OUT / "dan_axis.json").read_text()))
    tb = OUT / "think_ban.json"
    if tb.exists():
        res["think_ban"] = {k: v["total"] for k, v in json.loads(tb.read_text()).items()}
    dan = OUT / "dan_axis.json"
    if dan.exists():
        res["dan_axis"] = json.loads(dan.read_text())
    (HERE / "numbers.json").write_text(json.dumps(res, indent=1))
    lines = [
        "# knockout_grpo numbers\n",
        f"Bar (median E32 of the full bases, n 256): **{bar:.2f}** of 106 ({pct(bar):.1f} %); "
        f"Dan-matched single-draw bar {bar_d:g}. E32 = expected number of the 106 solved by a k 32 "
        "draw (unbiased pass@32 from 64 attempts per checkpoint, 256 per base). R = (6 trained + 2 "
        "sampled tokens) / (6 Stage-1 tokens); c = GRPO iterations (L2 39, then L3). First "
        "crossing: the first checkpoint at or above the bar, linear in log10 R from the one "
        "before (≤ when it is the first checkpoint). Held: Amendment 1's window (≥ 3 consecutive "
        "checkpoints ≥ bar, the last at ≥ 2 × the first's R). "
        "σ share: of the distinct Lean-accepted training proofs of an iteration (train_sigma.py), "
        "at L2 iteration 1 → the last logged iteration (max).\n",
        "| line | base E32 (% of 106) | first crossing R (bracket) | GPU-h to it | held window "
        "| last checkpoint (c, R, E32) | σ share of accepted training proofs |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, x in res["runs"].items():
        lines.append(_row(tag, x))
    if "think_ban" in res:
        lines += ["\n## Think-ban guard events (think_ban.py; distinct GRPO iterations)\n",
                  "| line | iterations | think-ban alarms (% of iterations) | rollbacks "
                  "| alarms with no rollback | stops |", "|---|---|---|---|---|---|"]  # fmt: skip
        for k, v in res["think_ban"].items():
            pc = 100.0 * v["alarm"] / max(1, v["its"])
            lines.append(f"| {k} | {v['its']} | {v['alarm']} ({pc:.1f} %) | {v['rollback']} | "
                         f"{v['no_rollback']} | {v['stop']} |")  # fmt: skip
    if "dan_axis" in res:
        lines += ["\n## Dan's EI ladders on our axis (dan_axis.py)\n",
                  "His x ÷ our R: " + ", ".join(
                      f"round 1 (sampling only; his x counts the re-read state prefill) "
                      f"{a:.2f}–{b:.2f}" for a, b in [res["facts"]["dan_x_over_ours_round1"]])
                  + "; every later round (fine-tuning and replay added) "
                  + "{:.2f}–{:.2f}.\n".format(*res["facts"]["dan_x_over_ours_later"]),
                  "| ladder | arm | crossing round | his x | our R | his ÷ ours "
                  "| not reached: his / our last |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for o in res["dan_axis"]:
            c, la = o["cross"], o["last"]
            if c:
                q = c["x_dan"] / c["x_ours"]
                cells = [
                    f"r{c['round']}",
                    f"{c['x_dan']:.2f}",
                    f"{c['x_ours']:.3f}",
                    f"{q:.2f}",
                    "",
                ]
            else:
                q = la["x_dan"] / la["x_ours"]
                cells = ["—", "", "", f"{q:.2f}", f"{la['x_dan']:.2f} / {la['x_ours']:.2f}"]
            lines.append("| " + " | ".join([o["name"], o["arm"], *cells]) + " |")
    (HERE / "numbers.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
