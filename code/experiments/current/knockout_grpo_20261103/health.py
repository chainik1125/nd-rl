"""Training health of every line (coordinator, 2026-11-03), from the collected metrics. Stdlib only.

    python3 health.py      # rewrites KO/STATUS.md (lines, health table every 25 iterations, reads)

Per line and every 25 cumulative iterations c: training reward (`reward_pass1`, % of attempts
Lean-accepted), entropy (per scored token; think / action), the no-signal share (`groups_skipped`),
the all-fail share (1 - `solved_frac`) and the all-correct share estimated as skipped - all-fail
(with --max_rej 0 a failed group with a rejected draw is not skipped, so this is an estimate).
ALERT lines (stdout): entropy collapse = the last 10 iterations' mean entropy below 0.4 x the
line's first-10 L3 iterations' mean; reward falling = the 10-iteration moving mean of the reward
more than 3 points below its best so far for each of the last 50 iterations.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import statistics as st
from pathlib import Path

KO = Path(os.environ.get("KO", "/scratch/cowifornia/nd-rl/knockout_grpo"))
TAGS = ["full_s0", "d1_s0", "d1_s1", "d01_s0", "d01_s1", "d01_s2", "ko_s0", "ko_s1", "ko_s2"]
HOST = {"full_s0": "serrano", "d1_s0": "serrano", "d1_s1": "serrano", "d01_s0": "temescal",
        "d01_s1": "temescal", "d01_s2": "ynez", "ko_s0": "ynez", "ko_s1": "ynez", "ko_s2": "ynez"}  # fmt: skip  # noqa: E501


def rows(tag: str) -> list[dict]:
    out = []
    for st_ in ("L2", "L3"):
        p = KO / "runs" / f"{tag}_{st_}" / "metrics.jsonl"
        if not p.exists():
            continue
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                r["_stage"] = st_
                out.append(r)
    for i, r in enumerate(out):
        r["_c"] = i + 1
    return out


def mean(xs: list[float]) -> float:
    return st.mean(xs) if xs else float("nan")


def alerts(tag: str, rs: list[dict]) -> list[str]:
    out = []
    l3 = [r for r in rs if r["_stage"] == "L3"]
    if len(l3) >= 20:
        e0 = mean([r["entropy"] for r in l3[:10]])
        e1 = mean([r["entropy"] for r in l3[-10:]])
        if e1 < 0.4 * e0:
            out.append(
                f"ALERT {tag}: entropy collapse ({e1:.3f} vs L3 start {e0:.3f}) at c {rs[-1]['_c']}"
            )
    rw = [100 * r["reward_pass1"] for r in rs]
    ma = [mean(rw[max(0, i - 9) : i + 1]) for i in range(len(rw))]
    if len(ma) >= 60:
        best, below = -1.0, 0
        for v in ma:
            if v > best:
                best = v
            below = below + 1 if v < best - 3.0 else 0
        if below >= 50:
            out.append(
                f"ALERT {tag}: reward falling for {below} iterations (now {ma[-1]:.1f}, best {best:.1f})"  # noqa: E501
            )
    return out


def main() -> None:
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# knockout_grpo status (all machines)\n",
        f"Updated {now} from serrano's collected copy.\n",
    ]
    lines += ["## Lines\n", "| line | host | c | R | last E32 (of 106) | best E32 | stop | done |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    health = ["## Health (every 25 iterations)\n",
              "| line | c | stage | reward % | entropy (think / action) | no-signal % | all-fail % | all-correct % (est.) |",  # noqa: E501
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    al = []
    for tag in TAGS:
        rs = rows(tag)
        cv = KO / "evals" / tag / "curve.jsonl"
        cur = (
            [json.loads(x) for x in cv.read_text().splitlines() if x.strip()] if cv.exists() else []
        )
        stop = (KO / "evals" / tag / "STOP_LINE").exists()
        done = (KO / "evals" / tag / "line_done.json").exists()
        last = cur[-1] if cur else None
        lines.append(
            f"| {tag} | {HOST[tag]} | {len(rs)} | {last['R']:.3f} | {last['e32']:.2f} (c {last['c']}) | "  # noqa: E501
            f"{max(x['e32'] for x in cur):.2f} | {'yes' if stop else ''} | {'yes' if done else ''} |"  # noqa: E501
            if last
            else f"| {tag} | {HOST[tag]} | {len(rs)} | | | | | |"
        )
        for r in rs:
            if r["_c"] % 25 == 0 or r is rs[-1]:
                af = 1 - r["solved_frac"]
                ac = max(0.0, r["groups_skipped"] - af)
                health.append(
                    f"| {tag} | {r['_c']} | {r['_stage']} | {100 * r['reward_pass1']:.1f} | "
                    f"{r['entropy']:.3f} ({r['entropy_think']:.3f} / {r['entropy_action']:.3f}) | "
                    f"{100 * r['groups_skipped']:.1f} | {100 * af:.1f} | {100 * ac:.1f} |"
                )
        al += alerts(tag, rs)
    txt = "\n".join(lines) + "\n\n" + "\n".join(health) + "\n"
    if al:
        txt += "\n## Alerts\n\n" + "\n".join(f"- {a}" for a in al) + "\n"
    (KO / "STATUS.md").write_text(txt)
    for a in al:
        print(a)


if __name__ == "__main__":
    main()
