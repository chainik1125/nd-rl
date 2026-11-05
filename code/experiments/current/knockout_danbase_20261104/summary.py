"""Fill the generated sections of the summary README from numbers.json.

    python3 summary.py      # rewrites the text between <!-- AUTO:name --> and <!-- /AUTO:name -->

Sections: `lines` (one row per line), `dan` (his EI ladders on the same bases), `compute`.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = (HERE / "../../../../experiment-summaries/2026-11-04-grpo-on-dan-knockout-base"
           / "README.md").resolve()  # fmt: skip
ROLE = {"primary": "**primary**: knockout base, his 4,495 targets",
        "secondary": "secondary: knockout base, our σ-rich pool",
        "reference": "reference: full K12 base, his targets",
        "dose01": "**dose 0.1 %**: his 0.1 % σ base, his 4,495 targets",
        "dose1": "**dose 1 %**: his 1 % σ base, his 4,495 targets"}  # fmt: skip


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.2f} %" if d else "—"


def lines_table(N: dict) -> str:
    out = ["| line | arm | base (first 32 of the 106) | his measure, by matched round | reached his bar (14)? "
           "| x_dan at the end (his round-16 x) | ended | GPU-h | σ in accepted training proofs: all / last 50 iterations |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t, v in N["lines"].items():
        base = next((c for c in v["curve"] if c["it"] == 0), {})
        cr = v["cross_dan"]
        if cr is None:
            reached = f"no: {max((m['cum'] for m in v['matched']), default=0)} of 106 by x_dan {v['x_dan']:.2f}"
        elif cr["round"] == 1:
            reached = f"at the bar before RL ({cr['cum']} at round 1)"
        else:
            reached = (f"**yes**: round {cr['round']}, x_dan {cr['x_dan']:.2f} (x_ours "
                       f"{cr['x_ours']:.3f})")  # fmt: skip
        sg = v.get("sigma") or []
        tot, s_ = sum(x["accepted"] for x in sg), sum(x["sigma"] for x in sg)
        tail = sg[-50:]
        tn, ts = sum(x["accepted"] for x in tail), sum(x["sigma"] for x in tail)
        why = v.get("stop_reason") or v["status"]
        why = re.sub(r" \(\{.*\}\)", "", why)
        out.append(f"| {t} | {ROLE[v.get('dose') if v['arm'] == 'dose' else v['arm']]} | {base.get('first32', '—')} | "
                   f"{', '.join(str(m['cum']) for m in v['matched']) or '—'} | {reached} | "
                   f"{v['x_dan']:.2f} ({v['x_cap']:.2f}) | {why} | {v['gpuh']:.1f} | "
                   f"{_pct(s_, tot)} / {_pct(ts, tn)} |")  # fmt: skip
    return "\n".join(out)


def dan_table(N: dict) -> str:
    out = [
        "| his ladder | held-out count by round (rounds 1–16) | x at round 16 | crossing |",
        "|---|---|---|---|",
    ]
    for o in N["dan"]:
        if o["name"] in (
            "la_ko_s0",
            "la_ko_s1",
            "la_ko_s2",
            "la_full_s3",
            "la_dose01_s0",
            "la_dose01_s1",
            "la_dose01_s2",
            "la_dose1_s0",
            "la_dose1_s1",
            "la_dose1_s2",
        ):
            ys = [r["y"] for r in o["rounds"][:16]]
            c = o["cross"]
            cs = (f"round {c['round']}, x {c['x_dan']:.2f}" if c else "not reached by round 16"
                  if max(ys) < 14 else "after round 16")  # fmt: skip
            out.append(f"| {o['name']} | {', '.join(map(str, ys))} | {o['rounds'][15]['x_dan']:.2f} "
                       f"| {cs} |")  # fmt: skip
    return "\n".join(out)


def compute(N: dict) -> str:
    g = sum(v["gpuh"] for v in N["lines"].values())
    lost = {t: v["lost"] for t, v in N["lines"].items() if v["lost"]}
    return (
        f"- Lines: {g:.1f} GPU-h in all (segment wall time, reads included), plus about 0.4 "
        f"GPU-h for the base check and 0.2 for the pilot.\n"
        f"- Iterations lost to deaths and moves (superseded rows): {lost or 'none'}."
    )


def main() -> None:
    N = json.loads((HERE / "numbers.json").read_text())
    s = SUMMARY.read_text()
    for name, text in (("lines", lines_table(N)), ("dan", dan_table(N)), ("compute", compute(N))):
        pat = re.compile(rf"(<!-- AUTO:{name} -->\n)(?:.*?\n)?(<!-- /AUTO:{name} -->)", re.S)
        assert pat.search(s), name
        s = pat.sub(lambda m, t=text: m.group(1) + t + "\n" + m.group(2), s)
    SUMMARY.write_text(s)
    print(f"updated {SUMMARY}")


if __name__ == "__main__":
    main()
