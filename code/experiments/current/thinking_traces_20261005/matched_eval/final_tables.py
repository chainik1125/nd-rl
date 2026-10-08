"""Final RL comparison tables for the 2026-10-07 runs: pass@1 and pass@5 per arm with 95% intervals,
and paired differences on the same theorems.

    code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.final_tables [--out PATH]

Reads the stored eval rows under /scratch/cowifornia/nd-rl/think/runs/eval/matched (CPU only).
Intervals are the hierarchical bootstrap of `code.thinking.stats` (theorems, and seeds when an arm
pools two seeds). The 13-20 reads use the 986 valid items (see `figs.valid_1320`). Thinking arms are
compared at B 256, their full budget, which was fixed before any eval. B 64 rows are reported
but not used for the headline differences. Reads that do not exist yet are skipped.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import M, valid_1320
from code.thinking import stats

SETS = [("h12", "hard12 test (1,000)"), ("tb72", "textbook72 (72)"), ("t1320", "13-20 lines (986)")]
# (label, dir tag, step tags, budgets)
ARMS = [
    ("EI no-think", "eiT0", ("r1", "r2"), (0,)),
    ("EI effort levels", "eiT1v1e", ("r1", "r2"), (0, 256)),
    ("GRPO no-think, seed 0", "grpoT0", ("it39", "it78"), (0,)),
    ("GRPO no-think, seed 1", "grpoT0s1", ("it39", "it78"), (0,)),
    ("GRPO effort levels, seed 0", "grpoT1v1e", ("it39", "it78"), (0, 64, 256)),
    ("GRPO effort levels, seed 1", "grpoT1v1es1", ("it39", "it78"), (0, 64, 256)),
    ("GRPO one budget", "T1v1_b256", ("it39", "it78"), (256,)),
    ("GRPO one budget + token cost", "T1v1_b256_lp02", ("it39", "it78"), (256,)),
    ("GRPO graded traces", "T1v1g4e", ("it39", "it78"), (0, 64, 256)),
    ("GRPO graded + LCPO", "T1v1g4e_lcpo", ("it39", "it78"), (0, 64, 256)),
]
# thinking arms (seed 0 unless named) compared with no-think GRPO of the same seed
VS_NOTHINK = [
    ("effort levels, seed 0", ["grpoT1v1e"], ["grpoT0"]),
    ("effort levels, seed 1", ["grpoT1v1es1"], ["grpoT0s1"]),
    ("effort levels, seeds 0+1", ["grpoT1v1e", "grpoT1v1es1"], ["grpoT0", "grpoT0s1"]),
    ("one budget", ["T1v1_b256"], ["grpoT0"]),
    ("one budget + token cost", ["T1v1_b256_lp02"], ["grpoT0"]),
    ("graded traces", ["T1v1g4e"], ["grpoT0"]),
    ("graded + LCPO", ["T1v1g4e_lcpo"], ["grpoT0"]),
]
TOKENS = (1000, 2000)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ival(e: stats.Estimate, signed: bool = False) -> str:
    f = (lambda v: f"{100 * v:+.1f}") if signed else pct
    return f"{f(e.est)} [{f(e.lo)}, {f(e.hi)}]"


class Loader:
    """One arm from one rows file per seed, filtered to `keep` names on the 13-20 set."""

    def __init__(self, keep: set[str], tmp: Path) -> None:
        self.keep, self.tmp, self.n = keep, tmp, 0

    def arm(self, s: str, tags: Sequence[str], step: str, b: int) -> stats.Arm | None:
        paths = [M / f"{s}_{t}_{step}" / f"rows_plain_B{b}.jsonl" for t in tags]
        if not all(p.exists() for p in paths):
            return None
        if s == "t1320":
            out = []
            for p in paths:
                self.n += 1
                q = self.tmp / f"{self.n}_{p.parent.name}_{p.name}"
                with open(p) as f, open(q, "w") as g:
                    g.writelines(x for x in f if json.loads(x)["name"] in self.keep)
                out.append(q)
            paths = out
        return stats.load_arm("+".join(tags) + f"_{step}_B{b}", [str(p) for p in paths])


def per_arm(ld: Loader) -> list[str]:
    out = ["## Per arm", ""]
    for s, title in SETS:
        out += [
            f"### {title}",
            "",
            "| arm | step | budget | pass@1 | pass@5 | tokens / attempt |",
            "|---|---|---|---|---|---|",
        ]
        for lab, tag, steps, budgets in ARMS:
            for step in steps:
                for b in budgets:
                    a = ld.arm(s, [tag], step, b)
                    if a is None:
                        continue
                    tok = f"{a.tokens.sum() / a.n.sum():.0f}" if a.tokens is not None else "-"
                    p1, p5 = stats.ci(a, 1), stats.ci(a, 5)
                    out.append(f"| {lab} | {step} | B {b} | {ival(p1)} | {ival(p5)} | {tok} |")
        out.append("")
    return out


def diff_rows(a: stats.Arm, b: stats.Arm) -> str:
    d1, d5 = stats.paired_diff(a, b, 1), stats.paired_diff(a, b, 5)
    return f"{ival(d1, True)} | {ival(d5, True)} | {d1.n_pos}/{d1.n_neg}"


def grpo_vs_ei(ld: Loader) -> list[str]:
    out = [
        "## GRPO minus EI at matched attempts (seed 0; it39 vs r1, it78 vs r2)",
        "",
        "| set | arm | point | Δ pass@1 | Δ pass@5 | theorems + / − (pass@1) |",
        "|---|---|---|---|---|---|",
    ]
    for s, title in SETS:
        for lab, g, e, b in [
            ("no-think", "grpoT0", "eiT0", 0),
            ("effort levels, B 256", "grpoT1v1e", "eiT1v1e", 256),
        ]:
            for it, r in [("it39", "r1"), ("it78", "r2")]:
                x, y = ld.arm(s, [g], it, b), ld.arm(s, [e], r, b)
                if x is None or y is None:
                    continue
                out.append(f"| {title} | {lab} | {it} − {r} | {diff_rows(x, y)} |")
    return [*out, ""]


def vs_nothink(ld: Loader) -> list[str]:
    out = [
        "## Thinking (B 256) minus no-think GRPO, same seed and iteration",
        "",
        "Seeds 0+1 pools two independent runs per side, so seeds are resampled too.",
        "",
        "| set | thinking arm | step | Δ pass@1 | Δ pass@5 | theorems + / − (pass@1) |",
        "|---|---|---|---|---|---|",
    ]
    for s, title in SETS:
        for lab, think, base in VS_NOTHINK:
            for it in ("it39", "it78"):
                x, y = ld.arm(s, think, it, 256), ld.arm(s, base, it, 0)
                if x is None or y is None:
                    continue
                out.append(f"| {title} | {lab} | {it} | {diff_rows(x, y)} |")
    return [*out, ""]


def at_tokens(ld: Loader) -> list[str]:
    out = [
        "## Thinking (B 256) minus no-think at matched sampled tokens per problem",
        "",
        "pass@k is read at the k where each arm has spent the target tokens per problem (log-k "
        "interpolation). 'clamped' marks a target outside an arm's sampled range.",
        "",
        "| set | thinking arm | step | " + " | ".join(f"Δ at {t} tokens" for t in TOKENS) + " |",
        "|---|---|---|" + "---|" * len(TOKENS),
    ]
    for s, title in SETS:
        for lab, think, base in VS_NOTHINK:
            for it in ("it39", "it78"):
                x, y = ld.arm(s, think, it, 256), ld.arm(s, base, it, 0)
                if x is None or y is None or x.tokens is None or y.tokens is None:
                    continue
                es = stats.paired_diff_at_tokens(x, y, TOKENS)
                cells = [ival(e, True) + ("" if e.in_range else " (clamped)") for e in es]
                out.append(f"| {title} | {lab} | {it} | " + " | ".join(cells) + " |")
    return [*out, ""]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="code/experiments/current/thinking_traces_20261005/matched_eval/final_tables.md",
    )
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as td:
        ld = Loader(valid_1320(), Path(td))
        lines = [
            "---",
            "written_on: 2026-10-08",
            "written_by: agent:claude",
            "---",
            "",
            "# Final RL tables (generated by `final_tables.py`; do not edit by hand)",
            "",
            "pass@k in % with 95% hierarchical-bootstrap intervals (4,000 draws). Differences are "
            "paired on theorems; '+ / −' counts theorems where the first arm's pass@1 is higher / "
            "lower.",
            "",
            *grpo_vs_ei(ld),
            *vs_nothink(ld),
            *at_tokens(ld),
            *per_arm(ld),
        ]
    Path(a.out).write_text("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
