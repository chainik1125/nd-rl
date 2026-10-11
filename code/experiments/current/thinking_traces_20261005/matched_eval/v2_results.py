"""GRPO recipe v2: results tables and figures, re-runnable as the night's reads land.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.v2_results \
        [--draws 4000] [--v2 RUNS/eval/v2] [--out FIGDIR] [--table PATH]

The v2 runs (`runs/grpo_v2/line.sh`, branch `leon/grpo-v2`; Leon's design answers of 2026-10-09)
train from Stage 1 (S1g3 thinking, T0S no-think) on the length curriculum L2 (10-12 lines) then L3
(13-15), 39 GRPO iterations or one k-32 EI round per stage: 78 iterations = 2 EI rounds in all, the
same training budget as the v1 references (GRPO v1 g2 = rl_hard12 then hard712 a; EI round 2).
"+ verifier" arms sample with the Lean-sound step check (invalid steps redrawn, at most 10 per
attempt); GRPO's also penalises each rejected draft (P 0.1). Thinking models are read at B64, the
no-think models at B0, with the plain arm and (on 13-20, textbook72 and Charles) the guided arm.

Sets: 13-20 (the 986 valid items of `t1320_n1000`), its 16-20-line part (out of distribution now
that training covers 13-15), textbook72, Charles's release 2026-10-02 reduced to the 149 problems
that share no class with any training pool (`charles_eval/build_clean.py`), hand_proved_14 and a
1,000-item sample of 21-30. pass@1 and pass@5 come from `stats.ci` (hierarchical bootstrap over
seeds and problems, 95% intervals, % of the set); differences from `stats.paired_diff` on a shared
problem resample, with seeds paired when both arms have the same Stage-1 seeds (same inits). The v1
reads have k 32 on 13-20 and the v2 reads k 16, so the chain compares pass@1 and pass@5 only.
Rows that are not there yet are skipped with a note.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import valid_1320
from code.ndbench.figures import style as S
from code.thinking import stats

R = Path("/scratch/cowifornia/nd-rl/think/runs")
E = R / "eval"
TTS, MATCHED, CHARLES = E / "tts", E / "matched", E / "charles"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#6b6b6b"
SEEDS = (0, 1)
TOKENS = (1000.0, 2000.0, 4000.0, 8000.0)


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    method: str  # "grpo" or "ei"
    think: bool
    v2: bool = True  # False: a v1 reference read

    @property
    def budget(self) -> int:
        return 64 if self.think else 0


# "redraw": the logical arm redraws a rejected step (at most 10); "stop": the first rejected step
# ends the attempt (--max_rej 0). Both put the 0.1 step penalty on rejected steps. "+ 3-premise
# rows": a third of each iteration's groups from pools/curriculum_3p (goal antecedents moved into
# the premises, `curriculum/uncurry3.py`). "half lr": GRPO lr 7.5e-5 instead of 1.5e-4 in both
# stages (`line.sh` arms ending `_h`). Thinking arms come first (fig_summary splits the rows there).
ARMS = [
    ArmSpec("grpo_think", "GRPO", "grpo", True),
    ArmSpec("grpo_think_h", "GRPO, half lr", "grpo", True),
    ArmSpec("grpo_think_v", "GRPO + verifier (redraw)", "grpo", True),
    ArmSpec("grpo_think_vs", "GRPO + verifier (stop)", "grpo", True),
    ArmSpec("grpo_think_vs_h", "GRPO + verifier (stop), half lr", "grpo", True),
    ArmSpec("ei_think", "EI", "ei", True),
    ArmSpec("ei_think_v", "EI + verifier", "ei", True),
    ArmSpec("grpo_nothink", "GRPO", "grpo", False),
    ArmSpec("grpo_nothink_h", "GRPO, half lr", "grpo", False),
    ArmSpec("grpo_nothink_v", "GRPO + verifier (redraw)", "grpo", False),
    ArmSpec("grpo_nothink_vs", "GRPO + verifier (stop)", "grpo", False),
    ArmSpec("grpo_nothink_vs3", "GRPO + verifier (stop) + 3-premise rows", "grpo", False),
    ArmSpec("ei_nothink", "EI", "ei", False),
    ArmSpec("ei_nothink_v", "EI + verifier", "ei", False),
]
REFS = [
    ArmSpec("grpo_v1", "GRPO v1 (old problems)", "grpo", True, v2=False),
    ArmSpec("ei_v1", "EI v1 (old problems)", "ei", True, v2=False),
    ArmSpec("ei_nothink_v1", "EI v1, no thinking (old problems)", "ei", False, v2=False),
]
SPEC = {a.key: a for a in ARMS + REFS}


@dataclass(frozen=True)
class SetSpec:
    key: str
    title: str
    read: str  # the read's set name in eval/v2
    k: int
    guided: bool  # also read with the guided arm


SETS = [
    SetSpec("t1320", "13–20 lines (986)", "t1320", 16, True),
    SetSpec("t1620", "16–20 lines", "t1320", 16, True),
    SetSpec("tb72", "textbook72", "tb72", 64, True),
    SetSpec("charles149", "Charles149", "charles173", 64, True),
    SetSpec("hand14", "hand-proved 14", "hand14", 64, False),
    SetSpec("t2130", "21–30 lines (1,000)", "t2130", 8, False),
]
SET = {s.key: s for s in SETS}
FIG_SETS = ("t1620", "tb72", "charles149")

COMPARE = [
    ("thinking: GRPO + verifier (redraw) − EI + verifier", "grpo_think_v", "ei_think_v"),
    ("thinking, curriculum only: GRPO − EI", "grpo_think", "ei_think"),
    ("no thinking: GRPO + verifier (redraw) − EI + verifier", "grpo_nothink_v", "ei_nothink_v"),
    ("no thinking, curriculum only: GRPO − EI", "grpo_nothink", "ei_nothink"),
    ("no thinking: GRPO + verifier (redraw) − GRPO", "grpo_nothink_v", "grpo_nothink"),
    ("thinking: GRPO + verifier (redraw) − GRPO", "grpo_think_v", "grpo_think"),
    ("no thinking: GRPO + verifier (stop) − GRPO", "grpo_nothink_vs", "grpo_nothink"),
    ("thinking: GRPO + verifier (stop) − GRPO", "grpo_think_vs", "grpo_think"),
    ("thinking: GRPO + verifier (stop) − EI + verifier", "grpo_think_vs", "ei_think_v"),
    ("no thinking: GRPO + verifier (stop) − EI + verifier", "grpo_nothink_vs", "ei_nothink_v"),
    (
        "no thinking: GRPO + verifier (stop) + 3-premise rows − GRPO + verifier (stop)",
        "grpo_nothink_vs3",
        "grpo_nothink_vs",
    ),
    (
        "no thinking: GRPO + verifier (stop) + 3-premise rows − GRPO",
        "grpo_nothink_vs3",
        "grpo_nothink",
    ),
    (
        "no thinking: GRPO + verifier (stop) + 3-premise rows − EI + verifier",
        "grpo_nothink_vs3",
        "ei_nothink_v",
    ),
    ("thinking: EI + verifier − EI", "ei_think_v", "ei_think"),
    ("no thinking: EI + verifier − EI", "ei_nothink_v", "ei_nothink"),
    ("thinking: GRPO, half lr − GRPO", "grpo_think_h", "grpo_think"),
    (
        "thinking: GRPO + verifier (stop), half lr − GRPO + verifier (stop)",
        "grpo_think_vs_h",
        "grpo_think_vs",
    ),
    ("no thinking: GRPO, half lr − GRPO", "grpo_nothink_h", "grpo_nothink"),
    (
        "thinking: GRPO + verifier (stop), half lr − GRPO, half lr",
        "grpo_think_vs_h",
        "grpo_think_h",
    ),
    ("thinking, curriculum only: GRPO, half lr − EI", "grpo_think_h", "ei_think"),
    ("thinking: GRPO + verifier (stop), half lr − EI", "grpo_think_vs_h", "ei_think"),
    (
        "thinking: GRPO + verifier (stop), half lr − EI + verifier",
        "grpo_think_vs_h",
        "ei_think_v",
    ),
    ("no thinking, curriculum only: GRPO, half lr − EI", "grpo_nothink_h", "ei_nothink"),
]
CHAIN = {
    "grpo": [
        ("v1: old problems", "grpo_v1"),
        ("+ curriculum", "grpo_think"),
        ("+ verifier", "grpo_think_v"),
    ],
    "ei": [
        ("v1: old problems", "ei_v1"),
        ("+ curriculum", "ei_think"),
        ("+ verifier", "ei_think_v"),
    ],
}


def subsets() -> dict[str, set[str] | None]:
    valid = valid_1320()
    t1620 = set()
    with open(MATCHED / "sets/t1320_n1000.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["name"] in valid and int(r["reference_lines"]) >= 16:
                t1620.add(r["name"])
    with open(R / "pools/charles_1002/charles_clean.jsonl") as f:
        ch = {json.loads(line)["name"] for line in f}
    return {
        "t1320": valid,
        "t1620": t1620,
        "tb72": None,
        "charles149": ch,
        "hand14": None,
        "t2130": None,
    }


def rows_path(arm: ArmSpec, s: SetSpec, seed: int, ra: str, v2: Path) -> Path | None:
    """The rows file of one (arm, set, seed, read arm), or None where no such read exists."""
    b = arm.budget
    if arm.v2:
        return v2 / f"{s.read}_{arm.key}_s{seed}_{ra}_k{s.k}" / f"rows_{ra}_B{b}.jsonl"
    if ra != "plain":
        return None
    sfx = "" if seed == 0 else f"s{seed}"
    if arm.key == "grpo_v1":
        return {
            "t1320": TTS / f"t1320_grpo_g2s{seed}_k32" / "rows_plain_B64.jsonl",
            "tb72": TTS / f"tb72_grpo_g2s{seed}_k64" / "rows_plain_B64.jsonl",
            "charles173": CHARLES / f"ch149_grpo_g2s{seed}" / "rows_plain_B64.jsonl",
        }.get(s.read)
    kind = "think" if arm.think else "nothink"
    tb_s0 = MATCHED / ("tb72_eiS1g3_h712_r2" if arm.think else "tb72_eiT0S_h712_r2")
    return {
        "t1320": TTS
        / f"t1320_{kind}_r2{sfx}_k{32 if arm.think else 64}"
        / f"rows_plain_B{b}.jsonl",
        "tb72": (tb_s0 if seed == 0 else TTS / f"tb72_{kind}_r2s{seed}_k64")
        / f"rows_plain_B{b}.jsonl",
        "charles173": CHARLES / f"ch149_{kind}_r2s{seed}" / f"rows_plain_B{b}.jsonl",
    }.get(s.read)


class Loader:
    def __init__(self, v2: Path, keep: dict[str, set[str] | None]) -> None:
        self.v2, self.keep = v2, keep
        self.missing: list[str] = []
        self._cache: dict[tuple[str, str, tuple[int, ...], str], stats.Arm | None] = {}

    def seeds(self, arm: str, s: str, ra: str) -> list[int]:
        out = []
        for seed in SEEDS:
            p = rows_path(SPEC[arm], SET[s], seed, ra, self.v2)
            if p is not None and p.exists():
                out.append(seed)
            elif p is not None:
                self.missing.append(str(p))
        return out

    def arm(self, arm: str, s: str, seeds: Sequence[int], ra: str = "plain") -> stats.Arm | None:
        key = (arm, s, tuple(seeds), ra)
        if key not in self._cache:
            paths = [rows_path(SPEC[arm], SET[s], seed, ra, self.v2) for seed in seeds]
            if not seeds or any(p is None or not p.exists() for p in paths):
                self._cache[key] = None
            else:
                a = stats.load_arm(
                    f"{arm} {s} s{','.join(map(str, seeds))}", [str(p) for p in paths]
                )
                names = self.keep[s]
                self._cache[key] = (
                    a.subset(sorted(set(a.names) & names)) if names is not None else a
                )
        return self._cache[key]


def fmt(e: stats.Estimate, sign: bool = False) -> str:
    f = "+.1f" if sign else ".1f"
    return f"{100 * e.est:{f}} [{100 * e.lo:{f}}, {100 * e.hi:{f}}]"


def estimate(
    ld: Loader, arm: str, s: str, seeds: Sequence[int], ra: str, k: int, draws: int
) -> stats.Estimate | None:
    a = ld.arm(arm, s, seeds, ra)
    return None if a is None else stats.ci(a, k, draws=draws)


def diff(
    ld: Loader, a: str, b: str, s: str, seeds: Sequence[int], ra: str, k: int, draws: int
) -> stats.Estimate | None:
    x, y = ld.arm(a, s, seeds, ra), ld.arm(b, s, seeds, ra)
    if x is None or y is None:
        return None
    return stats.paired_diff(x, y, k, paired_seeds=True, draws=draws)


def common_seeds(ld: Loader, arms: Sequence[str], s: str, ra: str = "plain") -> list[int]:
    have = [set(ld.seeds(a, s, ra)) for a in arms]
    return sorted(set.intersection(*have)) if have else []


def tables(ld: Loader, draws: int) -> list[str]:
    md = [
        "# GRPO recipe v2: results (generated by `v2_results.py`; do not edit by hand)",
        "",
        "pass@1 / pass@5 in % of the set, 95% hierarchical-bootstrap intervals (seeds and",
        "problems). v2 = length curriculum (10–12 then 13–15 lines), 78 GRPO iterations or 2 EI",
        "rounds; v1 references = GRPO g2 / EI round 2 on rl_hard12 then hard712 a (same budget).",
        "Plain reads unless marked guided. Thinking at B64, no thinking at B0.",
        "",
        "## 1. Every arm, plain reads (all available seeds pooled)",
        "",
        "| arm | "
        + " | ".join(f"{SET[s].title}: pass@1 / pass@5" for s in [x.key for x in SETS])
        + " |",
        "|---|" + "---|" * len(SETS),
    ]
    for arm in ARMS + REFS:
        cells = []
        for s in SETS:
            seeds = ld.seeds(arm.key, s.key, "plain")
            p1 = estimate(ld, arm.key, s.key, seeds, "plain", 1, draws)
            p5 = estimate(ld, arm.key, s.key, seeds, "plain", 5, draws)
            cells.append(
                "—"
                if p1 is None or p5 is None
                else f"{fmt(p1)} / {fmt(p5)} (s{','.join(map(str, seeds))})"
            )
        tag = "thinking" if arm.think else "no thinking"
        md.append(f"| {arm.label}, {tag} | " + " | ".join(cells) + " |")
    md += ["", "## 2. Per seed (plain and guided reads)", ""]
    md += ["| arm | seed | read | set | n | pass@1 | pass@5 |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS + REFS:
        for s in SETS:
            for ra in ("plain", "logical") if s.guided and arm.v2 else ("plain",):
                for seed in ld.seeds(arm.key, s.key, ra):
                    a = ld.arm(arm.key, s.key, [seed], ra)
                    if a is None:
                        continue
                    p1, p5 = stats.ci(a, 1, draws=draws), stats.ci(a, 5, draws=draws)
                    read = "guided" if ra == "logical" else "plain"
                    md.append(
                        f"| {arm.label}{'' if arm.think else ', no thinking'} | {seed} | {read} | "
                        f"{s.title} | {a.T} | {fmt(p1)} | {fmt(p5)} |"
                    )
    md += ["", "## 3. Paired differences at the same training budget (seeds paired by init)", ""]
    md += ["| comparison | read | set | seeds | Δ pass@1 | Δ pass@5 |", "|---|---|---|---|---|---|"]
    for title, a, b in COMPARE:
        for s in SETS:
            for ra in ("plain", "logical") if s.guided else ("plain",):
                seeds = common_seeds(ld, [a, b], s.key, ra)
                for group in [[x] for x in seeds] + ([seeds] if len(seeds) > 1 else []):
                    d1 = diff(ld, a, b, s.key, group, ra, 1, draws)
                    d5 = diff(ld, a, b, s.key, group, ra, 5, draws)
                    if d1 is None or d5 is None:
                        continue
                    read = "guided" if ra == "logical" else "plain"
                    md.append(
                        f"| {title} | {read} | {SET[s.key].title} | {','.join(map(str, group))} | "
                        f"{fmt(d1, True)} | {fmt(d5, True)} |"
                    )
    md += ["", "## 4. What each change buys (thinking, plain reads, seeds paired)", ""]
    md += ["| method | change | set | seeds | Δ pass@1 | Δ pass@5 |", "|---|---|---|---|---|---|"]
    for method, steps in CHAIN.items():
        for (_, prev), (name, cur) in zip(steps, steps[1:], strict=False):
            for s in FIG_SETS + ("t1320",):
                seeds = common_seeds(ld, [prev, cur], s)
                if not seeds:
                    continue
                d1 = diff(ld, cur, prev, s, seeds, "plain", 1, draws)
                d5 = diff(ld, cur, prev, s, seeds, "plain", 5, draws)
                if d1 is None or d5 is None:
                    continue
                md.append(
                    f"| {method.upper()} | {name} (vs {SPEC[prev].label}) | {SET[s].title} | "
                    f"{','.join(map(str, seeds))} | {fmt(d1, True)} | {fmt(d5, True)} |"
                )
    md += ["", "## 5. GRPO − EI at matched sampled tokens per problem (plain reads)", ""]
    md += ["‡: outside one arm's sampled range (not drawn in figures).", ""]
    md += ["| comparison | set | seeds | " + " | ".join(f"{x / 1000:g}k" for x in TOKENS) + " |"]
    md += ["|---|---|---|" + "---|" * len(TOKENS)]
    for title, a, b in COMPARE:
        for s in FIG_SETS:
            seeds = common_seeds(ld, [a, b], s)
            x, y = ld.arm(a, s, seeds), ld.arm(b, s, seeds)
            if x is None or y is None:
                continue
            es = stats.paired_diff_at_tokens(x, y, list(TOKENS), paired_seeds=True, draws=draws)
            cells = [fmt(e, True) + ("" if e.in_range else " ‡") for e in es]
            md.append(
                f"| {title} | {SET[s].title} | {','.join(map(str, seeds))} | "
                + " | ".join(cells)
                + " |"
            )
    md += by_premises(ld, draws)
    md += [
        "",
        "## Notes",
        "",
        "- 12 of the 986 13–20 items contain a part (main proof or lemma of the composite)",
        "  that is, up to renaming or currying, a row of L2 / L3; dropping them moves no v2 arm's",
        "  pass@1 by more than 0.2 points (checked 2026-10-10; 8 of the 1,000 21–30 items too).",
        "- The v1 references trained on hard712 sample a, which holds curried twins of 3",
        "  textbook72 problems (found 2026-10-09 by the curriculum build), so textbook72 slightly",
        "  favours v1.",
        "  The v2 pools exclude every class shared with textbook72 and Charles's release.",
        "- The v1 Charles149 points are the reads of `followup_20261007/charles_reads.sh` (k 64).",
        "- `groups_skipped` means different things with and without the step penalty; compare the",
        "  arms' training signal with the dashboards' `with_grad` and `by_pass` panels instead.",
    ]
    if ld.missing:
        md += ["", f"Rows not there yet ({len(set(ld.missing))}):", ""]
        md += [f"- `{p}`" for p in sorted(set(ld.missing))]
    return md


def n_prem(prompt: str) -> int:
    """The premises of `THM p1 , p2 SEQ goal PRF` (commas at depth 0)."""
    t = prompt.split()
    body = t[1 : t.index("SEQ")]
    depth, n = 0, int(bool(body))
    for x in body:
        depth += (x == "(") - (x == ")")
        n += x == "," and depth == 0
    return n


def by_premises(ld: Loader, draws: int) -> list[str]:
    """13–20, plain reads, split by premise count (the v2 pools have 0–2 premises, 78% of 13–20
    has 3): pass@1 on each part and, on 3 premises, the share of attempts that end at a premise
    line out of place (`S:PR position`: premise lines come first, as h1, h2, h3 in order)."""
    with open(MATCHED / "sets/t1320_n1000.jsonl") as f:
        prem = {r["name"]: n_prem(r["prompt"]) for r in map(json.loads, f)}
    valid = ld.keep["t1320"] or set(prem)
    few = sorted(n for n, p in prem.items() if p <= 2 and n in valid)
    three = sorted(n for n, p in prem.items() if p >= 3 and n in valid)
    md = ["", "## 6. 13–20 by premise count (plain reads; the v2 pools have 0–2 premises)", ""]
    md += [
        f"| arm | seeds | 0–2 premises ({len(few)}): pass@1 | 3 premises ({len(three)}): pass@1 | "
        "3 premises: attempts ending at `S:PR position` |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS + REFS:
        seeds = ld.seeds(arm.key, "t1320", "plain")
        a = ld.arm(arm.key, "t1320", seeds)
        if a is None:
            continue
        cells = [
            fmt(stats.ci(a.subset(sorted(set(a.names) & set(g))), 1, draws=draws))
            for g in (few, three)
        ]
        att = pr = 0
        for seed in seeds:
            p = rows_path(arm, SET["t1320"], seed, "plain", ld.v2)
            assert p is not None
            with open(p) as f:
                for r in map(json.loads, f):
                    if r["name"] in three:
                        st = r.get("status") or []
                        att += len(st)
                        pr += sum(x == "fail:S:PR position" for x in st)
        share = f"{100 * pr / att:.0f}%" if att else "—"
        md.append(
            f"| {arm.label}{'' if arm.think else ', no thinking'} | {','.join(map(str, seeds))} | "
            f"{cells[0]} | {cells[1]} | {share} |"
        )
    return md


def _seed_means(ld: Loader, arm: str, s: str, k: int) -> tuple[list[float], float | None]:
    vals = []
    for seed in ld.seeds(arm, s, "plain"):
        a = ld.arm(arm, s, [seed])
        if a is not None:
            vals.append(float(a.values(k).mean()))
    return vals, (float(np.mean(vals)) if vals else None)


def fig_summary(ld: Loader, out: Path) -> None:
    rows = [(a, f"{a.label}{'' if a.think else ' (no thinking)'}") for a in ARMS]
    n_think = sum(a.think for a in ARMS)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 0.6 + 0.17 * len(rows), layout="constrained")
        axs = fig.subplots(1, len(FIG_SETS), sharey=True, squeeze=False)[0]
        rng = np.random.default_rng(0)
        for j, s in enumerate(FIG_SETS):
            ax: Axes = axs[j]
            for i, (arm, _) in enumerate(rows):
                y = len(rows) - 1 - i
                col = BLUE if arm.method == "grpo" else ORANGE
                means = {}
                for k, mk in ((1, "o"), (5, "s")):
                    vals, m = _seed_means(ld, arm.key, s, k)
                    for v in vals:
                        ax.plot(
                            [v],
                            [y + rng.uniform(-0.12, 0.12)],
                            ls="none",
                            marker=mk,
                            ms=2.6,
                            mfc=S.WHITE,
                            mec=col,
                            mew=0.6,
                            zorder=2,
                        )
                    if m is not None:
                        ax.plot(
                            [m],
                            [y],
                            ls="none",
                            marker=mk,
                            ms=4.2,
                            mfc=col,
                            mec=S.WHITE,
                            mew=0.5,
                            zorder=3,
                        )
                        means[k] = m
                if len(means) == 2:
                    ax.plot([means[1], means[5]], [y, y], color=col, lw=1.0, alpha=0.5, zorder=1)
                if not means:
                    ax.text(
                        0.02,
                        y,
                        "pending",
                        transform=ax.get_yaxis_transform(),
                        va="center",
                        fontsize=S.FS_TICK,
                        color=GRAY,
                    )
            ax.axhline(len(rows) - n_think - 0.5, color=S.GRID, lw=0.8)  # thinking | no thinking
            S.panel_title(ax, "abc"[j], SET[s].title)
            ax.set_xlim(0, 1)
            S.pct_axis(ax, "x", nbins=2)
            S.light_grid(ax, "x")
            ax.set_ylim(-0.6, len(rows) - 0.4)
        axs[0].set_yticks(range(len(rows)), [lab for _, lab in rows][::-1])
        fig.supxlabel(
            "problems solved (plain sampling; seeds hollow, mean filled)", fontsize=S.FS_LABEL
        )
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=4,
                mfc=GRAY,
                mec=S.WHITE,
                label="first try (pass@1)",
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="s",
                ms=4,
                mfc=GRAY,
                mec=S.WHITE,
                label="within 5 tries (pass@5)",
            ),
            Line2D([], [], color=BLUE, lw=2, label="GRPO"),
            Line2D([], [], color=ORANGE, lw=2, label="EI"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
        S.save(fig, out / "fig_v2_summary")
        S.close(fig)


def _seed_value(ld: Loader, arm: str, s: str, seed: int, k: int) -> float | None:
    a = ld.arm(arm, s, [seed]) if seed in ld.seeds(arm, s, "plain") else None
    return None if a is None else float(a.values(k).mean())


def fig_chain(ld: Loader, out: Path) -> None:
    """One line per (method, Stage-1 seed) through the three steps, so a step that only one seed
    has does not shift a seed-mean line."""
    looks = {"grpo": (BLUE, "o", "GRPO"), "ei": (ORANGE, "s", "EI")}
    seed_ls = {0: "-", 1: (0, (1.2, 1.4))}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 3.6, layout="constrained")
        axs = fig.subplots(2, len(FIG_SETS), sharex=True, squeeze=False)
        for r, k in enumerate((1, 5)):
            for j, s in enumerate(FIG_SETS):
                ax: Axes = axs[r][j]
                for method, steps in CHAIN.items():
                    col, mk, _ = looks[method]
                    dx = -0.05 if method == "grpo" else 0.05
                    for seed in SEEDS:
                        pts = [
                            (x + dx, v)
                            for x, (_, arm) in enumerate(steps)
                            if (v := _seed_value(ld, arm, s, seed, k)) is not None
                        ]
                        if not pts:
                            continue
                        ax.plot(
                            [p[0] for p in pts],
                            [p[1] for p in pts],
                            color=col,
                            ls=seed_ls[seed],
                            lw=1.2,
                            marker=mk,
                            ms=3.6,
                            mfc=col if seed == 0 else S.WHITE,
                            mec=col if seed else S.WHITE,
                            mew=0.8 if seed else 0.5,
                            zorder=3,
                        )
                S.panel_title(
                    ax,
                    "abcdef"[3 * r + j],
                    f"{SET[s].title}, {'first try' if k == 1 else 'within 5 tries'}",
                )
                ax.set_xlim(-0.4, 2.4)
                ax.set_xticks([0, 1, 2], ["v1", "+ curriculum", "+ verifier"])
                S.pct_axis(ax)
                S.light_grid(ax)
        axs[0][0].set_ylabel("pass@1")
        axs[1][0].set_ylabel("pass@5")
        fig.supxlabel(
            "thinking models at 78 GRPO iterations = 2 EI rounds. v1: old problems (rl_hard12,\n"
            "hard712 a); + curriculum: 10–12 then 13–15 lines; + verifier: guided rollouts",
            fontsize=S.FS_LABEL,
        )
        handles = [
            Line2D([], [], color=c, marker=m, ms=4, lw=1.2, label=lab)
            for c, m, lab in looks.values()
        ] + [
            Line2D([], [], color=GRAY, ls=seed_ls[0], lw=1.2, label="seed 0"),
            Line2D([], [], color=GRAY, ls=seed_ls[1], lw=1.2, label="seed 1"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
        S.save(fig, out / "fig_v2_chain")
        S.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=stats.DRAWS)
    ap.add_argument("--v2", default=str(E / "v2"), help="the v2 reads (a copy for tests)")
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument(
        "--table",
        default="code/experiments/current/thinking_traces_20261005/matched_eval/v2_results.md",
    )
    a = ap.parse_args(argv)
    ld = Loader(Path(a.v2), subsets())
    md = tables(ld, a.draws)
    Path(a.table).write_text("\n".join(md) + "\n")
    out = Path(a.out)
    fig_summary(ld, out)
    fig_chain(ld, out)
    n_missing = len(set(ld.missing))
    print(f"wrote {a.table}, {out}/fig_v2_summary and fig_v2_chain")
    print(f"{n_missing} expected rows files not there yet")


if __name__ == "__main__":
    main()
