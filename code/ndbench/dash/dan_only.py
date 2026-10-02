"""The Dan-only results page: Dan Pandori's twelve best-state checkpoints on the ndbench panel,
laid out like a paper's results section (`dash --summary --dan-only`, writing
`dash-dir/index.html`; `figures --dan-only` writes the same figures as PDF/SVG/PNG).

Leon (2026-10-02): the shared page shows only the new run, none of our own base/rl/armC results,
and it is the only page of the site (no links to other pages). So nothing here reads, draws,
legends or tabulates a base/rl/armC run: every chart is built for Dan's labels only
(`analysis.models_registry`'s `dan12`/`dan6` families).

Structure: title, a three-sentence lede, a Figure 1 overview grid (one panel per benchmark's
headline fz vs t1 comparison), Section 1 "Eliciting proofs", Section 2 "Internals", Limitations.
Every caption says what is plotted, n, the CI method (or "point estimates"), the judge and the
unseen filter where one applies, then a one-sentence takeaway computed from the plotted numbers.

Numbers come from the audited Dan views wherever they exist -- `dash.dan_pages`'s builders
(called with `families=("dan12", "dan6")`, so "ours" is never computed) and `analysis.dan_heldout`
/ `analysis.judges` -- and from `analysis.dan_views` for the views this page adds (textbook72
counts, padding by pad type, textbook_law unseen-only, steering, interpolation, probes,
likelihood, KL). Each chart's numbers are computed once into a `ChartData`; the figure
(`figures/dan_only.py`), its caption and its data table read that object.
"""

from __future__ import annotations

import html
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..analysis import dan_heldout as heldout_view
from ..analysis import dan_views as dv
from ..analysis import judges as judges_view
from ..analysis import models_registry as mr
from ..analysis._common import (
    default_version,
    items_by_pid,
    latest_run_id,
    problems_by_pid,
    run_records,
)
from ..core.io import read_jsonl
from ..core.types import JSON
from ..figures import dan_only as F
from ..metrics import pass_at_k
from ..store import Store
from . import dan_pages, summary
from . import primitives as P
from .chart_data import ChartData
from .chart_data import missing_chart as _missing

FAMS: tuple[str, ...] = ("dan12", "dan6")
JUDGE = dv.DAN_JUDGE_ID
PAIR: tuple[str, str] = ("dan12-fz-s0", "dan12-t1-s0")
#: Item-bootstrap replicates for the textbook curves (B = 1000; the CI the caption names).
B_TEXTBOOK = 1000
_BOOT = "95% item-level bootstrap"


def _pct(x: float | None, d: int = 1) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{100 * x:.{d}f}%"


def _pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def _run_k(store: Store, suite: str, model: str) -> int | None:
    version = default_version(store, suite=suite, models=[model], judge_id=JUDGE) or "v1"
    run_id = latest_run_id(store, suite=suite, model=model, version=version, judge_id=JUDGE)
    if run_id is None:
        return None
    for r in run_records(store, suite=suite, model=model, judge_id=JUDGE):
        if r.get("run_id") == run_id:
            return int(cast(int, r.get("k") or 0)) or None
    return None


def _seen_note(store: Store, suite: str, family: str = "dan12") -> tuple[int, int] | None:
    version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE) or "v1"
    counts = dv.seen_counts(store, family, [(suite, version)])
    return (counts[0][3], counts[0][2]) if counts else None


def _curve_at(curve: Sequence[tuple[float, float]], x: float) -> float | None:
    return next((y for xx, y in curve if xx == x), None)


# ========================================================================= 2. textbook


def _c_textbook(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_textbook", "Pass@k on textbook theorems, by cap and seed"
    base = dan_pages._data_dan_yue(store, reg, families=FAMS, B=B_TEXTBOOK)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no textbook_dev runs for Dan's models yet")
    panels = {
        f: p
        for f, p in cast("dict[str, ChartData]", base.extra["panels"]).items()
        if f in FAMS and p.series
    }
    curves = {s.name: s.pts for p in panels.values() for s in p.series}
    n_items = next((n for n in _textbook_n(store)), "?")
    pairs = []
    for fam in FAMS:
        for seed in range(3):
            fz, t1 = curves.get(f"{fam}-fz-s{seed}"), curves.get(f"{fam}-t1-s{seed}")
            if fz and t1:
                common = sorted(set(dict(fz)) & set(dict(t1)))
                pairs.append(all(dict(t1)[k] >= dict(fz)[k] for k in common))
    bits = []
    for fam in FAMS:
        fz, t1 = curves.get(f"{fam}-fz-s0"), curves.get(f"{fam}-t1-s0")
        if not fz or not t1:
            continue
        k_hi = max(k for k in set(dict(fz)) & set(dict(t1)) if k <= 256)
        a1, b1 = dict(fz)[1.0], dict(t1)[1.0]
        ah, bh = dict(fz)[k_hi], dict(t1)[k_hi]
        bits.append(
            f"cap {fam[3:]} seed 0: pass@1 {_pct(a1)} → {_pct(b1)}, pass@{k_hi:g} {_pct(ah)} → "
            f"{_pct(bh)} (gap {100 * (b1 - a1):.0f} → {100 * (bh - ah):.0f} points)"
        )
    lead = (
        f"RL helps at every sampling budget: t1 is at or above fz at every k in all {len(pairs)} "
        "seed pairs"
        if pairs and all(pairs)
        else f"t1 is at or above fz at every k in {sum(pairs)} of {len(pairs)} seed pairs"
    )
    k_max = {m: int(max(x for x, _y in pts)) for m, pts in curves.items()}
    kmax_note = "; ".join(
        f"cap {fam[3:]} "
        + ", ".join(
            f"{F.S.parse_model_label(m)[1]}-s{F.S.parse_model_label(m)[2]} {k}"
            for m, k in k_max.items()
            if m.startswith(fam + "-")
        )
        for fam in FAMS
        if any(m.startswith(fam + "-") for m in k_max)
    )
    seen12 = _seen_note(store, "textbook_dev", "dan12")
    seen6 = _seen_note(store, "textbook_dev", "dan6")
    seen_txt = ", ".join(
        f"{s[0]:,} of {s[1]:,} in the cap-{c} tiers" for c, s in (("12", seen12), ("6", seen6)) if s
    )
    caption = (
        f"Pass@k against samples per theorem k (log scale) on textbook_dev (n = {n_items} "
        "textbook theorems), (a) cap 12 and (b) cap 6 on one y scale; fz (Stage-1, pretrained) "
        "darker, t1 (after the T1 RL ladder) full hue. Seed 0 solid with a "
        f"{_BOOT} band (B = {B_TEXTBOOK}); seeds 1 and 2 dashed and dotted, no band. Unbiased "
        "pass@k from each item's first 256 draws up to k = 256, above that from every draw of the "
        "adaptive run, with the 1-(1-c/n)^k extrapolation past an item's own draw count; each "
        f"curve stops at its model's largest per-item draw count ({kmax_note}). Judge: Lean "
        "(lean_v2). Not filtered for training overlap"
        + (f" ({seen_txt})" if seen_txt else "")
        + "."
    )
    takeaway = lead + ("; " + "; ".join(bits) if bits else "") + "."
    pair = [s for s in panels.get("dan12", ChartData(key, title)).series if s.name in PAIR]
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=[s for p in panels.values() for s in p.series],
        opts={"x_label": "k", "y_fmt": "pct"},
        extra={"panels": panels, "n": n_items, "pair": pair},
    )


def _textbook_n(store: Store) -> list[int]:
    version = default_version(store, suite="textbook_dev", models=list(PAIR), judge_id=JUDGE)
    probs = problems_by_pid(store, "textbook_dev", version or "v1")
    return [len(probs)] if probs else []


# ========================================================================== Table 1


_ROLES: tuple[tuple[str, str, str], ...] = (
    ("dan12", "fz", "base"),
    ("dan12", "t1", "rl"),
    ("dan6", "fz", "base"),
    ("dan6", "t1", "rl"),
)


def _c_table1(store: Store, reg: Mapping[str, mr.ModelEntry], tb: ChartData) -> ChartData:
    key, title = "do_table1", "Headline textbook numbers against Dan's published counts"
    models = dv.dan_models(reg)
    lean = dv.textbook72_counts(store, models, JUDGE)
    nd = dv.textbook72_counts(store, models, dv.ND_JUDGE_ID)
    if not lean:
        return _missing(key, title, "no textbook_dev/textbook_train14 runs for Dan's models yet")
    curves = {s.name: dict(s.pts) for s in tb.series} if tb.kind != "missing" else {}
    headers = ["Model", "pass@1 (%)", "pass@256 (%)", "Solved / 72", "Dan's count"]
    rows: list[list[str]] = []
    raw: list[dict[str, Any]] = []
    details: list[list[str]] = []
    diffs: list[int] = []
    nd_diffs: list[int] = []
    t72: dict[str, dict[str, Any]] = {}
    k_first: set[int] = set()
    for fam, role, reg_role in _ROLES:
        labels = mr.seed_labels(reg, fam, reg_role)
        have = [m for m in labels if m in lean and "dev" in lean[m] and "train14" in lean[m]]
        if not have:
            continue
        ours = [lean[m]["dev"][0] + lean[m]["train14"][0] for m in have]
        k_first |= {lean[m]["dev"][2] for m in have} | {lean[m]["train14"][2] for m in have}
        pub = dv.DAN_PUBLISHED["textbook72"].get((fam, role), ())
        pub_dev = dv.DAN_PUBLISHED["dev58"].get((fam, role), ())
        seeds = [cast(int, reg[m].seed) for m in have]
        for m, o, s in zip(have, ours, seeds, strict=True):
            if s < len(pub):
                diffs.append(o - int(pub[s]))
            if m in nd and "dev" in nd[m] and "train14" in nd[m]:
                nd_diffs.append(o - (nd[m]["dev"][0] + nd[m]["train14"][0]))
            details.append(
                [
                    m,
                    str(lean[m]["dev"][0]),
                    str(int(pub_dev[s])) if s < len(pub_dev) else "-",
                    str(lean[m]["train14"][0]),
                    str(int(pub[s]) - int(pub_dev[s]))
                    if s < len(pub) and s < len(pub_dev)
                    else "-",
                    str(o),
                    str(int(pub[s])) if s < len(pub) else "-",
                    (
                        str(nd[m]["dev"][0] + nd[m]["train14"][0])
                        if m in nd and "train14" in nd[m]
                        else "not rejudged"
                    ),
                ]
            )
        p1 = [curves[m][1.0] for m in have if m in curves and 1.0 in curves[m]]
        p256 = [curves[m][256.0] for m in have if m in curves and 256.0 in curves[m]]
        raw.append(
            {
                "fam": fam,
                "role": role,
                "p1": p1,
                "p256": p256,
                "ours": ours,
                "pub": [int(x) for x in pub],
            }
        )
        t72[f"{fam}-{role}"] = {"ours": ours, "dan": [int(x) for x in pub]}
    best = {
        c: max((statistics.mean(r[c]) for r in raw if r[c]), default=None)
        for c in ("p1", "p256", "ours")
    }

    def _cell(vals: list[float], col: str, scale: float = 100.0) -> str:
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        txt = f"{scale * mean:.1f}"
        txt = f"<b>{txt}</b>" if best[col] is not None and mean == best[col] else txt
        lo, hi = f"{scale * min(vals):.1f}", f"{scale * max(vals):.1f}"
        if len(vals) > 1 and lo != hi:
            txt += f' <span class="rng">{lo}–{hi}</span>'
        return txt

    for r in raw:
        solved = " / ".join(str(o) for o in r["ours"])
        if best["ours"] is not None and statistics.mean(r["ours"]) == best["ours"]:
            solved = f"<b>{solved}</b>"
        rows.append(
            [
                f"{r['role']}, cap {r['fam'][3:]}",
                _cell(r["p1"], "p1"),
                _cell(r["p256"], "p256"),
                solved,
                " / ".join(str(x) for x in r["pub"]) or "-",
            ]
        )
    max_abs = max((abs(d) for d in diffs), default=None)
    mean_abs = statistics.mean(abs(d) for d in diffs) if diffs else None
    gains = {
        fam: statistics.mean(next(r["ours"] for r in raw if r["fam"] == fam and r["role"] == "t1"))
        - statistics.mean(next(r["ours"] for r in raw if r["fam"] == fam and r["role"] == "fz"))
        for fam in FAMS
        if {"fz", "t1"} <= {r["role"] for r in raw if r["fam"] == fam}
    }
    k_txt = "/".join(str(k) for k in sorted(k_first))
    nd_txt = (
        "nd_v2 on the same samples changes no count"
        if nd_diffs and all(d == 0 for d in nd_diffs)
        else f"nd_v2 on the same samples changes counts by at most {max(map(abs, nd_diffs))}"
        if nd_diffs
        else "no nd_v2 rejudge yet"
    )
    caption = (
        "textbook_dev (58 theorems) pass@1 and pass@256 (Figure 2's estimator; mean over seeds, "
        "seed range in grey) and textbook72 = textbook_dev + textbook_train14 (14) theorems "
        f"solved within each item's first {k_txt} draws, listed for seeds 0 / 1 / 2 "
        "(T = 0.8, max_action 512, max_steps 96) "
        "per seed, next to Dan's own per-seed counts from his best-state summary (same "
        "checkpoints and protocol, his sampler and seeds). Judge: Lean (lean_v2); "
        f"{nd_txt}. Point estimates per seed; bold: best per column."
    )
    takeaway = (
        (
            f"Our counts reproduce Dan's to within {max_abs} theorems of 72 per checkpoint "
            f"(mean |difference| {mean_abs:.1f})"
            if max_abs is not None and mean_abs is not None
            else "Dan's published counts are not available for these checkpoints"
        )
        + (
            "; RL adds "
            + " and ".join(f"{g:.1f} theorems at cap {fam[3:]}" for fam, g in gains.items())
            + " (seed means)"
            if gains
            else ""
        )
        + "."
    )
    cd = ChartData(
        key,
        title,
        "table",
        takeaway,
        caption,
        extra={"table": {"headers": headers, "rows": rows, "align": "lrrrr"}, "t72": t72},
    )
    cd.extra_tables.append(
        (
            "per seed, dev58 / train14 split",
            [
                "model",
                "dev58 ours",
                "dev58 Dan",
                "train14 ours",
                "train14 Dan",
                "72 ours",
                "72 Dan",
                "72 nd_v2",
            ],
            details,
        )
    )
    return cd


# ========================================================================= 3. families


def _c_families(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_families", "Length horizon on restricted families (unseen theorems)"
    base = dan_pages._data_dan_families(store, reg, families=FAMS)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no families runs for Dan's models yet")
    panels = cast("dict[str, ChartData]", base.extra["panels"])
    rows_by: dict[str, list[tuple[str, float | None, str, bool]]] = {}
    for fam in FAMS:
        for r in panels.get(fam, ChartData(key, title)).rows:
            rows_by.setdefault(r.label, []).extend(r.points)
    order = [o for o in cast("list[str]", base.extra["row_order"]) if o in rows_by]
    rows = [P.DotRow(o, rows_by[o]) for o in order]
    x_hi = max([v or 0.0 for r in rows for _n, v, _c, _z in r.points] + [1.0])
    longer = shorter = equal = 0
    fz_fail = t1_fail = 0
    both_cens: list[str] = []
    for r in rows:
        pts = {n: (v, z) for n, v, _c, z in r.points}
        if PAIR[0] not in pts or PAIR[1] not in pts:
            continue
        (a, za), (b, zb) = pts[PAIR[0]], pts[PAIR[1]]
        fz_fail += a is None
        t1_fail += b is None
        av, bv = a or 0.0, b or 0.0
        if za and zb and av == bv:
            both_cens.append(f"{r.label} (>= {av:.0f} lines)")
            equal += 1
        elif bv > av:
            longer += 1
        elif bv < av:
            shorter += 1
        else:
            equal += 1
    n_cmp = longer + shorter + equal
    probs = problems_by_pid(
        store,
        "families",
        default_version(store, suite="families", models=list(PAIR), judge_id=JUDGE) or "v1",
    )
    seen = _seen_note(store, "families")
    k = _run_k(store, "families", PAIR[1])
    takeaway = (
        f"t1's horizon is longer than fz's in {longer} of {n_cmp} families, equal in {equal} "
        f"and shorter in {shorter}; fz fails at the smallest unseen size in {fz_fail} families, "
        f"t1 in {t1_fail}"
        + (f"; both are censored at the same length in {', '.join(both_cens)}" if both_cens else "")
        + "."
    )
    caption = (
        "Horizon of each restricted family (one row per family): the mean written length "
        "(lines) of the items at n*, the largest family size n at which pass@8 is still at "
        f"least 50%, over size cells with >= {summary._MIN_ITEMS_PER_N} theorems unseen in Dan's "  # noqa: SLF001
        "cap-12 training tiers (his Stage-1 set plus the T1 ladder's RL targets"
        + (f"; {seen[0]:,} of {seen[1]:,} items are seen and excluded" if seen else "")
        + f"). k = {k or '?'} draws per item ({len(probs):,} items in the suite). Hollow marker "
        "at 0: fails at the smallest unseen size; arrow: censored (never below 50% in the tested "
        "range, a lower bound). Point estimates. Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "written lines at n*", "none_label": "fails at smallest size"},
        extra={"row_order": order, "x_hi": x_hi},
    )


# ============================================================================ 4. laws


def _c_laws(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_laws", "Instances of known laws against substitution size"
    base = dan_pages._data_dan_laws(store, reg, families=FAMS)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no laws runs for Dan's models yet")
    series = [s for s in base.series if s.pts]
    curves = {s.name: s.pts for s in series}
    fz, t1 = curves.get(PAIR[0], []), curves.get(PAIR[1], [])
    common = sorted(set(dict(fz)) & set(dict(t1)))
    takeaway = "pass@1 against s for the models with runs."
    if common:
        above = sum(dict(t1)[s] >= dict(fz)[s] for s in common)
        s0 = common[0]
        floor = next((s for s in common if dict(t1)[s] < 0.05), None)
        takeaway = (
            f"t1 is at or above fz at {above} of {len(common)} substitution sizes, from "
            f"{_pct(dict(t1)[s0])} vs {_pct(dict(fz)[s0])} at s = {s0:g}"
            + (
                f"; both are below 5% from s = {floor:g} (t1 {_pct(dict(t1)[floor])}, fz "
                f"{_pct(dict(fz)[floor])})"
                if floor is not None
                else ""
            )
            + "."
        )
    version = default_version(store, suite="laws", models=list(PAIR), judge_id=JUDGE) or "v1"
    n_items = len(problems_by_pid(store, "laws", version))
    seen = _seen_note(store, "laws")
    caption = (
        "Pass@1 against the substitution size s (connectives substituted per metavariable) for "
        f"instances of 8 known law schemas ({n_items:,} items), pooled per item over the schemas "
        f"at each s; s cells with < {summary._MIN_ITEMS_PER_BIN} items omitted. Instances in Dan's "  # noqa: SLF001
        "cap-12 training tiers are excluded"
        + (f" ({seen[0]:,} of {seen[1]:,})" if seen else "")
        + f". k = {_run_k(store, 'laws', PAIR[1]) or '?'} draws per item. Point estimates. "
        "Judge: Lean (lean_v2)."
    )
    return ChartData(
        key, title, "line", takeaway, caption, series=series, opts={"x_label": "s", "y_fmt": "pct"}
    )


# ============================================================================ 5. glue


def _c_glue(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_glue", "Compositionality of solved parts"
    from ..analysis import glue as glue_view

    suite = glue_view.DEFAULT_SUITE
    version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE) or "v1"
    problems = problems_by_pid(store, suite, version)
    pids_by_op: dict[str, list[str]] = {}
    for pid, p in problems.items():
        op = summary._OP_GROUP.get(cast(str, p.tags.get("op", "")))  # noqa: SLF001
        if op:
            pids_by_op.setdefault(op, []).append(pid)
    ops = [op for op in summary._OP_ORDER if op in pids_by_op]  # noqa: SLF001
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    bars: list[P.Bar] = []
    used: list[list[str]] = []
    for op in ops:
        for m in models:
            comp = dan_pages._glue_ratio(store, m, pids_by_op[op], JUDGE)  # noqa: SLF001
            if comp is None or comp["R"] != comp["R"]:
                continue
            lo, hi = comp["ci"]
            bars.append(P.Bar(f"{op}: {m}", float(comp["R"]), lo, hi))
            used.append([op, m, str(len(pids_by_op[op])), str(comp["n_used"])])
    if not bars:
        return _missing(key, title, "no glue runs for Dan's models yet")
    by = {b.name: b for b in bars}
    below = [
        op
        for op in ops
        if all(
            (b := by.get(f"{op}: {m}")) is None or (b.hi is not None and b.hi < 1.0) for m in PAIR
        )
        and any(f"{op}: {m}" in by for m in PAIR)
    ]
    higher = [
        op
        for op in ops
        if f"{op}: {PAIR[0]}" in by
        and f"{op}: {PAIR[1]}" in by
        and by[f"{op}: {PAIR[1]}"].value > by[f"{op}: {PAIR[0]}"].value
    ]
    best_op = max(
        ops, key=lambda op: by[f"{op}: {PAIR[1]}"].value if f"{op}: {PAIR[1]}" in by else -1.0
    )
    takeaway = (
        f"Joined theorems are harder than their mastered parts predict for both models (R < 1 "
        f"with the CI below 1 on {len(below)} of {len(ops)} operators: {', '.join(below) or '-'})"
        f"; t1's R is higher than fz's on {len(higher)} of {len(ops)}, closest to 1 on {best_op} "
        f"(t1 {by[f'{best_op}: {PAIR[1]}'].value:.2f}"
        + (f", fz {by[f'{best_op}: {PAIR[0]}'].value:.2f}" if f"{best_op}: {PAIR[0]}" in by else "")
        + ")."
    )
    caption = (
        "Compositionality ratio R per glue operator: joined pass@1 divided by the product of "
        "its parts' pass@1, pooled over the joined items whose parts each have pass@1 >= 0.5 "
        "for that model (parts sampled as they appear inside the join, glue_parts_renamed); "
        + "; ".join(f"{op} n = {len(pids_by_op[op])}" for op in ops)
        + " joined items, of which the mastered subsets are in the data table (CASE pools the "
        "dilemma and same-conclusion joins). Dotted line: R = 1. Error bars: 95% item-level "
        f"bootstrap CIs (B = 1000). k = {_run_k(store, 'glue', PAIR[1]) or '?'} draws per joined "
        "item. Not filtered for training overlap. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "f3"},
        extra={"ops": ops, "counts": {op: len(v) for op, v in pids_by_op.items()}},
    )
    cd.extra_tables.append(("items used", ["op", "model", "joined items", "parts mastered"], used))
    return cd


# =========================================================================== 6. skills


def _skills_by_length(store: Store, model: str, *, min_items: int = 20) -> list[P.Series]:
    """Mean pass@1 per (skill_group, L) cell on skill_pool_sub, cells with < `min_items` items
    dropped (the main page's `_MIN_SKILL_CELL_ITEMS` rule)."""
    suite = "skill_pool_sub"
    run_id = latest_run_id(store, suite=suite, model=model, version="v1", judge_id=JUDGE)
    if run_id is None:
        return []
    problems = problems_by_pid(store, suite, "v1")
    cells: dict[str, dict[int, list[float]]] = {}
    for pid, it in items_by_pid(store, run_id).items():
        p = problems.get(pid)
        if p is None or it.n <= 0 or p.length.L is None:
            continue
        v = pass_at_k(it.n, it.c, 1)
        if v == v:
            cells.setdefault(str(p.tags.get("skill_group", "plain")), {}).setdefault(
                int(p.length.L), []
            ).append(v)
    out = []
    for group, by_l in cells.items():
        pts = [
            (float(L), statistics.mean(v)) for L, v in sorted(by_l.items()) if len(v) >= min_items
        ]
        if pts:
            out.append(P.Series(group, pts, "var(--s1)"))
    return out


def _c_skills(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_skills", "Pass@1 by required skill"
    base = dan_pages._data_dan_skills(store, reg, families=FAMS)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no skill_pool_sub runs for Dan's models yet")
    bars = list(base.bars)
    by = {b.name: b.value for b in bars}
    models = list(dict.fromkeys(b.name.rpartition(": ")[2] for b in bars))
    by_len = {m: _skills_by_length(store, m) for m in models}
    problems = problems_by_pid(store, "skill_pool_sub", "v1")
    sizes: dict[str, int] = {}
    for p in problems.values():
        g = str(p.tags.get("skill_group", "plain"))
        sizes[g] = sizes.get(g, 0) + 1

    def v(group: str, m: str) -> float | None:
        return by.get(f"{group}: {m}")

    t1, fz = PAIR[1], PAIR[0]
    takeaway = "pass@1 by skill group."
    if v("case_split", t1) is not None and v("plain", t1) is not None:
        cs, pl = cast(float, v("case_split", t1)), cast(float, v("plain", t1))
        lines = {s.name: dict(s.pts) for s in by_len.get(t1, [])}
        common = sorted(set(lines.get("case_split", {})) & set(lines.get("plain", {})))
        at_or_above = sum(lines["case_split"][L] >= lines["plain"][L] for L in common)
        hardest = min(
            (g for g in sizes if v(g, t1) is not None), key=lambda g: cast(float, v(g, t1))
        )
        verb = "do not cost t1 anything" if cs >= pl else "cost t1"
        takeaway = (
            f"Case splits {verb}: it solves {_pct(cs)} of case-split items vs {_pct(pl)} of "
            f"plain ones"
            + (
                f" (fz {_pct(v('case_split', fz))} vs {_pct(v('plain', fz))})"
                if v("case_split", fz) is not None
                else ""
            )
            + (
                f", and is at or above plain at {at_or_above} of {len(common)} matched lengths"
                if common
                else ""
            )
            + f"; {hardest.replace('_', ' ')} is the hardest group (t1 "
            f"{_pct(v(hardest, t1))}"
            + (f", fz {_pct(v(hardest, fz))}" if v(hardest, fz) is not None else "")
            + ")."
        )
    caption = (
        "Pass@1 by required skill on skill_pool_sub, a stratified subsample of skill_pool (up to "
        f"25 items per (L, group) cell; {len(problems)} items: "
        + ", ".join(f"{g.replace('_', ' ')} {n}" for g, n in sorted(sizes.items()))
        + "). (a) Mean pass@1 over each group's items; (b, c) the same by written length L, "
        f"cells with < {summary._MIN_SKILL_CELL_ITEMS} items omitted, so the groups are compared "  # noqa: SLF001
        "at matched length. Groups: plain (neither skill), case split (needs an or-elimination), "
        "classical (needs a classical step), both. k = "
        f"{_run_k(store, 'skill_pool_sub', t1) or '?'} draws per item. Point estimates. Not "
        "filtered for training overlap. Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "pct"},
        extra={"by_length": {m: s for m, s in by_len.items() if s}},
    )
    for m, ss in by_len.items():
        if ss:
            cd.extra_tables.append(
                (f"{m} by length (b, c)", *P.line_table_rows(ss, x_label="L", y_fmt="pct"))
            )
    return cd


# ========================================================================== 7. padding


def _span(ns: Sequence[int]) -> str:
    """`"34"`, `"37-40"` or `"-"` (no padded points)."""
    if not ns:
        return "-"
    lo, hi = min(ns), max(ns)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _c_padding(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_padding", "Padding: a longer proof against more distractors"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    data = {m: d for m in models if (d := dv.padding_by_pad_type(store, m, judge_id=JUDGE))}
    if not data:
        return _missing(key, title, "no padding runs for Dan's models yet")
    pad_panels: dict[str, list[P.Series]] = {}
    series: list[P.Series] = []
    mini: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for m, by_type in data.items():
        p1_acc: dict[int, list[float]] = {}
        for t, pts in by_type.items():
            s = P.Series(m, [(float(mm), p) for mm, p, _n, _w in pts], "var(--s1)")
            pad_panels.setdefault(t, []).append(s)
            series.append(P.Series(f"{t}: {m}", s.pts, "var(--s1)"))
            if t.startswith("P1"):
                for mm, p, n, _w in pts:
                    acc = p1_acc.setdefault(mm, [0.0, 0.0])
                    acc[0] += p * n
                    acc[1] += n
        mini[m] = {
            "P2": [(float(mm), p) for mm, p, _n, _w in by_type.get("P2", [])],
            "P1": [(float(mm), t / n) for mm, (t, n) in sorted(p1_acc.items()) if n],
        }
    fz, t1 = PAIR
    takeaway = "pass@1 against pads added, by pad type."
    if t1 in data and {"P2", "P1-shared", "P1-disjoint"} <= set(data[t1]):
        d = {t: {mm: p for mm, p, _n, _w in pts} for t, pts in data[t1].items()}
        m_cmp = (
            2 if all(2 in d[t] for t in d) else max(set.intersection(*(set(x) for x in d.values())))
        )
        p2, p1s, p1d = d["P2"][m_cmp], d["P1-shared"][m_cmp], d["P1-disjoint"][m_cmp]
        worse = (
            "more distractors hurt t1 more than a longer proof"
            if max(p1s, p1d) < p2
            else "a longer proof hurts t1 more than more distractors"
        )
        fz_txt = ""
        if fz in data:
            e = {t: {mm: p for mm, p, _n, _w in pts} for t, pts in data[fz].items()}
            if all(m_cmp in e.get(t, {}) for t in ("P2", "P1-shared", "P1-disjoint")):
                fz_txt = (
                    f"; fz goes from {_pct(e['P2'][0])} to {_pct(e['P2'][m_cmp])} (P2) and "
                    f"{_pct(e['P1-shared'][m_cmp])} / {_pct(e['P1-disjoint'][m_cmp])} (P1)"
                )
        takeaway = (
            f"At m = {m_cmp} pads, {worse}: from {_pct(d['P2'][0])} unpadded, {m_cmp} extra "
            f"∧-introduction lines (P2) leave {_pct(p2)} pass@1, {m_cmp} distractor premises "
            f"{_pct(p1s)} (P1-shared) and {_pct(p1d)} (P1-disjoint){fz_txt}."
        )
    n_by = {t: [n for _m, _p, n, _w in pts] for t, pts in next(iter(data.values())).items()}
    ns = ", ".join(f"{t} {_span(v[1:])}" for t, v in n_by.items())
    m0 = next(iter(n_by.values()))[0] if n_by else 0
    caption = (
        "Pass@1 of padded versions of short theorems against the number of pads m, by pad type. "
        "(a) P2 extends the goal C to ((C ∧ X1) ∧ X2) ... with existing premises Xi: each pad "
        "adds one real ∧-introduction line, so the proof is longer. (b, c) P1 appends m extra "
        "premises over the theorem's own atoms (shared) or fresh atoms (disjoint). Under Lean "
        "premises need no restating, so in Dan's environment P1 pads are only extra hypotheses "
        "in the proof state (distractors): the natural-deduction written length W = L + m "
        "overstates the lines his models must write, so the x-axis is m, not W. "
        f"m = 0 is the same {m0} unpadded theorems in every panel, each counted once; items per "
        f"padded point: {ns}. k = {_run_k(store, 'padding', t1) or '?'} draws per item. Point "
        "estimates. Not filtered for training overlap. Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "m", "y_fmt": "pct"},
        extra={"pad_panels": pad_panels, "mini": mini, "n_bases": m0},
    )


# =========================================================================== 8. prefix


def _c_prefix(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_prefix", "Completing a given proof prefix"
    base = dan_pages._data_dan_prefix_state(store, reg, families=FAMS)  # noqa: SLF001
    if base.kind == "missing":
        return _missing(key, title, "no prefix_state runs for Dan's models yet")
    series = [s for s in base.series if s.pts]
    curves = {s.name: dict(s.pts) for s in series}
    fz, t1 = curves.get(PAIR[0], {}), curves.get(PAIR[1], {})
    common = sorted(set(fz) & set(t1))
    takeaway = "prefix completion by lines left."
    if common:
        above = sum(t1[x] >= fz[x] for x in common)
        lo, hi = common[0], common[-1]
        takeaway = (
            f"t1 completes more prefixes than fz at {above} of {len(common)} distances: "
            f"{_pct(t1[lo])} vs {_pct(fz[lo])} with {lo:g} line{'s' if lo != 1 else ''} left, "
            f"{_pct(t1[hi])} vs {_pct(fz[hi])} with {hi:g} left."
        )
    probs = problems_by_pid(store, "prefix_state", "v1")
    caption = (
        "Pass@1 of completing a reference proof whose first j lines are given, against lines "
        "left to write (L - j): the first j lines are replayed as a forced action prefix in "
        f"Dan's environment (prefix_state, {len(probs):,} (proof, j) items); bins with < "
        f"{summary._MIN_PREFIX_BIN_ITEMS} items omitted. k = "  # noqa: SLF001
        f"{_run_k(store, 'prefix_state', PAIR[1]) or '?'} draws per item. Point estimates. Not "
        "filtered for training overlap. Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "lines left", "y_fmt": "pct"},
    )


# ========================================================================== 9. held-out


def _c_heldout(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_heldout", "Greedy accuracy on Dan's held-out set"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    data = heldout_view.view(store, cast("dict[str, JSON]", {"models": models, "judge_id": JUDGE}))
    if data.get("empty"):
        return _missing(key, title, str(data.get("reason", "no dan_heldout runs yet")))
    per = cast("dict[str, dict[str, Any]]", data["models"])
    bars = [P.Bar(m, float(i["accuracy"]), i["ci"][0], i["ci"][1]) for m, i in per.items()]
    published: dict[str, float] = {}
    for m in per:
        e = reg.get(m)
        if e is None or e.seed is None:
            continue
        role = dv.role_of(m)
        vals = dv.DAN_PUBLISHED["heldout"].get((e.family, role), ())
        if e.seed < len(vals):
            published[m] = float(vals[e.seed])
    parts = []
    for m in F._role_order(list(per)):  # noqa: SLF001
        i = per[m]
        parts.append(
            f"{F.display(m)} {i['n_solved']:,}/{i['n_items']:,} ({_pct(i['accuracy'])}"
            + (f"; Dan {_pct(published[m])}" if m in published else "")
            + ")"
        )
    takeaway = "greedy held-out accuracy per model."
    if PAIR[0] in per and PAIR[1] in per:
        a, b = per[PAIR[0]], per[PAIR[1]]
        gap = max((abs(per[m]["accuracy"] - published[m]) for m in published), default=None)
        takeaway = (
            f"RL cuts greedy errors from {_pct(1 - a['accuracy'])} to "
            f"{_pct(1 - b['accuracy'])} of the held-out theorems ({a['n_items'] - a['n_solved']:,}"
            f" → {b['n_items'] - b['n_solved']:,} of {b['n_items']:,})"
            + (
                f", within {100 * gap:.1f} points of Dan's reported accuracies"
                if gap is not None
                else ""
            )
            + "."
        )
    seen = _seen_note(store, "dan_heldout")
    caption = (
        f"Fraction of Dan's {next(iter(per.values()))['n_items']:,}-theorem held-out set "
        "(heldout.jsonl, his Stage-1 validation split) solved by one greedy rollout (k = 1, "
        f"T = 0), with a 95% Clopper-Pearson CI: {'; '.join(parts)}. Open triangles: Dan's "
        "reported accuracy for the same checkpoint. Not filtered for training overlap"
        + (f" ({seen[0]:,} of {seen[1]:,} in the cap-12 tiers)" if seen else "")
        + ". Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "bar",
        takeaway,
        caption,
        bars=bars,
        opts={"y_fmt": "pct"},
        extra={"published": published},
    )


# ===================================================================== 10. textbook_law


def _c_textbook_law(store: Store, reg: Mapping[str, mr.ModelEntry]) -> ChartData:
    key, title = "do_textbook_law", "Pass@k on textbook_law, unseen theorems only"
    models = [m for fam in FAMS for m in mr.labels_in_family(reg, fam)]
    data = dv.textbook_law_unseen(store, models, registry=reg, judge_id=JUDGE, B=B_TEXTBOOK)
    if not data:
        return _missing(key, title, "no textbook_law runs for Dan's models yet")
    series = [
        P.Series(
            m,
            [(float(k), mean) for k, mean, _l, _h in d["curve"]],
            "var(--s1)",
            band=[(float(k), lo, hi) for k, _m, lo, hi in d["curve"]],
        )
        for m, d in data.items()
    ]
    takeaway = "pass@k on unseen textbook_law theorems."
    if PAIR[0] in data and PAIR[1] in data:
        a = {k: m for k, m, _l, _h in data[PAIR[0]]["curve"]}
        b = {k: m for k, m, _l, _h in data[PAIR[1]]["curve"]}
        k_hi = max(set(a) & set(b))
        takeaway = (
            f"t1's pass@1 ({_pct(b[1])}) is "
            + ("above" if b[1] > a[k_hi] else "below")
            + f" fz's pass@{k_hi} ({_pct(a[k_hi])}); at k = {k_hi} t1 reaches {_pct(b[k_hi])}, so "
            "sampling more from fz does not close the gap on these theorems."
            if b[1] > a[k_hi]
            else f"t1 is ahead at k = 1 ({_pct(b[1])} vs {_pct(a[1])}) and at k = {k_hi} "
            f"({_pct(b[k_hi])} vs {_pct(a[k_hi])})."
        )
    d0 = next(iter(data.values()))
    caption = (
        f"Pass@k on the textbook_law theorems unseen in Dan's cap-12 training tiers (n = "
        f"{d0['n']}; {d0['n_seen']} seen theorems excluded), k = 1 to {d0['k_max']} draws per "
        f"theorem: unbiased pass@k with {_BOOT} bands (B = {B_TEXTBOOK}). Judge: Lean (lean_v2)."
    )
    return ChartData(
        key, title, "line", takeaway, caption, series=series, opts={"x_label": "k", "y_fmt": "pct"}
    )


# ====================================================================== 11. likelihood

_ORIGIN_LABEL = {
    "reserve": "neutral reserve proofs",
    "family": "family witnesses",
    "reference": "neutral minimal proofs",
}


def _c_likelihood(store: Store) -> ChartData:
    key, title = "do_likelihood", "Log-likelihood of neutral proofs under fz and t1"
    data = dv.dan_likelihood(store)
    if data is None:
        return _missing(key, title, "no interp likelihood run scoring fz/t1 yet")
    origins = cast("dict[str, dict[str, Any]]", data["origins"])
    order = [o for o in ("reserve", "reference", "family") if o in origins] + sorted(
        set(origins) - {"reserve", "reference", "family"}
    )
    cfg = cast("dict[str, JSON]", data["cfg"])
    base = cast(str, cfg.get("base_model") or PAIR[0])
    rl = cast(str, cfg.get("rl_model") or PAIR[1])
    rows = [
        P.DotRow(
            f"{_ORIGIN_LABEL.get(o, o)} (n={origins[o]['n']})",
            [
                (base, origins[o]["median_fz"], "var(--s1)", False),
                (rl, origins[o]["median_t1"], "var(--s2)", False),
            ],
        )
        for o in order
    ]
    bits = [
        f"{origins[o]['median_diff']:+.2f} nats/line on {_ORIGIN_LABEL.get(o, o)} (t1 higher on "
        f"{100 * origins[o]['frac_t1_higher']:.0f}%)"
        for o in order
    ]
    small = max(abs(origins[o]["median_diff"]) for o in order) < 0.5
    takeaway = (
        ("RL barely moves" if small else "RL changes")
        + " the likelihood of neutral proofs: the paired median change t1 - fz is "
        + "; ".join(bits)
        + "."
    )
    n_tot = sum(origins[o]["n"] for o in order)
    multi = sum(origins[o]["n_multi_order"] for o in order)
    names = (
        "The environment-assigned defining-name tokens are excluded from the score, as in Dan's "
        "trajectory study."
        if data["exclude_defining_names"]
        else "The scored tokens include the environment-assigned defining-name tokens, which the "
        "environment overrides at sampling time and Dan's trajectory study excludes."
    )
    caption = (
        "Median log-likelihood per written line (nats) of fixed proof sets under fz and t1 (the "
        f"cap-12 seed-0 pair): {n_tot} proofs ("
        + ", ".join(f"{origins[o]['n']} {_ORIGIN_LABEL.get(o, o)}" for o in order)
        + f") of theorems with L = {cfg.get('l_min', '?')}-{cfg.get('l_max', '?')}, unseen in "
        f"Dan's training tiers; run {data['run_id']}"
        + ("; small n, so the medians are noisy" if n_tot < 100 else "")
        + ". Each proof's log p is marginalised over its valid line orders and the environment's "
        f"name bases ({multi} proofs have more than one order) and divided by its written lines. "
        + names
        + " Medians, no CI."
    )
    return ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "median log p per line"},
        extra={
            "exclude_defining_names": data["exclude_defining_names"],
            "n": n_tot,
            "run_id": data["run_id"],
        },
    )


# ============================================================================= 12. KL

#: Elimination steps (the rules that consume a line), for the KL takeaway.
_ELIM_KINDS: tuple[str, ...] = ("NEGE", "IMPE", "ANDE1", "ANDE2", "BOTE")
_KIND_GLOSS = (
    "AS opens a box (assumption), IMPI/NEGI/ORE close one, PR/R restate a premise/line, FINAL is "
    "the closing exact, ANDI/ORI1/ORI2 introductions, ANDE1/ANDE2/IMPE/NEGE/BOTE eliminations, "
    "DN double negation"
)


def _c_kl(store: Store) -> ChartData:
    key, title = "do_kl", "Token-level KL between t1 and fz, by action kind"
    data = dv.dan_kl(store)
    if data is None:
        cd = _missing(
            key,
            title,
            "not yet run: no interp kl run in the store, so the token-"
            "level KL between t1 and fz is not measured on this page yet",
        )
        cd.extra["missing_label"] = "not yet run"
        return cd
    cfg = cast("dict[str, JSON]", data["cfg"])
    base = cast(str, cfg.get("base_model") or PAIR[0])
    rl = cast(str, cfg.get("rl_model") or PAIR[1])
    dirs = cast("dict[str, dict[str, Any]]", data["directions"])
    short = {"rl||base": "KL(t1 ‖ fz)", "base||rl": "KL(fz ‖ t1)"}
    name = {
        "rl||base": f"KL(t1 ‖ fz) on t1 samples|{rl}",
        "base||rl": f"KL(fz ‖ t1) on fz samples|{base}",
    }
    order = [d for d in ("base||rl", "rl||base") if d in dirs] + sorted(
        set(dirs) - {"base||rl", "rl||base"}
    )
    kinds = sorted(
        {k for d in dirs.values() for k in d["by_kind"]},
        key=lambda k: -max(d["by_kind"].get(k, (0.0, 0))[0] for d in dirs.values()),
    )
    rows = [
        P.DotRow(
            "all tokens",
            [(name.get(dn, dn), dirs[dn]["per_token"][0], "var(--s1)", False) for dn in order],
        )
    ]
    for k in kinds:
        rows.append(
            P.DotRow(
                k,
                [
                    (name.get(dn, dn), dirs[dn]["by_kind"][k][0], "var(--s1)", False)
                    for dn in order
                    if k in dirs[dn]["by_kind"]
                ],
            )
        )
    overall = "; ".join(
        f"{short.get(dn, dn)} {dirs[dn]['per_token'][0]:.3f} nats/token "
        f"({dirs[dn]['per_sample']:.1f} per sample)"
        for dn in order
    )
    elim = [k for k in _ELIM_KINDS if all(k in dirs[dn]["by_kind"] for dn in order)]
    low = [k for k in kinds if all(dirs[dn]["by_kind"].get(k, (1.0, 0))[0] < 0.02 for dn in order)]
    elim_txt = ""
    if elim:
        ranges = []
        for dn in order:
            vals = [dirs[dn]["by_kind"][k][0] for k in elim]
            ranges.append(f"{min(vals):.2f}-{max(vals):.2f} ({short.get(dn, dn)})")
        elim_txt = (
            f"; the eliminations ({', '.join(elim)}) carry "
            + " and ".join(ranges)
            + " nats/token"
            + (f", while {', '.join(low)} stay under 0.02 in both directions" if low else "")
        )
    tops = [
        max(dirs[dn]["by_kind"].items(), key=lambda kv: kv[1][0])[0]
        for dn in order
        if dirs[dn]["by_kind"]
    ]
    lead = (
        "t1 and fz differ most on elimination steps: "
        if elim and tops and all(t in _ELIM_KINDS for t in tops)
        else "Per-token KL: "
    )
    takeaway = lead + overall + elim_txt + "."
    excl = cfg.get("exclude_defining_names")
    suites = ", ".join(
        str(cast("dict[str, JSON]", s_).get("suite"))
        for s_ in cast("list[JSON]", cfg.get("suites") or [])
        if isinstance(s_, dict)
    ) or ", ".join(sorted({s for d in dirs.values() for s in d["suites"]}))
    n_rows, n_skip = cast(int, data["n_rows"]), cast("int | None", data["n_skipped"])
    skipped = (
        f" {n_skip:,} of {n_rows + n_skip:,} sampled attempts "
        f"({100 * n_skip / (n_rows + n_skip):.0f}%) were skipped: they are attempts the "
        "environment rejected as unparseable, which store no action tokens to score, so this KL "
        "covers in-grammar attempts only."
        if n_skip
        else ""
    )
    caption = (
        "Exact per-token KL divergence between the cap-12 seed-0 pair's next-action "
        "distributions, teacher-forced along each model's own stored samples (KL(t1 ‖ fz) along "
        "t1's, KL(fz ‖ t1) along fz's) from its "
        f"{suites} runs, at most {cfg.get('max_samples_per_suite', '?')} samples per (model, "
        f'suite) (run {data["run_id"]}). "all tokens": the sum of KL over scored tokens '
        f"divided by their count, with a 95% bootstrap CI over samples (B = {data['B']}); then "
        f"the same per action kind, largest first, point estimates ({_KIND_GLOSS}). "
        + "; ".join(
            f"{short.get(dn, dn)}: {dirs[dn]['n']:,} samples ({dirs[dn]['n_ok']:,} correct), "
            f"{dirs[dn]['n_tokens']:,} tokens"
            for dn in order
        )
        + ". Environment-assigned defining-name tokens are "
        + ("excluded" if excl else "included" if excl is not None else "of unrecorded status")
        + "."
        + skipped
    )
    cd = ChartData(
        key,
        title,
        "dot",
        takeaway,
        caption,
        rows=rows,
        opts={"x_label": "nats per token", "x_fmt": "f3"},
        extra={"n_skipped": n_skip, "n_rows": n_rows},
    )
    cd.extra_tables.append(
        (
            "tokens per kind",
            ["kind", *(short.get(dn, dn) + " tokens" for dn in order)],
            [[k, *(f"{dirs[dn]['by_kind'].get(k, (0, 0))[1]:,}" for dn in order)] for k in kinds],
        )
    )
    return cd


# ========================================================================= 13. steering


def _short_suite(suite: str) -> str:
    return {"interp_eval_dev": "dev", "interp_eval_sealed": "sealed"}.get(suite, suite)


def _c_steering(store: Store) -> ChartData:
    key, title = "do_steering", "Steering fz along the t1 - fz activation difference"
    data = dv.dan_steering(store, base=PAIR[0], rl=PAIR[1])
    rows_in = cast("list[dict[str, Any]]", data["rows"])
    if not rows_in:
        return _missing(key, title, "no steering runs for Dan's models yet")
    # fz rows first (unsteered, then each setting), then t1 unsteered next to t1 - delta
    rows_in = sorted(
        rows_in,
        key=lambda r: (r["model"] != PAIR[0], r["steered"], r["layer"] or 0, r["alpha"] or 0.0),
    )
    suites = cast("list[str]", data["suites"])

    def cond(r: dict[str, Any]) -> str:
        role = "fz" if r["model"] == PAIR[0] else "t1"
        if not r["steered"]:
            return f"{role} (unsteered)"
        sign = "+" if r["alpha"] >= 0 else "−"
        a = abs(r["alpha"])
        return f"{role} {sign} δ (layer {r['layer']}" + (
            f", α = {a:g})" if a != 1 or role == "fz" else ")"
        )

    fz_rows = [r for r in rows_in if r["model"] == PAIR[0]]
    best_col = {
        s: max((r["per_suite"][s][0] for r in fz_rows if r["per_suite"].get(s)), default=None)
        for s in suites
    }
    headers = ["Condition", *(f"{_short_suite(s)} (%)" for s in suites)]
    rows: list[list[str]] = []
    bars: list[P.Bar] = []
    raw: list[list[str]] = []
    for r in rows_in:
        cells = [html.escape(cond(r))]
        for s in suites:
            v = r["per_suite"].get(s)
            if v is None:
                cells.append('<span class="na">not run</span>')
                continue
            mean, lo, hi, n = v
            num = f"{100 * mean:.1f}"
            if r["model"] == PAIR[0] and mean == best_col[s]:
                num = f"<b>{num}</b>"
            cells.append(
                f'{num}<span class="pm"><sup>+{100 * (hi - mean):.1f}</sup>'
                f"<sub>&minus;{100 * (mean - lo):.1f}</sub></span>"
            )
            raw.append(
                [
                    cond(r),
                    _short_suite(s),
                    str(n),
                    f"{100 * mean:.2f}",
                    f"{100 * lo:.2f}",
                    f"{100 * hi:.2f}",
                ]
            )
        rows.append(cells)
    # Overview bars: unsteered fz, the best fz + delta (by the mean over suites), t1 - delta.
    steered = [r for r in fz_rows if r["steered"] and all(r["per_suite"].get(s) for s in suites)]
    best = max(steered, key=lambda r: sum(r["per_suite"][s][0] for s in suites), default=None)
    worst = min(steered, key=lambda r: sum(r["per_suite"][s][0] for s in suites), default=None)
    unsteered = next((r for r in fz_rows if not r["steered"]), None)
    control = next((r for r in rows_in if r["model"] == PAIR[1] and r["steered"]), None)
    t1_plain = next((r for r in rows_in if r["model"] == PAIR[1] and not r["steered"]), None)
    has_t1 = t1_plain is not None and all(t1_plain["per_suite"].get(s) for s in suites)
    for r, lab in (
        (unsteered, "fz"),
        (best, "fz + δ"),
        (control, "t1 − δ"),
        (t1_plain if has_t1 else None, "t1"),
    ):
        if r is None:
            continue
        for s in suites:
            v = r["per_suite"].get(s)
            if v is not None:
                bars.append(P.Bar(f"{_short_suite(s)}: {lab}|{r['model']}", v[0], v[1], v[2]))
    takeaway = "pass@1 under steering."
    if unsteered is not None and best is not None:
        inside = all(
            best["per_suite"][s][0] <= unsteered["per_suite"][s][2]
            for s in suites
            if unsteered["per_suite"].get(s)
        )
        vals = lambda r: " / ".join(f"{100 * r['per_suite'][s][0]:.1f}%" for s in suites)  # noqa: E731
        takeaway = (
            "Adding the t1 - fz direction to fz does not make it t1: the best of "
            f"{len(steered)} settings (layer {best['layer']}, α = {best['alpha']:g}) scores "
            f"{vals(best)} "
            f"(dev / sealed) against {vals(unsteered)} unsteered"
            + (", inside fz's 95% CI" if inside else "")
            + (
                f", and the most harmful (layer {worst['layer']}, α = {worst['alpha']:g}) lowers "
                f"it to {vals(worst)}"
                if worst is not None and worst is not best
                else ""
            )
            + _t1_control_text(control, t1_plain if has_t1 else None, suites, vals)
            + "."
        )
    ns = {
        s: next((r["per_suite"][s][3] for r in rows_in if r["per_suite"].get(s)), 0) for s in suites
    }
    caption = (
        "Pass@1 of fz with the activation difference δ_l (mean t1 minus fz residual activation "
        "at layer l over the action tokens of 300 unseen reference proofs, L = 6-10) added at "
        "every generated action token with weight α, and of t1 with δ subtracted; layers 0 and "
        "1 of 6. interp_eval_dev = generated theorems L = 8-11 (n = "
        f"{ns.get('interp_eval_dev', '?')}), interp_eval_sealed = L = 12-14 (n = "
        f"{ns.get('interp_eval_sealed', '?')}); k = {data['k']} draws per theorem at T = 0.8; "
        f"superscript/subscript: 95% item-level bootstrap CI (B = {data['B']}); bold: best fz "
        "condition per column. "
        + (
            "t1 (unsteered) is sampled on the same suites with the same k, T and seed. "
            if has_t1
            else "t1 has no unsteered run on these two suites. "
        )
        + "Judge: Lean (lean_v2)."
    )
    cd = ChartData(
        key,
        title,
        "table",
        takeaway,
        caption,
        bars=bars,
        extra={"table": {"headers": headers, "rows": rows, "align": "l" + "r" * len(suites)}},
        opts={"has_t1_baseline": has_t1},
    )
    cd.extra_tables.append(
        ("values", ["condition", "suite", "n", "pass@1 (%)", "CI lo", "CI hi"], raw)
    )
    return cd


def _t1_control_text(
    control: dict[str, Any] | None,
    t1_plain: dict[str, Any] | None,
    suites: Sequence[str],
    vals: Callable[[dict[str, Any]], str],
) -> str:
    """The t1 - delta clause of the steering takeaway, against unsteered t1 when it was run."""
    if control is None:
        return ""
    if t1_plain is None:
        return (
            f"; subtracting it from t1 leaves {vals(control)} (no unsteered t1 run on these "
            "suites to compare)"
        )
    shifts, verdicts = [], []
    for s in suites:
        c, t = control["per_suite"].get(s), t1_plain["per_suite"].get(s)
        if c is None or t is None:
            continue
        shifts.append(f"{100 * (c[0] - t[0]):+.1f}".replace("-", "−"))
        where = "inside" if t[1] <= c[0] <= t[2] else "below" if c[0] < t[1] else "above"
        verdicts.append((_short_suite(s), where))
    if all(w == "inside" for _s, w in verdicts):
        ci = "inside t1's 95% CI on every suite"
    else:
        ci = ", ".join(f"{name} {w} t1's 95% CI" for name, w in verdicts)
    return (
        f"; subtracting it from t1 gives {vals(control)} against {vals(t1_plain)} unsteered "
        f"({' / '.join(shifts)} points; {ci})"
    )


# ===================================================================== 14. interpolation


def _c_weights(store: Store) -> ChartData:
    key, title = "do_weights", "Interpolating weights from fz to t1"
    data = dv.dan_interpolation(store, base=PAIR[0], rl=PAIR[1], family="dan12")
    curves = cast("dict[float, list[tuple[int, float, float, float]]]", data["curves"])
    pk: dict[float, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for a, c in curves.items():
        by_k = {k: (m, lo, hi) for k, m, lo, hi in c}
        if 1 in by_k and 128 in by_k:
            pk[a] = (by_k[1], by_k[128])
    if not pk:
        return _missing(key, title, "no fz/t1/interp runs on the k = 128 subset yet")
    alphas = sorted(pk)
    series = [
        P.Series(
            "pass@1",
            [(a, pk[a][0][0]) for a in alphas],
            "var(--ink)",
            band=[(a, pk[a][0][1], pk[a][0][2]) for a in alphas],
        ),
        P.Series(
            "pass@128",
            [(a, pk[a][1][0]) for a in alphas],
            "var(--ink2)",
            muted=True,
            band=[(a, pk[a][1][1], pk[a][1][2]) for a in alphas],
        ),
    ]
    p1 = {a: pk[a][0][0] for a in alphas}
    p128 = {a: pk[a][1][0] for a in alphas}
    a0, a1 = alphas[0], min(alphas, key=lambda a: abs(a - 1.0))
    mids = [a for a in alphas if a0 < a < a1]
    half = next((a for a in alphas if p1[a] - p1[a0] >= 0.5 * (p1[a1] - p1[a0])), a1)
    sat = next((a for a in alphas if p128[a] >= max(p128.values()) - 1e-9), a1)
    beyond = [a for a in alphas if a > a1]
    takeaway = (
        f"pass@1 rises from {_pct(p1[a0])} (α = {a0:g}, fz) to {_pct(p1[a1])} (α = {a1:g}, t1), "
        f"half of the gain already by α = {half:g}"
        + (f" ({_pct(p1[half])})" if half in mids else "")
        + f", while pass@128 rises from {_pct(p128[a0])} and saturates at {_pct(p128[sat])} from "
        f"α = {sat:g}"
        + (
            f"; extrapolating to α = {beyond[-1]:g} lowers pass@1 to {_pct(p1[beyond[-1]])} with "
            f"pass@128 at {_pct(p128[beyond[-1]])}"
            if beyond
            else ""
        )
        + "."
    )
    takeaway = takeaway[0].upper() + takeaway[1:]
    n = next(iter(cast("dict[float, int]", data["n"]).values()))
    seen = _seen_note(store, "interp_eval_k128")
    caption = (
        "Pass@1 and pass@128 of θ_fz + α(θ_t1 − θ_fz) against α (cap-12 seed-0 pair; α > 1 "
        f"extrapolates past t1) on a fixed subset of n = {n} generated theorems (L = 8-14), 128 "
        f"draws each at T = 0.8; shaded: {_BOOT} bands (B = {data['B']}). Not filtered for "
        "training overlap"
        + (f" ({seen[0]:,} of {seen[1]:,} in the cap-12 tiers)" if seen else "")
        + ". Judge: Lean (lean_v2)."
    )
    return ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "alpha (0 = fz, 1 = t1)", "y_fmt": "pct"},
    )


# ============================================================================ 15. probes

_BASELINE_LABEL = {
    "step_index_premises": "step + premises",
    "step_index_premises_depth": "step + premises + depth",
}


def _c_probes(store: Store) -> ChartData:
    key, title = "do_probes", "What linear probes read from fz and t1"
    data = dv.dan_probes(store)
    if data is None:
        return _missing(key, title, "no interp probes run for fz/t1 yet")
    cfg = cast("dict[str, JSON]", data["cfg"])
    label = {
        "fz": cast(str, cfg.get("base_model") or PAIR[0]),
        "t1": cast(str, cfg.get("rl_model") or PAIR[1]),
    }
    per = cast(
        "dict[str, list[tuple[int, float | None, float | None, int | None]]]", data["models"]
    )
    series: list[P.Series] = []
    for metric, idx in (("next_rule", 1), ("actions_to_go", 2)):
        for role in ("fz", "t1"):
            pts = [
                (float(row[0]), cast(float, row[idx]))
                for row in per.get(role, [])
                if row[idx] is not None
            ]
            if pts:
                series.append(P.Series(f"{metric}: {label[role]}", pts, "var(--s1)"))
    bl = cast("dict[str, float]", data["baselines"])
    baselines: dict[str, list[tuple[str, float]]] = {}
    for metric, prefix in (("next_rule", "next_rule_acc_"), ("actions_to_go", "actions_to_go_r2_")):
        for name in ("step_index_premises", "step_index_premises_depth"):
            v = bl.get(prefix + name)
            if v is not None:
                baselines.setdefault(metric, []).append((_BASELINE_LABEL[name], float(v)))
    by = {s.name: dict(s.pts) for s in series}
    rule = {r: by.get(f"next_rule: {label[r]}", {}) for r in ("fz", "t1")}
    atg = {r: by.get(f"actions_to_go: {label[r]}", {}) for r in ("fz", "t1")}
    best_rule = max((v for _n, v in baselines.get("next_rule", [])), default=None)
    best_atg = max((v for _n, v in baselines.get("actions_to_go", [])), default=None)
    takeaway = "next_rule accuracy and actions_to_go R2 by layer."
    common = sorted(set(rule["fz"]) & set(rule["t1"]))
    if common and best_rule is not None:
        last = common[-1]
        first_above = next(
            (x for x in common if min(rule["fz"][x], rule["t1"][x]) > best_rule), None
        )
        gap = max(abs(rule["t1"][x] - rule["fz"][x]) for x in common)
        takeaway = (
            "RL leaves the linearly decodable plan where it was: both models' next_rule probes "
            + (
                f"beat the best position-only lookup ({_pct(best_rule)}) from layer "
                f"{first_above:g} on"
                if first_above is not None
                else f"stay at or below the best position-only lookup ({_pct(best_rule)})"
            )
            + f", reaching {_pct(rule['fz'][last])} (fz) and {_pct(rule['t1'][last])} (t1) at "
            f"layer {last:g}, within {100 * gap:.1f} points of each other at every layer"
        )
        if atg["fz"] and atg["t1"] and best_atg is not None:
            la = max(set(atg["fz"]) & set(atg["t1"]))
            beats = max(atg["fz"][la], atg["t1"][la]) > best_atg
            takeaway += (
                f"; actions_to_go R² at layer {la:g} (fz {atg['fz'][la]:.2f}, t1 "
                f"{atg['t1'][la]:.2f}) {'beats' if beats else 'does not beat'} the lookup "
                f"({best_atg:.2f})"
            )
        takeaway += "."
    n_test = next((row[3] for row in per.get("fz", []) if row[3]), None)
    caption = (
        "Held-out accuracy of a linear probe for next_rule (the ND rule of the next "
        "line-creating action, (a)) and R² of a linear probe for actions_to_go (actions left in "
        "the proof, (b)) on the residual stream at each action's decision token, layers 0-5, "
        f"for fz and t1 (cap-12 seed-0 pair; {cfg.get('n_train', '?')} reference proofs with L = "
        f"{cfg.get('l_min', '?')}-{cfg.get('l_max', '?')} unseen in training, "
        f"{data['n_train_examples']} steps, split by theorem; {n_test} test steps). Grey lines: "
        "lookup tables fit on the same training split that predict from the step index and "
        "premise count (dotted) and also the frame depth (dashed), all visible from the state "
        "alone"
        + (
            f"; majority class {_pct(bl['next_rule_acc_majority'])}"
            if bl.get("next_rule_acc_majority") is not None
            else ""
        )
        + ". Point estimates on one split."
    )
    cd = ChartData(
        key,
        title,
        "line",
        takeaway,
        caption,
        series=series,
        opts={"x_label": "layer", "y_fmt": "f3"},
        extra={"baselines": baselines},
    )
    cd.extra_tables.append(
        (
            "position-only baselines",
            ["target", "lookup", "value"],
            [[m, n, f"{v:.3f}"] for m, vs in baselines.items() for n, v in vs],
        )
    )
    return cd


# ============================================================================ assembly


def dan_only_chart_data(
    store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None
) -> list[ChartData]:
    """Every chart of the page, in page order (shared by the page and `figures --dan-only`)."""
    reg = registry if registry is not None else mr.load_registry()
    tb = _c_textbook(store, reg)
    return [
        tb,
        _c_table1(store, reg, tb),
        _c_families(store, reg),
        _c_laws(store, reg),
        _c_glue(store, reg),
        _c_skills(store, reg),
        _c_padding(store, reg),
        _c_prefix(store, reg),
        _c_heldout(store, reg),
        _c_textbook_law(store, reg),
        _c_likelihood(store),
        _c_kl(store),
        _c_steering(store),
        _c_weights(store),
        _c_probes(store),
    ]


SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Eliciting proofs",
        "Pass rates of the pretrained (fz) and RL (t1) checkpoints: textbook theorems across "
        "sampling budgets, seeds and caps, unseen textbook law instances and Dan's held-out "
        "set; then generated benchmarks that vary one thing at a time: length within a family, "
        "substitution size in known laws, composing solved parts, finishing a given prefix, "
        "the skill a proof needs, and padding.",
        (
            "do_textbook",
            "do_table1",
            "do_textbook_law",
            "do_heldout",
            "do_families",
            "do_laws",
            "do_glue",
            "do_prefix",
            "do_skills",
            "do_padding",
        ),
    ),
    (
        "Internals",
        "Whether the change from fz to t1 shows up in the likelihood of fixed proofs, along a "
        "weight interpolation, in a token-level KL between their next-action distributions, "
        "under activation steering and in linear probes (cap-12 seed-0 pair).",
        ("do_likelihood", "do_weights", "do_kl", "do_steering", "do_probes"),
    ),
)


def numbering(charts: Sequence[ChartData]) -> dict[str, str]:
    """`key -> "Figure N" | "Table N"` in page order; Figure 1 is the overview."""
    out = {"overview": "Figure 1"}
    n_fig, n_tab = 1, 0
    by = {c.key: c for c in charts}
    for _h, _i, keys in SECTIONS:
        for k in keys:
            cd = by.get(k)
            if cd is None:
                continue
            if cd.kind == "table":
                n_tab += 1
                out[k] = f"Table {n_tab}"
            else:
                n_fig += 1
                out[k] = f"Figure {n_fig}"
    return out


def caption_text(cd: ChartData, label: str | None) -> str:
    head = f"{label}: " if label else ""
    if cd.kind == "missing":
        return f"{head}{cd.title}. Not available: {cd.missing}."
    return f"{head}{cd.title}. {cd.caption} {_cap_first(cd.takeaway)}".strip()


def _cap_first(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text and not text.startswith(("fz", "t1")) else text


def _caption_html(cd: ChartData, label: str | None) -> str:
    lab = f'<span class="fignum">{P.esc(label)}:</span> ' if label else ""
    if cd.kind == "missing":
        return f"{lab}<b>{P.esc(cd.title)}.</b> {P.esc(cd.missing)}."
    return (
        f"{lab}<b>{P.esc(cd.title)}.</b> {P.esc(cd.caption)} "
        f'<span class="takeaway">{P.esc(_cap_first(cd.takeaway))}</span>'
    )


def _table_html(cd: ChartData) -> str:
    t = cast("dict[str, Any]", cd.extra["table"])
    align = cast(str, t.get("align", ""))

    def cls(i: int) -> str:
        return ' class="l"' if i < len(align) and align[i] == "l" else ""

    th = "".join(f"<th{cls(i)}>{P.esc(h)}</th>" for i, h in enumerate(t["headers"]))
    body = "".join(
        "<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in t["rows"]
    )
    return (
        f'<table class="booktabs multi"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'
    )


def render_card(cd: ChartData, label: str | None) -> str:
    fid = f"{'tab' if cd.kind == 'table' else 'fig'}-{cd.key}"
    if cd.kind == "missing":
        return (
            f'<figure class="card chart-card" id="{fid}"><div class="missing">'
            f"{P.esc(cd.extra.get('missing_label') or 'no data yet')}</div>"
            f"<figcaption>{_caption_html(cd, label)}</figcaption></figure>"
        )
    details = "".join(summary._details_table(s, h, r) for s, h, r in summary.data_tables(cd))  # noqa: SLF001
    if cd.kind == "table":
        return (
            f'<figure class="card chart-card table-card" id="{fid}">'
            f'<figcaption class="above">{_caption_html(cd, label)}</figcaption>'
            f'<div class="tw">{_table_html(cd)}</div>{details}</figure>'
        )
    image = F.figure_uri(cd)
    if image is None:
        return render_card(ChartData(cd.key, cd.title, missing="no figure"), label)
    pic = summary._picture(image, F.figure_uri(cd, narrow=True), f"{label}: {cd.title}")  # noqa: SLF001
    return (
        f'<figure class="card chart-card" id="{fid}"><div class="figbox">{pic}</div>'
        f"<figcaption>{_caption_html(cd, label)}</figcaption>{details}</figure>"
    )


def _half(cd: ChartData) -> bool:
    if cd.kind == "missing":
        return True
    if cd.kind == "table":
        return bool(cd.opts.get("half"))
    w = F.width_of(cd.key)
    return w is not None and w < F.S.FULL_W - 0.5


def _section_html(cards: Sequence[tuple[ChartData, str]]) -> str:
    out: list[str] = []
    i = 0
    while i < len(cards):
        cd, h = cards[i]
        if _half(cd) and i + 1 < len(cards) and _half(cards[i + 1][0]):
            out.append(f'<div class="pair">{h}{cards[i + 1][1]}</div>')
            i += 2
            continue
        out.append(f'<div class="{"single half" if _half(cd) else "single"}">{h}</div>')
        i += 1
    return "".join(out)


OVERVIEW_TITLE = "Overview: every benchmark's fz vs t1 comparison"


def overview_caption(charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    def ref(key: str, sub: str = "") -> str:
        return nums.get(key, "?") + sub

    steer = next((c for c in charts if c.key == "do_steering"), None)
    t1_bar = ", t1" if steer is not None and steer.opts.get("has_t1_baseline") else ""
    return (
        "One panel per benchmark, each the headline comparison of the figure or table named, in "
        "page order; fz darker, t1 full hue (hatched bars), cap 12 reddish purple, cap 6 sky "
        "blue; that figure or table gives n, the CI and the takeaway. Cap-12 seed-0 pair unless "
        f"noted. (a) Pass@k on textbook_dev, log k ({ref('do_textbook', 'a')}). (b) textbook72 "
        "theorems solved at k = 256 by cap: bars = mean of three seeds, dots = seeds, black "
        f"ticks = Dan's published counts ({ref('do_table1')}). (c) Pass@k on unseen textbook_law "
        f"theorems ({ref('do_textbook_law')}). (d) Held-out greedy accuracy with 95% CI; open "
        f"triangles: Dan's reported value ({ref('do_heldout')}). (e) Written lines at the "
        f"horizon n* per restricted family, unseen theorems ({ref('do_families')}). (f) Pass@1 "
        f"against substitution size s ({ref('do_laws')}). (g) Compositionality ratio R per glue "
        f"operator, dotted R = 1 ({ref('do_glue')}). (h) Prefix completion against lines left "
        f"({ref('do_prefix')}). (i) Pass@1 by required skill ({ref('do_skills', 'a')}). (j) "
        "Pass@1 against pads added: P2 (extra ∧I lines, solid) and the two P1 kinds pooled "
        f"(distractor premises, hollow, dotted) ({ref('do_padding')}). (k) Median log p per line "
        f"of neutral proof sets ({ref('do_likelihood')}). (l) Pass@1 (black) and pass@128 (grey, "
        f"dashed) along the fz-to-t1 weight interpolation ({ref('do_weights')}). (m) Per-token "
        f"KL, pooled and for the four kinds with the largest KL ({ref('do_kl')}). (n) Pass@1 of "
        f"fz, the best fz + δ{t1_bar} and t1 − δ on dev and sealed generated theorems "
        f"({ref('do_steering')}). (o) next_rule probe accuracy by layer; "
        f"dashed: best position-only lookup ({ref('do_probes', 'a')}). (p) Key."
    )


def _overview_card(charts: Sequence[ChartData], nums: Mapping[str, str]) -> str:
    image = F.overview_uri(charts)
    if image is None:
        return ""
    pic = summary._picture(  # noqa: SLF001
        image, F.overview_uri(charts, narrow=True), f"Figure 1: {OVERVIEW_TITLE}"
    )
    return (
        f'<figure class="card overview-card" id="fig-overview"><div class="figbox">{pic}</div>'
        f'<figcaption><span class="fignum">Figure 1:</span> <b>{P.esc(OVERVIEW_TITLE)}.</b> '
        f"{P.esc(overview_caption(charts, nums))}</figcaption></figure>"
    )


# ===================================================================== lede, limitations


def _judge_totals(store: Store, reg: Mapping[str, mr.ModelEntry]) -> dict[str, int] | None:
    data = judges_view.view(store, cast("dict[str, JSON]", {"models": dv.dan_models(reg)}))
    if data.get("empty"):
        return None
    per = cast("dict[str, dict[str, Any]]", data["models"])
    keys = ("n_attempts", "agree_count", "lean_only_count", "nd_only_count")
    out = {k: sum(int(i[k]) for i in per.values()) for k in keys}
    out["missing_premise"] = sum(
        int(i["lean_only_breakdown"]["missing_premise_restatement"]["count"]) for i in per.values()
    )
    return out


def _agree_pct(j: Mapping[str, int]) -> str:
    rate = j["agree_count"] / j["n_attempts"] if j["n_attempts"] else float("nan")
    return f"{100 * rate:.2f}%"


def lede(j: Mapping[str, int] | None, models_with_textbook: int) -> str:
    agree = (
        f" Every attempt is judged by Lean (lean_v2, Dan's own judge); re-judging the same "
        f"{j['n_attempts']:,} attempts with our natural-deduction checker nd_v2 agrees on "
        f"{_agree_pct(j)}."
        if j
        else " Every attempt is judged by Lean (lean_v2, Dan's own judge)."
    )
    return (
        "Dan Pandori's best-state recipe trains a 9.56M-parameter ALiBiGPT (6 layers, width 384) "
        "to prove propositional theorems through a proof-state interface: at each step the model "
        "reads the current proof state and writes one action, the environment applies it, and "
        "Lean checks the finished proof. We compare each pretrained Stage-1 checkpoint (fz) "
        "with the same checkpoint after his 8-round T1 RL ladder (t1), for Stage-1 proofs of at "
        f"most 6 and at most 12 lines, three seeds each ({models_with_textbook} checkpoints), on "
        "our benchmark panel." + agree
    )


MODELS_NOTE = (
    "Labels: cap 12 / cap 6 (dan12 / dan6) = Stage-1 training proofs of at most 12 / 6 lines; "
    "fz = Stage-1 only (frozen); t1 = after the T1 ladder; s0-s2 = seeds. Figure 2 and Table 1 "
    "cover every checkpoint; every other result is the cap-12 seed-0 pair (dan12-fz-s0 vs "
    "dan12-t1-s0)."
)

_SEEN_SUITES: tuple[tuple[str, str], ...] = (
    ("textbook_dev", "textbook_dev"),
    ("textbook_train14", "textbook_train14"),
    ("families", "families"),
    ("laws", "laws"),
    ("glue", "glue"),
    ("glue_parts_renamed", "glue parts"),
    ("padding", "padding"),
    ("prefix_state", "prefix_state"),
    ("skill_pool_sub", "skill_pool_sub"),
    ("dan_heldout", "held-out"),
    ("textbook_law", "textbook_law"),
    ("interp_eval_dev", "interp_eval_dev"),
    ("interp_eval_sealed", "interp_eval_sealed"),
    ("interp_eval_k128", "k = 128 subset"),
)
_FILTERED = frozenset({"families", "laws", "textbook_law"})


def _truncation(
    store: Store, reg: Mapping[str, mr.ModelEntry]
) -> tuple[float, str, dict[str, float]] | None:
    """`(worst fraction, which run, {role: pooled fraction})` of textbook attempts that hit the
    action or step cap (`ItemResult.hit_max`: attempts that did not finish)."""
    worst: tuple[float, str] | None = None
    pooled: dict[str, list[int]] = {}
    for m in dv.dan_models(reg):
        for suite in ("textbook_dev", "textbook_train14"):
            run_id = latest_run_id(store, suite=suite, model=m, version="v1", judge_id=JUDGE)
            if run_id is None:
                continue
            its = list(items_by_pid(store, run_id).values())
            n = sum(i.n for i in its)
            hit = sum(i.hit_max for i in its)
            acc = pooled.setdefault(dv.role_of(m), [0, 0])
            acc[0] += hit
            acc[1] += n
            if n and (worst is None or hit / n > worst[0]):
                worst = (hit / n, f"{m} on {suite}")
    if worst is None:
        return None
    return worst[0], worst[1], {r: h / n for r, (h, n) in pooled.items() if n}


def limitations(
    store: Store,
    reg: Mapping[str, mr.ModelEntry],
    charts: Sequence[ChartData],
    nums: Mapping[str, str],
    j: Mapping[str, int] | None,
) -> list[str]:
    by = {c.key: c for c in charts}
    out: list[str] = []
    t72 = (
        cast("dict[str, dict[str, Any]]", by["do_table1"].extra.get("t72") or {})
        if "do_table1" in by
        else {}
    )
    spread = max(
        (max(v["ours"]) - min(v["ours"]) for v in t72.values() if len(v["ours"]) > 1), default=None
    )
    out.append(
        f"Only textbook pass@k ({nums.get('do_textbook', '?')}, {nums.get('do_table1', '?')}) "
        "covers all twelve checkpoints. Every other benchmark and measure is the cap-12 seed-0 "
        "pair alone: one seed, so those figures carry no seed-to-seed uncertainty"
        + (
            f" (seeds of one cell differ by up to {spread} of the 72 textbook theorems)"
            if spread is not None
            else ""
        )
        + ", and nothing there speaks to cap 6."
    )
    pad_cd = by.get("do_padding")
    n_base = cast("int | None", pad_cd.extra.get("n_bases")) if pad_cd is not None else None
    out.append(
        f"Padding: the {n_base or 'unpadded'} {'unpadded ' if n_base else ''}bases were chosen as "
        "short theorems a different prover (our "
        "earlier RL model) had mastered (pass@1 >= 0.7), not for Dan's models; and P1 pads add "
        "premises that Lean does not need restated, so for his models they add distractor "
        f"hypotheses, not lines ({nums.get('do_padding', '?')} plots pads added, not written "
        "length)."
    )
    seen_rows = []
    for suite, name in _SEEN_SUITES:
        version = default_version(store, suite=suite, models=list(PAIR), judge_id=JUDGE)
        if version is None:
            continue
        c = dv.seen_counts(store, "dan12", [(suite, version)])
        if c:
            seen_rows.append((suite, name, c[0][3], c[0][2]))
    if seen_rows:
        filt = [f"{n} {s:,}/{t:,}" for su, n, s, t in seen_rows if su in _FILTERED]
        rest = [f"{n} {s:,}/{t:,}" for su, n, s, t in seen_rows if su not in _FILTERED]
        out.append(
            "Training overlap is judged against Dan's cap-12 tiers (his Stage-1 set plus the T1 "
            "ladder's RL targets, matched by sequent). Families, laws and textbook_law exclude "
            "seen theorems (" + ", ".join(filt) + "); the other suites are not filtered, with "
            "seen counts " + ", ".join(rest) + "."
        )
    if j:
        out.append(
            "Judge: every number uses Lean on the literal sampled text (lean_v2, Dan's protocol "
            f"and RL reward). Re-judging the same {j['n_attempts']:,} attempts with nd_v2 agrees "
            f"on {_agree_pct(j)}: {j['lean_only_count']:,} Lean-only accepts, "
            f"{j['missing_premise']:,} of them rejected by nd_v2 with a premise error (premises "
            "not restated as natural-deduction lines, which Lean does not require), and "
            f"{j['nd_only_count']:,} nd_v2-only accepts."
        )
    trunc = _truncation(store, reg)
    if trunc is not None and trunc[0] > 0.001:
        worst, where, by_role = trunc
        roles = ", ".join(f"{r} {100 * f:.2f}%" for r, f in sorted(by_role.items()))
        out.append(
            "Attempts that hit the action or step cap (max_action 512, max_steps 96) count as "
            f"failures: up to {100 * worst:.1f}% of attempts in one textbook run ({where}); "
            f"pooled over every textbook run, {roles}. Dan reports the same pattern, and that "
            "doubling both caps raised his counts by at most 2 theorems where he tested it."
        )
    steer = by.get("do_steering")
    if steer is not None and steer.kind != "missing" and not steer.opts.get("has_t1_baseline"):
        out.append(
            "Steering has no unsteered t1 run on interp_eval_dev/sealed, so t1 − δ "
            f"({nums.get('do_steering', '?')}) has no paired baseline; it shows only that t1 "
            "keeps most of its accuracy without the direction."
        )
    lik = by.get("do_likelihood")
    if lik is not None and lik.kind != "missing":
        n_lik = sum(n for r in lik.rows if (n := _row_n(r.label)) is not None)
        out.append(
            f"Likelihood ({nums.get('do_likelihood', '?')}, run {lik.extra.get('run_id')}) rests "
            f"on {n_lik} fixed proofs"
            + (" (a small pilot)" if n_lik < 100 else "")
            + ", not model samples"
            + (
                "; it excludes the environment-assigned defining-name tokens."
                if lik.extra.get("exclude_defining_names")
                else ", and its scores include the environment-assigned defining-name tokens, "
                "which the environment overrides at sampling time and Dan's trajectory study "
                "excludes."
            )
        )
    kl = by.get("do_kl")
    if kl is not None and kl.kind == "missing":
        out.append(f"Token-level KL ({nums.get('do_kl', '?')}) has not been run yet.")
    elif kl is not None and kl.extra.get("n_skipped"):
        n_skip, n_rows = int(kl.extra["n_skipped"]), int(kl.extra["n_rows"])
        out.append(
            f"Token-level KL ({nums.get('do_kl', '?')}) covers in-grammar attempts only: "
            f"{n_skip:,} of {n_skip + n_rows:,} sampled attempts "
            f"({100 * n_skip / (n_skip + n_rows):.0f}%) were rejected by the environment as "
            "unparseable and have no action tokens to score."
        )
    out.append(
        f"Weight interpolation ({nums.get('do_weights', '?')}) is evaluated on the fixed "
        "150-theorem k = 128 subset only, not on the full interp_eval suites."
    )
    return out


def _row_n(label: str) -> int | None:
    """`"family witnesses (n=30)"` -> 30."""
    _head, sep, tail = label.rpartition("(n=")
    return int(tail.rstrip(")")) if sep and tail.rstrip(")").isdigit() else None


def _gpu_hours(store: Store) -> float:
    if not store.ledger_path.exists():
        return 0.0
    return (
        sum(
            float(cast(float, row.get("gpu_s") or 0.0))
            for row in read_jsonl(store.ledger_path)
            if "dan" in str(row.get("job", "")).lower()
        )
        / 3600.0
    )


_CSS_EXTRA = """
.rng{color:var(--muted);font-size:.8em;margin-left:3px;white-space:nowrap}
@media (max-width:700px){.rng{display:block;margin-left:0}}
.na{color:var(--muted);font-style:italic}
table.booktabs.multi{font-size:.9rem}
table.booktabs.multi th{text-align:right;vertical-align:bottom}
table.booktabs.multi th.l,table.booktabs.multi td.l{text-align:left}
table.booktabs.multi td{white-space:nowrap}
.takeaway{font-style:normal}
.limits li{margin:4px 0}
@media (max-width:700px){table.booktabs.multi{font-size:.8rem}
table.booktabs.multi th,table.booktabs.multi td{padding:3px 5px}}
"""


def render_page(store: Store, *, registry: Mapping[str, mr.ModelEntry] | None = None) -> str:
    reg = registry if registry is not None else mr.load_registry()
    charts = dan_only_chart_data(store, registry=reg)
    nums = numbering(charts)
    by = {c.key: c for c in charts}
    sections = []
    for i, (heading, intro, keys) in enumerate(SECTIONS, start=1):
        cards = [(by[k], render_card(by[k], nums.get(k))) for k in keys if k in by]
        sections.append(
            f'<section><h2><span class="secnum">{i}</span>{P.esc(heading)}</h2>'
            f'<p class="intro">{P.esc(intro)}</p>{_section_html(cards)}</section>'
        )
    j = _judge_totals(store, reg)
    n_ckpt = len(dv.dan_models(reg))
    limits = "".join(f"<li>{P.esc(t)}</li>" for t in limitations(store, reg, charts, nums, j))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Best-state Provers</title>"
        f"<style>{summary._CSS}{_CSS_EXTRA}</style></head><body>"  # noqa: SLF001
        '<div class="col"><header><h1>What RL changes in Dan\'s best-state provers</h1>'
        f'<p class="lede">{P.esc(lede(j, n_ckpt))}</p>'
        f'<p class="models">{P.esc(MODELS_NOTE)}</p></header>'
        f"<main>{_overview_card(charts, nums)}{''.join(sections)}"
        f'<section class="limits"><h2>Limitations</h2><ul>{limits}</ul></section></main>'
        f"<footer>GPU time on the ledger for jobs on Dan's models (benchmarks, interp, gates "
        f"and smoke tests): {_gpu_hours(store):.2f} GPU-hours. Source of Dan's published "
        "numbers: "
        f"{P.esc(dv.DAN_SOURCE)}.</footer></div>"
        f"<script>{summary._JS}</script></body></html>"  # noqa: SLF001
    )


def build_dan_only(
    store: Store,
    out_dir: Path,
    detail_href: str | None = None,
    *,
    registry: Mapping[str, mr.ModelEntry] | None = None,
) -> Path:
    """`dash --summary --dan-only`: write the page to `out_dir/index.html` (the site's only
    page, so it links nowhere else; `detail_href` is accepted for CLI symmetry and ignored)."""
    del detail_href
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(render_page(store, registry=registry))
    return path


#: Builders by key, for callers and tests that compute a single chart.
BUILDERS: dict[str, Callable[..., ChartData]] = {
    "do_textbook": _c_textbook,
    "do_families": _c_families,
    "do_laws": _c_laws,
    "do_glue": _c_glue,
    "do_skills": _c_skills,
    "do_padding": _c_padding,
    "do_prefix": _c_prefix,
    "do_heldout": _c_heldout,
    "do_textbook_law": _c_textbook_law,
    "do_likelihood": _c_likelihood,
    "do_kl": _c_kl,
    "do_steering": _c_steering,
    "do_weights": _c_weights,
    "do_probes": _c_probes,
}
