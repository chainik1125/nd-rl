"""Is thinking adaptive?  Think length and thinking gain against target difficulty.

    # 1. the fixed, labelled problem set (stratified RL-pool targets + all of textbook72)
    python -m code.thinking.adaptivity build-set --pool POOL.jsonl --textbook DIR/textbook72 \
        --per_cell 75 --set SET.jsonl --labels SET.labels.json
    # 2. one checkpoint: k attempts per target per effort level, plain arm, then the analysis
    python -m code.thinking.adaptivity --ckpt CKPT.pt --sets SET.jsonl --labels SET.labels.json \
        --k 16 --out RUNDIR [--budgets 0,16,64,256] [--batch 512] [--analyze_only]
    # 3. checkpoints side by side
    python -m code.thinking.adaptivity compare --runs "it0=RUNDIR0,it39=RUNDIR1" --out FILE.md

**Measurement.** One `code.thinking.eval` run per effort level B (arm plain, temperature 0.8, line
forcing, eval's defaults otherwise) into RUNDIR/eval/B<B>/.  A level whose rows exist is skipped,
so a measurement split over several `--budgets` calls resumes.

**Analysis** (RUNDIR/adaptivity.json and adaptivity.md) joins the rows with the labels by theorem
name.  Difficulty is read four ways: the pool filter's T0 pass@1 (k = 16, the Stage-1 T0 model, no
think) in bins 0, (0, .25], (.25, .5], (.5, .625]; the generated proof length (<= 12, 13-20);
textbook72 dev and train as their own groups; and the target's own e0 pass@1 under the measured
checkpoint.  Per level:

* think tokens per attempt (summed over the attempt's blocks, the failing draw's included) and per
  block (think / draws, a ratio of sums per target, so a long proof's many blocks do not count as
  long thinking), by bin with bootstrap CIs over targets; Spearman of the per-target mean against
  each difficulty measure, CI from a target bootstrap;
* thinking gain pass@1(eB) - pass@1(e0) by bin, paired on targets (one target resample shared by
  both levels).  Bins of the checkpoint's own e0 pass rate are formed from attempts [0, k/2) of e0
  and the e0 side of the gain uses attempts [k/2, k), so a bin is not selected on the noise it is
  compared with (regression to the mean);
* pass@1 and pass@5 (unbiased estimator, Chen et al. 2021), overall and per group;
* think length by attempt outcome: pooled over attempts, within targets (targets with both
  outcomes: failed minus accepted), and the pass rate by quartile of think per block.

This module imports torch and the vendored fork only to measure (`measure`); the analysis is numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
Bool = NDArray[np.bool_]

BUDGETS = (0, 16, 64, 256)
DRAWS = 2000
T0_BINS = ("0", "(0,.25]", "(.25,.5]", "(.5,.625]")  # hardest first
LEN_BINS = ("<=12", "13-20")
OWN_BINS = ("0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]")
GROUPS = ("rl", "textbook_dev", "textbook_train")

# ------------------------------------------------------------------------------------ the set


def t0_bin(p: float | None) -> str | None:
    """T0 pass@1 bin of a pool target (the pool keeps p <= 0.625)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return None
    if p == 0:
        return T0_BINS[0]
    if p <= 0.25:
        return T0_BINS[1]
    if p <= 0.5:
        return T0_BINS[2]
    if p <= 0.625:
        return T0_BINS[3]
    return None


def len_bin(n: float | None) -> str | None:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return None
    if n <= 12:
        return LEN_BINS[0]
    if n <= 20:
        return LEN_BINS[1]
    return None


def own_bin(p: float) -> str:
    """Bin of a pass rate in [0, 1] (the checkpoint's own e0 pass@1)."""
    if p == 0:
        return OWN_BINS[0]
    for hi, b in zip((0.25, 0.5, 0.75), OWN_BINS[1:4], strict=True):
        if p <= hi:
            return b
    return OWN_BINS[4]


def build_set(
    pool: Sequence[Mapping[str, Any]],
    textbook: Mapping[str, Sequence[Mapping[str, Any]]],
    per_cell: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """`per_cell` pool targets per (T0 bin, length bin) cell, drawn without replacement, plus every
    textbook record.  -> (eval records, labels by name).  A cell short of `per_cell` is an error."""
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in pool:
        tb, lb = t0_bin(r.get("t0_pass1")), len_bin(r.get("n_lines"))
        if tb is not None and lb is not None:
            cells.setdefault((tb, lb), []).append(r)
    rng = random.Random(seed)
    recs: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    for tb in T0_BINS:
        for lb in LEN_BINS:
            have = sorted(cells.get((tb, lb), []), key=lambda r: str(r["name"]))
            if len(have) < per_cell:
                raise ValueError(f"cell ({tb}, {lb}) has {len(have)} < {per_cell} targets")
            for r in rng.sample(have, per_cell):
                recs.append({"name": r["name"], "prompt": r["prompt"], "group": "rl"})
                labels[r["name"]] = dict(
                    group="rl",
                    t0_pass1=float(r["t0_pass1"]),
                    t0_n_ok=r.get("t0_n_ok"),
                    t0_k=r.get("t0_k"),
                    t0_bin=tb,
                    n_lines=int(r["n_lines"]),
                    len_bin=lb,
                    band=r.get("band"),
                )
    for group, rs in textbook.items():
        for r in rs:
            rec: dict[str, Any] = {"name": r["name"], "prompt": r["prompt"], "group": group}
            if r.get("reference_lines") is not None:
                rec["reference_lines"] = r["reference_lines"]
            recs.append(rec)
            labels[r["name"]] = dict(
                group=group,
                t0_pass1=None,
                t0_bin=None,
                n_lines=None,
                len_bin=None,
                reference_lines=r.get("reference_lines"),
            )
    if len(labels) != len(recs):
        raise ValueError("duplicate theorem names in the set")
    return recs, labels


# ---------------------------------------------------------------------------------- the table


@dataclass
class Level:
    """One effort level's attempts, T targets x k attempts."""

    ok: I64
    think: F64  # think tokens per attempt
    draws: F64  # think blocks (decoded draws) per attempt
    forced: F64  # blocks closed by the budget


@dataclass
class Table:
    names: list[str]
    group: list[str]
    t0: F64  # pool T0 pass@1 (nan: textbook)
    n_lines: F64  # generated proof length (nan: textbook)
    levels: dict[int, Level]

    @property
    def k(self) -> int:
        return int(next(iter(self.levels.values())).ok.shape[1])


def load_rows(run: str | Path) -> dict[int, list[dict[str, Any]]]:
    """RUNDIR/eval/B<B>/rows_plain_B<B>.jsonl -> rows by level."""
    out: dict[int, list[dict[str, Any]]] = {}
    for p in sorted(Path(run, "eval").glob("B*/rows_plain_B*.jsonl")):
        B = int(p.stem.rsplit("_B", 1)[1])
        out[B] = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def make_table(
    rows: Mapping[int, Sequence[Mapping[str, Any]]], labels: Mapping[str, Mapping[str, Any]]
) -> Table:
    """Align every level's rows on theorem name (every level must cover the same targets)."""
    if not rows:
        raise ValueError("no rows")
    names = [str(r["name"]) for r in next(iter(rows.values()))]
    if len(set(names)) != len(names):
        raise ValueError("duplicate theorem names in the rows")
    missing = [n for n in names if n not in labels]
    if missing:
        raise ValueError(f"{len(missing)} targets without labels, e.g. {missing[:3]}")
    levels: dict[int, Level] = {}
    for B, rs in sorted(rows.items()):
        by = {str(r["name"]): r for r in rs}
        if set(by) != set(names):
            raise ValueError(f"level {B} covers different targets")
        ordered = [by[n] for n in names]

        def arr(key: str, ordered: list[Mapping[str, Any]] = ordered) -> F64:
            return np.asarray([r[key] for r in ordered], dtype=np.float64)

        levels[B] = Level(
            ok=np.asarray([r["ok"] for r in ordered], dtype=np.int64),
            think=arr("think"),
            draws=arr("draws"),
            forced=arr("forced"),
        )
    ks = {lv.ok.shape[1] for lv in levels.values()}
    if len(ks) != 1:
        raise ValueError(f"levels have different k: {sorted(ks)}")

    def lab(key: str) -> F64:
        v = [labels[n].get(key) for n in names]
        return np.asarray([math.nan if x is None else float(x) for x in v], dtype=np.float64)

    return Table(
        names, [str(labels[n]["group"]) for n in names], lab("t0_pass1"), lab("n_lines"), levels
    )


# ------------------------------------------------------------------------------------- stats


def pass_at_k(n: int, c: I64, k: int) -> F64:
    """Unbiased pass@k per target from n attempts with c accepted (Chen et al. 2021)."""
    out = np.ones(c.shape, dtype=np.float64)
    for i, ci in enumerate(c.tolist()):
        if n - ci >= k:
            out[i] = 1.0 - math.prod((n - ci - j) / (n - j) for j in range(k))
    return out


def ranks(x: F64) -> F64:
    """Ranks 1..n, ties at their average rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new = np.r_[True, xs[1:] != xs[:-1]]  # first of each run of ties
    group = np.cumsum(new) - 1
    edges = np.flatnonzero(np.r_[new, True])  # run starts, then n
    avg = (edges[:-1] + edges[1:] - 1) / 2.0 + 1.0
    r = np.empty(len(x), dtype=np.float64)
    r[order] = avg[group]
    return r


def spearman(x: F64, y: F64) -> float:
    """Spearman's rho (Pearson on average ranks); nan when either side is constant."""
    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return math.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def boot_index(n: int, draws: int, seed: int) -> I64:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def ci(est: float, boot: F64, n: int, level: float = 0.95) -> dict[str, float]:
    b = boot[np.isfinite(boot)]
    if len(b) == 0:
        return dict(est=est, lo=math.nan, hi=math.nan, n=n)
    a = (1 - level) / 2
    return dict(est=est, lo=float(np.quantile(b, a)), hi=float(np.quantile(b, 1 - a)), n=n)


def mean_ci(v: F64, draws: int = DRAWS, seed: int = 0) -> dict[str, float]:
    """Mean over targets with a percentile bootstrap over targets (nan entries dropped)."""
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(est=math.nan, lo=math.nan, hi=math.nan, n=0)
    idx = boot_index(len(v), draws, seed)
    return ci(float(v.mean()), v[idx].mean(axis=1), len(v))


def spearman_ci(x: F64, y: F64, draws: int = DRAWS, seed: int = 0) -> dict[str, float]:
    """Spearman's rho with a percentile bootstrap over (x, y) pairs (pairs with nan dropped)."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return dict(est=math.nan, lo=math.nan, hi=math.nan, n=len(x))
    idx = boot_index(len(x), draws, seed)
    boot = np.asarray([spearman(x[i], y[i]) for i in idx], dtype=np.float64)
    return ci(spearman(x, y), boot, len(x))


# ----------------------------------------------------------------------------------- analysis


def per_target_think(lv: Level) -> tuple[F64, F64]:
    """-> (mean think tokens per attempt, think tokens per block) per target."""
    per_attempt = lv.think.mean(axis=1)
    blocks = lv.draws.sum(axis=1)
    per_block = np.divide(
        lv.think.sum(axis=1), blocks, out=np.full(len(blocks), np.nan), where=blocks > 0
    )
    return per_attempt, per_block


def p1(lv: Level, cols: slice = slice(None)) -> F64:
    return lv.ok[:, cols].mean(axis=1).astype(np.float64)


def difficulty_bins(t: Table) -> list[tuple[str, Bool]]:
    """The fixed breakdown: T0 bin x length cells, their margins, the textbook groups, all."""
    g = np.asarray(t.group)
    tb = np.asarray([t0_bin(None if math.isnan(x) else float(x)) or "" for x in t.t0])
    lb = np.asarray([len_bin(None if math.isnan(x) else float(x)) or "" for x in t.n_lines])
    out: list[tuple[str, Bool]] = []
    for b in T0_BINS:
        for ln in LEN_BINS:
            out.append((f"T0 {b} / len {ln}", (tb == b) & (lb == ln)))
    for b in T0_BINS:
        out.append((f"T0 {b}", tb == b))
    for ln in LEN_BINS:
        out.append((f"len {ln}", lb == ln))
    out.append(("RL pool (all)", g == "rl"))
    out.append(("textbook dev", g == "textbook_dev"))
    out.append(("textbook train", g == "textbook_train"))
    out.append(("all", np.ones(len(g), dtype=np.bool_)))
    return [(n, m) for n, m in out if m.any()]


def own_bins(t: Table, cols: slice = slice(None)) -> list[tuple[str, Bool]]:
    """Bins of the checkpoint's own e0 pass@1 over attempts `cols`."""
    lab = np.asarray([own_bin(float(p)) for p in p1(t.levels[0], cols)])
    return [(f"own e0 {b}", lab == b) for b in OWN_BINS if (lab == b).any()]


def think_by_bin(t: Table, draws: int, seed: int) -> dict[str, Any]:
    """Per level > 0, per bin: think per attempt and per block, blocks per attempt, forced share."""
    out: dict[str, Any] = {}
    bins = difficulty_bins(t) + own_bins(t)
    for B, lv in t.levels.items():
        if B == 0:
            continue
        pa, pb = per_target_think(lv)
        nblk = lv.draws.mean(axis=1)
        forced = np.divide(
            lv.forced.sum(axis=1),
            lv.draws.sum(axis=1),
            out=np.full(len(pa), np.nan),
            where=lv.draws.sum(axis=1) > 0,
        )
        out[str(B)] = {
            name: dict(
                think_attempt=mean_ci(pa[m], draws, seed),
                think_block=mean_ci(pb[m], draws, seed),
                blocks=float(nblk[m].mean()),
                forced_share=float(np.nanmean(forced[m])),
                pass1=float(p1(lv)[m].mean()),
            )
            for name, m in bins
        }
    return out


def gain_by_bin(t: Table, draws: int, seed: int) -> dict[str, Any]:
    """Per level > 0, per bin: pass@1(eB) - pass@1(e0), bootstrap paired on targets.  Own-e0 bins
    use the split halves of e0 (bin on the first half, gain against the second)."""
    out: dict[str, Any] = {}
    e0 = t.levels[0]
    h = t.k // 2
    for B, lv in t.levels.items():
        if B == 0:
            continue
        pB = p1(lv)
        cells: dict[str, Any] = {}
        for name, m in difficulty_bins(t):
            d = pB[m] - p1(e0)[m]
            cells[name] = dict(
                gain=mean_ci(d, draws, seed), p_e0=float(p1(e0)[m].mean()), p_eB=float(pB[m].mean())
            )
        if h >= 1:
            p0b = p1(e0, slice(h, None))
            for name, m in own_bins(t, slice(0, h)):
                d = pB[m] - p0b[m]
                cells[name + " (split half)"] = dict(
                    gain=mean_ci(d, draws, seed),
                    p_e0=float(p0b[m].mean()),
                    p_eB=float(pB[m].mean()),
                )
        out[str(B)] = cells
    return out


def passk(t: Table, draws: int, seed: int, ks: Sequence[int] = (1, 5)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g = np.asarray(t.group)
    groups = [("all", np.ones(len(g), dtype=np.bool_))] + [
        (x, g == x) for x in GROUPS if (g == x).any()
    ]
    for B, lv in t.levels.items():
        c = lv.ok.sum(axis=1)
        out[str(B)] = {
            name: {
                f"pass@{j}": mean_ci(100.0 * pass_at_k(t.k, c[m], j), draws, seed)
                for j in ks
                if j <= t.k
            }
            for name, m in groups
        }
    return out


def correlations(t: Table, draws: int, seed: int) -> dict[str, Any]:
    """Spearman of per-target think (per attempt, per block) and gain against difficulty."""
    out: dict[str, Any] = {}
    rl = np.asarray(t.group) == "rl"
    own = p1(t.levels[0])
    nan = np.full(len(own), np.nan)
    for B, lv in t.levels.items():
        if B == 0:
            continue
        pa, pb = per_target_think(lv)
        gain = p1(lv) - own
        cell: dict[str, Any] = {}
        for yname, y in (("think_attempt", pa), ("think_block", pb), ("gain_vs_e0", gain)):
            for xname, x in (
                ("t0_pass1 (RL)", np.where(rl, t.t0, nan)),
                ("n_lines (RL)", np.where(rl, t.n_lines, nan)),
                ("own_e0_pass1 (all)", own),
            ):
                if yname == "gain_vs_e0" and xname.startswith("own"):
                    continue  # shares e0's noise with the gain: see gain_by_bin's split halves
                cell[f"{yname} ~ {xname}"] = spearman_ci(x, y, draws, seed)
        out[str(B)] = cell
    return out


def by_outcome(t: Table, draws: int, seed: int) -> dict[str, Any]:
    """Per level > 0: think by attempt outcome (pooled, within target), pass by think quartile."""
    out: dict[str, Any] = {}
    for B, lv in t.levels.items():
        if B == 0:
            continue
        ok = lv.ok.astype(bool)
        blk = np.divide(lv.think, lv.draws, out=np.zeros_like(lv.think), where=lv.draws > 0)
        cell: dict[str, Any] = {
            "attempts_ok": int(ok.sum()),
            "attempts_fail": int((~ok).sum()),
        }
        for name, v in (("think_attempt", lv.think), ("think_block", blk)):
            cell[f"{name}_ok"] = float(v[ok].mean()) if ok.any() else math.nan
            cell[f"{name}_fail"] = float(v[~ok].mean()) if (~ok).any() else math.nan
            mixed = ok.any(axis=1) & (~ok).any(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                f_mean = np.where(~ok, v, 0).sum(axis=1) / (~ok).sum(axis=1)
                o_mean = np.where(ok, v, 0).sum(axis=1) / ok.sum(axis=1)
            cell[f"{name}_within_fail_minus_ok"] = mean_ci((f_mean - o_mean)[mixed], draws, seed)
        cell["forced_share_ok"] = (
            float(lv.forced[ok].sum() / max(lv.draws[ok].sum(), 1)) if ok.any() else math.nan
        )
        cell["forced_share_fail"] = (
            float(lv.forced[~ok].sum() / max(lv.draws[~ok].sum(), 1)) if (~ok).any() else math.nan
        )
        qs = np.quantile(blk, [0.25, 0.5, 0.75])
        q = np.searchsorted(qs, blk, side="right")
        cell["pass_by_think_block_quartile"] = [
            dict(
                quartile=i + 1,
                lo=float(blk[q == i].min()) if (q == i).any() else math.nan,
                hi=float(blk[q == i].max()) if (q == i).any() else math.nan,
                n=int((q == i).sum()),
                pass_pct=100.0 * float(ok[q == i].mean()) if (q == i).any() else math.nan,
            )
            for i in range(4)
        ]
        out[str(B)] = cell
    return out


def analyze(t: Table, draws: int = DRAWS, seed: int = 0) -> dict[str, Any]:
    if 0 not in t.levels:
        raise ValueError("the analysis needs the e0 level")
    g = np.asarray(t.group)
    return dict(
        n_targets=len(t.names),
        k=t.k,
        levels=sorted(t.levels),
        groups={x: int((g == x).sum()) for x in GROUPS},
        draws=draws,
        passk=passk(t, draws, seed),
        think=think_by_bin(t, draws, seed),
        gain=gain_by_bin(t, draws, seed),
        corr=correlations(t, draws, seed),
        outcome=by_outcome(t, draws, seed),
    )


# ----------------------------------------------------------------------------------- markdown


NA = dict(est=math.nan, lo=math.nan, hi=math.nan, n=0)


def fmt_ci(e: Mapping[str, float], scale: float = 1.0, signed: bool = False, nd: int = 1) -> str:
    if not math.isfinite(e["est"]):
        return "n/a"
    s = "+" if signed else ""
    v, lo, hi = e["est"] * scale, e["lo"] * scale, e["hi"] * scale
    return f"{v:{s}.{nd}f} [{lo:{s}.{nd}f}, {hi:{s}.{nd}f}]"


def md_table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render(res: Mapping[str, Any], title: str) -> str:
    """One checkpoint's analysis as markdown tables."""
    lv = [str(lvl) for lvl in res["levels"]]
    pos = [lvl for lvl in lv if lvl != "0"]
    parts = [
        f"# {title}",
        "",
        f"{res['n_targets']} targets ({res['groups']}), k = {res['k']} attempts per target per "
        f"effort level, plain arm.  Intervals: 95% percentile bootstrap over targets "
        f"({res['draws']} draws).  Think tokens are summed over an attempt's blocks (per attempt) "
        "or divided by its blocks (per block).",
        "",
        "## pass@1 and pass@5 (% of targets)",
        "",
    ]
    rows = []
    for lvl in lv:
        for gname, cell in res["passk"][lvl].items():
            rows.append(
                [f"e{lvl}", gname] + [fmt_ci(cell.get(x, NA)) for x in ("pass@1", "pass@5")]
            )
    parts += [md_table(["level", "group", "pass@1", "pass@5"], rows), ""]
    for lvl in pos:
        parts += [f"## e{lvl}: think length and thinking gain by difficulty", ""]
        rows = []
        for name, c in res["think"][lvl].items():
            g = res["gain"][lvl].get(name)
            rows.append(
                [
                    name,
                    str(int(c["think_attempt"]["n"])),
                    fmt_ci(c["think_attempt"]),
                    fmt_ci(c["think_block"]),
                    f"{c['blocks']:.1f}",
                    f"{100 * c['forced_share']:.0f}%",
                    f"{100 * g['p_e0']:.1f}" if g else "",
                    f"{100 * g['p_eB']:.1f}" if g else "",
                    fmt_ci(g["gain"], 100, signed=True) if g else "(see split half)",
                ]
            )
        for name, g in res["gain"][lvl].items():
            if name.endswith("(split half)"):
                rows.append(
                    [
                        name,
                        str(int(g["gain"]["n"])),
                        "",
                        "",
                        "",
                        "",
                        f"{100 * g['p_e0']:.1f}",
                        f"{100 * g['p_eB']:.1f}",
                        fmt_ci(g["gain"], 100, signed=True),
                    ]
                )
        head = [
            "bin",
            "n",
            "think/attempt",
            "think/block",
            "blocks",
            "forced",
            "pass@1 e0 %",
            f"pass@1 e{lvl} %",
            f"gain e{lvl}-e0 (pp)",
        ]
        parts += [md_table(head, rows), ""]
    parts += ["## Spearman correlations (per-target means)", ""]
    rows = []
    for lvl in pos:
        for name, e in res["corr"][lvl].items():
            rows.append([f"e{lvl}", name, str(int(e["n"])), fmt_ci(e, signed=True, nd=3)])
    parts += [md_table(["level", "y ~ x", "n", "rho [95% CI]"], rows), ""]
    parts += ["## Think length by attempt outcome", ""]
    rows = []
    for lvl in pos:
        c = res["outcome"][lvl]
        rows.append(
            [
                f"e{lvl}",
                f"{c['attempts_ok']}/{c['attempts_fail']}",
                f"{c['think_attempt_ok']:.1f} / {c['think_attempt_fail']:.1f}",
                f"{c['think_block_ok']:.1f} / {c['think_block_fail']:.1f}",
                fmt_ci(c["think_block_within_fail_minus_ok"], signed=True),
                fmt_ci(c["think_attempt_within_fail_minus_ok"], signed=True),
                f"{100 * c['forced_share_ok']:.0f}% / {100 * c['forced_share_fail']:.0f}%",
            ]
        )
    head = [
        "level",
        "attempts ok/fail",
        "think/attempt ok/fail",
        "think/block ok/fail",
        "within-target fail-ok, per block",
        "within-target fail-ok, per attempt",
        "forced ok/fail",
    ]
    parts += [md_table(head, rows), ""]
    rows = []
    for lvl in pos:
        for q in res["outcome"][lvl]["pass_by_think_block_quartile"]:
            rows.append(
                [
                    f"e{lvl}",
                    str(q["quartile"]),
                    f"{q['lo']:.1f}-{q['hi']:.1f}",
                    str(q["n"]),
                    f"{q['pass_pct']:.1f}",
                ]
            )
    parts += [
        "Pass rate by quartile of think per block (attempts pooled over targets):",
        "",
        md_table(["level", "quartile", "think/block range", "attempts", "pass %"], rows),
        "",
    ]
    return "\n".join(parts)


COMPACT_BINS = (
    "T0 0",
    "T0 (0,.25]",
    "T0 (.25,.5]",
    "T0 (.5,.625]",
    "len <=12",
    "len 13-20",
    "textbook dev",
    "textbook train",
    "all",
)


def compare(results: Mapping[str, Mapping[str, Any]], B: str = "256") -> str:
    """Checkpoints side by side at one level: think per block and gain by the compact bins, pass@1
    and pass@5 per level, the correlations and the outcome split."""
    names = list(results)
    parts = ["### pass@1 / pass@5 (% of all targets)", ""]
    levels = [str(x) for x in next(iter(results.values()))["levels"]]
    rows = []
    for lvl in levels:
        rows.append(
            [f"e{lvl}"]
            + [
                " / ".join(
                    fmt_ci(results[n]["passk"][lvl]["all"].get(x, NA)) for x in ("pass@1", "pass@5")
                )
                for n in names
            ]
        )
    parts += [md_table(["level"] + names, rows), ""]
    for metric, label in (
        ("think_block", "think tokens per block"),
        ("think_attempt", "think tokens per attempt"),
    ):
        parts += [f"### e{B}: {label} by difficulty", ""]
        rows = []
        for b in COMPACT_BINS:
            rows.append(
                [b]
                + [
                    fmt_ci(results[n]["think"][B][b][metric]) if b in results[n]["think"][B] else ""
                    for n in names
                ]
            )
        parts += [md_table(["bin"] + names, rows), ""]
    for lvl in [x for x in levels if x != "0"]:
        parts += [f"### gain pass@1(e{lvl}) - pass@1(e0), pp", ""]
        rows = []
        bins = list(COMPACT_BINS) + [
            k for k in next(iter(results.values()))["gain"][lvl] if k.endswith("(split half)")
        ]
        for b in bins:
            rows.append(
                [b]
                + [
                    fmt_ci(results[n]["gain"][lvl][b]["gain"], 100, signed=True)
                    if b in results[n]["gain"][lvl]
                    else ""
                    for n in names
                ]
            )
        parts += [md_table(["bin"] + names, rows), ""]
    parts += ["### Spearman correlations", ""]
    keys = list(next(iter(results.values()))["corr"][B])
    rows = [
        [f"e{B}: {k}"] + [fmt_ci(results[n]["corr"][B][k], signed=True, nd=3) for n in names]
        for k in keys
    ]
    parts += [md_table(["y ~ x"] + names, rows), ""]
    parts += [f"### e{B}: think by attempt outcome", ""]
    rows = []
    for key, lab in (("think_block", "think/block"), ("think_attempt", "think/attempt")):
        rows.append(
            [f"{lab} ok / fail (pooled)"]
            + [
                f"{results[n]['outcome'][B][key + '_ok']:.1f} / "
                f"{results[n]['outcome'][B][key + '_fail']:.1f}"
                for n in names
            ]
        )
        rows.append(
            [f"{lab} fail - ok (within target)"]
            + [
                fmt_ci(results[n]["outcome"][B][key + "_within_fail_minus_ok"], signed=True)
                for n in names
            ]
        )
    rows.append(
        ["pass % by think/block quartile (Q1..Q4)"]
        + [
            " / ".join(
                f"{q['pass_pct']:.0f}"
                for q in results[n]["outcome"][B]["pass_by_think_block_quartile"]
            )
            for n in names
        ]
    )
    parts += [md_table(["measure"] + names, rows), ""]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------- measuring


def measure(a: argparse.Namespace, budgets: Sequence[int]) -> None:
    """One `code.thinking.eval` run per level not yet in RUNDIR/eval/B<B>/."""
    from . import eval as E  # torch and the vendored fork: only to measure

    for B in budgets:
        d = Path(a.out) / "eval" / f"B{B}"
        rows = d / f"rows_plain_B{B}.jsonl"
        if rows.exists():
            print(f"[adaptivity] e{B}: rows exist, skipped ({rows})", flush=True)
            continue
        argv = [
            "--ckpt",
            a.ckpt,
            "--sets",
            a.sets,
            "--out",
            str(d),
            "--k",
            str(a.k),
            "--budgets",
            str(B),
            "--arms",
            "plain",
            "--temperature",
            str(a.temperature),
            "--batch",
            str(a.batch),
            "--seed",
            str(a.seed),
            "--force_mode",
            "line",
        ]
        if a.limit:
            argv += ["--limit", str(a.limit)]
        if a.lean:
            argv += ["--lean", a.lean]
        E.run(E.parser().parse_args(argv))
        if not rows.exists():
            raise SystemExit(f"error: eval wrote no {rows.name} (a checkpoint without think?)")


def load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text())["targets"]


def analyze_run(
    run: str | Path, labels: Mapping[str, Mapping[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    return analyze(make_table(load_rows(run), labels), draws, seed)


def write_outputs(run: Path, res: Mapping[str, Any], title: str) -> None:
    (run / "adaptivity.json").write_text(json.dumps(res, indent=1))
    (run / "adaptivity.md").write_text(render(res, title) + "\n")


# ---------------------------------------------------------------------------------------- CLI


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default=None, help="checkpoint (not needed with --analyze_only)")
    ap.add_argument("--sets", required=True, help="the eval set (build-set's --set)")
    ap.add_argument("--labels", required=True, help="the labels sidecar (build-set's --labels)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", required=True, help="RUNDIR")
    ap.add_argument("--budgets", default="0,16,64,256", help="levels to measure in this call")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N targets only (a pilot)")
    ap.add_argument("--lean", default=None)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--title", default=None)
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--no_analyze", action="store_true", help="measure only")
    return ap


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="the stratified, labelled set")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--textbook", required=True, help="textbook72 directory")
    ap.add_argument("--per_cell", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", required=True)
    ap.add_argument("--labels", required=True)
    return ap


def compare_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="checkpoints side by side")
    ap.add_argument("--runs", required=True, help="NAME=RUNDIR,...")
    ap.add_argument("--level", default="256")
    ap.add_argument("--out", required=True)
    return ap


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main_build(argv: list[str]) -> int:
    a = build_parser().parse_args(argv)
    tb = Path(a.textbook)
    recs, labels = build_set(
        read_jsonl(a.pool),
        {
            "textbook_dev": read_jsonl(tb / "textbook_dev.jsonl"),
            "textbook_train": read_jsonl(tb / "textbook_train.jsonl"),
        },
        a.per_cell,
        a.seed,
    )
    Path(a.set).parent.mkdir(parents=True, exist_ok=True)
    Path(a.set).write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta = dict(
        pool=a.pool,
        textbook=a.textbook,
        per_cell=a.per_cell,
        seed=a.seed,
        t0_bins=T0_BINS,
        len_bins=LEN_BINS,
        n=len(recs),
        t0_source="pool record fields t0_pass1 = t0_n_ok / t0_k (Stage-1 T0, k = 16, "
        "temperature 0.8, plain, no think)",
        length_source="pool record field n_lines (the generator's proof length)",
    )
    Path(a.labels).write_text(json.dumps({"meta": meta, "targets": labels}, indent=1))
    print(f"{len(recs)} targets -> {a.set}, labels -> {a.labels}")
    return 0


def main_compare(argv: list[str]) -> int:
    a = compare_parser().parse_args(argv)
    results = {}
    for item in a.runs.split(","):
        name, d = item.split("=", 1)
        results[name] = json.loads(Path(d, "adaptivity.json").read_text())
    Path(a.out).write_text(compare(results, a.level) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sub: dict[str, Callable[[list[str]], int]] = {"build-set": main_build, "compare": main_compare}
    if argv and argv[0] in sub:
        return sub[argv[0]](argv[1:])
    a = parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        if a.ckpt is None:
            raise SystemExit("error: --ckpt is required unless --analyze_only")
        measure(a, [int(b) for b in a.budgets.split(",")])
    if a.no_analyze:
        return 0
    res = analyze_run(out, load_labels(a.labels), a.draws, 0)
    res["ckpt"] = a.ckpt
    write_outputs(out, res, a.title or f"Adaptivity: {out.name}")
    print(f"[adaptivity] -> {out / 'adaptivity.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
