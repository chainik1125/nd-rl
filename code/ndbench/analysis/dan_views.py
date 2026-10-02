"""Dan-only data views for the best-state results page (`dash/dan_only.py`, `dash --summary
--dan-only`).

The audited "State-env models" builders (`dash/dan_pages.py`, pinned by
`code/tests/ndbench/test_dan_section_audit.py`) already cover textbook pass@k, families, laws,
glue, padding (pooled), prefix_state, skills, held-out greedy and judge agreement; the Dan-only
page calls those. This module adds only what they do not compute, each with the estimator of the
nearest existing view:

- `textbook72_counts`: theorems solved within each item's first sampling round (k = 256 on the
  real store) on textbook_dev + textbook_train14, Dan's own headline statistic, next to his
  published counts (`DAN_PUBLISHED`);
- `padding_by_pad_type`: `dash.summary._padded_curve_dedup`'s per-item pass@1, split by pad
  type and indexed by pads added `m` instead of pooled by written length;
- `textbook_law_unseen`: `analysis.yue`'s own pass@k estimator (`_curve_for_model`) on the
  textbook_law items unseen in the model's training tiers;
- `dan_steering` / `dan_interpolation`: `analysis.interp.view_steering` / `view_interpolation`
  for Dan's labels (`steer:<model>:<layer>:<alpha>`, `<family>-interp:<alpha>`) and judge;
- `dan_probes`, `dan_likelihood`, `dan_kl`: the newest interp artefact run whose rows score
  Dan's fz/t1 pair, plus the probes run's position-only baselines (printed in the run's own
  output, `interp/danstate_support.run_probes_dan`, not stored as a separate artefact);
- `seen_counts`: how many items of each suite Dan's own training tiers contain
  (`logic.exclusion.seen_for`).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..core.io import read_jsonl
from ..core.types import JSON, ItemResult
from ..interp import eval_suites
from ..interp._store import interp_runs, load_interp_rows
from ..logic.exclusion import seen_for
from ..metrics import pass_curve
from ..store import Store
from . import models_registry as mr
from . import yue as yue_view
from ._common import (
    default_version,
    first_chunk_by_pid,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
    run_records,
)

DAN_JUDGE_ID = mr.DAN_JUDGE_ID
ND_JUDGE_ID = "nd_v2"

#: Dan's published per-seed numbers (seeds 0/1/2) for these twelve checkpoints, from his
#: best-state summary (`origin/dan:experiment-summaries/2026-09-30-best-state-robbies-recipe-
#: wins-after-rl-level-frozen-at-cap12/README.md`, "Per-seed counts"; the dev58 split from
#: `artifacts/bs/analysis_stdout.txt`, `DANSTATE_PLAN.md` §0). Lean, k = 256, T = 0.8.
DAN_PUBLISHED: dict[str, dict[tuple[str, str], tuple[float, ...]]] = {
    "textbook72": {
        ("dan6", "fz"): (19, 18, 20),
        ("dan6", "t1"): (39, 40, 33),
        ("dan12", "fz"): (32, 27, 27),
        ("dan12", "t1"): (52, 51, 52),
    },
    "dev58": {
        ("dan6", "fz"): (13, 13, 15),
        ("dan6", "t1"): (34, 33, 26),
        ("dan12", "fz"): (27, 22, 21),
        ("dan12", "t1"): (42, 44, 42),
    },
    "heldout": {
        ("dan6", "fz"): (0.963, 0.967, 0.955),
        ("dan6", "t1"): (0.992, 0.996, 0.994),
        ("dan12", "fz"): (0.926, 0.951, 0.903),
        ("dan12", "t1"): (0.992, 0.990, 0.993),
    },
}
DAN_SOURCE = (
    "Dan's best-state summary (origin/dan, experiment-summaries/2026-09-30-best-state-robbies-"
    "recipe-wins-after-rl-level-frozen-at-cap12)"
)


def dan_models(registry: Mapping[str, mr.ModelEntry]) -> list[str]:
    """Every Dan label in the registry, cap 12 before cap 6, then role, then seed."""
    out: list[str] = []
    for family in ("dan12", "dan6"):
        for role in ("base", "rl"):
            out += mr.seed_labels(registry, family, role)
    return out


def role_of(label: str) -> str:
    """`"dan12-t1-s0"` -> `"t1"` (Dan's own role names, as in the labels)."""
    parts = label.split("-")
    return parts[1] if len(parts) >= 3 else label


def cap_of(label: str) -> int | None:
    family = label.split("-", 1)[0]
    return int(family[3:]) if family.startswith("dan") and family[3:].isdigit() else None


# ======================================================================= textbook72 counts


def _first_round_solved(store: Store, run_id: str) -> tuple[int, int, int]:
    """`(solved, n_items, k)`: items with a success in their first sampling round, the item count,
    and that round's (smallest) draw count -- 256 for every textbook run on the real store, so
    `solved` is Dan's own "solved at k = 256" count (his read-outs draw exactly 256)."""
    first = first_chunk_by_pid(store, run_id)
    if not first:
        return 0, 0, 0
    k = min(int(f["n"]) for f in first.values())
    solved = sum(1 for f in first.values() if int(f["c"]) > 0)
    return solved, len(first), k


def textbook72_counts(
    store: Store, models: Sequence[str], judge_id: str = DAN_JUDGE_ID
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """`{model: {"dev": (solved, n, k), "train14": (...)}}` for every model with both runs."""
    out: dict[str, dict[str, tuple[int, int, int]]] = {}
    for m in models:
        row: dict[str, tuple[int, int, int]] = {}
        for key, suite in (("dev", "textbook_dev"), ("train14", "textbook_train14")):
            version = default_version(store, suite=suite, models=[m], judge_id=judge_id) or "v1"
            run_id = latest_run_id(store, suite=suite, model=m, version=version, judge_id=judge_id)
            if run_id is not None:
                row[key] = _first_round_solved(store, run_id)
        if row:
            out[m] = row
    return out


# ================================================================================ padding

PAD_TYPES: tuple[str, ...] = ("P2", "P1-shared", "P1-disjoint")


def padding_by_pad_type(
    store: Store, model: str, *, judge_id: str = DAN_JUDGE_ID, suite: str = "padding"
) -> dict[str, list[tuple[int, float, int, float]]]:
    """`{pad_type: [(m, mean pass@1, n items, mean written lines W), ...]}`, m = pads added.

    `dash.summary._padded_curve_dedup`'s estimator (mean of each item's c/n), split by
    `tags.pad_type`: every unpadded theorem (m = 0) appears once per pad type in the suite, so
    m = 0 is the mean over its copies, counted once per base and shared by every pad type."""
    version = default_version(store, suite=suite, models=[model], judge_id=judge_id) or "v1"
    run_id = latest_run_id(store, suite=suite, model=model, version=version, judge_id=judge_id)
    if run_id is None:
        return {}
    items = items_by_pid(store, run_id)
    bases: dict[str, list[float]] = {}
    base_w: dict[str, float] = {}
    by: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for pid, prob in problems_by_pid(store, suite, version).items():
        item = items.get(pid)
        if item is None or not item.n:
            continue
        m = int(cast(int, prob.tags.get("m", 0)))
        w = float(prob.length.L) if prob.length.L is not None else float("nan")
        if m == 0:
            key = str(prob.tags.get("base_pid", pid))
            bases.setdefault(key, []).append(item.c / item.n)
            base_w[key] = w
            continue
        pad_type = str(prob.tags.get("pad_type", ""))
        by.setdefault(pad_type, {}).setdefault(m, []).append((item.c / item.n, w))
    out: dict[str, list[tuple[int, float, int, float]]] = {}
    m0 = (
        (
            0,
            statistics.mean(statistics.mean(v) for v in bases.values()),
            len(bases),
            statistics.mean(base_w.values()),
        )
        if bases
        else None
    )
    for pad_type in [t for t in PAD_TYPES if t in by] + sorted(set(by) - set(PAD_TYPES)):
        pts = [
            (m, statistics.mean(p for p, _w in v), len(v), statistics.mean(w for _p, w in v))
            for m, v in sorted(by[pad_type].items())
        ]
        out[pad_type] = ([m0] if m0 is not None else []) + pts
    return out


# ========================================================================= textbook_law


def textbook_law_unseen(
    store: Store,
    models: Sequence[str],
    *,
    registry: Mapping[str, mr.ModelEntry],
    judge_id: str = DAN_JUDGE_ID,
    B: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """`{model: {"curve": [(k, mean, lo, hi)], "n": unseen items, "n_seen": excluded}}` on
    textbook_law: `analysis.yue`'s pass@k estimator and item bootstrap on the items unseen in the
    model's own training tiers (`models_registry.seen_lookup`), k = 1, 2, 4, ... up to the run's
    draw count."""
    suite = "textbook_law"
    out: dict[str, dict[str, Any]] = {}
    for m in models:
        version = default_version(store, suite=suite, models=[m], judge_id=judge_id) or "v1"
        run_id = latest_run_id(store, suite=suite, model=m, version=version, judge_id=judge_id)
        if run_id is None:
            continue
        problems = problems_by_pid(store, suite, version)
        seen = mr.seen_lookup(store, suite, version, m, problems, registry=registry)
        first = first_chunk_by_pid(store, run_id)
        rows: dict[str, tuple[int, int, int, int]] = {}
        n_seen = 0
        for pid, item in items_by_pid(store, run_id).items():
            if seen.get(pid, problems[pid].seen_train if pid in problems else False):
                n_seen += 1
                continue
            f = first.get(pid)
            n_first, c_first = (int(f["n"]), int(f["c"])) if f is not None else (item.n, item.c)
            rows[pid] = (n_first, c_first, item.n, item.c)
        if not rows:
            continue
        k_max = yue_view._model_k_max(rows)  # noqa: SLF001
        ks = [k for k in yue_view.KS if k <= k_max]
        curve, _lower = yue_view._curve_for_model(rows, ks, B, seed)  # noqa: SLF001
        out[m] = {"curve": curve, "n": len(rows), "n_seen": n_seen, "k_max": k_max}
    return out


# ================================================================================ steering

_STEER_RE = re.compile(r"^steer:(?P<model>.+):(?P<layer>-?\d+):(?P<alpha>-?[0-9.]+)$")
STEER_SUITES: tuple[str, ...] = (eval_suites.DEV_GEN_SUITE, eval_suites.SEALED_GEN_SUITE)
SWEEP_K = 8


def _sweep_run_id(store: Store, *, suite: str, model: str, k: int, judge_id: str) -> str | None:
    """`analysis.interp._sweep_run_id` with a judge filter: the model's run at the sweep's own k
    (most recently finished), else its largest-k run."""
    version = default_version(store, suite=suite, models=[model], judge_id=judge_id) or "v1"
    cands = [
        r
        for r in run_records(store, suite=suite, version=version, model=model, judge_id=judge_id)
        if int(cast("int | None", r.get("k")) or 0) == k
    ]
    if cands:
        cands.sort(key=lambda r: cast(str, r.get("finished_utc") or r.get("started_utc") or ""))
        return cast(str, cands[-1]["run_id"])
    return latest_run_id(store, suite=suite, model=model, version=version, judge_id=judge_id)


def _pass1(items: Sequence[ItemResult], B: int, seed: int) -> tuple[float, float, float, int]:
    _k, mean, lo, hi = pass_curve(list(items), [1], B=B, seed=seed)[0]
    return mean, lo, hi, len(items)


def dan_steering(
    store: Store,
    *,
    base: str = "dan12-fz-s0",
    rl: str = "dan12-t1-s0",
    suites: Sequence[str] = STEER_SUITES,
    k: int = SWEEP_K,
    judge_id: str = DAN_JUDGE_ID,
    B: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Pass@1 (95% item bootstrap, `metrics.pass_curve`) per suite of the unsteered pair and of
    every `steer:<base|rl>:<layer>:<alpha>` run, at the sweep's own k. Rows: `{"label", "model",
    "steered", "layer", "alpha", "per_suite": {suite: (mean, lo, hi, n) | None}}`."""
    present: set[str] = set()
    for suite in suites:
        present |= set(models_present(store, suite=suite, judge_id=judge_id))
    rows: list[dict[str, Any]] = []

    def _per_suite(model: str) -> dict[str, tuple[float, float, float, int] | None]:
        out: dict[str, tuple[float, float, float, int] | None] = {}
        for suite in suites:
            run_id = _sweep_run_id(store, suite=suite, model=model, k=k, judge_id=judge_id)
            items = list(items_by_pid(store, run_id).values()) if run_id else []
            out[suite] = _pass1(items, B, seed) if items else None
        return out

    steered: list[tuple[str, int, float, str]] = []
    for label in sorted(present):
        match = _STEER_RE.match(label)
        if match and match.group("model") in (base, rl):
            layer, alpha = int(match.group("layer")), float(match.group("alpha"))
            steered.append((match.group("model"), layer, alpha, label))
    for model in (base, rl):
        rows.append(
            {
                "label": model,
                "model": model,
                "steered": False,
                "layer": None,
                "alpha": None,
                "per_suite": _per_suite(model),
            }
        )
    for model, layer, alpha, label in sorted(steered, key=lambda t: (t[0] != base, t[1], t[2])):
        rows.append(
            {
                "label": label,
                "model": model,
                "steered": True,
                "layer": layer,
                "alpha": alpha,
                "per_suite": _per_suite(label),
            }
        )
    any_data = any(v is not None for r in rows for v in r["per_suite"].values())
    return {"rows": rows if any_data else [], "suites": list(suites), "k": k, "B": B}


# =========================================================================== interpolation

INTERP_KS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


def dan_interpolation(
    store: Store,
    *,
    base: str = "dan12-fz-s0",
    rl: str = "dan12-t1-s0",
    family: str = "dan12",
    suite: str = eval_suites.K128_SUITE,
    judge_id: str = DAN_JUDGE_ID,
    B: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """`analysis.interp.view_interpolation`'s pass@k on the fixed k = 128 subset for
    theta_fz + alpha (theta_t1 - theta_fz): `base` is alpha 0, `rl` alpha 1, `<family>-interp:<a>`
    the rest. `{"curves": {alpha: [(k, mean, lo, hi)]}, "n": {alpha: items}}`."""
    pat = re.compile(rf"^{re.escape(family)}-interp:(-?[0-9.]+)$")
    labels: dict[float, str] = {}
    for label in models_present(store, suite=suite, judge_id=judge_id):
        match = pat.match(label)
        if match:
            labels[float(match.group(1))] = label
        elif label == base:
            labels[0.0] = label
        elif label == rl:
            labels[1.0] = label
    curves: dict[float, list[tuple[int, float, float, float]]] = {}
    n_items: dict[float, int] = {}
    for alpha, label in sorted(labels.items()):
        version = default_version(store, suite=suite, models=[label], judge_id=judge_id) or "v1"
        run_id = latest_run_id(store, suite=suite, model=label, version=version, judge_id=judge_id)
        items = list(items_by_pid(store, run_id).values()) if run_id else []
        if not items:
            continue
        ks = [k for k in INTERP_KS if k <= max(it.n for it in items)]
        curves[alpha] = pass_curve(items, ks, B=B, seed=seed)
        n_items[alpha] = len(items)
    return {"curves": curves, "n": n_items, "labels": labels, "B": B}


# ========================================================================= interp artefacts


def _is_dan_pair_rows(rows: Sequence[Mapping[str, JSON]], key: str | None) -> bool:
    if not rows:
        return False
    if key is None:  # likelihood: every row carries both sides
        return all("fz" in r and "t1" in r for r in rows)
    return any(r.get(key) in ("fz", "t1") for r in rows)


def run_output_json(store: Store, run_id: str) -> dict[str, JSON] | None:
    """The JSON object an interp CLI run printed for `run_id`, read from the run logs
    (`<out>/logs/*.log`): the Dan probes run reports its position-only baselines only there."""
    logs = Path(store.root) / "logs"
    if not logs.is_dir():
        return None
    needle = f'"run_id": "{run_id}"'
    decoder = json.JSONDecoder()
    for path in sorted(logs.glob("*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        idx = text.rfind(needle)
        while idx >= 0:
            start = text.rfind("{", 0, idx)
            while start >= 0:
                try:
                    obj, _end = decoder.raw_decode(text, start)
                except ValueError:
                    start = text.rfind("{", 0, start)
                    continue
                if isinstance(obj, dict) and obj.get("run_id") == run_id:
                    return cast("dict[str, JSON]", obj)
                start = text.rfind("{", 0, start)
            idx = text.rfind(needle, 0, idx)
    return None


def dan_probes(store: Store) -> dict[str, Any] | None:
    """The newest `probes` run with fz/t1 rows: per model, per layer, the held-out next_rule
    accuracy and actions_to_go R2, plus the run's position-only baselines (step index; + premise
    count; + frame depth) on the same split."""
    for rec in reversed(interp_runs(store, "probes")):
        run_id = cast(str, rec["run_id"])
        rows = load_interp_rows(store, "probes", run_id)
        if not _is_dan_pair_rows(rows, "model"):
            continue
        per_model: dict[str, list[tuple[int, float | None, float | None, int | None]]] = {}
        for r in rows:
            model = cast(str, r.get("model"))
            if model not in ("fz", "t1"):
                continue
            rule = cast("dict[str, JSON]", r.get("next_rule") or {})
            atg = cast("dict[str, JSON]", r.get("actions_to_go") or {})
            per_model.setdefault(model, []).append(
                (
                    int(cast(int, r["layer"])),
                    cast("float | None", rule.get("test_acc")),
                    cast("float | None", atg.get("test_r2")),
                    cast("int | None", rule.get("n_test")),
                )
            )
        for v in per_model.values():
            v.sort()
        output = run_output_json(store, run_id) or {}
        return {
            "run_id": run_id,
            "cfg": cast("dict[str, JSON]", rec.get("cfg") or {}),
            "models": per_model,
            "baselines": cast("dict[str, JSON]", output.get("baselines") or {}),
            "n_train_examples": next(
                (r.get("n_train_examples") for r in rows if r.get("n_train_examples")), None
            ),
        }
    return None


def _likelihood_candidates(store: Store) -> list[dict[str, JSON]]:
    """Dan likelihood runs to try, best first: the newest run whose cfg records
    `exclude_defining_names: true` (scores without the environment-assigned name tokens, as
    Dan's trajectory study), then every other run newest first."""
    runs = list(reversed(interp_runs(store, "likelihood")))
    excl = [
        r
        for r in runs
        if cast("dict[str, JSON]", r.get("cfg") or {}).get("exclude_defining_names") is True
    ]
    return excl + [r for r in runs if r not in excl]


def dan_likelihood(store: Store) -> dict[str, Any] | None:
    """The preferred `likelihood` run whose rows score fz and t1 (`_likelihood_candidates`): per
    origin, each proof's ordering-marginalised log p divided by its written lines (`lines`, else
    `L`), under both. `exclude_defining_names` says whether the name tokens were scored."""
    for rec in _likelihood_candidates(store):
        run_id = cast(str, rec["run_id"])
        rows = load_interp_rows(store, "likelihood", run_id)
        if not _is_dan_pair_rows(rows, None):
            continue
        by_origin: dict[str, dict[str, Any]] = {}
        for r in rows:
            norm = cast("int | None", r.get("lines")) or cast("int | None", r.get("L"))
            fz = cast("dict[str, JSON]", r["fz"])
            t1 = cast("dict[str, JSON]", r["t1"])
            a, b = fz.get("logp_marginal"), t1.get("logp_marginal")
            if not norm or a is None or b is None:
                continue
            fa, fb = float(cast(float, a)) / norm, float(cast(float, b)) / norm
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            d = by_origin.setdefault(
                cast(str, r["origin"]), {"fz": [], "t1": [], "lines": [], "multi": 0}
            )
            d["fz"].append(fa)
            d["t1"].append(fb)
            d["lines"].append(norm)
            if int(cast(int, fz.get("n_orderings") or 1)) > 1:
                d["multi"] += 1
        if not by_origin:
            continue
        summary: dict[str, dict[str, Any]] = {}
        for origin, d in by_origin.items():
            diffs = [b - a for a, b in zip(d["fz"], d["t1"], strict=True)]
            summary[origin] = {
                "n": len(diffs),
                "median_fz": statistics.median(d["fz"]),
                "median_t1": statistics.median(d["t1"]),
                "median_diff": statistics.median(diffs),
                "frac_t1_higher": sum(x > 0 for x in diffs) / len(diffs),
                "median_lines": statistics.median(d["lines"]),
                "min_lines": min(d["lines"]),
                "max_lines": max(d["lines"]),
                "n_multi_order": d["multi"],
            }
        cfg = cast("dict[str, JSON]", rec.get("cfg") or {})
        return {
            "run_id": run_id,
            "cfg": cfg,
            "origins": summary,
            "exclude_defining_names": cfg.get("exclude_defining_names") is True,
        }
    return None


def _kl_rows(store: Store, run_id: str) -> list[dict[str, JSON]]:
    path = store.store_dir / "interp" / "kl" / f"{run_id}.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def _ratio_ci(num: np.ndarray, den: np.ndarray, B: int, seed: int) -> tuple[float, float, float]:
    """Token-weighted mean sum(num)/sum(den) with a 95% percentile bootstrap over samples."""
    total = float(den.sum())
    mean = float(num.sum()) / total if total else float("nan")
    if len(num) < 2 or not total:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    dens = den[idx].sum(axis=1)
    boots = num[idx].sum(axis=1) / np.where(dens > 0, dens, np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def dan_kl(store: Store, *, B: int = 1000, seed: int = 0) -> dict[str, Any] | None:
    """The newest `kl` run (`interp/kl.py`'s rows: one per scored sample with `direction`
    "rl||base" | "base||rl", `kl_sum`, `n_tokens`, `by_kind`): per direction the token-weighted
    per-token KL with a 95% bootstrap over samples, and per token kind; `None` if no run."""
    runs = interp_runs(store, "kl")
    for rec in reversed(runs):
        run_id = cast(str, rec["run_id"])
        rows = [r for r in _kl_rows(store, run_id) if r.get("kl_sum") is not None]
        if not rows:
            continue
        directions: dict[str, dict[str, Any]] = {}
        for direction in sorted({cast(str, r.get("direction")) for r in rows}):
            rs = [r for r in rows if r.get("direction") == direction]
            num = np.array([float(cast(float, r["kl_sum"])) for r in rs])
            den = np.array([float(cast(float, r.get("n_tokens") or 0)) for r in rs])
            mean, lo, hi = _ratio_ci(num, den, B, seed)
            kinds: dict[str, list[float]] = {}
            for r in rs:
                for kind, pair in cast("dict[str, list[float]]", r.get("by_kind") or {}).items():
                    acc = kinds.setdefault(kind, [0.0, 0.0])
                    acc[0] += float(pair[0])
                    acc[1] += float(pair[1])
            directions[direction] = {
                "n": len(rs),
                "n_ok": sum(1 for r in rs if r.get("ok")),
                "n_tokens": int(den.sum()),
                "per_token": (mean, lo, hi),
                "per_sample": float(num.mean()) if len(num) else float("nan"),
                "by_kind": {k: (v[0] / v[1], int(v[1])) for k, v in kinds.items() if v[1] > 0},
                "policies": sorted({str(r.get("policy")) for r in rs if r.get("policy")}),
                "suites": sorted({str(r.get("suite")) for r in rs if r.get("suite")}),
            }
        output = run_output_json(store, run_id) or {}
        n_skipped = output.get("n_skipped")
        return {
            "run_id": run_id,
            "cfg": cast("dict[str, JSON]", rec.get("cfg") or {}),
            "directions": directions,
            "B": B,
            "n_rows": len(rows),
            "n_skipped": int(cast(int, n_skipped)) if n_skipped is not None else None,
        }
    return None


# ============================================================================== seen counts


def seen_counts(
    store: Store, family: str, suite_versions: Sequence[tuple[str, str]]
) -> list[tuple[str, str, int, int]]:
    """`(suite, version, n items, n seen in `family`'s training tiers)` for each built sidecar."""
    out: list[tuple[str, str, int, int]] = []
    for suite, version in suite_versions:
        try:
            seen = seen_for(store, suite, version, family)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue
        out.append((suite, version, len(seen), sum(1 for v in seen.values() if v)))
    return out


def run_version(store: Store, suite: str, models: Sequence[str], judge_id: str) -> str | None:
    """The suite version the given models' runs target (`_common.default_version`)."""
    return default_version(store, suite=suite, models=list(models), judge_id=judge_id)
