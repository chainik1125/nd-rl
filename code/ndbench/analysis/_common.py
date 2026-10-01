"""Shared, private helpers for the WP6 analysis views (plan §3's `analysis/{yue,...}.py` row).

Not itself one of the plan's eight named analysis files: every `view_<bench>` module needs the
same handful of `Store`-querying and bootstrap-CI patterns (pick the runs for a suite/model, merge
their `ItemResult`s, resample items for a CI), so this module factors them out once instead of
repeating ~150 lines of glue in each of the eight files. It adds no new *statistics*: every
estimator here either calls straight through to `code.ndbench.metrics` or (for `irls_binomial`,
needed by `skills.py`'s multi-covariate GLM, where `metrics.logit_slope` is single-covariate)
generalises `metrics.logit_slope`'s own IRLS loop to an arbitrary design matrix using the same
algorithm.

Every `view_<bench>(store, cfg)` follows the same shape: look up which suite/version/models `cfg`
names (falling back to a benchmark-specific default and to whatever the `Store` actually has), and
return `{"empty": True, "reason": ...}` (see `empty_view`) if nothing matches, so `dash/pages.py`
can render the plan's "no data yet" panel uniformly.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

import numpy as np

from ..core.io import read_jsonl
from ..core.types import JSON, ItemResult, Problem
from ..metrics import logit_slope
from ..store import Store

#: Run statuses `run_records` includes by default: real sampling results, not in-flight or legacy
#: rows (callers that specifically want those, e.g. the `runs` dashboard page, pass `statuses=None`
#: or their own tuple).
LIVE_STATUSES: tuple[str, ...] = ("complete", "partial")

#: `judge_id` `run_records`/`latest_run_id`/`all_run_ids`/`models_present` filter to by default
#: (plan §2 WP-D2 item 5): `judge.nd.NDJudge.judge_id`, the only judge any run in the real store
#: predates (plan §0: "No lean* runs exist in the store yet"), so this default is exactly today's
#: behaviour for every existing view/dashboard call site -- none of which pass `judge_id` -- both
#: now and once `lean_v2` runs land (a bare call stays pinned to `nd_v2` rather than silently
#: starting to mix two judges' verdicts together once both exist for the same suite/model). Pass
#: `judge_id=None` to disable the filter entirely (every judge), or an explicit id (e.g.
#: `"lean_v2"`) for a judge-specific view.
DEFAULT_JUDGE_ID: str | None = "nd_v2"


def cfg_get(cfg: Mapping[str, JSON], key: str, default: JSON) -> JSON:
    """`cfg[key]` if present and not `None`, else `default` -- the view config convention used
    throughout (every `cfg` entry is optional; a benchmark-specific default always applies)."""
    v = cfg.get(key)
    return default if v is None else v


def run_records(
    store: Store,
    *,
    suite: str | None = None,
    version: str | None = None,
    model: str | None = None,
    statuses: Sequence[str] | None = LIVE_STATUSES,
    judge_id: str | None = DEFAULT_JUDGE_ID,
) -> list[dict[str, JSON]]:
    """Folded run records (`Store.runs()`) matching every non-`None` filter, restricted to
    `statuses` (`None` disables the status filter entirely) and to `judge_id` (default
    `DEFAULT_JUDGE_ID`, `"nd_v2"`; `None` disables this filter entirely too, matching every
    judge)."""
    out: list[dict[str, JSON]] = []
    for rec in store.runs():
        if suite is not None and rec.get("suite") != suite:
            continue
        if version is not None and rec.get("version") != version:
            continue
        if model is not None and rec.get("model") != model:
            continue
        if statuses is not None and rec.get("status") not in statuses:
            continue
        if judge_id is not None and rec.get("judge_id") != judge_id:
            continue
        out.append(rec)
    return out


def models_present(
    store: Store,
    *,
    suite: str,
    version: str | None = None,
    statuses: Sequence[str] | None = LIVE_STATUSES,
    judge_id: str | None = DEFAULT_JUDGE_ID,
) -> list[str]:
    """Every distinct `model` with at least one matching run, sorted (deterministic page order)."""
    return sorted(
        {
            cast(str, r["model"])
            for r in run_records(
                store, suite=suite, version=version, statuses=statuses, judge_id=judge_id
            )
        }
    )


def latest_run_id(
    store: Store,
    *,
    suite: str,
    model: str,
    version: str | None = None,
    statuses: Sequence[str] | None = LIVE_STATUSES,
    judge_id: str | None = DEFAULT_JUDGE_ID,
) -> str | None:
    """The matching run with the most total samples (`k`), ties broken by most recently finished
    (or, failing that, started); `None` if nothing matches.

    **Why `k` first, not just recency.** A suite can legitimately be sampled more than once against
    the same model under different configs that both finish as `"complete"` -- e.g. `textbook_dev`
    carries both the Yue benchmark's adaptive `k=4096` run for `base`/`rl` *and* a small `k=8`
    "unsteered baseline" measurement for the interp steering/interpolation sweep (`INTERP_SPEC.md`'s
    steering/interpolation eval suites include `textbook_dev`), and the `k=8` baseline happens to
    finish *after* the big Yue run. Picking "most recently finished" alone would then silently hand
    an 8-sample run to every caller expecting the real 4096-sample one. Preferring the largest `k`
    asks for "the most complete/informative run" directly; a `--rerun` at the same `k` as an earlier
    attempt still wins on the recency tie-break, so resume/rerun behaviour is unchanged, and a suite
    with only one run per (suite, model, version) (the common case) is unaffected either way.
    """
    cands = run_records(
        store, suite=suite, version=version, model=model, statuses=statuses, judge_id=judge_id
    )
    if not cands:
        return None
    cands.sort(
        key=lambda r: (
            int(cast("int | None", r.get("k")) or 0),
            cast(str, r.get("finished_utc") or r.get("started_utc") or ""),
        )
    )
    return cast(str, cands[-1]["run_id"])


def all_run_ids(
    store: Store,
    *,
    suite: str,
    model: str,
    version: str | None = None,
    statuses: Sequence[str] | None = LIVE_STATUSES,
    judge_id: str | None = DEFAULT_JUDGE_ID,
) -> list[str]:
    return [
        cast(str, r["run_id"])
        for r in run_records(
            store, suite=suite, version=version, model=model, statuses=statuses, judge_id=judge_id
        )
    ]


def items_by_pid(store: Store, run_id: str) -> dict[str, ItemResult]:
    return {it.pid: it for it in store.items(run_id)}


def merged_items_by_pid(store: Store, run_ids: Sequence[str]) -> dict[str, ItemResult]:
    """`items_by_pid` unioned over several run ids (e.g. every run matching a (suite, model) once
    several seeds/configs have accumulated); a `pid` sampled by more than one run keeps whichever
    run's row is folded in last (callers needing a true merge should instead point at one run)."""
    out: dict[str, ItemResult] = {}
    for run_id in run_ids:
        out.update(items_by_pid(store, run_id))
    return out


def problems_by_pid(store: Store, suite: str, version: str) -> dict[str, Problem]:
    try:
        return {p.pid: p for p in store.load_suite(suite, version)}
    except (FileNotFoundError, OSError):
        return {}


def problem_versions(store: Store, suite: str) -> list[str]:
    """Every version directory `store.load_suite(suite, ...)` could load, newest last is not
    guaranteed (directory listing order), sorted for determinism."""
    d = store.suites_dir / suite
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def default_version(
    store: Store,
    *,
    suite: str,
    models: Sequence[str] | None = None,
    statuses: Sequence[str] | None = LIVE_STATUSES,
) -> str | None:
    """Which `suite` version a view should read when `cfg` doesn't pin one explicitly: the
    version with the most run coverage, rather than an arbitrary hard-coded `"v1"` (fix list item
    1 -- a real suite can accumulate a small demo version nobody ran alongside the version the
    real runs target, e.g. `padding`'s 132-item `v1` demo vs. the 51-base `v2` that `base`/`rl`
    were actually run on, or `glue`'s 185-item `demo_v1` vs. the 792-item `v1` the real `base`/
    `rl`/`armC` runs target).

    Scores each `problem_versions(store, suite)` entry by `(n_models_matched, version_string)` --
    `n_models_matched` is "how many of `models` have >= 1 matching run at this version" when
    `models` is given, else simply "how many distinct models have any run at this version" (so a
    caller that doesn't yet know its model list still prefers a version something was actually run
    on) -- and returns the best-scoring version, ties broken by the lexicographically-greatest
    version string (so "v2" beats "v1" when both have identical coverage, e.g. both zero). Falls
    back to the only version when there's just one, and to `None` when `suite` has no version
    directory at all (mirrors `problems_by_pid`'s own not-found handling, so callers can still
    produce an `empty_view` either way).
    """
    versions = problem_versions(store, suite)
    if not versions:
        return None
    if len(versions) == 1:
        return versions[0]

    def score(version: str) -> tuple[int, str]:
        present = set(models_present(store, suite=suite, version=version, statuses=statuses))
        n_matched = sum(1 for m in models if m in present) if models is not None else len(present)
        return (n_matched, version)

    return max(versions, key=score)


class FirstChunk(TypedDict):
    n: int
    c: int


def first_chunk_by_pid(store: Store, run_id: str) -> dict[str, FirstChunk]:
    """Each item's *first* chunk only: `chunk.jsonl` rows filtered to each `pid`'s minimum
    `chunk` index, read directly from `store.items_dir` (a public `Store` attribute) since
    `Store.items()` always merges every chunk for a `pid` into one row. Plan §2 B8 needs the
    unbiased fixed-`n` estimator from just the first sampling round for `k <= 256`; later chunks
    are only drawn for items an `AdaptiveRule` decided were still short of `stop_c`, so mixing them
    into a `k <= 256` estimate would compare unlike `n` across items."""
    path = store.items_dir / f"{run_id}.jsonl"
    if not path.exists():
        return {}
    best: dict[str, dict[str, JSON]] = {}
    for row in read_jsonl(path):
        pid = cast(str, row["pid"])
        chunk = cast(int, row["chunk"])
        if pid not in best or chunk < cast(int, best[pid]["chunk"]):
            best[pid] = row
    return {pid: {"n": cast(int, row["n"]), "c": cast(int, row["c"])} for pid, row in best.items()}


def bootstrap_ci(
    values: Sequence[float], seed: int = 0, B: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI over already-per-item `values` (resampled with replacement, `nan`s
    dropped within each resample, matching `metrics.pass_curve`'s convention)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boot = nanmean_ignore(arr[idx], axis=1)
    lo, hi = nanpercentile_ignore(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def nanmean_ignore(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """`np.nanmean` with the "Mean of empty slice" `RuntimeWarning` suppressed: several views
    (e.g. `yue.py`'s per-`k` bootstrap, where a resample can leave every item `nan` at a `k` none
    of them reached) hit all-`nan` slices as an expected, harmless case, not a bug to warn about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return cast(np.ndarray, np.nanmean(arr, axis=axis))


def nanpercentile_ignore(arr: np.ndarray, q: Sequence[float]) -> np.ndarray:
    """`np.nanpercentile` with the "All-NaN slice encountered" `RuntimeWarning` suppressed (same
    rationale as `nanmean_ignore`)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(arr, q)


def is_nan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def group_by_tag(problems: Mapping[str, Problem], tag: str) -> dict[JSON, list[str]]:
    """`pid`s bucketed by `Problem.tags[tag]` (missing tag groups under `None`)."""
    out: dict[JSON, list[str]] = {}
    for pid, p in problems.items():
        out.setdefault(p.tags.get(tag), []).append(pid)
    return out


def item_length(problem: Problem) -> float | None:
    """A plotting-friendly length for `problem`: the point estimate `L` if certified exactly,
    else the midpoint of `(L_lb, L_ub)` when both bounds are known, else `L_ub`, else `None`."""
    length = problem.length
    if length.L is not None:
        return float(length.L)
    if length.L_lb is not None and length.L_ub is not None:
        return (length.L_lb + length.L_ub) / 2.0
    if length.L_ub is not None:
        return float(length.L_ub)
    return None


def item_tokens(problem: Problem) -> float | None:
    """The item's own size in model tokens: `tags["tokens"]` (prompt + reference proof, set by the
    suite builders) when present, else the whitespace-token count of the prompt plus the reference
    body (the abs tokenizer maps 1:1 to whitespace tokens), else `None`. Never a model's own
    generated length (`ItemResult.tokens_mean`): that depends on the outcome, failed attempts
    included, so a slope against it is not a slope against item size (audit 2026-10-01)."""
    tok = problem.tags.get("tokens")
    if isinstance(tok, int | float) and not isinstance(tok, bool):
        return float(tok)
    if problem.ref is not None:
        return float(len(problem.sequent.prompt().split()) + len(problem.ref.body.split()))
    return None


def logit_slope_safe(
    xs: Sequence[float], ns: Sequence[int], cs: Sequence[int]
) -> dict[str, JSON] | None:
    """`metrics.logit_slope` on `(x, n, c)` triples, first dropping `nan`/missing `x`s and
    zero-trial rows; `None` (rather than an unstable two-point fit) if fewer than 3 rows remain.
    Shared by every view that reports "a logit slope per added line/100 tokens" (plan §2 B2/B4/B5).
    """
    pairs = [(x, n, c) for x, n, c in zip(xs, ns, cs, strict=True) if not is_nan(x) and n > 0]
    if len(pairs) < 3:
        return None
    xs2, ns2, cs2 = (list(v) for v in zip(*pairs, strict=True))
    b, se = logit_slope(xs2, ns2, cs2)
    return {"slope": b, "se": se, "n_points": len(pairs)}


def empty_view(reason: str) -> dict[str, JSON]:
    """The uniform "nothing to show" return value every `view_<bench>` uses when its `cfg`/`Store`
    combination has no matching data; `dash/pages.py` renders this as an explanatory panel rather
    than an empty page (plan §3's "Empty views render an explanatory 'no data yet' panel")."""
    return {"empty": True, "reason": reason}


def irls_binomial(
    design: np.ndarray,
    n: np.ndarray,
    c: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
    ridge: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Binomial GLM (logit link) by ridge-penalised IRLS/Fisher scoring: `design` is
    `(n_obs, n_coef)` (a leading all-ones column gives an intercept), `n`/`c` are per-row
    trials/successes. Returns `(beta, se_beta)`, both length `n_coef`.

    Generalises `metrics.logit_slope`'s single-covariate IRLS loop (identical algorithm,
    convergence criterion, and `ridge` penalty -- see that function's docstring for why one is
    needed: a near-complete-separation design, e.g. `prefix.py`'s fitted logit when a model's
    prefix-success rate is near-0 almost everywhere, otherwise drives the unpenalised MLE and its
    SE to a huge, numerically unstable value -- confirmed on the real store's `prefix` page, a
    `base`-model intercept at se=1431.97) to an arbitrary number of covariates, for `skills.py`'s
    "binomial pass ~ L + skill indicators" GLM (plan §2 B3) and `prefix.py`'s fitted logit (plan §2
    B6); `metrics.logit_slope` itself stays single-covariate and is reused as-is wherever one
    covariate is all a view needs.
    """
    ns = np.asarray(n, dtype=float)
    cs = np.asarray(c, dtype=float)
    x = np.asarray(design, dtype=float)
    mask = ns > 0
    x, ns, cs = x[mask], ns[mask], cs[mask]
    n_coef = x.shape[1]
    beta = np.zeros(n_coef)
    if x.shape[0] == 0:
        return beta, np.full(n_coef, float("nan"))
    penalty = ridge * np.eye(n_coef)
    for _ in range(max_iter):
        eta = x @ beta
        p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
        w = ns * p * (1 - p)
        z = eta + (cs - ns * p) / (ns * p * (1 - p))
        xtwx = x.T @ (x * w[:, None]) + penalty
        xtwz = x.T @ (w * z)
        try:
            new_beta = np.linalg.solve(xtwx, xtwz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    eta = x @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
    w = ns * p * (1 - p)
    xtwx = x.T @ (x * w[:, None]) + penalty
    cov = np.linalg.pinv(xtwx)
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None))
