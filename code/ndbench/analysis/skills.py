"""B3: the skill profile -- dashboard page "skills".

Reads the `skill_pool` suite (WP2's `suites/existing.py`; `Problem.tags` carry skill indicators --
either a per-skill tag, e.g. `tags["classical"] = True`, or a `tags["skills"]` list naming every
skill that applies, both handled by `_skill_active`) and, per model:

- a skill x length heat-map (pass@1 and pass@32, pooled over items with that skill active at that
  length, with item counts -- plan §2 B3's "heat-map of skill x L (pass@1 and pass@32, with cell
  counts)");
- a binomial GLM `pass ~ L + skill indicators` fit by IRLS (`_common.irls_binomial`, the
  multi-covariate generalisation of `metrics.logit_slope`), one row per item (`ItemResult.n`/`.c`),
  reporting each coefficient with its standard error.

`cfg["skills"]` overrides the default skill-tag list (plan §2 B3's table): `classical`,
`case_split`, `derived_case`, `nested_case`, `case_in_box`, `negi_ande_hyp`, `depth_ge_3`. The
`uses_*` rule-count tags are descriptive (plan §2 B3) and not part of the GLM design. Of these,
`skill_pool`'s builder only tags `classical`/`case_split` unconditionally -- `necessity`/
`needs_depth` (the other five) are NOT precomputed over the pool (see `suites/existing.py`'s and
`logic/skills.py`'s module docstrings for the cost measurements that rule this out at 11.5k items)
-- so `view` narrows the default list to whichever skills the pool actually tagged on at least one
item, rather than showing always-empty heatmap rows/GLM columns for the rest; `meta["skills_note"]`
says so on the page.

**No run ever samples `skill_pool` itself (fix list item 2).** Its 11,524 items are plan §2 B3's
"existing per-theorem results [that] cover base+RL on dev/sealed/frontier and pretrain+C on
lengthexp sets" -- i.e. `import_legacy`'s legacy runs, under their own suite names/pids, not
`skill_pool`'s. This module joins them back to `skill_pool` by the underlying theorem:

- `skill_pool`'s own `pid` is `f"{source_set}:{row['name']}"` (`suites/existing.py:skill_pool`,
  tagged `tags["source_set"]` to one of `dev_L8-11`/`sealed_L12-14`/`frontier_L12-14`/
  `lengthexp_std`/`lengthexp_matched` -- the keys of its own `SKILL_POOL_SOURCES`).
- `import_legacy`'s imported `ItemResult.pid` is the bare `row["name"]` from those exact same
  source files (`suites/existing.py:_import_monitor_run`/`_import_lengthexp_run`), under legacy
  suite names: `dev_L8-11`/`sealed_L12-14`/`frontier_L12-14` import as one legacy suite each
  (`legacy_dev`, `legacy_sealed_L12-14`, `legacy_frontier` -- the monitor import's own naming,
  `f"legacy_{set_label}"`, does not always echo the source label verbatim); `lengthexp_std`/
  `lengthexp_matched` import as one legacy suite *per L bucket*
  (`legacy_lengthexp_{std,matched}_L<n>`, the lengthexp import's own per-file naming), so those two
  are matched by suite-name *prefix* rather than one literal name. `_join_pid` undoes the
  `f"{source_set}:"` prefix to recover the bare name a legacy run's `ItemResult.pid` uses.
- Legacy model labels differ from this dashboard's (fix list item 2): the monitor import's
  `"champion"` is this dashboard's `"rl"`; both the monitor import's `"base"` (dev/sealed/frontier)
  and the lengthexp import's `"pretrain"` (lengthexp std/matched) are the same physical pretrained
  checkpoint (plan §0: `ab2/frozen/pretrain.pt`) and both collapse onto this dashboard's `"base"`
  -- they cover disjoint `source_set`s of `skill_pool`, so this is a union, not a conflict;
  `"armC"` keeps its name. `meta["model_map"]`/`meta["legacy_note"]` say so on the page.
- A `(legacy suite, legacy model)` pair can have more than one matching run: either independent
  seeds of the *same* config (the lengthexp arm C import's three seeds `C_s101`/`C_s102`/`C_s103`,
  meant to be pooled) or a rerun at a different `k` superseding an earlier one (`sealed_L12-14` was
  run at both `k64` and `k256` -- the higher-`k` run supersedes, not a replicate to pool alongside
  the lower one). `_collect_legacy_items` tells these apart the only way the `Store` can: keep only
  the run(s) at each pair's *highest* `k` (dropping a strictly-lower-`k` "superseded" run), then sum
  `n`/`c` per `pid` across whatever remains (one run in the rerun case, three in the seed case).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.types import JSON, Problem
from ..metrics import pass_at_k
from ..store import Store
from ._common import (
    cfg_get,
    default_version,
    empty_view,
    irls_binomial,
    item_length,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
    run_records,
)

DEFAULT_SUITE = "skill_pool"
DEFAULT_SKILLS: tuple[str, ...] = (
    "classical",
    "case_split",
    "derived_case",
    "nested_case",
    "case_in_box",
    "negi_ande_hyp",
    "depth_ge_3",
)

#: `skill_pool`'s `tags["source_set"]` values that map to exactly one legacy suite (the monitor
#: import, `_import_monitor_run`): see this module's docstring for why the legacy suite name is
#: not always the source-set label verbatim.
_LEGACY_SUITE_EXACT: dict[str, str] = {
    "dev_L8-11": "legacy_dev",
    "sealed_L12-14": "legacy_sealed_L12-14",
    "frontier_L12-14": "legacy_frontier",
}
#: `source_set` values scattered across one legacy suite *per L bucket* (the lengthexp import,
#: `_import_lengthexp_run`): matched by suite-name prefix instead of one literal name.
_LEGACY_SUITE_PREFIX: dict[str, str] = {
    "lengthexp_std": "legacy_lengthexp_std_L",
    "lengthexp_matched": "legacy_lengthexp_matched_L",
}

#: Legacy run model label -> this dashboard's model label (module docstring's "Legacy model labels
#: differ").
LEGACY_MODEL_MAP: dict[str, str] = {
    "base": "base",
    "champion": "rl",
    "pretrain": "base",
    "armC": "armC",
}

_LEGACY_STATUSES: tuple[str, ...] = ("legacy",)

#: Legacy armC runs kept (their run `version` is the lengthexp seed). Audit 2026-10-01: this page
#: pooled seeds C_s101-C_s103 while "armC" everywhere else is C_s101 alone.
DEFAULT_ARMC_LEGACY_VERSIONS: tuple[str, ...] = ("C_s101",)

#: Model pairs compared on their common items only (audit 2026-10-01: base covers dev/sealed/
#: frontier *and* lengthexp, rl only the first three, armC only lengthexp, so per-model numbers
#: are on different item sets).
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (("base", "rl"), ("base", "armC"))


def _skill_active(problem: Problem, skill: str) -> bool:
    v = problem.tags.get(skill)
    if v is not None:
        return bool(v)
    skills_list = problem.tags.get("skills")
    return isinstance(skills_list, list) and skill in skills_list


def _mean_or_nan(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _legacy_suites_for_source(store: Store, source_set: str) -> list[str]:
    """Every legacy suite name joining `source_set` back to `skill_pool` needs: one exact name, or
    (lengthexp) every suite name actually present in the `Store` with the right prefix -- found by
    scanning `store.runs()` rather than hard-coding the L range, so a legacy suite for an L bucket
    this module doesn't already know about still joins."""
    exact = _LEGACY_SUITE_EXACT.get(source_set)
    if exact is not None:
        return [exact]
    prefix = _LEGACY_SUITE_PREFIX.get(source_set)
    if prefix is None:
        return []
    return sorted(
        {str(r["suite"]) for r in store.runs() if str(r.get("suite", "")).startswith(prefix)}
    )


def _collect_legacy_items(
    store: Store, source_set: str, armc_versions: Sequence[str] = DEFAULT_ARMC_LEGACY_VERSIONS
) -> dict[str, dict[str, tuple[int, int]]]:
    """`{dashboard_model: {bare_pid: (n, c)}}` for `source_set`, pooled per this module's docstring
    ("A (legacy suite, legacy model) pair can have more than one matching run"). `bare_pid` is the
    legacy `ItemResult.pid` as-is (the join back to `skill_pool`'s own `f"{source_set}:{pid}"` is
    `view`'s job, once results from every `source_set` are in hand)."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    suites = _legacy_suites_for_source(store, source_set)
    if not suites:
        return out
    groups: dict[tuple[str, str], list[dict[str, JSON]]] = {}
    for suite in suites:
        # `judge_id=None`: these are `import_legacy`'s pre-ndbench runs (`judge_id="legacy_import"`
        # per `suites/existing.py`), not `run_records`'s new default `"nd_v2"` filter (plan §2
        # WP-D2 item 5) -- an explicit override, not a relaxation of the filter for its own sake.
        for rec in run_records(store, suite=suite, statuses=_LEGACY_STATUSES, judge_id=None):
            legacy_model = cast(str, rec.get("model"))
            if legacy_model not in LEGACY_MODEL_MAP:
                continue
            if legacy_model == "armC" and str(rec.get("version")) not in armc_versions:
                continue
            groups.setdefault((suite, legacy_model), []).append(rec)
    for (_suite, legacy_model), recs in groups.items():
        max_k = max(int(cast(int, r.get("k", 0))) for r in recs)
        kept = [r for r in recs if int(cast(int, r.get("k", 0))) == max_k]
        dash_model = LEGACY_MODEL_MAP[legacy_model]
        bucket = out.setdefault(dash_model, {})
        for rec in kept:
            for item in store.items(cast(str, rec["run_id"])):
                prev = bucket.get(item.pid)
                bucket[item.pid] = (
                    (prev[0] + item.n, prev[1] + item.c) if prev is not None else (item.n, item.c)
                )
    return out


def _joined_items_by_model(
    store: Store,
    source_sets: Sequence[str],
    armc_versions: Sequence[str] = DEFAULT_ARMC_LEGACY_VERSIONS,
) -> dict[str, dict[str, tuple[int, int]]]:
    """`{dashboard_model: {skill_pool_pid: (n, c)}}`, unioned over every `source_set` (module
    docstring: `"base"` legitimately gets a union of the monitor import's `base` and the lengthexp
    import's `pretrain`, since those cover disjoint source sets -- no pid collides across
    `source_set`s because the `f"{source_set}:"` prefix is restored here)."""
    by_model: dict[str, dict[str, tuple[int, int]]] = {}
    for source_set in source_sets:
        for dash_model, bucket in _collect_legacy_items(store, source_set, armc_versions).items():
            dest = by_model.setdefault(dash_model, {})
            for bare_pid, nc in bucket.items():
                dest[f"{source_set}:{bare_pid}"] = nc
    return by_model


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    override_models = cast("list[str] | None", cfg.get("models"))
    # `skill_pool` is a tagged item pool, never sampled directly (module docstring) -- there is no
    # run to pick a version *by* the way `default_version` does elsewhere, so this just takes
    # whichever version `problem_versions` finds (there has only ever been one, `v1`); `cfg
    # ["version"]` still overrides.
    version = cast(str, cfg.get("version") or default_version(store, suite=suite) or "v1")
    skills = cast("list[str]", cfg_get(cfg, "skills", list(DEFAULT_SKILLS)))

    problems = problems_by_pid(store, suite, version)
    if not problems:
        return empty_view(f"suite {suite!r} version {version!r} not found in the store")

    # Primary source: a run that sampled `skill_pool` directly, if one ever exists (the suite's
    # own builder contract -- nothing in `suites/existing.py` rules this out for the future, it
    # just hasn't happened yet, module docstring's "No run ever samples skill_pool itself").
    by_model: dict[str, dict[str, tuple[int, int]]] = {}
    for model in models_present(store, suite=suite, version=version):
        run_id = latest_run_id(store, suite=suite, model=model, version=version)
        if run_id is None:
            continue
        for pid, item in items_by_pid(store, run_id).items():
            by_model.setdefault(model, {})[pid] = (item.n, item.c)

    # Supplementary source (fix list item 2, this module's main docstring): join the legacy-
    # imported per-theorem runs back to skill_pool by theorem name, for whichever (model, pid)
    # a direct skill_pool run above didn't already cover.
    source_sets = sorted(
        {str(v) for p in problems.values() if (v := p.tags.get("source_set")) is not None}
    )
    armc_versions = cast(
        "list[str]", cfg_get(cfg, "armc_legacy_versions", list(DEFAULT_ARMC_LEGACY_VERSIONS))
    )
    legacy_by_model = _joined_items_by_model(store, source_sets, armc_versions)
    used_legacy = any(bucket for bucket in legacy_by_model.values())
    for model, bucket in legacy_by_model.items():
        dest = by_model.setdefault(model, {})
        for pid, nc in bucket.items():
            dest.setdefault(pid, nc)

    models = override_models or sorted(by_model)
    if not models:
        return empty_view(
            f"suite {suite!r} has no run of its own and no joinable legacy run was found under "
            "legacy_dev/legacy_sealed_L12-14/legacy_frontier/legacy_lengthexp_*"
        )

    # Narrow to skills the pool actually tagged (module docstring): avoids always-empty heatmap
    # rows/GLM columns for the necessity()/needs_depth() skills skill_pool doesn't compute yet.
    tagged_skills = [s for s in skills if any(s in p.tags for p in problems.values())]
    untagged_skills = [s for s in skills if s not in tagged_skills]
    if tagged_skills:
        skills = tagged_skills

    per_model: dict[str, Any] = {}
    n_pool_items = len(problems)
    for model in models:
        item_ncs = by_model.get(model, {})
        rows: list[tuple[float, dict[str, bool], int, int]] = []
        for pid, (n, c) in item_ncs.items():
            p = problems.get(pid)
            if p is None or n <= 0:
                continue
            length = item_length(p)
            if length is None:
                continue
            flags = {s: _skill_active(p, s) for s in skills}
            rows.append((length, flags, n, c))
        if not rows:
            continue

        heat1: dict[str, dict[str, Any]] = {}
        heat32: dict[str, dict[str, Any]] = {}
        for s in skills:
            by_l: dict[float, list[tuple[int, int]]] = {}
            for length, flags, n, c in rows:
                if not flags[s]:
                    continue
                by_l.setdefault(length, []).append((n, c))
            row1: dict[str, Any] = {}
            row32: dict[str, Any] = {}
            for length, ncs in sorted(by_l.items()):
                p1 = [pass_at_k(n, c, 1) for n, c in ncs]
                p32 = [pass_at_k(n, c, 32) for n, c in ncs if n >= 32]
                row1[str(length)] = {"pass1": _mean_or_nan(p1), "n_items": len(ncs)}
                row32[str(length)] = {"pass32": _mean_or_nan(p32), "n_items": len(p32)}
            heat1[s] = row1
            heat32[s] = row32

        design_cols = ["intercept", "L", *skills]
        design = np.zeros((len(rows), len(design_cols)))
        n_arr = np.zeros(len(rows))
        c_arr = np.zeros(len(rows))
        for i, (length, flags, n, c) in enumerate(rows):
            design[i, 0] = 1.0
            design[i, 1] = length
            for j, s in enumerate(skills):
                design[i, 2 + j] = 1.0 if flags[s] else 0.0
            n_arr[i] = n
            c_arr[i] = c
        beta, se = irls_binomial(design, n_arr, c_arr)
        glm = {
            name: {"coef": float(b), "se": float(s)}
            for name, b, s in zip(design_cols, beta, se, strict=True)
        }

        per_model[model] = {
            "n_items": len(rows),
            "n_pool_items": n_pool_items,
            "heatmap_pass1": heat1,
            "heatmap_pass32": heat32,
            "glm": glm,
        }

    if not per_model:
        return empty_view(
            f"suite {suite!r} has joinable legacy runs but no items had a known length"
        )

    coverage = {
        m: dict(
            Counter(
                str(problems[pid].tags.get("source_set")) for pid in by_model[m] if pid in problems
            )
        )
        for m in by_model
    }
    comparisons: dict[str, Any] = {}
    for a, b in COMPARISON_PAIRS:
        if a not in by_model or b not in by_model:
            continue
        common = sorted(set(by_model[a]) & set(by_model[b]) & set(problems))
        common = [p for p in common if by_model[a][p][0] > 0 and by_model[b][p][0] > 0]
        if not common:
            continue
        groups: dict[str, list[str]] = {"no special skill": []}
        for pid in common:
            active = [s for s in skills if _skill_active(problems[pid], s)]
            if not active:
                groups["no special skill"].append(pid)
            for sk in active:
                groups.setdefault(sk, []).append(pid)
        comparisons[f"{a}_vs_{b}"] = {
            "models": [a, b],
            "n_common": len(common),
            "sources": dict(Counter(str(problems[p].tags.get("source_set")) for p in common)),
            "by_skill": {
                g: {
                    "n_items": len(pids),
                    a: _mean_or_nan([pass_at_k(*by_model[a][p], 1) for p in pids]),
                    b: _mean_or_nan([pass_at_k(*by_model[b][p], 1) for p in pids]),
                }
                for g, pids in groups.items()
                if pids
            },
        }
    return {
        "empty": False,
        "meta": {
            "suite": suite,
            "version": version,
            "skills": skills,
            "untagged_skills": untagged_skills,
            "models": models,
            "model_map": LEGACY_MODEL_MAP,
            "used_legacy_join": used_legacy,
            "armc_legacy_versions": armc_versions,
            "coverage_by_source": coverage,
            "legacy_note": (
                "skill_pool has no run of its own; these results are legacy-imported per-theorem "
                "runs (status=legacy) joined back to skill_pool by theorem name. Legacy label "
                "'champion' -> rl; legacy labels 'base' and 'pretrain' both -> base (same "
                "pretrained checkpoint, disjoint source sets)."
                if used_legacy
                else "skill_pool was sampled directly; no legacy join was needed."
            ),
        },
        "models": per_model,
        "comparisons": comparisons,
    }
