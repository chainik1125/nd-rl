"""nd_v2 vs lean_v2 judge agreement (`DANSTATE_PLAN.md` §2, WP-D5 item 2).

Dan's models sample online under the headline judge `"lean_v2"`; `cli.cmd_rejudge --judge nd`
recomputes the *same already-sampled attempts* under `"nd_v2"` without resampling, writing a
derived run `f"{run_id}~nd_v2"` (`store.new_derived_run`'s own naming convention). This view reads
both runs' `samples/<run_id>.jsonl.gz` rows (never resampled, so every attempt lines up one-to-one
by `(pid, chunk, i)`) and reports:

- overall agreement / lean-only-accept / nd-only-accept rates over every attempt both runs judged;
- a heuristic breakdown of the lean-only-accept attempts into the three buckets plan §2 names from
  Dan's own `LEAN_JUDGE.md` analysis -- missing premise restatement, `.elim` on not-A, other
  (`_classify_lean_only`). **This is a heuristic over the two signals this `Store` actually
  records per attempt** (the nd_v2 judge's own rejection `reason`, and whether the attempt's
  native Lean text contains `.elim`), not a re-derivation of Dan's own reported percentages
  (41.7% / 25.9% / 6.2%), which came from his manual read of the mismatches; those numbers belong
  in a caption as cited context, never asserted as this function's output.

A "lean-only accept" (lean_v2 ok, nd_v2 not) is the expected direction -- Lean denotes proofs
nd_verify's ND grammar does not (plan §0: "24.6% have no ND denotation"). An "nd-only accept" (the
other direction) is the soundness check plan V7 asks to assert is empty ("nothing nd_v2 accepts is
Lean-rejected"); this view reports its rate rather than hiding it, in case it ever isn't.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..core.io import read_jsonl
from ..core.types import JSON
from ..store import Store
from ._common import empty_view, run_records, safe_div

#: Plan §2's three named buckets for a lean-only accept, in the order it quotes them.
BUCKETS: tuple[str, ...] = ("missing_premise_restatement", "elim_on_not_a", "other")

#: The headline judge pairing this view compares (plan §2): online-sampled Dan runs are judged
#: `LEAN_JUDGE_ID`; `rejudge --judge nd` derives the `NDJudge`-judged counterpart.
LEAN_JUDGE_ID = "lean_v2"
ND_JUDGE_ID = "nd_v2"


def _classify_lean_only(nd_reason: str, native: str | None) -> str:
    """Heuristic bucket for one lean-only-accept attempt (module docstring)."""
    reason = (nd_reason or "").lower()
    if "premise" in reason or "restat" in reason:
        return "missing_premise_restatement"
    lean_text = ""
    if native:
        try:
            decoded = json.loads(native)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict):
            lean_text = str(decoded.get("lean") or "")
    if ".elim" in lean_text:
        return "elim_on_not_a"
    return "other"


def _samples_rows(store: Store, run_id: str) -> dict[tuple[str, int, int], dict[str, JSON]]:
    """`{(pid, chunk, i): row}` for `run_id`'s own samples file (empty if the run has none, e.g.
    it was sampled without `keep_samples=True`)."""
    path = store.samples_dir / f"{run_id}.jsonl.gz"
    if not path.exists():
        return {}
    return {
        (cast(str, row["pid"]), cast(int, row["chunk"]), cast(int, row["i"])): row
        for row in read_jsonl(path)
    }


def agreement_for_pair(store: Store, lean_run_id: str, nd_run_id: str) -> dict[str, Any] | None:
    """Agreement between `lean_run_id` (judged `lean_v2`) and `nd_run_id` (its `rejudge --judge
    nd` derivative), matched attempt-for-attempt; `None` if either has no samples file or they
    share no attempt (e.g. the suite changed between sampling and rejudging)."""
    lean_rows = _samples_rows(store, lean_run_id)
    nd_rows = _samples_rows(store, nd_run_id)
    common = sorted(set(lean_rows) & set(nd_rows))
    if not common:
        return None
    agree = lean_only = nd_only = 0
    bucket_counts: dict[str, int] = dict.fromkeys(BUCKETS, 0)
    for key in common:
        lean_ok = bool(lean_rows[key]["ok"])
        nd_ok = bool(nd_rows[key]["ok"])
        if lean_ok == nd_ok:
            agree += 1
            continue
        if lean_ok and not nd_ok:
            lean_only += 1
            bucket = _classify_lean_only(
                cast(str, nd_rows[key].get("reason") or ""),
                cast("str | None", lean_rows[key].get("native")),
            )
            bucket_counts[bucket] += 1
        else:
            nd_only += 1
    n = len(common)
    return {
        "lean_run_id": lean_run_id,
        "nd_run_id": nd_run_id,
        "n_attempts": n,
        "agree_rate": safe_div(float(agree), float(n)),
        "lean_only_rate": safe_div(float(lean_only), float(n)),
        "nd_only_rate": safe_div(float(nd_only), float(n)),
        "agree_count": agree,
        "lean_only_count": lean_only,
        "nd_only_count": nd_only,
        "lean_only_breakdown": {
            name: {
                "count": count,
                "frac": safe_div(float(count), float(lean_only)),
            }
            for name, count in bucket_counts.items()
        },
    }


def _pool(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`agreement_for_pair` results for one model's runs, summed into one record of the same
    shape (rates over all attempts), plus `pairs`: each run pair's own result."""
    n = sum(int(r["n_attempts"]) for r in results)
    lean_only = sum(int(r["lean_only_count"]) for r in results)
    nd_only = sum(int(r["nd_only_count"]) for r in results)
    agree = sum(int(r["agree_count"]) for r in results)
    counts = {
        name: sum(int(r["lean_only_breakdown"][name]["count"]) for r in results) for name in BUCKETS
    }
    return {
        "n_attempts": n,
        "n_pairs": len(results),
        "agree_rate": safe_div(float(agree), float(n)),
        "lean_only_rate": safe_div(float(lean_only), float(n)),
        "nd_only_rate": safe_div(float(nd_only), float(n)),
        "agree_count": agree,
        "lean_only_count": lean_only,
        "nd_only_count": nd_only,
        "lean_only_breakdown": {
            name: {"count": c, "frac": safe_div(float(c), float(lean_only))}
            for name, c in counts.items()
        },
        "pairs": list(results),
    }


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    """`cfg["pairs"]` (an explicit `[(model, lean_run_id, nd_run_id), ...]`) overrides discovery;
    otherwise every run in the `Store` judged `"lean_v2"` (optionally narrowed by `cfg["suite"]`/
    `cfg["models"]`) is checked for a matching `f"{run_id}~nd_v2"` rejudge (`cli.cmd_rejudge`'s own
    naming), and only pairs where one exists are compared."""
    explicit = cast("Sequence[Sequence[str]] | None", cfg.get("pairs"))
    suite = cast("str | None", cfg.get("suite"))
    override_models = cast("list[str] | None", cfg.get("models"))

    pairs: list[tuple[str, str, str]] = []
    if explicit:
        pairs = [(str(m), str(a), str(b)) for m, a, b in explicit]
    else:
        for rec in run_records(store, suite=suite, judge_id=LEAN_JUDGE_ID):
            model = cast(str, rec.get("model"))
            if override_models is not None and model not in override_models:
                continue
            lean_run_id = cast(str, rec["run_id"])
            nd_run_id = f"{lean_run_id}~{ND_JUDGE_ID}"
            if store.runs(run_id=nd_run_id):
                pairs.append((model, lean_run_id, nd_run_id))

    if not pairs:
        return empty_view(
            f"no {LEAN_JUDGE_ID!r} run has a matching {ND_JUDGE_ID!r} rejudge yet "
            f"(run `rejudge --run-id <id> --judge nd` first)"
        )

    # Every pair of a model is pooled (audit 2026-10-01: keying `per_model[model] = result` per
    # pair kept only the model's last run, e.g. one suite out of eleven).
    by_model: dict[str, list[dict[str, Any]]] = {}
    for model, lean_run_id, nd_run_id in pairs:
        result = agreement_for_pair(store, lean_run_id, nd_run_id)
        if result is not None:
            by_model.setdefault(model, []).append(result)
    per_model = {model: _pool(results) for model, results in by_model.items()}
    if not per_model:
        return empty_view(
            "matching lean_v2/nd_v2 run pairs exist but share no sampled attempts "
            "(samples recorded without keep_samples=True?)"
        )
    return {
        "empty": False,
        "meta": {"suite": suite, "models": sorted(per_model)},
        "models": per_model,
    }
