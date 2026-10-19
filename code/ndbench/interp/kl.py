"""`interp kl`: exact, token-level KL divergence between a base/rl model pair, computed from both
models' full next-token distributions (not a single-sample log-ratio) along *stored* samples from
an existing benchmark run -- no new sampling (`DanStateAdapter.masked_next_logprobs`/
`AbsGptAdapter.masked_next_logprobs`, the two adapters' own new teacher-forcing primitives, do the
per-model forward passes this module combines).

For a prompt `x` and a sample `y` drawn from a policy's own stored attempts, `y` decomposes into a
sequence of scored positions `t` (Dan: one `(state, action)` step's action tokens incl. `<eos>`,
minus `exclude_defining_names`'s positions by default; absgpt: one proof-token position, in the
literal `native` text the model actually emitted). At each scored position this module asks both
models for their full next-token distribution `p(.|x, y<t)` (teacher-forced, same tokens, same
mask) and sums the pointwise KL `KL(p_rl(.|x,y<t) || p_base(.|x,y<t))` over every scored `t` --
*exact*, not a Monte-Carlo estimate. `KL(rl||base)` is computed along rl's own stored samples;
`KL(base||rl)` along base's own stored samples (the reverse direction needs its own, differently
distributed samples -- it is not the same sum with signs flipped). The plain log-ratio `log
p_rl(y) - log p_base(y)` over the same scored tokens is reported alongside (`logratio_sum`): its
mean over `rl||base` rows is the Monte-Carlo estimate of the *sequence*-level KL (the familiar
`E_{y~rl}[log p_rl(y) - log p_base(y)]` form), connecting this measurement to the likelihood
benchmark (`interp/likelihood.py`/`danstate_support.marginal_logprob_dan`) and to `logprob` itself
(`logratio_sum` reproduces exactly one `valid_bases` term of `DanStateAdapter.logprob`'s own
per-encoding sum at the sample's own base, modulo the uniform `name_base_max`-prior offset
`logprob` adds and `masked_next_logprobs` does not -- see that method's own docstring).

**Two independent pairs, two private drivers.** `_run_kl_dan` (Dan's `dan12-t1-s0`/`dan12-fz-s0`,
judge `lean_v2`, stored `lean_seq` actions) and `_run_kl_absgpt` (our `rl`/`base`, judge `nd_v2`,
stored literal ND proof text) share no model-specific code beyond `_combine_row`/`_pool_rows`/
`_by_kind_summary` (pure arithmetic over each adapter's `MaskedLogProbs`, which both adapters
return in the same shape -- duck-typed, no shared base class, matching `adapters/absgpt.py`'s and
`adapters/danstate.py`'s own "duplicated rather than imported" convention for this kind of small
per-module helper). `run_kl` dispatches on the resolved adapters' concrete type (same `isinstance`
convention `interp/__main__.py._is_danstate` already uses for `likelihood`/`probes`).

**Sample selection.** Each `(policy, suite)`'s own stored samples (`Store.samples_dir`, the
`<run_id>.jsonl.gz` the latest matching `judge_id` run wrote -- `analysis._common.latest_run_id`,
read-only import, the same convention `interp/data.py.load_verified_samples` uses) are
deterministically subsampled to at most `max_samples_per_suite` (default 2,000) *per (model,
suite)*, stratified by `pid` (round-robin over sorted pids, each pid's own rows kept in their
stored `(chunk, i)` order -- no RNG needed, every pid contributes before any pid contributes a
second row). Every attempt in the run is a candidate (not just `ok=True` ones): `ok` is instead
carried through to each row and the pooled `by_ok` breakdown, matching the plan's "report ...
by whether the sample was accepted" breakdown.

**Skips.** A sample whose ND body is `UNPARSEABLE` (Dan's own `_attempt_fields` marker) or
otherwise outside the grammar (`DanStateAdapter.masked_next_logprobs` returns `None` for it -- the
same exception set `logprob` filters, plus the uncited-box check) is skipped and counted
(`n_skipped`), never silently dropped. For absgpt, a `native` with no trailing `QED` (truncated
generation) or that fails `encode_proof` (an out-of-vocabulary token, or an `N<k>` index outside
`1..MAXN`) is skipped the same way.

**Breakdown by action kind.** Dan: `interp.danstate_support.classify_steps`'s own per-step `rule`
label (`RULE_CLASSES` plus `"FINAL"` for the proof-closing `exact`) -- a box opener (`imp`/`neg`
lambda, or `Or.elim`'s first branch) is labelled `"AS"` (the assumption line it creates), matching
that module's own dispatch table; a box *closer* is `"IMPI"`/`"NEGI"`/`"ORE"` depending which frame
it closes. absgpt: the literal `RULE` token each `'N<i> ... : RULE ...;'` line already carries
(`core.proof.split_lines` + `line.split()[line.split().index(':') + 1]` -- the same rule-name
vocabulary, since both model families denote the same ND proof grammar), plus `"FINAL"` for the
trailing `QED` token itself (mirroring Dan's own closing-step convention).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..adapters.base import ModelAdapter
from ..core.io import read_jsonl
from ..core.types import JSON, Sequent
from ..store import Store
from ._store import new_interp_run, write_interp_rows

#: Default cap on how many of a (model, suite)'s own stored samples are scored, stratified by
#: `pid` (module docstring). A real run on the full `name_base_max`-sized Dan checkpoints scores
#: roughly `n_samples * mean_steps_per_sample * mean_tokens_per_step` teacher-forced *rows* per
#: model (textbook_dev L6-10 proofs: a handful of steps, a handful of tokens/action each -- see
#: `configs/interp_kl_dan12.yaml`'s own scale estimate), so this cap bounds wall-clock, not just
#: memory (`MaskedLogProbs.log_probs` is `(n_scored, vocab)` float32 per model; `vocab` is tiny --
#: 111 for Dan, 99 for absgpt -- so memory itself is not the binding constraint here).
DEFAULT_MAX_SAMPLES_PER_SUITE = 2000

#: `interp_kl_dan12.yaml`'s own default suite list (module docstring's "Sample selection"): the two
#: suites both `dan12-fz-s0`/`dan12-t1-s0` have `lean_v2`-judged runs with `keep_samples=True` for
#: (plan's "Samples" bullet). A config's own `suites:` list overrides this entirely.
DEFAULT_DAN_SUITES: tuple[dict[str, str], ...] = (
    {"suite": "textbook_dev", "version": "v1"},
    {"suite": "skill_pool_sub", "version": "v1"},
)

#: `interp_kl_ours.yaml`'s own default: only `textbook_dev` has `nd_v2`-judged `base`/`rl` runs
#: with stored samples (module docstring's absgpt section).
DEFAULT_ABSGPT_SUITES: tuple[dict[str, str], ...] = ({"suite": "textbook_dev", "version": "v1"},)


# --------------------------------------------------------------------------- raw sample loading


@dataclass(frozen=True)
class _RawSample:
    """One row of a `Store` samples file (`runner.runner.ItemAccum.record`'s own shape), narrowed
    to the fields this module needs. `native` is kept as the raw, un-interpreted JSON value (a
    dict-shaped `str` for Dan's `Attempt.native`, a plain proof-text `str` for absgpt's) -- each
    family's own driver interprets it its own way."""

    pid: str
    i: int
    chunk: int
    body: str
    native: JSON
    ok: bool


def _load_raw_samples(store: Store, run_id: str) -> list[_RawSample]:
    path = store.samples_dir / f"{run_id}.jsonl.gz"
    if not path.exists():
        return []
    out: list[_RawSample] = []
    for row in read_jsonl(path):
        out.append(
            _RawSample(
                pid=str(row["pid"]),
                i=int(row["i"]),
                chunk=int(row.get("chunk", -1) or -1),
                body=str(row.get("body", "")),
                native=cast(JSON, row.get("native")),
                ok=bool(row.get("ok")),
            )
        )
    return out


def _stratified_subsample(samples: Sequence[_RawSample], max_n: int | None) -> list[_RawSample]:
    """Deterministic, stratified-by-`pid` subsample (module docstring): round-robin over `pid`s
    (sorted, for determinism), each pid's own samples kept in their stored `(chunk, i)` order,
    until `max_n` total rows are picked or every pid is exhausted. No RNG is needed -- the
    round-robin order alone already guarantees every pid contributes a row before any pid
    contributes a second one."""
    if max_n is None or len(samples) <= max_n:
        return list(samples)
    by_pid: dict[str, list[_RawSample]] = {}
    for s in samples:
        by_pid.setdefault(s.pid, []).append(s)
    for lst in by_pid.values():
        lst.sort(key=lambda s: (s.chunk, s.i))
    pids = sorted(by_pid)
    out: list[_RawSample] = []
    round_idx = 0
    while len(out) < max_n:
        progressed = False
        for pid in pids:
            lst = by_pid[pid]
            if round_idx < len(lst):
                out.append(lst[round_idx])
                progressed = True
                if len(out) >= max_n:
                    break
        if not progressed:
            break
        round_idx += 1
    return out


# --------------------------------------------------------------------------- shared combination


def _combine_row(
    *,
    pid: str,
    suite: str,
    direction: str,
    policy_label: str,
    ok: bool,
    p_res: Any,
    q_res: Any,
    kinds: Sequence[str],
) -> dict[str, JSON]:
    """One output row (module docstring's row schema): `p_res`/`q_res` are the `p`/`q` side's own
    `MaskedLogProbs` (Dan's or absgpt's -- duck-typed, same field names) on the *same* teacher-
    forced tokens (`p_res.target_ids == q_res.target_ids`, checked by the caller); `kinds[step]` is
    that step's action-kind label. `direction` is `"rl||base"` (`p`=rl, `q`=base, scored along
    rl's own samples) or `"base||rl"` (`p`=base, `q`=rl, along base's own samples) -- `kl_sum` is
    always `sum KL(p || q)`, matching whichever `direction` the caller is building.
    """
    n_tokens = p_res.n_tokens
    kl_sum = 0.0
    logratio_sum = 0.0
    by_kind: dict[str, list[float]] = {}
    p_logp = p_res.log_probs.astype(np.float64, copy=False)
    q_logp = q_res.log_probs.astype(np.float64, copy=False)
    for k in range(n_tokens):
        p_row = p_logp[k]
        diff = p_row - q_logp[k]
        kl_k = float(np.sum(np.exp(p_row) * diff))
        t = int(p_res.target_ids[k])
        lr_k = float(p_row[t] - q_logp[k, t])
        kl_sum += kl_k
        logratio_sum += lr_k
        kind = kinds[int(p_res.step_of[k])]
        acc = by_kind.setdefault(kind, [0.0, 0.0])
        acc[0] += kl_k
        acc[1] += 1
    return {
        "pid": pid,
        "suite": suite,
        "direction": direction,
        "policy": policy_label,
        "ok": ok,
        "n_steps": int(p_res.n_steps),
        "n_tokens": int(n_tokens),
        "kl_sum": kl_sum,
        "kl_per_token": (kl_sum / n_tokens) if n_tokens else None,
        "logratio_sum": logratio_sum,
        "by_kind": {k: [v[0], int(v[1])] for k, v in by_kind.items()},
    }


def _pool_rows(rows: Sequence[Mapping[str, JSON]]) -> dict[str, JSON]:
    """Pooled summary stats over `rows` sharing one direction (or `ok`/kind subset): the mean
    `kl_sum` per sample, the pooled (token-weighted) mean KL per token (`sum(kl_sum) /
    sum(n_tokens)` -- a long proof's tokens count equally regardless of which proof they came
    from), and the unweighted mean of each row's own `kl_per_token` (every *sample* counts equally
    regardless of length) -- both are useful and generally differ."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    kl_sums = [cast(float, r["kl_sum"]) for r in rows]
    n_tokens = [cast(int, r["n_tokens"]) for r in rows]
    logratios = [cast(float, r["logratio_sum"]) for r in rows]
    total_tokens = sum(n_tokens)
    per_token_each = [ks / nt for ks, nt in zip(kl_sums, n_tokens, strict=True) if nt]
    return {
        "n": n,
        "n_tokens_total": total_tokens,
        "mean_kl_sum_per_sample": sum(kl_sums) / n,
        "mean_kl_per_token_pooled": (sum(kl_sums) / total_tokens) if total_tokens else None,
        "mean_kl_per_token_per_sample": (
            (sum(per_token_each) / len(per_token_each)) if per_token_each else None
        ),
        "mean_logratio_sum_per_sample": sum(logratios) / n,
    }


def _by_kind_summary(rows: Sequence[Mapping[str, JSON]]) -> dict[str, JSON]:
    totals: dict[str, list[float]] = {}
    for r in rows:
        by_kind = cast("dict[str, JSON]", r["by_kind"])
        for kind, pair in by_kind.items():
            kl_sum, n_tok = cast("list[float]", pair)
            acc = totals.setdefault(kind, [0.0, 0.0])
            acc[0] += kl_sum
            acc[1] += n_tok
    return {
        kind: {
            "kl_sum": kl_sum,
            "n_tokens": int(n_tok),
            "mean_kl_per_token": (kl_sum / n_tok) if n_tok else None,
        }
        for kind, (kl_sum, n_tok) in totals.items()
    }


def _direction_summary(rows: Sequence[Mapping[str, JSON]]) -> dict[str, JSON]:
    ok_rows = [r for r in rows if r["ok"]]
    not_ok_rows = [r for r in rows if not r["ok"]]
    summary = dict(_pool_rows(rows))
    summary["by_ok"] = {"ok": _pool_rows(ok_rows), "not_ok": _pool_rows(not_ok_rows)}
    summary["by_kind"] = _by_kind_summary(rows)
    return summary


def _summarize(
    rows: Sequence[dict[str, JSON]], n_skipped: int, notes: Sequence[str]
) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_skipped": n_skipped,
        "notes": list(notes),
        "pooled": {
            direction: _direction_summary([r for r in rows if r["direction"] == direction])
            for direction in ("rl||base", "base||rl")
        },
    }


def _int_cfg(cfg: Mapping[str, JSON], key: str, default: int) -> int:
    v = cfg.get(key)
    return default if v is None else int(cast(float, v))


def _suites_cfg(
    cfg: Mapping[str, JSON], default: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    raw = cfg.get("suites")
    if not raw:
        return [dict(s) for s in default]
    return [
        {"suite": str(s["suite"]), "version": str(s.get("version", "v1"))}
        for s in cast("list[Mapping[str, JSON]]", raw)
    ]


# ------------------------------------------------------------------------------------ Dan driver


def _dan_base_of(native: JSON) -> int | None:
    """`native["base"]` (module docstring's "use the sample's own name base" requirement) --
    `None` if `native` doesn't parse as the `{"lean", "base", "steps", "end"}` dict `DanStateAdapter
    .sample`'s `_attempt_fields` writes."""
    if not isinstance(native, str):
        return None
    try:
        obj = json.loads(native)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    base = obj.get("base")
    return int(cast(float, base)) if isinstance(base, (int, float)) else None


def _run_kl_dan(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter,
    rl_adapter: ModelAdapter,
    base_label: str,
    rl_label: str,
) -> dict[str, Any]:
    from ..analysis._common import latest_run_id  # WP6 module; read-only import, not an edit
    from ..vendor import load_danstate
    from .danstate_support import classify_steps

    ds = load_danstate()
    judge_id = str(cfg.get("judge_id", "lean_v2"))
    max_n = _int_cfg(cfg, "max_samples_per_suite", DEFAULT_MAX_SAMPLES_PER_SUITE)
    exclude_defining_names = bool(cfg.get("exclude_defining_names", True))
    allow_reencode = bool(cfg.get("allow_reencode", False))
    suites_cfg = _suites_cfg(cfg, DEFAULT_DAN_SUITES)

    rows: list[dict[str, JSON]] = []
    notes: list[str] = []
    n_skipped = 0
    resolved_run_ids: dict[str, str] = {}
    suites_used: list[dict[str, str]] = []

    for suite_spec in suites_cfg:
        suite, version = suite_spec["suite"], suite_spec["version"]
        try:
            problems = {p.pid: p for p in store.load_suite(suite, version)}
        except (FileNotFoundError, OSError):
            notes.append(f"{suite}/{version}: suite not found; skipping")
            continue
        if not problems:
            notes.append(f"{suite}/{version}: no problems found; skipping")
            continue
        suites_used.append({"suite": suite, "version": version})

        for direction, policy_label, p_adapter, q_adapter in (
            ("rl||base", rl_label, rl_adapter, base_adapter),
            ("base||rl", base_label, base_adapter, rl_adapter),
        ):
            sample_run_id = latest_run_id(
                store, suite=suite, version=version, model=policy_label, judge_id=judge_id
            )
            if sample_run_id is None:
                notes.append(
                    f"{suite}/{version}: no {judge_id} run for {policy_label!r}; "
                    f"skipping {direction}"
                )
                continue
            resolved_run_ids[f"{suite}:{version}:{policy_label}"] = sample_run_id
            raw = _load_raw_samples(store, sample_run_id)
            sub = _stratified_subsample(raw, max_n)

            kept: list[_RawSample] = []
            items: list[tuple[Sequent, str, int]] = []
            for s in sub:
                if s.pid not in problems:
                    n_skipped += 1
                    continue
                if s.body.startswith("UNPARSEABLE"):
                    n_skipped += 1
                    continue
                base = _dan_base_of(s.native)
                if base is None:
                    n_skipped += 1
                    continue
                kept.append(s)
                items.append((problems[s.pid].sequent, s.body, base))

            if not items:
                continue
            p_results = p_adapter.masked_next_logprobs(  # type: ignore[attr-defined]
                items, exclude_defining_names=exclude_defining_names, allow_reencode=allow_reencode
            )
            q_results = q_adapter.masked_next_logprobs(  # type: ignore[attr-defined]
                items, exclude_defining_names=exclude_defining_names, allow_reencode=allow_reencode
            )
            for s, (seq, body, _b), p_res, q_res in zip(
                kept, items, p_results, q_results, strict=True
            ):
                if p_res is None or q_res is None:
                    n_skipped += 1
                    continue
                if p_res.n_tokens != q_res.n_tokens or not np.array_equal(
                    p_res.target_ids, q_res.target_ids
                ):
                    raise ValueError(
                        f"kl: base/rl masked_next_logprobs disagree on tokenisation for "
                        f"pid={s.pid!r} -- base/rl checkpoints do not share one tokenizer/vocab"
                    )
                kinds = [m["rule"] for m in classify_steps(ds, seq.prompt(), body)]
                rows.append(
                    _combine_row(
                        pid=s.pid,
                        suite=suite,
                        direction=direction,
                        policy_label=policy_label,
                        ok=s.ok,
                        p_res=p_res,
                        q_res=q_res,
                        kinds=kinds,
                    )
                )

    summary = _summarize(rows, n_skipped, notes)
    effective_cfg: dict[str, JSON] = {
        **{k: v for k, v in dict(cfg).items() if k != "suites"},
        "base_model": base_label,
        "rl_model": rl_label,
        "exclude_defining_names": exclude_defining_names,
        "allow_reencode": allow_reencode,
        "max_samples_per_suite": max_n,
        "suites": cast(JSON, suites_used),
        "run_ids": cast(JSON, resolved_run_ids),
        "judge_id": judge_id,
        "family": "dan",
    }
    run_id = new_interp_run(store, "kl", effective_cfg)
    write_interp_rows(store, "kl", run_id, rows)
    summary["run_id"] = run_id
    summary["suites"] = suites_used
    summary["run_ids"] = resolved_run_ids
    return summary


# --------------------------------------------------------------------------------- absgpt driver


def _absgpt_kinds(native: str) -> list[str]:
    """Per-"step" kind labels for one absgpt `native` proof text (module docstring's "Breakdown by
    action kind" section): each line's own literal `RULE` token (`'N<i> ... : RULE ...;'`), plus
    `"FINAL"` for the trailing `QED` token (mirroring `MaskedLogProbs.step_of`'s own convention --
    `n_steps = n_lines + 1`)."""
    from ..core.proof import split_lines

    kinds: list[str] = []
    for line in split_lines(native):
        toks = line.split()
        try:
            kinds.append(toks[toks.index(":") + 1])
        except (ValueError, IndexError):
            kinds.append("UNKNOWN")
    kinds.append("FINAL")
    return kinds


def _run_kl_absgpt(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter,
    rl_adapter: ModelAdapter,
    base_label: str,
    rl_label: str,
) -> dict[str, Any]:
    from ..analysis._common import latest_run_id  # WP6 module; read-only import, not an edit

    judge_id = str(cfg.get("judge_id", "nd_v2"))
    max_n = _int_cfg(cfg, "max_samples_per_suite", DEFAULT_MAX_SAMPLES_PER_SUITE)
    suites_cfg = _suites_cfg(cfg, DEFAULT_ABSGPT_SUITES)

    rows: list[dict[str, JSON]] = []
    notes: list[str] = []
    n_skipped = 0
    resolved_run_ids: dict[str, str] = {}
    suites_used: list[dict[str, str]] = []

    for suite_spec in suites_cfg:
        suite, version = suite_spec["suite"], suite_spec["version"]
        try:
            problems = {p.pid: p for p in store.load_suite(suite, version)}
        except (FileNotFoundError, OSError):
            notes.append(f"{suite}/{version}: suite not found; skipping")
            continue
        if not problems:
            notes.append(f"{suite}/{version}: no problems found; skipping")
            continue
        suites_used.append({"suite": suite, "version": version})

        for direction, policy_label, p_adapter, q_adapter in (
            ("rl||base", rl_label, rl_adapter, base_adapter),
            ("base||rl", base_label, base_adapter, rl_adapter),
        ):
            sample_run_id = latest_run_id(
                store, suite=suite, version=version, model=policy_label, judge_id=judge_id
            )
            if sample_run_id is None:
                notes.append(
                    f"{suite}/{version}: no {judge_id} run for {policy_label!r}; "
                    f"skipping {direction}"
                )
                continue
            resolved_run_ids[f"{suite}:{version}:{policy_label}"] = sample_run_id
            raw = _load_raw_samples(store, sample_run_id)
            sub = _stratified_subsample(raw, max_n)

            kept: list[_RawSample] = []
            items: list[tuple[Sequent, str]] = []
            for s in sub:
                if s.pid not in problems:
                    n_skipped += 1
                    continue
                native = s.native
                if not isinstance(native, str) or not native.split() or native.split()[-1] != "QED":
                    n_skipped += 1
                    continue
                kept.append(s)
                items.append((problems[s.pid].sequent, native))

            if not items:
                continue
            p_results = p_adapter.masked_next_logprobs(items)  # type: ignore[attr-defined]
            q_results = q_adapter.masked_next_logprobs(items)  # type: ignore[attr-defined]
            for s, (_seq, native), p_res, q_res in zip(
                kept, items, p_results, q_results, strict=True
            ):
                if p_res is None or q_res is None:
                    n_skipped += 1
                    continue
                if p_res.n_tokens != q_res.n_tokens or not np.array_equal(
                    p_res.target_ids, q_res.target_ids
                ):
                    raise ValueError(
                        f"kl: base/rl masked_next_logprobs disagree on tokenisation for "
                        f"pid={s.pid!r} -- base/rl checkpoints do not share one tokenizer/vocab"
                    )
                kinds = _absgpt_kinds(native)
                rows.append(
                    _combine_row(
                        pid=s.pid,
                        suite=suite,
                        direction=direction,
                        policy_label=policy_label,
                        ok=s.ok,
                        p_res=p_res,
                        q_res=q_res,
                        kinds=kinds,
                    )
                )

    summary = _summarize(rows, n_skipped, notes)
    effective_cfg: dict[str, JSON] = {
        **{k: v for k, v in dict(cfg).items() if k != "suites"},
        "base_model": base_label,
        "rl_model": rl_label,
        "max_samples_per_suite": max_n,
        "suites": cast(JSON, suites_used),
        "run_ids": cast(JSON, resolved_run_ids),
        "judge_id": judge_id,
        "family": "absgpt",
    }
    run_id = new_interp_run(store, "kl", effective_cfg)
    write_interp_rows(store, "kl", run_id, rows)
    summary["run_id"] = run_id
    summary["suites"] = suites_used
    summary["run_ids"] = resolved_run_ids
    return summary


# -------------------------------------------------------------------------------- dispatch (CLI)


def run_kl(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter | None,
    rl_adapter: ModelAdapter | None,
    base_label: str = "base",
    rl_label: str = "rl",
) -> dict[str, Any]:
    """`interp kl` (module docstring): dispatches to `_run_kl_dan` or `_run_kl_absgpt` by the
    resolved adapters' own concrete type (same `isinstance` convention `interp/__main__.py
    ._is_danstate` already uses for `likelihood`/`probes`) -- both `base_adapter`/`rl_adapter`
    must be given and of the same (supported) family."""
    from ..adapters.absgpt import AbsGptAdapter
    from ..adapters.danstate import DanStateAdapter

    if base_adapter is None or rl_adapter is None:
        raise ValueError("kl needs both a base and an rl adapter")
    if isinstance(base_adapter, DanStateAdapter) and isinstance(rl_adapter, DanStateAdapter):
        return _run_kl_dan(
            store,
            cfg,
            base_adapter=base_adapter,
            rl_adapter=rl_adapter,
            base_label=base_label,
            rl_label=rl_label,
        )
    if isinstance(base_adapter, AbsGptAdapter) and isinstance(rl_adapter, AbsGptAdapter):
        return _run_kl_absgpt(
            store,
            cfg,
            base_adapter=base_adapter,
            rl_adapter=rl_adapter,
            base_label=base_label,
            rl_label=rl_label,
        )
    raise ValueError(
        f"kl: unsupported adapter pair ({type(base_adapter).__name__}, "
        f"{type(rl_adapter).__name__}) -- both must be DanStateAdapter or both AbsGptAdapter"
    )
