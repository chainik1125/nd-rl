"""Dan-pair support for the interp measurements (`DANSTATE_PLAN.md` §7 WP-D6, §1, §3 rows I1-I4,
§4 rows D11-D13): a `DanStateAdapter`-specific `likelihood` (I1) and `probes` (I4) driver, plus a
`steering` (I2) vector-fitting function. Weight interpolation (I3) needs **no** Dan-specific code
here at all: `interp/interpolate.py`'s `interpolate()` already preserves every non-`"state"` key of
a checkpoint dict byte-for-byte (`out_ck = dict(base_ck); out_ck["state"] = new_state`), so it
writes a valid `{"cfg", "state", "tok_mode", "extra"}` Dan checkpoint unchanged; `run_interpolation`
is generalised in `interp/interpolate.py` itself (`base_adapter_label`/`rl_adapter_label`/
`label_prefix`/`judge`) and `interp/__main__.py`'s `cmd_interpolate` resolves Dan's pair through
`configs/models.yaml` the same way `cli._resolve_model` does. See this work package's final report
for the exact wiring.

Why this module exists at all rather than reusing `interp/likelihood.py`/`probes.py` unchanged:
their activation extraction assumes a whole-sequence, per-token `activations` call (`absgpt`'s own
shape, with "proof"/"prompt"/"line_start" *position types* sliced out of it) -- that does not apply
to `DanStateAdapter.activations`, which already does its own per-step position selection (module
docstring's "Probes" section). Everything *else* is format-agnostic and reused directly below,
**including the neutral/reserve proof pool itself**: `load_reference_proofs` calls straight into
`interp/data.py`'s `load_neutral_proofs`/`load_sealed_l12_14`/`load_family_witnesses` -- the exact
same `ab2/data/reserve.jsonl` + `ab2/sealed/sealed_L12-14.jsonl` + `families` suite pool the absgpt
measures use (`DANSTATE_PLAN.md` §7 WP-D6's own configs: "reserve proofs as ours") -- since the ND
proof *format*/theorem pool is shared across model families (plan §0: identical prompts; "100% of
our suites are expressible" under Dan's own grammar) and a neutral proof is not "owned" by any one
model. The only Dan-specific piece is *which* theorems count as `seen_train` for Dan (his own
training tiers differ from ours): pass `logic.exclusion.DanExcluder(family)` as `excluder` instead
of `data.build_exclusion_index()`'s absgpt tiers. `likelihood._combine_exact_or_importance`,
`data.load_verified_samples`, `likelihood._load_base_success`, and `probes.probe_classification`/
`probe_regression`/`split_by_pid` are reused verbatim for the same reason.

## Likelihood (I1): ordering dedup is *by canonical action sequence*, not order-equivalence

`orderings.py`'s box/PR reordering model (`enumerate_orderings`/`sample_orderings`) is proof-syntax
only -- it does not know about Dan's state-env grammar at all, so it is reused unchanged to
generate candidate line-orderings of a given ND proof. What *is* Dan-specific: two line-orderings
of the same underlying proof can denote the exact same `lean_seq` action sequence (the box/PR
model's permutation freedom is coarser than the state-env's own name-assignment-independent action
grammar), in which case scoring both under `logsumexp` would double-count a single real sample from
the policy. `DanStateAdapter.canonical_actions(seq, body)` (a base-0, hashable action-sequence key;
`DANSTATE_PLAN.md` §1) is exactly the dedup key the plan asks for ("deduplicate orderings by action
sequence"); `_dedupe_orderings_by_canonical_actions` below enumerates/samples via `orderings.py`
and keeps one representative body per distinct key, then scores only the survivors and combines
them with `likelihood._combine_exact_or_importance` (reused verbatim: once duplicates are dropped,
the remaining math -- log-sum-exp in the exact branch, self-normalised importance weighting in the
sampled one -- is identical to the absgpt case).

## Probes (I4): classifying an action's own ND rule

`DanStateAdapter.activations` already extracts, for `positions="act"`, one feature vector per
*step* at the `<act>` token (the policy's decision point, immediately before the action is
written) -- this module only needs to label each step with the ND rule the *resulting* line uses
(`next_rule`) and how many further steps remain (`actions_to_go`). There is no public helper for
this in the vendor tree (`lean_tok.inverse` classifies a *complete* proof's lines, not one action
at a time), so `_classify_action` replicates its own per-action dispatch table directly (module
docstring continues in `_classify_action`'s own docstring) by replaying `canonicalise(prompt,
body)` + `split_actions` + `Env(prompt, canon=True)` exactly as `state_env.decompose` itself does
internally (so the step order this module assigns labels in is, by construction, the same order
`DanStateAdapter.activations`'s own `decompose` call produces rows in) and reading off, at each
step, the *pre*-apply `env.frames[-1].kind` (for an `exact` action) or the action's own term shape
(for a `have` action) -- precisely the two pieces of state `lean_tok.inverse`/`proof_tokens`
dispatch on (verified directly against both, see `_classify_action`'s docstring and
`test_dan_interp.py`).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..adapters.base import ModelAdapter
from ..adapters.danstate import DanStateAdapter
from ..core.io import sha256_str, utc_now_iso
from ..core.proof import split_lines
from ..core.types import JSON, Excluder, Sequent
from ..store import Store
from ..vendor import load_danstate
from . import data as absgpt_data
from . import likelihood as _likelihood
from . import orderings
from . import probes as _probes
from ._store import new_interp_run, write_interp_rows

#: `data.NeutralProof` is reused verbatim as this module's own neutral-proof type (module
#: docstring: the pool itself is shared across model families, not absgpt-specific).
DanNeutralProof = absgpt_data.NeutralProof

# ----------------------------------------------------------------------------------------- data


def build_dan_excluder(family: str, *, cache_dir: str | Path | None = None) -> Excluder:
    """Dan's own `seen_train` tier for `family` (`"dan6"` | `"dan12"`) -- `logic.exclusion.
    DanExcluder`, not `data.build_exclusion_index`'s absgpt tiers (module docstring)."""
    from ..logic.exclusion import DanExcluder

    return DanExcluder(family, cache_dir=cache_dir)


def load_reference_proofs(
    store: Store,
    *,
    excluder: Excluder | None = None,
    n: int | None = 2000,
    l_min: int = 6,
    l_max: int = 10,
    seed: int = 0,
    reserve_path: Path | None = None,
    sealed_path: Path | None = None,
    scan_limit: int | None = None,
    family_version: str = "v1",
    family_limit: int = _likelihood.DEFAULT_FAMILY_LIMIT,
) -> list[DanNeutralProof]:
    """The same reserve/sealed/family neutral-proof pool the absgpt interp measures use (module
    docstring): `data.load_neutral_proofs` (reserve, `L` in `[l_min, l_max]`, capped at `n`),
    topped up from `data.load_sealed_l12_14` if short, plus up to `family_limit` family
    witnesses (`0` drops them, matching `likelihood.DEFAULT_FAMILY_LIMIT`'s own convention) --
    `[]` only if `n == 0` *and* `family_limit == 0`. `reserve_path`/`sealed_path` let a test point
    at a tiny fixture file instead of the real multi-hundred-MB `ab2` data.
    """
    neutral: list[DanNeutralProof] = []
    if n is None or n > 0:
        reserve_kwargs: dict[str, Any] = {
            "n": n if n is not None else 2000,
            "l_min": l_min,
            "l_max": l_max,
            "excluder": excluder,
            "seed": seed,
            "scan_limit": scan_limit,
        }
        if reserve_path is not None:
            reserve_kwargs["reserve_path"] = reserve_path
        neutral = absgpt_data.load_neutral_proofs(**reserve_kwargs)
        if n is not None and len(neutral) < n:
            sealed_kwargs: dict[str, Any] = {
                "n": n - len(neutral),
                "excluder": excluder,
                "seed": seed,
            }
            if sealed_path is not None:
                sealed_kwargs["path"] = sealed_path
            neutral += absgpt_data.load_sealed_l12_14(**sealed_kwargs)
    if family_limit != 0:
        neutral += absgpt_data.load_family_witnesses(
            store,
            version=family_version,
            n=family_limit if family_limit > 0 else None,
            seed=seed,
        )
    return neutral


def reference_proofs_from_cfg(
    store: Store, cfg: Mapping[str, JSON], *, n: int | None, seed: int
) -> tuple[list[DanNeutralProof], list[str]]:
    """`load_reference_proofs` driven by the config keys `run_likelihood_dan`/`run_probes_dan`/
    `interp/__main__.py`'s `fit-vectors` Dan branch share: `family` (`"dan6"`/`"dan12"`, for
    `build_dan_excluder`'s `seen_train` tier -- `None`, the default, applies no exclusion at all,
    which is what every test fixture wants, since a tiny made-up `pid` is never `seen_train`
    under a real excluder either way, just more slowly), `exclude_train` (default `True`;
    `False` skips building an excluder even when `family` is given), `cache_dir`, `l_min`/`l_max`
    (6/10), `reserve_path`/`sealed_path` (a test fixture's own tiny JSONL in place of the real
    `ab2` files), `scan_limit`, `family_version` (`"v1"`), `family_limit`
    (`likelihood.DEFAULT_FAMILY_LIMIT` by default, `0` to drop family witnesses entirely).
    Returns `(proofs, notes)` -- `notes` is `["no reference proofs found; ..."]` when empty.
    """
    excluder = None
    if cfg.get("family") and cfg.get("exclude_train", True):
        excluder = build_dan_excluder(
            str(cfg["family"]), cache_dir=cast("str | None", cfg.get("cache_dir"))
        )
    family_limit_cfg = cfg.get("family_limit")
    family_limit = (
        _likelihood.DEFAULT_FAMILY_LIMIT
        if family_limit_cfg is None
        else int(cast(float, family_limit_cfg))
    )
    proofs = load_reference_proofs(
        store,
        excluder=excluder,
        n=n,
        l_min=int(cast(float, cfg.get("l_min", 6) or 6)),
        l_max=int(cast(float, cfg.get("l_max", 10) or 10)),
        seed=seed,
        reserve_path=Path(cast(str, cfg["reserve_path"])) if cfg.get("reserve_path") else None,
        sealed_path=Path(cast(str, cfg["sealed_path"])) if cfg.get("sealed_path") else None,
        scan_limit=cast("int | None", cfg.get("scan_limit")),
        family_version=str(cfg.get("family_version", "v1")),
        family_limit=family_limit,
    )
    notes = (
        []
        if proofs
        else [
            "no reference proofs found (reserve/sealed/family sources empty or n=0+family_limit=0)"
        ]
    )
    return proofs, notes


# ----------------------------------------------------------------------- I1: likelihood


def _dedupe_orderings_by_canonical_actions(
    adapter: DanStateAdapter,
    seq: Sequent,
    body: str,
    *,
    cap: int,
    m: int,
    rng: random.Random,
) -> tuple[list[str], list[float] | None, str, int | None, int]:
    """Enumerate (`orderings.count_orderings(body) <= cap`) or sample (`m` draws) candidate
    line-orderings exactly as `likelihood._plan_item` does, then drop every ordering whose
    `adapter.canonical_actions(seq, ·)` repeats an earlier one's (module docstring's "Likelihood"
    section): `body` itself is always among the exact-enumeration candidates (same guarantee
    `likelihood.marginal_logprob` relies on), so its own key is always present among the survivors
    unless something else already produced the identical action sequence first -- either way one
    representative body with that key survives, which is all a log-sum-exp/importance-weighted
    combination needs. Returns `(unique_bodies, unique_logqs_or_None, method, n_orderings,
    n_duplicates_dropped)`; `method` is `"single"` whenever only one unique body survives
    (including the trivial `n_orderings in (None-but-sampled-to-one-key, 1)` cases), `"exact"` for
    an exact enumeration with >1 surviving key, `"importance"` for the sampled branch.

    A candidate ordering whose action sequence cannot be computed at all (`canonical_actions`
    raises -- a malformed ordering `orderings.py`'s own `test_interp_orderings.py`-style
    verification against `nd_verify` has not been seen to produce, but this module does not
    re-verify every ordering against Lean the way that test suite does) is kept, under its own
    unique, never-repeating key, rather than silently dropped -- it is still a legitimate ordering
    to score, just not a candidate for dedup.
    """
    n = orderings.count_orderings(body)
    if n is not None and n <= cap:
        bodies = orderings.enumerate_orderings(body, cap)
        method_full = "exact" if n > 1 else "single"
        logqs: list[float] | None = None
    else:
        samples = orderings.sample_orderings(body, m, rng)
        bodies = [b for b, _q in samples]
        logqs = [q for _b, q in samples]
        method_full = "importance"

    seen: dict[tuple[tuple[str, ...], ...], int] = {}
    uniq_bodies: list[str] = []
    uniq_logqs: list[float] | None = [] if logqs is not None else None
    n_dup = 0
    for i, b in enumerate(bodies):
        try:
            key = adapter.canonical_actions(seq, b)
        except Exception:  # pragma: no cover -- defensive, see docstring
            key = (("__unkeyable__", str(i)),)
        if key in seen:
            n_dup += 1
            continue
        seen[key] = len(uniq_bodies)
        uniq_bodies.append(b)
        if uniq_logqs is not None:
            assert logqs is not None
            uniq_logqs.append(logqs[i])

    method = method_full if len(uniq_bodies) > 1 else "single"
    return uniq_bodies, uniq_logqs, method, n, n_dup


def marginal_logprob_dan(
    adapter: DanStateAdapter,
    seq: Sequent,
    body: str,
    *,
    cap: int = _likelihood.DEFAULT_CAP,
    m: int = _likelihood.DEFAULT_M,
    rng: random.Random | None = None,
) -> dict[str, JSON]:
    """`likelihood.marginal_logprob`'s Dan-pair analogue: dedupe orderings by canonical action
    sequence first (`_dedupe_orderings_by_canonical_actions`), then score and combine exactly as
    the absgpt version does (`likelihood._combine_exact_or_importance`, reused verbatim). Returns
    a plain dict (not `likelihood.MarginalResult`, since this also reports the dedup counts that
    dataclass has no fields for): `logp_single`, `logp_marginal`, `n_orderings` (before dedup),
    `n_unique_actions` (surviving distinct action sequences actually scored),
    `n_duplicate_orderings`, `method`, `ess`.
    """
    rng = rng if rng is not None else random.Random(0)
    uniq_bodies, uniq_logqs, method, n, n_dup = _dedupe_orderings_by_canonical_actions(
        adapter, seq, body, cap=cap, m=m, rng=rng
    )
    logps = [lp.nats for lp in adapter.logprob([(seq, b) for b in uniq_bodies])]
    if method != "importance":
        try:
            single = logps[uniq_bodies.index(body)]
        except ValueError:
            single = adapter.logprob([(seq, body)])[0].nats
    else:
        single = adapter.logprob([(seq, body)])[0].nats
    marginal, ess = _likelihood._combine_exact_or_importance(method, logps, uniq_logqs)
    return {
        "logp_single": single,
        "logp_marginal": marginal,
        "n_orderings": n,
        "n_unique_actions": len(uniq_bodies),
        "n_duplicate_orderings": n_dup,
        "method": method,
        "ess": ess,
    }


_NULL_ROW: dict[str, JSON] = {
    "logp_single": None,
    "logp_marginal": None,
    "n_orderings": None,
    "n_unique_actions": None,
    "n_duplicate_orderings": None,
    "method": None,
    "ess": None,
}


def run_likelihood_dan(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter | None,
    rl_adapter: ModelAdapter | None,
) -> dict[str, Any]:
    """`interp likelihood` for Dan's pair (`fz`=base, `t1`=rl; `DANSTATE_PLAN.md` §3 row I1, §4 row
    D11): neutral/reference proofs (`load_reference_proofs`, capped at `cfg["n_neutral"]`, default
    2000) plus each model's own verified `Store` samples (`cfg["model_samples"] = {"fz": {"suite",
    "version", "model"?, "limit"?}, "t1": {...}}`, reusing `data.load_verified_samples`, which is
    already format-agnostic). Scored with `marginal_logprob_dan` under whichever of
    `base_adapter`/`rl_adapter` is given; NaN scores (`DanStateAdapter.logprob`'s out-of-grammar
    convention) are kept in each row (so the row-per-pid record stays complete) but counted
    separately in the returned summary (`n_nan_fz`/`n_nan_t1`) -- callers filter them before
    aggregating, per the plan's "filter NaN with counts reported".

    `cfg["fz_success"] = {"suite", "version", "model"?}` marks the crux set (`p_hat_fz` per row,
    `likelihood._load_base_success` reused verbatim: it is already generic over any Store model
    label) -- theorems `t1` solves that `fz` (almost) never does.
    """

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    seed = _int_cfg("seed", 0)
    cap = _int_cfg("cap", _likelihood.DEFAULT_CAP)
    m = _int_cfg("m", _likelihood.DEFAULT_M)
    n_neutral = _int_cfg("n_neutral", 2000)

    neutral, notes = reference_proofs_from_cfg(store, cfg, n=n_neutral, seed=seed)

    items: list[tuple[str, str, Sequent, str, int | None, str]] = [
        (p.pid, p.origin, p.sequent, p.body, p.L, p.pid) for p in neutral
    ]

    model_samples_cfg = cfg.get("model_samples") or {}
    assert isinstance(model_samples_cfg, Mapping)
    for model_name in ("fz", "t1"):
        spec = model_samples_cfg.get(model_name)
        if not spec:
            continue
        assert isinstance(spec, Mapping)
        samples = absgpt_data.load_verified_samples(
            store,
            suite=str(spec["suite"]),
            version=str(spec["version"]),
            model=str(spec.get("model", model_name)),
            limit=int(cast(float, spec.get("limit", 500)) or 500),
        )
        if not samples:
            notes.append(f"no verified samples found for model_samples[{model_name}]; skipping")
            continue
        origin = f"{model_name}_verified"
        items += [(f"{origin}:{s.pid}", origin, s.sequent, s.body, None, s.pid) for s in samples]

    fz_success = _likelihood._load_base_success(store, cfg.get("fz_success"))

    run_id = new_interp_run(store, "likelihood", cfg)
    n_nan_fz = 0
    n_nan_t1 = 0
    rows: list[dict[str, JSON]] = []
    for pid, origin, seq, body, length_l, lookup_pid in items:
        rng = random.Random(f"{seed}:{pid}")
        row: dict[str, JSON] = {
            "pid": pid,
            "origin": origin,
            "L": length_l,
            "lines": len(split_lines(body)),
            "p_hat_fz": fz_success.get(lookup_pid),
        }
        for key, adapter in (("fz", base_adapter), ("t1", rl_adapter)):
            if adapter is None:
                row[key] = dict(_NULL_ROW)
                continue
            result = marginal_logprob_dan(
                cast(DanStateAdapter, adapter), seq, body, cap=cap, m=m, rng=rng
            )
            marginal = cast(float, result["logp_marginal"])
            if marginal != marginal:  # NaN
                if key == "fz":
                    n_nan_fz += 1
                else:
                    n_nan_t1 += 1
            row[key] = cast(JSON, result)
        rows.append(row)

    write_interp_rows(store, "likelihood", run_id, rows)
    return {
        "run_id": run_id,
        "n_items": len(rows),
        "n_neutral": len(neutral),
        "n_nan_fz": n_nan_fz,
        "n_nan_t1": n_nan_t1,
        "notes": notes,
    }


# -------------------------------------------------------------------------------- I2: steering


@dataclass(frozen=True)
class DanDiffVectorResult:
    layer: int
    vector: np.ndarray  # shape (d,)
    norm: float
    mean_residual_norm: float
    n_examples: int
    n_positions: int
    path: Path


def fit_diff_vectors_dan(
    fz_adapter: DanStateAdapter,
    t1_adapter: DanStateAdapter,
    pairs: Sequence[tuple[Sequent, str]],
    layers: Sequence[int],
    *,
    out_dir: Path,
) -> list[DanDiffVectorResult]:
    """`DANSTATE_PLAN.md` §3 row I2 / §1's `activations(..., positions="action")`: the mean
    (T1 - Fz) activation difference at every action token of `pairs` (teacher-forced reserve
    proofs), per layer. Unlike `steering.fit_diff_vectors` (absgpt: a full per-token sequence, so
    "proof"/"prompt"/"line_start" position *types* select slices of one `(T, d)` array per layer),
    `DanStateAdapter.activations` already does the position selection itself
    (`positions="action"` concatenates every step's action tokens, in order, across the whole
    proof) -- there is no separate position-type axis to add here, so this is the direct mean over
    everything `activations` returns, per layer, pooled across every pair.

    Saves `out_dir/diff_l<layer>_action.npy` + a `.json` sidecar (`layer`, `n_examples`,
    `n_positions`, `source_sha`, `norm`, `mean_residual_norm`, `relative_norm`), mirroring
    `steering.fit_diff_vectors`'s own sidecar shape so `steering.steering_sweep` (which only reads
    `diff_l<layer>_<position_type>.npy` off disk, format-agnostically) can consume these directly
    with `position_type="action"`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fz_acts = fz_adapter.activations(pairs, layers, positions="action")
    t1_acts = t1_adapter.activations(pairs, layers, positions="action")
    source_sha = sha256_str(
        json.dumps([[seq.prompt(), body] for seq, body in pairs], sort_keys=True)
    )[:16]

    results: list[DanDiffVectorResult] = []
    for li, layer in enumerate(layers):
        diffs: list[np.ndarray] = []
        base_norms: list[float] = []
        for fz_full, t1_full in zip(fz_acts, t1_acts, strict=True):
            fz_layer, t1_layer = fz_full[li], t1_full[li]  # (n_action_tokens, d) each
            if fz_layer.shape[0] == 0:
                continue
            diffs.append(t1_layer - fz_layer)
            base_norms.append(float(np.linalg.norm(fz_layer, axis=-1).mean()))
        if not diffs:
            continue
        all_diffs = np.concatenate(diffs, axis=0)
        delta = all_diffs.mean(axis=0)
        norm = float(np.linalg.norm(delta))
        mean_residual_norm = float(np.mean(base_norms)) if base_norms else float("nan")

        out_path = out_dir / f"diff_l{layer}_action.npy"
        np.save(out_path, delta.astype(np.float32))
        sidecar = {
            "layer": layer,
            "position_type": "action",
            "n": len(pairs),
            "n_positions": int(all_diffs.shape[0]),
            "source_sha": source_sha,
            "norm": norm,
            "mean_residual_norm": mean_residual_norm,
            "relative_norm": (norm / mean_residual_norm) if mean_residual_norm else None,
            "created_utc": utc_now_iso(),
        }
        out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, sort_keys=True))
        results.append(
            DanDiffVectorResult(
                layer=layer,
                vector=delta,
                norm=norm,
                mean_residual_norm=mean_residual_norm,
                n_examples=len(pairs),
                n_positions=int(all_diffs.shape[0]),
                path=out_path,
            )
        )
    return results


# ---------------------------------------------------------------------------------- I4: probes

#: Every `next_rule` class `_classify_action` can emit, in the order the module docstring's
#: dispatch table lists them (`"FINAL"` is excluded from the probe dataset itself -- see
#: `build_probe_dataset` -- but kept here since `_classify_action` can still return it).
RULE_CLASSES: tuple[str, ...] = (
    "AS",
    "PR",
    "ANDI",
    "ANDE1",
    "ANDE2",
    "IMPE",
    "NEGE",
    "ORI1",
    "ORI2",
    "BOTE",
    "DN",
    "R",
    "IMPI",
    "NEGI",
    "ORE",
)


def _classify_action(env: Any, act: Sequence[str], frame_kind_before: str) -> tuple[str, bool]:
    """`(next_rule, creates_line)` for one already-canonicalised `lean_seq` action, *before* it is
    applied to `env` (`env.lookup` is used for the IMPE/NEGE disambiguation below, which needs the
    pre-apply scope). Mirrors two things the vendor tree does per-complete-proof, applied one
    action at a time (module docstring):

    * `state_env.split_actions`' own per-action shapes (`vendor/danstate/state_env.py:81-150`):
      an `exact` action is always exactly `['exact', name]`; a `have` action is `['have', name,
      ':', <formula tokens>, ':=', <term tokens>, ';']` for an atomic rule, or ends at `'by'`
      (no trailing `';'`) for a box-opener (`imp`/`neg`: term head `'('`; `Or.elim`'s first
      branch: term head `'Or.elim'`).
    * `lean_tok.inverse`'s own per-rule term-shape dispatch (`vendor/danstate/lean_tok.py:99-112`,
      reproduced in reverse by `proof_tokens`, lines ~95-109): `'⟨'` -> ANDI; `name '.1'` ->
      ANDE1; `name '.2'` -> ANDE2; `name '.elim'` -> BOTE; `'Or.inl' name` -> ORI1; `'Or.inr'
      name` -> ORI2; `'Classical.byContradiction' ...` -> DN; a single bare token (term ends
      right there, i.e. the next token is `';'`) is `PR` (`lean_tok.proof_tokens`: `rule == 'PR':
      term = [f'h{n_pr}']` -- a bare *premise* token `h<k>`, restating a premise as an explicit
      proof step, unlike absgpt where `PR` lines live in the prompt and are never actions) or `R`
      (restate an earlier derived line: a bare `n<k>` name instead) -- told apart by the token's
      own leading letter (`'h'` vs `'n'`), exactly how `is_name`/premise tokens are told apart
      throughout the vendor tree; two bare names `[a, b]` -> IMPE (`forms[a]` is not a negation)
      or NEGE (`forms[a]` *is* one) -- `inverse` resolves this via its own tracked `forms` map;
      this function uses `env.lookup(a)` instead, which is the exact state-env equivalent (both
      are "the formula bound to name `a`, in scope at this point").

    An `exact` action never has its own term to inspect; it closes whatever box/frame is open
    (`frame_kind_before`, read *before* this action is applied -- `state_env.Frame.kind` is one of
    `top|imp|neg|or1|or2`), which `suites.danstate.materialize_profile`'s own docstring already
    works out line-for-line: closing `top` ends the whole proof (cites the final line; *creates*
    no new ND line -- `"FINAL"`, `creates_line=False`); closing `imp`/`neg` creates that box's own
    IMPI/NEGI conclusion line; closing `or1` creates the environment-injected `AS` line opening
    branch 2 (not an ORE line yet); closing `or2` creates the ORE conclusion line.
    """
    if act[0] == "exact":
        by_kind = {"top": ("FINAL", False), "imp": ("IMPI", True), "neg": ("NEGI", True)}
        by_kind["or1"] = ("AS", True)
        by_kind["or2"] = ("ORE", True)
        if frame_kind_before not in by_kind:
            raise ValueError(f"unknown frame kind {frame_kind_before!r}")
        return by_kind[frame_kind_before]
    if act[0] != "have":
        raise ValueError(f"unexpected action head {act[0]!r}")
    j = list(act).index(":=")
    head = act[j + 1]
    if head == "(" or head == "Or.elim":
        return "AS", True
    if head == "⟨":
        return "ANDI", True
    if head == "Or.inl":
        return "ORI1", True
    if head == "Or.inr":
        return "ORI2", True
    if head == "Classical.byContradiction":
        return "DN", True
    # `head` is a bare name/premise token: ANDE1/ANDE2/BOTE/PR/R, or the start of an IMPE/NEGE pair.
    nxt = act[j + 2]
    if nxt == ".1":
        return "ANDE1", True
    if nxt == ".2":
        return "ANDE2", True
    if nxt == ".elim":
        return "BOTE", True
    if nxt == ";":
        return ("PR" if head.startswith("h") else "R"), True
    formula = env.lookup(head)
    if formula is not None and formula[0] == "not":
        return "NEGE", True
    return "IMPE", True


def classify_steps(ds: Any, prompt: str, body: str) -> list[dict[str, Any]]:
    """One classification dict per step of `state_env.decompose(prompt, body, canon=True)`'s own
    step order (`canonicalise` -> `split_actions` -> replay through a fresh `Env(prompt,
    canon=True)`, exactly `decompose`'s own internal sequence -- module docstring): `{"rule":
    next_rule, "creates_line": bool, "frame_depth": int, "n_prem": int}`, `frame_depth`/`n_prem`
    read *before* the action is applied (the state the policy actually saw)."""
    toks = ds.state_env.canonicalise(prompt, body)
    acts = ds.state_env.split_actions(toks)
    env = ds.state_env.Env(prompt, canon=True)
    out: list[dict[str, Any]] = []
    for act in acts:
        frame_kind_before = env.frames[-1].kind
        rule, creates_line = _classify_action(env, act, frame_kind_before)
        out.append(
            {
                "rule": rule,
                "creates_line": creates_line,
                "frame_depth": len(env.frames) - 1,
                "n_prem": len(env.prem),
            }
        )
        ok, why = env.apply(act)
        if not ok:
            raise ValueError(f"classify_steps: replay failed on {act!r}: {why}")
    return out


@dataclass(frozen=True)
class DanProbeExample:
    pid: str
    step_idx: int  # 0-based index among this proof's *line-creating* steps
    n_actions: int  # total line-creating steps in this proof
    layer_features: dict[str, np.ndarray]  # str(layer) -> (d,)
    next_rule: str
    actions_to_go: int
    frame_depth: int
    n_prem: int


def build_probe_dataset(
    adapter: DanStateAdapter,
    items: Sequence[tuple[str, Sequent, str]],
    *,
    layers: Sequence[int],
) -> list[DanProbeExample]:
    """One `DanProbeExample` per line-creating step of every `(pid, sequent, body)` in `items`
    (module docstring): features = `adapter.activations(pairs, layers, positions="act")` (the
    `<act>` token, i.e. the state the policy conditioned on to pick that very action); labels from
    `classify_steps`. Steps classified `"FINAL"` (`creates_line=False` -- the proof-closing
    `exact` at the top frame) are dropped: there is no "line the action creates" for them, so
    `next_rule` has nothing to name (`_classify_action`'s docstring)."""
    ds = load_danstate()
    pairs = [(seq, body) for _pid, seq, body in items]
    acts_all = adapter.activations(pairs, layers, positions="act")
    out: list[DanProbeExample] = []
    for (pid, seq, body), acts in zip(items, acts_all, strict=True):
        steps_meta = classify_steps(ds, seq.prompt(), body)
        assert acts.shape[1] == len(steps_meta), (acts.shape, len(steps_meta))
        kept_meta = [s for s in steps_meta if s["creates_line"]]
        n_kept = len(kept_meta)
        kept_idx = 0
        for i, meta in enumerate(steps_meta):
            if not meta["creates_line"]:
                continue
            feats = {str(layer): acts[li, i].copy() for li, layer in enumerate(layers)}
            out.append(
                DanProbeExample(
                    pid=pid,
                    step_idx=kept_idx,
                    n_actions=n_kept,
                    layer_features=feats,
                    next_rule=meta["rule"],
                    actions_to_go=n_kept - (kept_idx + 1),
                    frame_depth=meta["frame_depth"],
                    n_prem=meta["n_prem"],
                )
            )
            kept_idx += 1
    return out


def _feature_matrix(examples: Sequence[DanProbeExample], layer: int) -> np.ndarray:
    return np.stack([e.layer_features[str(layer)] for e in examples], axis=0)


def run_layer_probes_dan(
    examples: Sequence[DanProbeExample], splits: Mapping[str, set[str]], layer: int
) -> dict[str, JSON]:
    """`next_rule` (multinomial logistic, `probes.probe_classification`) and `actions_to_go`
    (ridge, `probes.probe_regression`) at one `layer`, train/val/test split by theorem -- both
    reused verbatim from `interp/probes.py` (pure numpy/torch over a feature matrix, no absgpt
    dependency)."""
    by_split = {name: [e for e in examples if e.pid in pids] for name, pids in splits.items()}
    tr, va, te = by_split["train"], by_split["val"], by_split["test"]

    def xy_rule(exs: Sequence[DanProbeExample]) -> tuple[np.ndarray, np.ndarray]:
        idx = {c: i for i, c in enumerate(RULE_CLASSES)}
        y = np.array([idx[e.next_rule] for e in exs], dtype=np.int64)
        return _feature_matrix(exs, layer), y

    def xy_atg(exs: Sequence[DanProbeExample]) -> tuple[np.ndarray, np.ndarray]:
        return _feature_matrix(exs, layer), np.array([e.actions_to_go for e in exs], dtype=float)

    next_rule = _probes.probe_classification(
        xy_rule(tr), xy_rule(va), xy_rule(te), len(RULE_CLASSES)
    )
    actions_to_go = _probes.probe_regression(xy_atg(tr), xy_atg(va), xy_atg(te))
    return {"next_rule": next_rule, "actions_to_go": actions_to_go}


def position_baselines_dan(
    items: Sequence[tuple[str, Sequent, str]], splits: Mapping[str, set[str]]
) -> dict[str, JSON]:
    """`interp probe-baselines`'s Dan analogue (`probes.position_baselines`'s own docstring,
    adapted): non-neural lookup-table predictors fit on `train`, scored on `test`, keyed by step
    index alone, step index + `#premises`, and step index + `#premises` + frame depth (the plan's
    "step index + #premises + frame depth") -- all three visible from the state alone, without any
    model. A probe only shows information beyond position if it beats these."""
    ds = load_danstate()
    rows: list[tuple[str, int, int, int, str, int]] = []  # pid, k, n_prem, depth, rule, atg
    for pid, seq, body in items:
        meta = [s for s in classify_steps(ds, seq.prompt(), body) if s["creates_line"]]
        n = len(meta)
        for k, s in enumerate(meta):
            rows.append((pid, k, s["n_prem"], s["frame_depth"], s["rule"], n - (k + 1)))
    train = [r for r in rows if r[0] in splits["train"]]
    test = [r for r in rows if r[0] in splits["test"]]
    if not train or not test:
        return {"n_train": len(train), "n_test": len(test)}
    majority = Counter(r[4] for r in train).most_common(1)[0][0]
    mean_atg = float(np.mean([r[5] for r in train]))
    keys: dict[str, Any] = {
        "majority": lambda r: (),
        "step_index": lambda r: (r[1],),
        "step_index_premises": lambda r: (r[1], r[2]),
        "step_index_premises_depth": lambda r: (r[1], r[2], r[3]),
    }
    y_atg = np.array([r[5] for r in test], dtype=float)
    ss_tot = float(((y_atg - y_atg.mean()) ** 2).sum()) or 1.0
    out: dict[str, JSON] = {"n_train": len(train), "n_test": len(test)}
    for name, keyf in keys.items():
        rule_tab: dict[Any, Counter[str]] = {}
        atg_tab: dict[Any, list[int]] = {}
        for r in train:
            rule_tab.setdefault(keyf(r), Counter())[r[4]] += 1
            atg_tab.setdefault(keyf(r), []).append(r[5])
        acc = float(
            np.mean(
                [
                    (rule_tab[keyf(r)].most_common(1)[0][0] if keyf(r) in rule_tab else majority)
                    == r[4]
                    for r in test
                ]
            )
        )
        pred = np.array(
            [float(np.mean(atg_tab[keyf(r)])) if keyf(r) in atg_tab else mean_atg for r in test]
        )
        out[f"next_rule_acc_{name}"] = acc
        out[f"actions_to_go_r2_{name}"] = 1.0 - float(((y_atg - pred) ** 2).sum()) / ss_tot
    return out


def run_probes_dan(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter | None,
    rl_adapter: ModelAdapter | None,
) -> dict[str, Any]:
    """`interp probes` for Dan's pair (`DANSTATE_PLAN.md` §3 row I4, §4 row D11): builds the
    per-action-step dataset (`build_probe_dataset`) from `cfg["n_train"]` (default 300) reference
    proofs (`load_reference_proofs`) for each of `base_adapter`(=fz)/`rl_adapter`(=t1), runs
    `next_rule`/`actions_to_go` probes per `cfg["layers"]` (default every layer `0..n_layer-1` of
    whichever adapter is given -- callers on a real checkpoint should pass the actual layer count
    explicitly; the tiny test fixture does), split by theorem (`probes.split_by_pid`, reused
    verbatim). Writes one row per `(model, layer)` to `out/store/interp/probes/<run_id>.jsonl`,
    plus a `probe_baselines` row (`position_baselines_dan`) on the same split -- mirroring
    `probes.run_position_baselines`'s "like `probe-baselines`" convention the plan asks for.
    """

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    seed = _int_cfg("seed", 0)
    n_train = _int_cfg("n_train", 300)
    notes: list[str] = []

    proofs, notes = reference_proofs_from_cfg(store, cfg, n=n_train, seed=seed)

    items = [(p.pid, p.sequent, p.body) for p in proofs]
    splits = _probes.split_by_pid([pid for pid, *_r in items], seed=seed)

    layers_cfg = cfg.get("layers")
    rows: list[dict[str, JSON]] = []
    for model_name, adapter in (("fz", base_adapter), ("t1", rl_adapter)):
        if adapter is None:
            notes.append(f"no {model_name} adapter given; skipping its probes")
            continue
        if layers_cfg is not None:
            layers = tuple(cast("list[int]", layers_cfg))
        else:
            n_layer = int(getattr(adapter, "_cfg", {}).get("n_layer", 6))
            layers = tuple(range(n_layer))
        examples = build_probe_dataset(cast(DanStateAdapter, adapter), items, layers=layers)
        for layer in layers:
            layer_result = run_layer_probes_dan(examples, splits, layer)
            rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "n_train_examples": len(examples),
                    **layer_result,
                }
            )

    baseline_row = position_baselines_dan(items, splits) if items else {}

    run_id = new_interp_run(store, "probes", cfg)
    write_interp_rows(store, "probes", run_id, rows)
    return {
        "run_id": run_id,
        "n_rows": len(rows),
        "n_train": len(items),
        "baselines": baseline_row,
        "notes": notes,
    }
