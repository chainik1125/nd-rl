"""Prefix-completion suite (plan §2 B6 / WP4).

Takes verified long proofs (existing sources -- `ab3/sealed/*`, or any other already-certified
`Problem`s, e.g. this package's own `glue`/family suites once built), buckets them by written
length `L`, and for each one emits several `Problem`s that force a specific proof *prefix* (plan's
`PrefixSpec`: the first `j` lines, N1-based, no `QED`) at `j` in
`{0, n_prem, ⌈L/4⌉, ⌈L/2⌉, ⌈3L/4⌉, L-3, L-2, L-1}` (deduplicated, clipped to `[0, L-1]`). A model is
then sampled continuing from that prefix; success is "prefix + continuation passes `nd_verify`",
which `runner`/`judge` (not this module) checks.

Every source proof is worth reusing as-is (no new witness is constructed here, unlike `glue.py`),
so this module does not call `certifier` -- it only re-verifies each source with the vendored
`nd_verify` (a cheap integrity check) and applies the plan §2 dedup policy ("drop for ... prefix
items") via `excluder`.
"""

from __future__ import annotations

import collections
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..core import proof as core_proof
from ..core.io import read_jsonl
from ..core.types import (
    JSON,
    Certifier,
    Excluder,
    LengthLabel,
    PrefixSpec,
    Problem,
    ReferenceProof,
    Sequent,
)
from ..store import Store
from ..vendor import load_nd_verify
from .glue import DEFAULT_STORE_ROOT

#: Plan §2 B6's target length buckets.
DEFAULT_BUCKETS: tuple[int, ...] = (10, 14, 18, 24, 30, 40)

#: `label_kind` (ab3/sealed schema) -> `LengthLabel.method`.
_METHOD_MAP = {"exact": "minlen_exact", "ub_cert14": "ub_lb"}


def _excluded(seq: Sequent, excluder: Excluder | None) -> bool:
    """Plan §2's dedup policy: "drop for ... prefix items" on a match in either tier."""
    if excluder is None:
        return False
    in_train, eval_sets = excluder.status(seq)
    return bool(in_train or eval_sets)


def _dedup_key(seq: Sequent) -> str:
    return seq.pkey() if len(seq.premises) <= 5 else seq.key()


def _load_raw_source(spec: Mapping[str, JSON]) -> list[Problem]:
    """Load a jsonl file in the `ab3/sealed/*` schema (`prompt`, `minlen_proof`, `n_lines`,
    `label_kind`, `minlen_bound`, `name`; see plan §0/§2 B6) as source `Problem`s. `spec` keys
    (`path` required): `prompt_field`, `proof_field`, `length_field`, `method_field`,
    `bound_field`, `name_field` override the field names for a differently-shaped file.

    A `label_kind == "ub_cert14"` row (no exact minlen search completed) has `minlen_proof: null`
    in this schema; `n_lines`/`L_true` there is the *constructed* proof's own length (verified
    empirically: `n_lines == constructed_lines` on every `ub_cert14` row in `ab3/sealed/*`), so
    when `proof_field`'s value is `None`, this falls back to `fallback_proof_field`
    (`"constructed_proof"` by default) for the body, keeping the same `n_lines`.
    """
    path = Path(cast(str, spec["path"]))
    prompt_field = cast(str, spec.get("prompt_field", "prompt"))
    proof_field = cast(str, spec.get("proof_field", "minlen_proof"))
    fallback_proof_field = cast(str, spec.get("fallback_proof_field", "constructed_proof"))
    length_field = cast(str, spec.get("length_field", "n_lines"))
    method_field = cast(str, spec.get("method_field", "label_kind"))
    bound_field = cast(str, spec.get("bound_field", "minlen_bound"))
    name_field = cast(str, spec.get("name_field", "name"))
    pid_prefix = cast(str, spec.get("pid_prefix", path.stem))
    out: list[Problem] = []
    for i, row in enumerate(read_jsonl(path)):
        prompt = cast(str, row[prompt_field])
        body = cast("str | None", row.get(proof_field))
        if body is None:
            body = cast(str, row[fallback_proof_field])
        n_lines = cast(int, row[length_field])
        method = _METHOD_MAP.get(cast(str, row.get(method_field, "exact")), "minlen_exact")
        bound = cast("int | None", row.get(bound_field))
        label = LengthLabel(
            L=n_lines,
            L_ub=n_lines,
            L_lb=n_lines if method == "minlen_exact" else None,
            method=method,
            bound=bound,
        )
        ref = ReferenceProof(
            body=body, lines=n_lines, pruned=core_proof.pruned_length(body), source="minlen"
        )
        name = cast("str | int", row.get(name_field, i))
        out.append(
            Problem(
                pid=f"{pid_prefix}_{name}",
                suite="prefix_src",
                version="raw",
                sequent=Sequent.from_prompt(prompt),
                length=label,
                ref=ref,
                tags={},
            )
        )
    return out


def _load_store_source(spec: Mapping[str, JSON], store_root: str) -> list[Problem]:
    store = Store(Path(cast(str, spec.get("store_root", store_root))))
    suite = cast(str, spec["suite"])
    version = cast(str, spec.get("version", "v1"))
    return store.load_suite(suite, version)


def _prefix_indices(ln: int, n_prem: int) -> list[int]:
    """`{0, n_prem, ⌈L/4⌉, ⌈L/2⌉, ⌈3L/4⌉, L-3, L-2, L-1}`, deduplicated and clipped to `[0, L-1]`
    (plan §2 B6)."""
    raw = (0, n_prem, -(-ln // 4), -(-ln // 2), -(-3 * ln // 4), ln - 3, ln - 2, ln - 1)
    return sorted({j for j in raw if 0 <= j < ln})


def build(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> tuple[list[Problem], dict[str, Any]]:
    """Build the prefix-completion suite (plan §2 B6 / §3 WP4 row).

    `cfg` keys (all optional):
      - `injected_problems` (list[Problem]): source proofs supplied directly (test-only escape
        hatch -- these are not YAML-serialisable, so only a Python caller can use this key).
      - `sources` (list[{"suite","version","store_root"?}]): additional sources loaded from a
        `Store` (e.g. this package's own `glue` suite, once built: plan "glue ... witnesses when
        available").
      - `raw_sources` (list[{"path", ...}]): `ab3/sealed/*`-schema jsonl files (see
        `_load_raw_source`); default source for a real build is none (the caller must name the
        sealed files explicitly, since this module must not hardcode a read-only path as the only
        option -- see the WP4 report for the exact paths used for the real build).
      - `store_root` (str): default `Store` root used by `sources` entries that omit their own.
      - `buckets` (list[int]): default `DEFAULT_BUCKETS` = (10, 14, 18, 24, 30, 40).
      - `tolerance` (int): a source's `L` is assigned to the nearest bucket if within `tolerance`
        of it; default 0 (exact match only, per plan §2 B6's precise bucket values).
      - `per_bucket` (int): default 50 (plan: "50 per L bucket").
      - `version` (str): suite version tag; default `"v1"`.
    `certifier` is accepted for signature uniformity (plan §3's builder contract) but unused: every
    source proof is already certified upstream; this module only re-verifies it with `nd_verify`.
    """
    del certifier
    rng = random.Random(rng_seed)
    store_root = cast(str, cfg.get("store_root", DEFAULT_STORE_ROOT))

    sources: list[Problem] = list(cast(Sequence[Problem], cfg.get("injected_problems", ())))
    for spec in cast(list[Mapping[str, JSON]], cfg.get("sources", [])):
        sources.extend(_load_store_source(spec, store_root))
    for spec in cast(list[Mapping[str, JSON]], cfg.get("raw_sources", [])):
        sources.extend(_load_raw_source(spec))

    buckets = tuple(cast(list[int], cfg.get("buckets", list(DEFAULT_BUCKETS))))
    tolerance = int(cast(int, cfg.get("tolerance", 0)))
    per_bucket = int(cast(int, cfg.get("per_bucket", 50)))
    version = cast(str, cfg.get("version", "v1"))

    nd_verify = load_nd_verify()
    by_bucket: dict[int, list[Problem]] = {b: [] for b in buckets}
    drops: collections.Counter[str] = collections.Counter()
    seen_pkeys: set[str] = set()

    for src in sources:
        if src.ref is None:
            drops["no_ref"] += 1
            continue
        ln = src.ref.lines
        if not buckets:
            continue
        target = min(buckets, key=lambda b: abs(b - ln))
        if abs(target - ln) > tolerance:
            drops["no_matching_bucket"] += 1
            continue
        ok, _reason, nl = nd_verify.verify_text(src.sequent.prompt() + " " + src.ref.body)
        if not (ok and nl == ln):
            drops["self_check_fail"] += 1
            continue
        key = _dedup_key(src.sequent)
        if key in seen_pkeys:
            drops["duplicate"] += 1
            continue
        if _excluded(src.sequent, excluder):
            drops["excluded"] += 1
            continue
        seen_pkeys.add(key)
        by_bucket[target].append(src)

    problems: list[Problem] = []
    bucket_counts: dict[int, int] = {}
    for b in buckets:
        cands = list(by_bucket.get(b, ()))
        rng.shuffle(cands)
        chosen = cands[:per_bucket]
        bucket_counts[b] = len(chosen)
        for src in chosen:
            assert src.ref is not None
            ln = src.ref.lines
            n_prem = len(src.sequent.premises)
            for j in _prefix_indices(ln, n_prem):
                body = core_proof.take_prefix(src.ref.body, j, new_start=1)
                problems.append(
                    Problem(
                        pid=f"prefix_L{b}_{src.pid}_j{j:03d}",
                        suite="prefix",
                        version=version,
                        sequent=src.sequent,
                        length=src.length,
                        ref=src.ref,
                        tags={"j": j, "L": ln, "src_pid": src.pid, "bucket": b},
                        prefix=PrefixSpec(j=j, body=body),
                    )
                )

    stats: dict[str, Any] = {
        "n_sources_considered": len(sources),
        "bucket_counts": cast(JSON, dict(bucket_counts)),
        "short_buckets": cast(JSON, {b: n for b, n in bucket_counts.items() if n < per_bucket}),
        "drops": cast(JSON, dict(drops)),
        "n_items": len(problems),
    }
    return problems, stats
