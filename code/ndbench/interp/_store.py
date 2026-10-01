"""Storage layout for the interp package's own (non-`runner.run`) artefacts.

`INTERP_SPEC.md`'s "Storage" section: rows for the four non-benchmark-run analyses (`likelihood`,
`vectors`, `probes`, `lens`) go under `out/store/interp/<kind>/<run_id>.jsonl`, indexed by
`out/store/interp/runs.jsonl` -- separate from (but nested under the same root as) `store.py`'s own
`runs.jsonl`/`items/`/`samples/` layout, since these rows are not `ItemResult`s and were never
produced by `runner.run`. Steering-sweep and weight-interpolation *evaluations* are ordinary
`Store` runs instead (`interp/steering.py`, `interp/interpolate.py`) and never touch this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..core.io import append_jsonl, read_jsonl, sha256_str, utc_now_iso, write_jsonl
from ..core.types import JSON
from ..store import Store

KINDS: tuple[str, ...] = ("likelihood", "vectors", "probes", "probe_baselines", "lens")


def interp_dir(store: Store, kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown interp artefact kind {kind!r}, expected one of {KINDS}")
    return store.store_dir / "interp" / kind


def _index_path(store: Store) -> Path:
    return store.store_dir / "interp" / "runs.jsonl"


def new_interp_run(store: Store, kind: str, cfg: Mapping[str, JSON]) -> str:
    """A stable id for one artefact run (`<kind>-<cfg hash>`, so re-running an identical `cfg` is
    idempotent and visible as such, mirroring `Store.new_run`'s `config_key` convention) and an
    index row appended to `runs.jsonl`."""
    key = sha256_str(json.dumps(dict(cfg), sort_keys=True, default=str))[:16]
    run_id = f"{kind}-{key}"
    append_jsonl(
        _index_path(store),
        [{"run_id": run_id, "kind": kind, "cfg": dict(cfg), "created_utc": utc_now_iso()}],
    )
    return run_id


def write_interp_rows(
    store: Store, kind: str, run_id: str, rows: Sequence[Mapping[str, JSON]]
) -> None:
    write_jsonl(interp_dir(store, kind) / f"{run_id}.jsonl", rows)


def load_interp_rows(store: Store, kind: str, run_id: str) -> list[dict[str, JSON]]:
    path = interp_dir(store, kind) / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return list(read_jsonl(path))


def interp_runs(store: Store, kind: str | None = None) -> list[dict[str, JSON]]:
    """Every indexed artefact run, newest last (append order), optionally restricted to `kind`."""
    path = _index_path(store)
    if not path.exists():
        return []
    rows = list(read_jsonl(path))
    if kind is not None:
        rows = [r for r in rows if r.get("kind") == kind]
    return rows


def latest_interp_run(store: Store, kind: str) -> str | None:
    rows = interp_runs(store, kind)
    return str(rows[-1]["run_id"]) if rows else None
