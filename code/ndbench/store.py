"""`Store`: append-only JSONL persistence for ndbench suites and run results (plan §1).

`Store(root)` treats `root` as the overall ndbench output directory (the plan's
`/scratch/cowifornia/nd-rl/bench/out`, which also holds `cache/`, `dash/`, `logs/` that this class
does not manage) and lays out two subtrees under it:

    root/suites/<suite>/<version>/items.jsonl     one Problem.to_json() per line
    root/suites/<suite>/<version>/manifest.json   the SuiteManifest
    root/store/runs.jsonl                         run start/finish events (see below)
    root/store/items/<run_id>.jsonl               one ItemResult per (item, chunk)
    root/store/samples/<run_id>.jsonl.gz          one raw sample (body + verdict) per line
    root/store/edl/<run_id>.jsonl                 EDL rows (plan §B7 / WP5)
    root/store/ledger.jsonl                       GPU-lease bookkeeping (plan §A2 / WP1)

`runs.jsonl` is event-sourced rather than mutated in place, to keep every file in this class
append-only: `new_run` appends a `{"event": "start", ...}` row and `finish_run` appends a
`{"event": "finish", ...}` row for the same `run_id`; `runs()`/`latest()` fold a run's rows (in
file order) into one record before matching/returning it. `items/<run_id>.jsonl` rows are also
appended per chunk as the run progresses; `items()` merges chunks for the same `pid` into one
`ItemResult` (its `chunk` field is `-1` as a "merged across chunks" sentinel).

`ledger.jsonl` rows are free-form JSON entries; `ledger_total_s` sums each entry's `"gpu_s"` field
(`0` if absent), matching plan §4's "the ledger counts lock-held wall time only" stop rule.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

from .core.io import append_jsonl, read_jsonl, sha256_str, utc_now_iso, write_jsonl
from .core.types import JSON, ItemResult, Problem, SuiteManifest


class _AdaptiveRuleLike(Protocol):
    @property
    def chunk(self) -> int: ...
    @property
    def n_max(self) -> int: ...
    @property
    def stop_c(self) -> int: ...


class _RunSpecLike(Protocol):
    """The subset of `runner.runner.RunSpec`'s shape `config_key`/`new_run` need.

    Defined structurally (rather than importing `RunSpec` itself, even under `TYPE_CHECKING`) so
    `store.py` never imports anything from `runner/`, and `runner.py` (which does import `Store`
    for its own type hints) stays a one-directional dependency -- `runner.runner.RunSpec`
    satisfies this Protocol automatically, no inheritance needed. Declared as read-only
    properties (rather than plain attributes) so a *frozen* dataclass like `RunSpec` -- whose
    fields are read-only from a type checker's point of view -- structurally matches; a Protocol
    with plain mutable attributes would demand read/write access `RunSpec` cannot offer.
    """

    @property
    def suite(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def k(self) -> int: ...
    @property
    def temperature(self) -> float: ...
    @property
    def max_new_tokens(self) -> int | None: ...
    @property
    def seed(self) -> int: ...
    @property
    def adaptive(self) -> _AdaptiveRuleLike | None: ...
    @property
    def pids(self) -> tuple[str, ...] | None: ...
    @property
    def options(self) -> Mapping[str, JSON]: ...


class Store:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.suites_dir = self.root / "suites"
        self.store_dir = self.root / "store"
        self.items_dir = self.store_dir / "items"
        self.samples_dir = self.store_dir / "samples"
        self.edl_dir = self.store_dir / "edl"
        self.runs_path = self.store_dir / "runs.jsonl"
        self.ledger_path = self.store_dir / "ledger.jsonl"
        for d in (self.suites_dir, self.store_dir, self.items_dir, self.samples_dir, self.edl_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- suites

    def write_suite(self, problems: Sequence[Problem], manifest: SuiteManifest) -> None:
        d = self.suites_dir / manifest.suite / manifest.version
        write_jsonl(d / "items.jsonl", (p.to_json() for p in problems))
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(asdict(manifest), indent=1, sort_keys=True))

    def load_suite(self, suite: str, version: str) -> list[Problem]:
        d = self.suites_dir / suite / version
        return [Problem.from_json(row) for row in read_jsonl(d / "items.jsonl")]

    # ------------------------------------------------------------------ runs

    @staticmethod
    def config_key(spec: _RunSpecLike, identity: Mapping[str, JSON], judge_id: str) -> str:
        """A hash of everything that makes a run resamplable-identical: used as part of the
        `run_id` (so re-running the same `RunSpec` against the same model/judge is idempotent and
        visible as such) and as the lookup key for `latest`."""
        payload = {
            "suite": spec.suite,
            "version": spec.version,
            "model": spec.model,
            "k": spec.k,
            "temperature": spec.temperature,
            "max_new_tokens": spec.max_new_tokens,
            "seed": spec.seed,
            "adaptive": None
            if spec.adaptive is None
            else {
                "chunk": spec.adaptive.chunk,
                "n_max": spec.adaptive.n_max,
                "stop_c": spec.adaptive.stop_c,
            },
            "pids": None if spec.pids is None else list(spec.pids),
            "options": dict(spec.options),
            "identity": dict(identity),
            "judge_id": judge_id,
        }
        return sha256_str(json.dumps(payload, sort_keys=True, default=str))[:16]

    def new_run(self, spec: _RunSpecLike, identity: Mapping[str, JSON], judge_id: str) -> str:
        key = self.config_key(spec, identity, judge_id)
        run_id = f"{spec.suite}-{spec.version}-{spec.model}-{key}"
        append_jsonl(
            self.runs_path,
            [
                {
                    "event": "start",
                    "run_id": run_id,
                    "suite": spec.suite,
                    "version": spec.version,
                    "model": spec.model,
                    "config_key": key,
                    "judge_id": judge_id,
                    "identity": dict(identity),
                    "k": spec.k,
                    "started_utc": utc_now_iso(),
                    "status": "running",
                }
            ],
        )
        return run_id

    def finish_run(self, run_id: str, status: str, gpu_s: float) -> None:
        append_jsonl(
            self.runs_path,
            [
                {
                    "event": "finish",
                    "run_id": run_id,
                    "status": status,
                    "gpu_s": gpu_s,
                    "finished_utc": utc_now_iso(),
                }
            ],
        )

    def _folded_runs(self) -> list[dict[str, JSON]]:
        folded: dict[str, dict[str, JSON]] = {}
        order: list[str] = []
        if self.runs_path.exists():
            for row in read_jsonl(self.runs_path):
                rid = cast(str, row["run_id"])
                if rid not in folded:
                    folded[rid] = {}
                    order.append(rid)
                folded[rid].update(cast(dict[str, JSON], row))
        return [folded[rid] for rid in order]

    def runs(self, **match: JSON) -> list[dict[str, JSON]]:
        return [
            rec for rec in self._folded_runs() if all(rec.get(k) == v for k, v in match.items())
        ]

    def latest(self, suite: str, model: str, config_key: str) -> str | None:
        candidates = self.runs(suite=suite, model=model, config_key=config_key)
        if not candidates:
            return None
        candidates.sort(
            key=lambda r: cast(str, r.get("finished_utc") or r.get("started_utc") or "")
        )
        return cast(str, candidates[-1]["run_id"])

    # ----------------------------------------------------------------- items

    def append_items(self, run_id: str, rows: Sequence[ItemResult]) -> None:
        append_jsonl(self.items_dir / f"{run_id}.jsonl", (asdict(r) for r in rows))

    def item_chunks(self, run_id: str) -> list[dict[str, JSON]]:
        """Raw, unmerged per-(item, chunk) rows for `run_id`, in file (append) order -- the exact
        rows `append_items` wrote, before `items()` merges them into one `ItemResult` per pid
        (whose `chunk` field becomes the `-1` "merged" sentinel and so can no longer answer "what
        is this pid's own highest recorded chunk number"). Used by `runner.run`'s resume support
        (fix list item 2) to pick up each item's next unused chunk number after a killed run;
        returns `[]` if `run_id` has no recorded items at all.
        """
        path = self.items_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        return list(read_jsonl(path))

    def items(self, run_id: str) -> list[ItemResult]:
        path = self.items_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        rows = sorted(read_jsonl(path), key=lambda r: cast(int, r["chunk"]))
        merged: dict[str, ItemResult] = {}
        tokens_sum: dict[str, float] = {}
        for row in rows:
            pid = cast(str, row["pid"])
            m = merged.get(pid)
            if m is None:
                m = ItemResult(
                    run_id=run_id,
                    pid=pid,
                    chunk=-1,
                    n=0,
                    c=0,
                    first_success=None,
                    ok_written=[],
                    ok_pruned=[],
                    ok_tokens=[],
                    ok_starts=[],
                    fail_kinds={},
                    fail_lines=[],
                    long_enough=0,
                    tokens_mean=0.0,
                    hit_max=0,
                )
                merged[pid] = m
                tokens_sum[pid] = 0.0
            if m.first_success is None and row["first_success"] is not None:
                m.first_success = m.n + cast(int, row["first_success"])
            m.n += cast(int, row["n"])
            m.c += cast(int, row["c"])
            m.ok_written.extend(cast(list[int], row["ok_written"]))
            m.ok_pruned.extend(cast(list[int], row["ok_pruned"]))
            m.ok_tokens.extend(cast(list[int], row["ok_tokens"]))
            m.ok_starts.extend(cast(list[int], row["ok_starts"]))
            for fk, fv in cast(dict[str, int], row["fail_kinds"]).items():
                m.fail_kinds[fk] = m.fail_kinds.get(fk, 0) + fv
            m.fail_lines.extend(cast(list[int], row["fail_lines"]))
            m.long_enough += cast(int, row["long_enough"])
            m.hit_max += cast(int, row["hit_max"])
            tokens_sum[pid] += cast(float, row["tokens_mean"]) * cast(int, row["n"])
        for pid, m in merged.items():
            m.tokens_mean = tokens_sum[pid] / m.n if m.n else 0.0
        return list(merged.values())

    # --------------------------------------------------------------- samples

    def append_samples(self, run_id: str, rows: Sequence[Mapping[str, JSON]]) -> None:
        """Append `rows` (one dict per sample) as a new gzip member of
        `samples/<run_id>.jsonl.gz`. Opening in append mode writes an independent gzip stream per
        call; `gzip`'s reader transparently concatenates multiple members, so the file is always
        readable as one continuous JSONL stream (e.g. via `core.io.read_jsonl`) regardless of how
        many chunks wrote to it.
        """
        path = self.samples_dir / f"{run_id}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ edl

    def append_edl(self, run_id: str, rows: Sequence[Mapping[str, JSON]]) -> None:
        append_jsonl(self.edl_dir / f"{run_id}.jsonl", rows)

    # --------------------------------------------------------------- ledger

    def ledger_append(self, entry: Mapping[str, JSON]) -> None:
        append_jsonl(self.ledger_path, [dict(entry)])

    def ledger_total_s(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        return sum(
            float(cast(float, row.get("gpu_s", 0.0) or 0.0)) for row in read_jsonl(self.ledger_path)
        )
