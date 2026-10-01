"""Model registry helpers (`DANSTATE_PLAN.md` §6, WP-D5 item 1): pairs, families and the right
judge/"seen in training" tier for any model in `configs/models.yaml`, generalising the handful of
hard-coded `base`/`rl` pairings that predate Dan's models (`analysis/yue.py:_pick_pair` and
friends).

Reads the same `configs/models.yaml` `cli.py:_load_models_registry` resolves, duplicated here as
a small, dependency-light loader (`cli.py` pulls in the adapter/judge/suite registries, which this
module -- imported from `analysis/*` views and `dash/*` pages -- has no reason to need) and
exposes:

- `load_registry()`: `{label: ModelEntry}`, `family`/`role`/`seed`/`cap` read straight from the
  yaml; `base`/`rl`/`armC` (which predate those fields) are given the synthetic family `"ours"`
  with their own label as `role`, so every entry has a family/role to group and pair on.
- `families()` / `labels_in_family()` / `seed_labels()` / `pair_for_family()`: group and pair
  models by family (+ role, + seed); `pair_for_family` is the direct generalisation of
  `yue._pick_pair`'s hard-coded `("base", "rl")`.
- `default_judge_for()`: `"lean_v2"` for a Dan family, else `DEFAULT_JUDGE_ID` ("nd_v2") -- plan
  §2's "the headline judge for Dan's models is Lean; nd_v2 alongside".
- `seen_lookup()`: `{pid: seen_in_training}` for one `(suite, version, model)` -- Dan's own
  per-family "seen" sidecar (`logic.exclusion.seen_for`) for a Dan model, else each `Problem`'s own
  `seen_train` (set by the suite's own excluder at build time) for everyone else, so a caller
  computing an "unseen only" variant of a benchmark never needs to special-case Dan's models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from ..core.types import JSON, Problem
from ..logic.exclusion import DAN_FAMILIES, seen_for
from ..store import Store
from ._common import DEFAULT_JUDGE_ID

#: The same file `cli.py`'s `_load_models_registry`/`DEFAULT_MODELS_CONFIG` resolve (plan §6's
#: model table): `analysis/models_registry.py` -> parents[3] is the repo root.
DEFAULT_MODELS_YAML = (
    Path(__file__).resolve().parents[3]
    / "code/experiments/current/benchmarking_20260930/configs/models.yaml"
)

#: Judge id for Dan's models (plan §2): his own sampling protocol and RL reward, and a superset of
#: `DEFAULT_JUDGE_ID` ("nd_v2") on denotable proofs.
DAN_JUDGE_ID = "lean_v2"

#: `configs/models.yaml` entries that predate `family`/`role`/`seed` (plan §0's three checkpoints):
#: grouped under one synthetic family so every piece of pairing/grouping code below has exactly
#: one code path, not a special case for "the three original models".
OURS_FAMILY = "ours"
_OURS_ROLES: dict[str, str] = {"base": "base", "rl": "rl", "armC": "armC"}

#: Role names meaning "pretrained / frozen" and "after RL" across both naming schemes in the
#: registry (ours: base/rl; Dan's: fz/t1 -- plan §6's model table).
BASE_ROLES: tuple[str, ...] = ("base", "fz")
RL_ROLES: tuple[str, ...] = ("rl", "t1")


@dataclass(frozen=True)
class ModelEntry:
    label: str
    adapter: str
    family: str
    role: str | None
    seed: int | None
    cap: int | None
    description: str


def load_registry(path: str | Path | None = None) -> dict[str, ModelEntry]:
    """`configs/models.yaml` -> `{label: ModelEntry}`, in file order. A missing file returns `{}`
    (matching `cli._load_models_registry`'s own "not an error" convention) rather than raising,
    since every caller here already treats "model absent from the registry" as a normal case
    (e.g. `seen_lookup` falls back to each `Problem`'s own `seen_train`)."""
    yaml_path = Path(path) if path is not None else DEFAULT_MODELS_YAML
    if not yaml_path.is_file():
        return {}
    raw = cast("dict[str, JSON]", yaml.safe_load(yaml_path.read_text()) or {})
    out: dict[str, ModelEntry] = {}
    for label, entry in raw.items():
        e = cast("dict[str, JSON]", entry)
        family = cast("str | None", e.get("family"))
        role = cast("str | None", e.get("role"))
        if family is None:
            family = OURS_FAMILY
            role = _OURS_ROLES.get(label, role)
        out[label] = ModelEntry(
            label=label,
            adapter=cast(str, e.get("adapter", "")),
            family=family,
            role=role,
            seed=cast("int | None", e.get("seed")),
            cap=cast("int | None", e.get("cap")),
            description=cast(str, e.get("description", "")),
        )
    return out


def families(registry: Mapping[str, ModelEntry]) -> list[str]:
    """Every distinct family in `registry`, sorted (deterministic page/panel order)."""
    return sorted({e.family for e in registry.values()})


def labels_in_family(registry: Mapping[str, ModelEntry], family: str) -> list[str]:
    return sorted(e.label for e in registry.values() if e.family == family)


def seed_labels(registry: Mapping[str, ModelEntry], family: str, role: str) -> list[str]:
    """`family`'s members with this `role`, sorted by seed (an entry with no seed sorts last)."""
    members = [e for e in registry.values() if e.family == family and e.role == role]

    def _key(e: ModelEntry) -> tuple[bool, int, str]:
        return (e.seed is None, e.seed if e.seed is not None else 0, e.label)

    return [e.label for e in sorted(members, key=_key)]


def pair_for_family(
    registry: Mapping[str, ModelEntry], family: str, seed: int = 0
) -> tuple[str, str] | None:
    """`(base_label, rl_label)` for `family` at `seed`: the direct generalisation of
    `analysis/yue.py:_pick_pair`'s hard-coded `("base", "rl")` to any family/seed the registry
    knows about. `None` if `family` has no member with one of the two roles at that seed (an
    entry with `seed=None` -- `base`/`rl`/`armC` -- is only matched at `seed=0`)."""
    members = [e for e in registry.values() if e.family == family]

    def _pick(roles: tuple[str, ...]) -> str | None:
        return next((e.label for e in members if e.role in roles and (e.seed or 0) == seed), None)

    base, rl = _pick(BASE_ROLES), _pick(RL_ROLES)
    return (base, rl) if base is not None and rl is not None else None


def is_dan_family(family: str) -> bool:
    return family in DAN_FAMILIES


def default_judge_for(
    label_or_family: str, registry: Mapping[str, ModelEntry] | None = None
) -> str:
    """`"lean_v2"` if `label_or_family` names (or, via `registry`, belongs to) one of Dan's
    families, else `DEFAULT_JUDGE_ID` ("nd_v2") -- plan §2's headline-judge rule."""
    family = label_or_family
    if registry is not None and label_or_family in registry:
        family = registry[label_or_family].family
    # `DEFAULT_JUDGE_ID` is typed `str | None` in `_common` (callers may pass `None` to disable
    # the judge filter entirely); its *value* is always the literal "nd_v2" string.
    return DAN_JUDGE_ID if is_dan_family(family) else cast(str, DEFAULT_JUDGE_ID)


def seen_lookup(
    store: Store,
    suite: str,
    version: str,
    model: str,
    problems: Mapping[str, Problem],
    registry: Mapping[str, ModelEntry] | None = None,
) -> dict[str, bool]:
    """`{pid: seen_in_training}` for `model` on `(suite, version)`.

    Dan's models train on different tiers per family (plan §A2/§3's `dan_tiers`), not the tiers
    `problems[pid].seen_train` was computed against when the suite was built (ours) -- so for a
    model belonging to one of `logic.exclusion.DAN_FAMILIES`, this reads that family's own
    pre-built sidecar (`logic.exclusion.seen_for`) instead. Falls back to each `problems[pid].
    seen_train` for every other model, and for a Dan model whose sidecar has not been built yet
    for this `(suite, version)` (not a crash -- just "no Dan-tier check available here").
    """
    reg = registry if registry is not None else load_registry()
    entry = reg.get(model)
    family = entry.family if entry is not None else None
    if family is not None and is_dan_family(family):
        try:
            return seen_for(store, suite, version, family)
        except (FileNotFoundError, OSError):
            pass
    return {pid: p.seen_train for pid, p in problems.items()}
