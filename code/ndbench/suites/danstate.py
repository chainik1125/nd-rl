"""Suites for Dan's state-env models (plan `DANSTATE_PLAN.md` §3's `prefix_state`/`dan_tiers` row,
§7 WP-D3): `dan_heldout`, `textbook_train14`, `prefix_state`, `skill_pool_sub`.

Every builder matches the plan's standard contract: `build(cfg, *, certifier, excluder,
rng_seed) -> tuple[list[Problem], dict]` (see `suites/existing.py`'s module docstring). Dan's
"seen in training" tiers (`dan_tiers`/sidecar files) live in `logic/exclusion.py`, not here.

## `prefix_state`'s `m(j)` construction

`prefix_state` repacks the already-built `prefix` suite (one item per `(theorem, j)`, `j` a target
written-line count) into Dan's state-env `STATEPREFIX` format: instead of forcing the first `j` ND
*lines* verbatim (which commits to box goals the model never chose, plan §1's "Prefix" paragraph),
it forces the first `m` **env actions** of the reference's `lean_seq` decomposition, where `m(j)`
is the smallest `m` whose replay has "materialised" (would already be `emit()`-ted by vendored
`lean_tok.inverse`, given a well-formed prefix of its input) at least `j` of the reference's ND
lines.

`materialize_profile` derives this by replaying the canonicalised action list through a fresh
`Env(prompt, canon=True, base=0, assign=True)` and, at each step, classifying the action just
applied by its effect on `inverse()`'s own `emit()` call sites (traced directly against
`vendor/danstate/lean_tok.py:inverse` and `vendor/danstate/state_env.py:Env._close`):

* a `'have ...'` action always completes exactly one ND line -- either the direct rule line (an
  atomic/PR term) or the `AS` line opening a box (`imp`/`neg`, or an `Or.elim`'s first branch);
* an `'exact ...'` action completes one ND line **unless** it closes the top-level frame (the
  proof's very last action, which only cites the final line -- no new line is created). This
  covers an `imp`/`neg` box's own conclusion line, an `Or.elim`'s `or2`-closing `ORE` conclusion
  line, and -- closing `or1` -- the environment-injected `AS` line that opens the second branch
  (`Env._close`'s `or1` case renders that opener as a side effect of applying the *model's*
  branch-1 `exact`, even though no model action is sampled for it; plan §1's "Or.elim second-
  branch opener is NOT an action" note).

Verified directly against `inverse`'s line count on two hand sources (one plain `IMPI`/`BOTE`
proof, one with an `ORE`): the resulting `counts` sequence is `0, 1, 2, ..., L, L` -- it reaches
the reference's full written length `L` one action *before* the end (the final top-level `exact`
adds nothing), confirmed for several references via the module's own replay (see
`test_dan_suites.py`). Because of that one-step lag, `m(j)` is **not** simply "smallest `m` with
`counts[m] >= j`" at the `j == L` boundary -- that would give `A - 1`, not `A`. `m_for_j` special-
cases `j >= L` to return `A` (`len(counts) - 1`) by convention: the full reference needs the full
action replay, even though its last ND line's *content* was already fixed one action earlier. For
every `j < L` (the only values `prefix`'s own buckets ever use -- see `suites/prefix.py`'s
`_prefix_indices`, always `< ln`), `m_for_j` is the ordinary "smallest crossing" rule, and the two
definitions agree in the limit: `m_for_j` is non-decreasing across the `j = L` boundary too, since
`A >= m_for_j(j)` for every `j < L`.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
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
    Sequent,
)
from ..store import Store
from ..vendor.danstate_loader import load_danstate
from .glue import DEFAULT_STORE_ROOT

__all__ = [
    "dan_heldout",
    "textbook_train14",
    "prefix_state",
    "skill_pool_sub",
    "materialize_profile",
    "m_for_j",
]

BuildResult = tuple[list[Problem], dict[str, JSON]]

#: Dan's downloaded Stage-1 held-out set (plan §A2; md5-checked `ae6daf49...`).
DEFAULT_DAN_HELDOUT_PATH = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl")

#: Dan's fork clone's textbook72 train14 (read-only; plan §A4). Prompt-identical dev58 half is our
#: own `textbook_dev` v1 (plan §0).
DEFAULT_TEXTBOOK_TRAIN14_PATH = Path(
    "/scratch/cowifornia/nd-rl/bench/nd-takehome-dan/data/eval_only/textbook72/textbook_train.jsonl"
)

#: plan §3: "about 1,200 stratified" subsample of `skill_pool` (11,524).
SKILL_POOL_SUB_N_ITEMS = 1200


def _status(excluder: Excluder | None, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
    if excluder is None:
        return False, ()
    return excluder.status(seq)


# --------------------------------------------------------------------------------- dan_heldout


def dan_heldout(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """Dan's held-out greedy-gate set (plan §3 "B8"/V5: `heldout.jsonl`, 5,000), as ordinary
    `Problem`s. Each row already carries its own (generator, not minlen-exact) proof and length;
    no relabelling (`certifier` accepted for signature uniformity, unused). `excluder`, if given,
    flags any row that happens to coincide with *our own* training/eval tiers (a cross-
    contamination diagnostic -- these items are Dan's, not drawn from our pools, so a hit here
    would be a genuine theorem-class collision, not an expected one).

    `cfg` keys: `path` (default `DEFAULT_DAN_HELDOUT_PATH`), `version` ("v1").
    """
    del certifier, rng_seed
    path = Path(cfg.get("path", DEFAULT_DAN_HELDOUT_PATH))
    version = str(cfg.get("version", "v1"))

    problems: list[Problem] = []
    n_seen_train = 0
    n_eval_flagged = 0
    for row in read_jsonl(path):
        seq = Sequent.from_thm(cast(str, row["thm"]))
        in_train, eval_sets = _status(excluder, seq)
        n_seen_train += int(in_train)
        n_eval_flagged += int(bool(eval_sets))
        n_lines = cast(int, row["n_lines"])
        problems.append(
            Problem(
                pid=str(row["name"]),
                suite="dan_heldout",
                version=version,
                sequent=seq,
                length=LengthLabel(
                    L=n_lines, L_ub=n_lines, L_lb=None, method="reference", bound=None
                ),
                ref=core_proof.make_reference(cast(str, row["proof"]), source="constructed"),
                tags={
                    "dan_name": row.get("name"),
                    "dan_key": row.get("key"),
                    "n_prem": row.get("n_prem"),
                    "rules": cast(JSON, row.get("rules")),
                    "pat": cast(JSON, row.get("pat")),
                },
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "n_seen_train": n_seen_train,
        "n_eval_flagged": n_eval_flagged,
        "source_path": str(path),
    }
    return problems, stats


# ----------------------------------------------------------------------------- textbook_train14


def textbook_train14(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """Dan's textbook72 train14 half (plan §3 "B8"; dev58 is our own `textbook_dev` v1, prompt-
    identical per plan §0). These 14 rows carry no `L_true`/proof at all (unlike `textbook_dev`'s
    source), so -- exactly like `existing.textbook_dev` -- this relabels every item with
    `certifier` (required) and reports its overlap with our already-built `textbook_dev`/
    `textbook_law` suites (plan §1: "Check whether our existing textbook suites already contain
    them; note overlaps") by `Sequent.key()`.

    `cfg` keys: `path` (default `DEFAULT_TEXTBOOK_TRAIN14_PATH`), `bound` (16), `time_s` (120.0),
    `store_root` (`DEFAULT_STORE_ROOT`, for the overlap check against already-built suites),
    `overlap_suites` (default `(("textbook_dev", "v1"), ("textbook_law", "v1"))`), `version`
    ("v1").
    """
    del rng_seed
    if certifier is None:
        raise ValueError("textbook_train14 requires a certifier to label its 14 items")

    path = Path(cfg.get("path", DEFAULT_TEXTBOOK_TRAIN14_PATH))
    bound = int(cfg.get("bound", 16))
    time_s = float(cfg.get("time_s", 120.0))
    version = str(cfg.get("version", "v1"))
    store_root = cast(str, cfg.get("store_root", DEFAULT_STORE_ROOT))
    overlap_suites = cast(
        "Sequence[tuple[str, str]]",
        cfg.get("overlap_suites", (("textbook_dev", "v1"), ("textbook_law", "v1"))),
    )

    rows = list(read_jsonl(path))
    seqs = [Sequent.from_prompt(cast(str, r["prompt"])) for r in rows]
    labels = certifier.certify(seqs, [None] * len(seqs), bound=bound, time_s=time_s)

    overlap: dict[str, JSON] = {}
    store = Store(Path(store_root))
    for suite, suite_version in overlap_suites:
        try:
            other = store.load_suite(suite, suite_version)
        except FileNotFoundError:
            overlap[f"{suite}/{suite_version}"] = None
            continue
        other_keys = {p.sequent.key() for p in other}
        overlap[f"{suite}/{suite_version}"] = sum(1 for s in seqs if s.key() in other_keys)

    problems: list[Problem] = []
    by_method: Counter[str] = Counter()
    n_seen_train = 0
    for row, seq, label in zip(rows, seqs, labels, strict=True):
        in_train, eval_sets = _status(excluder, seq)
        n_seen_train += int(in_train)
        by_method[label.method] += 1
        problems.append(
            Problem(
                pid=str(row.get("name", row.get("canonical_hash"))),
                suite="textbook_train14",
                version=version,
                sequent=seq,
                length=label,
                ref=None,
                tags={
                    "dan_name": row.get("name"),
                    "canonical_hash": row.get("canonical_hash"),
                    "sampling_bucket": row.get("sampling_bucket"),
                    "role": row.get("role"),
                },
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "by_method": dict(by_method),
        "n_seen_train": n_seen_train,
        "bound": bound,
        "time_s": time_s,
        "source_path": str(path),
        "overlap_with_our_textbook_suites": overlap,
    }
    return problems, stats


# ---------------------------------------------------------------------------------- prefix_state


def materialize_profile(
    prompt: str, body: str
) -> tuple[list[list[str]], tuple[int, ...], tuple[int, ...]]:
    """Canonicalise `body`'s `lean_seq` actions and replay them through a fresh
    `Env(prompt, canon=True, base=0, assign=True)`. Returns `(acts, counts, open_boxes)`:

    * `acts`: the `A = len(acts)` model-visible actions (`state_env.split_actions`'s decomposition
      of the canonicalised token stream).
    * `counts`: length `A + 1`; `counts[m]` is the number of reference ND lines materialised after
      replaying the first `m` actions (module docstring's case analysis). `counts[0] == 0`.
    * `open_boxes`: length `A + 1`; `open_boxes[m]` is the number of currently-open boxes
      (`len(env.frames) - 1`) after replaying the first `m` actions.

    Raises `ValueError` if the replay does not reproduce `body` exactly (`env.done`, `env.text ==
    toks`, `env.renamed == 0` -- the last of these is the plan §1 check that the canonicalised
    names are already what `assign=True`'s environment-assigned naming would produce).
    """
    ds = load_danstate()
    toks = ds.state_env.canonicalise(prompt, body)
    acts = cast("list[list[str]]", ds.state_env.split_actions(toks))
    env = ds.state_env.Env(prompt, canon=True, base=0, assign=True)
    counts = [0]
    open_boxes = [0]
    for act in acts:
        frame_kind_before = env.frames[-1].kind
        ok, why = env.apply(act)
        if not ok:
            raise ValueError(f"replay failed on canonicalised action {act!r}: {why}")
        if act[0] == "have":
            counts.append(counts[-1] + 1)
        elif act[0] == "exact":
            counts.append(counts[-1] + (0 if frame_kind_before == "top" else 1))
        else:
            raise ValueError(f"unexpected action head {act[0]!r}")
        open_boxes.append(len(env.frames) - 1)
    if not env.done or env.text != toks or env.renamed != 0:
        raise ValueError("replay did not reproduce the canonicalised reference exactly")
    return acts, tuple(counts), tuple(open_boxes)


def m_for_j(counts: Sequence[int], j: int, length_l: int) -> int:
    """Smallest `m` with `counts[m] >= j`, for `0 <= j < length_l`. By convention, `j >=
    length_l` returns `len(counts) - 1` (`A`, the full replay) -- see module docstring for why
    this is a deliberate special case, not derivable from the general rule."""
    if j >= length_l:
        return len(counts) - 1
    for m, c in enumerate(counts):
        if c >= j:
            return m
    raise AssertionError(f"counts {counts!r} never reaches j={j!r} < length_l={length_l!r}")


def prefix_state(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """Repack the already-built `prefix` suite into Dan's state-env `STATEPREFIX` format (module
    docstring): one `prefix_state` item per `prefix` item, same `(theorem, j)`, with
    `PrefixSpec.body = "STATEPREFIX m=<m> | <full reference ND body>"` (`m = m_for_j(...)`, module
    docstring) and tags `m`, `A` (total actions), `actions_left` (`A - m`), `open_boxes`, plus the
    original `prefix` tags (`j`, `L`, `src_pid`, `bucket`) so the prefix analysis view can pool by
    "lines left" across both suites.

    Every source item is already certified/filtered by the `prefix` builder itself, so `certifier`
    and `excluder` are accepted for signature uniformity but unused here.

    `cfg` keys: `store_root` (`DEFAULT_STORE_ROOT`), `source_suite` ("prefix"), `source_version`
    ("v1"), `version` ("v1").
    """
    del certifier, excluder, rng_seed
    store_root = cast(str, cfg.get("store_root", DEFAULT_STORE_ROOT))
    source_suite = str(cfg.get("source_suite", "prefix"))
    source_version = str(cfg.get("source_version", "v1"))
    version = str(cfg.get("version", "v1"))

    store = Store(Path(store_root))
    source = store.load_suite(source_suite, source_version)

    profile_cache: dict[tuple[str, str], tuple[list[list[str]], tuple[int, ...], tuple[int, ...]]]
    profile_cache = {}
    problems: list[Problem] = []
    drops: Counter[str] = Counter()

    for p in source:
        if p.ref is None:
            drops["no_ref"] += 1
            continue
        prompt = p.sequent.prompt()
        cache_key = (prompt, p.ref.body)
        profile = profile_cache.get(cache_key)
        if profile is None:
            try:
                profile = materialize_profile(prompt, p.ref.body)
            except ValueError:
                drops["replay_failed"] += 1
                continue
            profile_cache[cache_key] = profile
        acts, counts, open_boxes = profile
        length_l = p.ref.lines
        if counts[-1] != length_l:
            drops["line_count_mismatch"] += 1
            continue
        j = cast(int, p.tags.get("j"))
        a_total = len(acts)
        m = m_for_j(counts, j, length_l)
        tags: dict[str, JSON] = {
            **p.tags,
            "m": m,
            "A": a_total,
            "actions_left": a_total - m,
            "open_boxes": open_boxes[m],
        }
        problems.append(
            Problem(
                pid=f"state_{p.pid}",
                suite="prefix_state",
                version=version,
                sequent=p.sequent,
                length=p.length,
                ref=p.ref,
                tags=tags,
                prefix=PrefixSpec(j=j, body=f"STATEPREFIX m={m} | {p.ref.body}"),
                seen_train=p.seen_train,
                seen_eval=p.seen_eval,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "n_source": len(source),
        "n_distinct_references": len(profile_cache),
        "drops": dict(drops),
        "source_suite": source_suite,
        "source_version": source_version,
    }
    return problems, stats


# ------------------------------------------------------------------------------- skill_pool_sub


def _skill_group(tags: Mapping[str, JSON]) -> str:
    """The two skill tags `skill_pool` tags unconditionally (`classical`, `case_split` -- see
    `logic/skills.py`'s module docstring for why the other five are not precomputed over the full
    pool), collapsed into one 4-valued stratification label."""
    is_classical = bool(tags.get("classical"))
    is_case_split = bool(tags.get("case_split"))
    if is_classical and is_case_split:
        return "classical+case_split"
    if is_classical:
        return "classical"
    if is_case_split:
        return "case_split"
    return "plain"


def skill_pool_sub(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """A deterministic stratified subsample of `skill_pool` (plan §3: "about 1,200 stratified"),
    by `(L, skill_group)` (`skill_group` = `_skill_group`'s 4-valued classical/case_split
    collapse -- the only two skills `skill_pool` tags unconditionally; see `analysis/skills.py`'s
    module docstring for which skills the dashboard's skill view actually needs). One item per
    `skill_pool` pid is kept unchanged (same `pid`/`sequent`/`length`/`ref`/`tags`, so joins against
    `skill_pool`-keyed legacy runs via `analysis.skills`'s own machinery still work); each item's
    `tags` additionally carry `skill_group` and `orig_suite`.

    Every source item is already classified/excluded by the `skill_pool` builder itself, so
    `certifier`/`excluder` are accepted for signature uniformity but unused here.

    `cfg` keys: `store_root` (`DEFAULT_STORE_ROOT`), `source_suite` ("skill_pool"),
    `source_version` ("v1"), `n_items` (1200), `version` ("v1").
    """
    del certifier, excluder
    store_root = cast(str, cfg.get("store_root", DEFAULT_STORE_ROOT))
    source_suite = str(cfg.get("source_suite", "skill_pool"))
    source_version = str(cfg.get("source_version", "v1"))
    n_items = int(cfg.get("n_items", SKILL_POOL_SUB_N_ITEMS))
    version = str(cfg.get("version", "v1"))

    store = Store(Path(store_root))
    source = store.load_suite(source_suite, source_version)

    by_cell: dict[tuple[object, str], list[Problem]] = defaultdict(list)
    for p in source:
        by_cell[(p.tags.get("L"), _skill_group(p.tags))].append(p)

    cells = sorted(by_cell, key=lambda c: (c[0] is None, c[0], c[1]))
    base_per_cell, remainder = divmod(n_items, max(len(cells), 1))
    rng = random.Random(rng_seed)
    selected: list[Problem] = []
    stats_by_cell: dict[str, JSON] = {}
    for i, cell in enumerate(cells):
        cand = sorted(by_cell[cell], key=lambda p: p.pid)
        rng.shuffle(cand)
        target = base_per_cell + (1 if i < remainder else 0)
        take = cand[:target]
        selected.extend(take)
        stats_by_cell[f"L{cell[0]}_{cell[1]}"] = {"available": len(cand), "taken": len(take)}

    problems = [
        Problem(
            pid=p.pid,
            suite="skill_pool_sub",
            version=version,
            sequent=p.sequent,
            length=p.length,
            ref=p.ref,
            tags={**p.tags, "skill_group": _skill_group(p.tags), "orig_suite": source_suite},
            seen_train=p.seen_train,
            seen_eval=p.seen_eval,
        )
        for p in selected
    ]

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "n_target": n_items,
        "n_source": len(source),
        "n_cells": len(cells),
        "by_cell": stats_by_cell,
        "source_suite": source_suite,
        "source_version": source_version,
    }
    return problems, stats
