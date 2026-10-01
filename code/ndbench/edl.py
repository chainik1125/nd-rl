"""Excess description length (EDL): plan §B7 / WP5.

EDL measures how much of a model's proving capability was *elicited* by fine-tuning rather than
already latent in the base weights (Donoway, Joren, Roger & Leike, "Elicitation, Not Learning"
[arXiv:2601.04728]). Per that paper's Definition 3.2 (§3.2): ``EDL(D,θ0,A) = MDL(D,θ0,A) -
n*L_test(θ*)``, where the prequential minimum-description-length term (Eq. 6, §3.1) is
``MDL(D,θ0,A) = sum_{i=1..n} ℓ(θ_{i-1}, x_i, y_i)``, the online per-example loss (Eq. 1, §2.1)
``ℓ(θ,x,y) = -log p_θ(y|x)`` scored *before* the model is updated on that example. Proposition 5.1
(§5.1) is the null-control sanity check this module's acceptance test relies on: for a "model"
whose predictive loss never changes with training (the degenerate case includes, but is not
limited to, the paper's i.i.d.-random-label construction), ``E[EDL] = 0``.

Plan §B7 instantiates this for ND proofs: ``EDL = sum_{i=1..n} ℓ(θ_{i-1}; x_i, y_i) -
n*L_test(θ_n)`` (nats), where ``ℓ`` is the *shift-marginal* NLL of the proof tokens (including
QED) conditioned on the prompt: ``ℓ = -log sum_s p_θ(shift_s(y) | x)``, summed over every start
index shift the tokenizer allows (this is what ``external/nd-takehome/novelty.py``'s ``score()``
computes as ``logp_T1 = logsumexp_s(...)``, and what `adapters.base.ModelAdapter.logprob`/
`adapters.base.Learner.nll` are specified to return per plan §1's "How each adapter fits" table).
Marginalising over shifts removes the ~4 nats/example of pure start-index-guessing entropy a
single fixed encoding would otherwise charge the model for (the paper's §5.5/§6.4 recommendation
to "score only designated output tokens" when computing EDL for capability analysis); plan §B7
additionally asks for a start-token-excluded variant as a sensitivity check (see
`Learner`-protocol note 2 below).

This module implements the two WP5 API-contract functions (plan §3's WP5 row):

* `build_classes(cfg, **sources) -> (list[Problem], dict)`: assembles the 12 EDL classes from
  suites already written to the `Store` (families/laws/glue, WP2-WP4) plus two raw legacy JSONL
  files (`gen` L10, heldout cap-6), splitting each into disjoint train/test pools tagged
  `tags["class"]`/`tags["split"]`.
* `run_edl(adapter, problems, ...) -> list[dict]`: the online-learning protocol above, run once
  per (class, learning rate, data order), returning JSON-able rows for `Store.append_edl`.
* `summarise_edl(rows, ...) -> dict`: per (class, model) EDL*, the L_test(n') curve, and a
  bootstrap CI over data orders.

Interface notes / assumptions (documented per plan §A6, since WP0's interfaces are fixed and this
module programs against them rather than editing them):

1. **Shift-copy convention for `Learner.update` (fixed; previously double-copied).** `adapters.
   base.Learner.update(pairs, lr)` takes a plain `Sequence[tuple[Sequent, str]]`; there is no
   separate "how many random shifts" argument. Per plan §B7 ("one AdamW step ... on 8 random-shift
   copies of the example"), exactly one layer must make those 8 copies. `run_edl` passes each
   training pair to `update()` **once** (`update([pair], lr)`), and the `Learner` itself is
   responsible for turning that single pair into `update_shifts` independently-reshifted rows
   internally (as `adapters.absgpt._AbsGptLearner.update` and `test_wp5_fixtures.TinyGptLearner.
   update` both do). An earlier version of this module instead passed the same pair
   `shifts_per_update` times *and* relied on the `Learner` reshifting each occurrence -- with a
   `Learner` that (correctly, per plan §1's adapter table) already does its own internal
   `update_shifts`-many reshifts per pair, that produced `shifts_per_update * update_shifts` (64,
   not 8) copies per step. `run_edl` now also sets a `Learner`'s duck-typed `update_shifts`
   attribute (when present) to its own `shifts_per_update` parameter before training, so the two
   numbers can never drift apart silently.
2. **Start-token-excluded sensitivity variant.** The `Learner` Protocol has no token-level hook to
   drop the start-index token's contribution to `nll`. Rather than widen WP0's Protocol (out of
   scope for this WP), `run_edl` looks for an *optional*, duck-typed extension method
   `nll_start_excluded(pairs) -> list[LogProb]` (same shape as `nll`, but excluding the first proof
   token from every shift's summed log-probability before marginalising) via `getattr`, and records
   `None` for the corresponding row fields when a `Learner` does not provide it. `TinyGptLearner`
   (this WP's own CPU test double) implements it, so the sensitivity computation is exercised in
   tests even though no committed production `Learner` currently offers it; a future `AbsGptAdapter`
   Learner (WP1) should add the same method name if the sensitivity check is to run for real.
3. **`build_classes`'s suite/tag assumptions.** Suite-backed classes (families, laws, glue) read
   `Store.load_suite(suite, version)` and filter by the tag names plan §3 documents for WP3/WP4
   (`family`/`n`, `schema`/`s`, `op`/`k`/`part_len`/`cell`); WP4's glue suite has not landed at the
   time this module was written, so the "AND glue L=12" class additionally falls back to matching
   `Problem.length.L == 12` directly (rather than a specific, unconfirmed tag key) if no `"L"`/
   `"total_L"` tag is present. `Store.load_suite` raising `FileNotFoundError` for a suite that has
   not been built yet is treated as "skip with a warning" per plan §B7's "class list is
   config-driven" clause, not a hard failure.
4. **CLI passes `--ckpts` unparsed.** `cli.py`'s `cmd_edl` forwards `args.ckpts` (an
   argparse string, or `None`) straight through as `run_edl(..., ckpts=args.ckpts)` -- it does not
   split/int-cast it the way it does `--lrs`. `run_edl` therefore accepts `ckpts: Sequence[int] |
   str | None` and parses a comma-separated string itself (`_parse_ckpts`).
5. **2026-10-01 EDL-pool fix: `dev_gen_L10` renamed `gen_L10`, now reads `reserve.jsonl`.** The
   first EDL run built only 2 of 12 classes: the families/laws suites' default ~40 items/cell
   fell short of the >=64-distinct threshold `_split_train_test` requires, and `dev_gen_L10`'s
   source file (`ab2/data/dev_L8-11.jsonl`) has no `minlen_proof` field at all, so every row was
   dropped. The fix has two parts. (a) `build_classes` now reads the families/laws suite-backed
   classes at a dedicated, larger suite version (plan default `"edl_v1"`, built by
   `code/experiments/current/benchmarking_20260930/build_edl_pools.py` with `target_per_n`/
   `target_per_s=128`, same certifier/excluder as the main `"v1"` suites) via the same
   `families_version`/`laws_version` config keys as before -- no new per-class config key was
   needed since every family/law EDL class moves to the same new version together. (b) the
   `dev_gen_L10` class is replaced by `gen_L10`, reading `source=='gen'`, `L_true==10` rows (with
   a `minlen_proof`, never falling back to the non-minimal `gen_proof` field) directly from
   `ab2/data/reserve.jsonl` (`DEFAULT_GEN_L10_PATH`) instead of `dev_L8-11.jsonl`; any row whose
   sequent the default `Excluder` marks `seen_train` is dropped (not merely flagged, unlike the
   families/laws builders' own flag-only policy) before the generic dedup/split logic runs, per
   the main agent's 2026-10-01 instruction. `and_glue_L12` and `heldout_cap6` are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np

from .adapters.base import Capability, Learner, ModelAdapter, NotSupported
from .core.io import read_jsonl
from .core.proof import make_reference, renumber
from .core.types import (
    JSON,
    Excluder,
    LengthLabel,
    LogProb,
    Problem,
    ReferenceProof,
    SampleRequest,
    Sequent,
)
from .metrics import edl as edl_of
from .metrics import pass_at_k
from .store import Store

#: `run_edl`'s optional, duck-typed sensitivity-variant hook (module docstring note 2): a
#: `Learner` may implement this alongside `nll` to support the start-token-excluded EDL variant.
StartExcludedFn = Callable[[Sequence[tuple[Sequent, str]]], list[LogProb]]


def _j(x: object) -> JSON:
    """Assert (without a runtime check -- `JSON` is a documentation-level shape, not an enforced
    one) that `x` is JSON-able. Used when embedding an already-concretely-typed collection (e.g.
    `list[float]`) into a `dict[str, JSON]`-shaped row: `dict`/`list` are invariant in `x`'s
    element type, so pyright cannot see `list[float]` as a `list[JSON]` on its own even though
    every element trivially is one."""
    return cast(JSON, x)


#: The 12 EDL classes of plan §B7, in the order the plan lists them.
EDL_CLASSES: tuple[str, ...] = (
    "and_tower_n8",
    "and_tower_n16",
    "imp_nest_n4",
    "imp_nest_n8",
    "reorder_and_n5",
    "reorder_or_n3",
    "contraposition_s3",
    "dist_and_over_or_s1",
    "demorgan_nand_to_or_s1",
    "and_glue_L12",
    "gen_L10",
    "heldout_cap6",
)

DEFAULT_LRS: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
DEFAULT_ORDERS = 2
DEFAULT_N = 64
DEFAULT_M = 64
DEFAULT_CKPTS: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_SHIFTS_PER_UPDATE = 8

#: Plan §0/§B7's raw legacy sources for the two file-backed classes (read-only; never written to).
#: `gen_L10` reads `reserve.jsonl` rather than `dev_L8-11.jsonl` (2026-10-01 fix, module docstring
#: note 5): `dev_L8-11.jsonl` has no `minlen_proof` field at all, so `gen_L10` could never find a
#: proof body there; `reserve.jsonl` has one for the great majority of its `source=='gen'` rows.
DEFAULT_GEN_L10_PATH = "/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl"
DEFAULT_HELDOUT_CAP6_PATH = "/scratch/cowifornia/nd-rl/ab3/frozen/data/heldout_cap6.jsonl"


class _ClassUnavailable(Exception):
    """Raised by a class loader when its source suite/file is missing or has no qualifying items.

    Caught by `build_classes`, which records the message as a warning and moves on to the next
    class (plan §B7: "The class list is config-driven, so missing suites just skip with a
    warning.").
    """


# --------------------------------------------------------------------------------- build_classes


def _problem_from_raw_row(
    row: Mapping[str, JSON], proof_body: str, *, suite: str, method: str
) -> Problem | None:
    """One `Problem` from a raw legacy JSONL row (dev-gen / heldout-cap6 sources), or `None` if the
    row's `prompt` does not round-trip through `Sequent.from_prompt` (defensive; every row seen in
    practice does)."""
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        return None
    try:
        seq = Sequent.from_prompt(prompt)
    except Exception:
        return None
    body = renumber(proof_body, new_start=1)
    ref_source = "minlen" if method == "minlen_exact" else "witness"
    ref: ReferenceProof = make_reference(body, source=ref_source)
    raw_l = row.get("L_true", row.get("n_lines"))
    length_l = int(raw_l) if isinstance(raw_l, int | float) else ref.lines
    length = LengthLabel(L=length_l, L_ub=None, L_lb=None, method=method, bound=None)
    name = row.get("name")
    pid = f"{suite}:{name if isinstance(name, str) else seq.key()}"
    return Problem(
        pid=pid,
        suite=suite,
        version="raw",
        sequent=seq,
        length=length,
        ref=ref,
        tags={"source_name": name},
    )


def _load_suite_class(
    store: Store, suite: str, version: str, match: Callable[[Problem], bool], label: str
) -> list[Problem]:
    try:
        items = store.load_suite(suite, version)
    except FileNotFoundError as e:
        raise _ClassUnavailable(f"suite {suite!r}/{version!r} not built yet ({label}): {e}") from e
    if not items:
        raise _ClassUnavailable(f"suite {suite!r}/{version!r} is empty ({label})")
    return [p for p in items if match(p)]


def _family_match(family: str, n_value: int) -> Callable[[Problem], bool]:
    def match(p: Problem) -> bool:
        return p.tags.get("family") == family and p.tags.get("n") == n_value

    return match


def _law_match(schema: str, s_value: int) -> Callable[[Problem], bool]:
    def match(p: Problem) -> bool:
        return p.tags.get("schema") == schema and p.tags.get("s") == s_value

    return match


def _and_glue_match(target_l: int) -> Callable[[Problem], bool]:
    def match(p: Problem) -> bool:
        if p.tags.get("op") != "AND":
            return False
        tagged_l = p.tags.get("L", p.tags.get("total_L"))
        if isinstance(tagged_l, int | float):
            return int(tagged_l) == target_l
        return target_l == p.length.L

    return match


def _default_excluder() -> Excluder:
    """Lazily imports `logic.exclusion.DefaultExcluder` (module docstring note 5): avoids paying
    the cost of scanning the real train/eval tier files -- or even importing this module -- for
    every `build_classes` call that never reaches `_load_gen_l10` with a real path (e.g. every
    test that points `gen_path` at a nonexistent file)."""
    from .logic.exclusion import DefaultExcluder

    return DefaultExcluder()


def _load_gen_l10(path: Path, excluder: Excluder | None) -> list[Problem]:
    """`gen_L10` (module docstring note 5): `source=='gen'`, `L_true==10` rows from `reserve.jsonl`
    that have a `minlen_proof` (never falling back to the non-minimal `gen_proof` field -- a row
    without `minlen_proof` is simply dropped, since `reserve.jsonl` has thousands of qualifying
    rows that do have one) and whose sequent the (default, unless `excluder` overrides it for a
    test) `Excluder` does *not* mark `seen_train` (main agent instruction, 2026-10-01: dropped
    here, not merely flagged the way the families/laws suite builders flag-only their own
    `seen_train` items)."""
    if not path.is_file():
        raise _ClassUnavailable(f"{path} not found")
    exc = excluder if excluder is not None else _default_excluder()
    out: list[Problem] = []
    for row in read_jsonl(path):
        if row.get("source") != "gen":
            continue
        raw_l = row.get("L_true", row.get("n_lines"))
        if not (isinstance(raw_l, int | float) and int(raw_l) == 10):
            continue
        minlen_proof = row.get("minlen_proof")
        if not isinstance(minlen_proof, str) or not minlen_proof.strip():
            continue
        p = _problem_from_raw_row(row, minlen_proof, suite="edl_gen_l10", method="minlen_exact")
        if p is None:
            continue
        in_train, eval_sets = exc.status(p.sequent)
        if in_train:
            continue
        out.append(replace(p, seen_eval=tuple(eval_sets)))
    if not out:
        raise _ClassUnavailable(
            f"no source=='gen' rows with L_true==10, a minlen_proof, and not seen_train in {path}"
        )
    return out


def _load_heldout_cap6(path: Path) -> list[Problem]:
    if not path.is_file():
        raise _ClassUnavailable(f"{path} not found")
    out: list[Problem] = []
    for row in read_jsonl(path):
        proof_body = row.get("proof")
        if not isinstance(proof_body, str) or not proof_body.strip():
            continue
        p = _problem_from_raw_row(row, proof_body, suite="edl_heldout_cap6", method="reference")
        if p is not None:
            out.append(p)
    if not out:
        raise _ClassUnavailable(f"no rows with a 'proof' field in {path}")
    return out


def _dedup_by_key(pool: Sequence[Problem]) -> list[Problem]:
    """First occurrence per distinct `Sequent.key()` (atom-renaming class), preserving order."""
    seen: dict[str, Problem] = {}
    for p in pool:
        k = p.sequent.key()
        if k not in seen:
            seen[k] = p
    return list(seen.values())


def _split_train_test(
    pool: Sequence[Problem], class_name: str, rng_seed: int, warnings: list[str]
) -> tuple[list[Problem], list[Problem]] | None:
    """n=64/m=64 (32/32 below 128 distinct items), deterministic given `rng_seed` (plan §B7)."""
    total = len(pool)
    if total < 64:
        warnings.append(
            f"{class_name}: only {total} certified, reference-bearing, distinct items (need >= 64)"
            " -- class skipped"
        )
        return None
    per_side = DEFAULT_N if total >= 128 else 32
    order = list(range(total))
    random.Random(f"edl-split:{rng_seed}:{class_name}").shuffle(order)
    shuffled = [pool[i] for i in order]
    return shuffled[:per_side], shuffled[per_side : 2 * per_side]


def _tag_split(problems: Sequence[Problem], class_name: str, split: str) -> list[Problem]:
    out = []
    for p in problems:
        tags = dict(p.tags)
        tags["class"] = class_name
        tags["split"] = split
        out.append(replace(p, tags=tags))
    return out


def _class_specs(
    families_version: str,
    laws_version: str,
    glue_version: str,
    gen_path: Path,
    heldout_cap6_path: Path,
    excluder: Excluder | None,
) -> list[tuple[str, Callable[[Store], list[Problem]]]]:
    def suite_loader(
        suite: str, version: str, match: Callable[[Problem], bool], label: str
    ) -> Callable[[Store], list[Problem]]:
        def loader(store: Store) -> list[Problem]:
            return _load_suite_class(store, suite, version, match, label)

        return loader

    return [
        (
            "and_tower_n8",
            suite_loader(
                "families", families_version, _family_match("and_tower", 8), "and_tower n=8"
            ),
        ),
        (
            "and_tower_n16",
            suite_loader(
                "families", families_version, _family_match("and_tower", 16), "and_tower n=16"
            ),
        ),
        (
            "imp_nest_n4",
            suite_loader(
                "families", families_version, _family_match("imp_nest", 4), "imp_nest n=4"
            ),
        ),
        (
            "imp_nest_n8",
            suite_loader(
                "families", families_version, _family_match("imp_nest", 8), "imp_nest n=8"
            ),
        ),
        (
            "reorder_and_n5",
            suite_loader(
                "families", families_version, _family_match("reorder_and", 5), "reorder_and n=5"
            ),
        ),
        (
            "reorder_or_n3",
            suite_loader(
                "families", families_version, _family_match("reorder_or", 3), "reorder_or n=3"
            ),
        ),
        (
            "contraposition_s3",
            suite_loader(
                "laws", laws_version, _law_match("contraposition", 3), "contraposition s=3"
            ),
        ),
        (
            "dist_and_over_or_s1",
            suite_loader(
                "laws", laws_version, _law_match("dist_and_over_or", 1), "dist_and_over_or s=1"
            ),
        ),
        (
            "demorgan_nand_to_or_s1",
            suite_loader(
                "laws",
                laws_version,
                _law_match("demorgan_nand_to_or", 1),
                "demorgan_nand_to_or s=1",
            ),
        ),
        ("and_glue_L12", suite_loader("glue", glue_version, _and_glue_match(12), "AND glue L=12")),
        ("gen_L10", lambda _store: _load_gen_l10(gen_path, excluder)),
        ("heldout_cap6", lambda _store: _load_heldout_cap6(heldout_cap6_path)),
    ]


def build_classes(
    cfg: Mapping[str, JSON],
    *,
    store: Store | None = None,
    gen_path: str | Path | None = None,
    heldout_cap6_path: str | Path | None = None,
    excluder: Excluder | None = None,
    rng_seed: int = 0,
) -> tuple[list[Problem], dict[str, JSON]]:
    """Assemble the 12 EDL classes of plan §B7 (see the module docstring's note 3 for the tag/suite
    assumptions, and note 5 for the 2026-10-01 `gen_L10` fix), split each into train/test, and
    return the combined `Problem` list plus build stats. Callable as `build_classes(cfg)` alone (as
    `cli.cmd_edl` does); `store`/`gen_path`/`heldout_cap6_path`/`excluder` are test/override hooks,
    falling back to `cfg` keys and then plan defaults (`excluder` falls back to `_default_excluder`
    lazily, only if `gen_L10`'s loader actually runs against a real file -- see that function).
    """
    if store is None:
        store = Store(Path(str(cfg.get("out", "out"))))
    gen_path_resolved = Path(gen_path or cast(str, cfg.get("gen_path", DEFAULT_GEN_L10_PATH)))
    heldout_path = Path(
        heldout_cap6_path or cast(str, cfg.get("heldout_cap6_path", DEFAULT_HELDOUT_CAP6_PATH))
    )
    families_version = str(cfg.get("families_version", "v1"))
    laws_version = str(cfg.get("laws_version", "v1"))
    glue_version = str(cfg.get("glue_version", "v1"))

    warnings: list[str] = []
    problems: list[Problem] = []
    class_counts: dict[str, JSON] = {}

    specs = _class_specs(
        families_version, laws_version, glue_version, gen_path_resolved, heldout_path, excluder
    )
    for name, loader in specs:
        try:
            pool = loader(store)
        except _ClassUnavailable as e:
            warnings.append(f"{name}: {e}")
            continue
        pool = _dedup_by_key([p for p in pool if p.ref is not None])
        split = _split_train_test(pool, name, rng_seed, warnings)
        if split is None:
            continue
        train, test = split
        problems.extend(_tag_split(train, name, "train"))
        problems.extend(_tag_split(test, name, "test"))
        class_counts[name] = {"n_train": len(train), "n_test": len(test)}

    stats: dict[str, JSON] = {
        "n_classes_built": len(class_counts),
        "n_classes_total": len(EDL_CLASSES),
        "class_counts": class_counts,
        "warnings": _j(warnings),
    }
    return problems, stats


# -------------------------------------------------------------------------------------- run_edl


def _parse_ckpts(ckpts: Sequence[int] | str | None) -> tuple[int, ...]:
    if ckpts is None:
        return DEFAULT_CKPTS
    if isinstance(ckpts, str):
        return tuple(int(x) for x in ckpts.split(",") if x.strip())
    return tuple(int(c) for c in ckpts)


def _combo_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _must_ref(p: Problem) -> ReferenceProof:
    if p.ref is None:
        raise ValueError(f"problem {p.pid!r} has no reference proof; run_edl needs one per item")
    return p.ref


def _train_test_from_tags(
    class_problems: Sequence[Problem], n: int, m: int
) -> tuple[list[Problem], list[Problem]]:
    train = [p for p in class_problems if p.tags.get("split") == "train"]
    test = [p for p in class_problems if p.tags.get("split") == "test"]
    if train or test:
        return train, test
    return list(class_problems[:n]), list(class_problems[n : n + m])


def _mean_nll(learner: Learner, pairs: Sequence[tuple[Sequent, str]]) -> float:
    lps = learner.nll(pairs)
    return sum(lp.nats for lp in lps) / len(lps) if lps else float("nan")


def _mean_nll_via(fn: StartExcludedFn, pairs: Sequence[tuple[Sequent, str]]) -> float:
    lps = fn(pairs)
    return sum(lp.nats for lp in lps) / len(lps) if lps else float("nan")


def _pass_at_8(
    sampling_adapter: ModelAdapter, test_problems: Sequence[Problem], seed: int
) -> float | None:
    """pass@8 of `sampling_adapter` on `test_problems`'s theorems (plan §B7's optional pass@8
    before/after), `None` if the adapter can't sample or judging is unavailable."""
    if Capability.SAMPLE not in sampling_adapter.capabilities or not test_problems:
        return None
    try:
        from .judge.nd import NDJudge

        judge = NDJudge()
        reqs = []
        for p in test_problems:
            max_new = max(1, 512 - sampling_adapter.token_count(p.sequent))
            reqs.append(
                SampleRequest(
                    pid=p.pid,
                    sequent=p.sequent,
                    n=8,
                    temperature=0.8,
                    max_new_tokens=max_new,
                    seed=seed,
                )
            )
        attempts_by_req = sampling_adapter.sample(reqs)
        vals: list[float] = []
        for p, attempts in zip(test_problems, attempts_by_req, strict=True):
            ref_lines = p.ref.lines if p.ref is not None else None
            verdicts = judge.judge(p.sequent, attempts, ref_lines)
            c = sum(1 for v in verdicts if v.ok)
            vals.append(pass_at_k(8, c, 8))
        return float(sum(vals) / len(vals)) if vals else None
    except NotSupported:
        return None


def run_edl(
    adapter: ModelAdapter,
    problems: Sequence[Problem],
    *,
    lrs: Sequence[float] = DEFAULT_LRS,
    orders: int = DEFAULT_ORDERS,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    ckpts: Sequence[int] | str | None = DEFAULT_CKPTS,
    shifts_per_update: int = DEFAULT_SHIFTS_PER_UPDATE,
    seed: int = 0,
    pass8: bool = False,
) -> list[dict[str, JSON]]:
    """The plan §B7 online-EDL protocol, once per (class, `lr`, data order).

    `problems` is grouped by `tags["class"]` (falling back to `p.suite` if absent, so a caller can
    pass a single hand-built class without going through `build_classes`); within a class, items
    tagged `tags["split"] in {"train", "test"}` are used as given (as `build_classes` produces),
    else the first `n`/next `m` items (in list order) are used. See the module docstring for the
    `shifts_per_update` and start-token-excluded-variant conventions this function relies on.
    """
    if Capability.FINETUNE not in adapter.capabilities:
        raise NotSupported(f"{type(adapter).__name__} does not advertise Capability.FINETUNE")
    ckpts_parsed = _parse_ckpts(ckpts)
    identity = adapter.identity()

    by_class: dict[str, list[Problem]] = {}
    for p in problems:
        cls = str(p.tags.get("class", p.suite))
        by_class.setdefault(cls, []).append(p)

    rows: list[dict[str, JSON]] = []
    for class_name, class_problems in by_class.items():
        train_all, test_all = _train_test_from_tags(class_problems, n, m)
        if not train_all or not test_all:
            continue
        test_pairs = [(p.sequent, _must_ref(p).body) for p in test_all]
        n_actual = len(train_all)
        m_actual = len(test_all)
        active_ckpts = sorted({c for c in ckpts_parsed if 0 < c <= n_actual} | {n_actual})

        for order in range(orders):
            perm = list(range(n_actual))
            random.Random(_combo_seed(seed, class_name, "order", order)).shuffle(perm)
            train_pairs = [(train_all[i].sequent, _must_ref(train_all[i]).body) for i in perm]

            for lr in lrs:
                combo_seed = _combo_seed(seed, class_name, lr, order)
                learner = adapter.learner(seed=combo_seed)
                # Exactly one layer makes the `shifts_per_update` random-shift copies (module
                # docstring note 1): if this `Learner` does its own internal reshifting, make its
                # count match ours rather than also repeating the pair ourselves below.
                # `setattr` (not `learner.update_shifts = ...`) since `update_shifts` is a
                # duck-typed extension, not part of the `Learner` Protocol pyright type-checks
                # against here.
                if hasattr(learner, "update_shifts"):
                    setattr(learner, "update_shifts", shifts_per_update)  # noqa: B010
                start_excluded_fn: StartExcludedFn | None = getattr(
                    learner, "nll_start_excluded", None
                )

                l_test_0 = _mean_nll(learner, test_pairs)
                l_test_0_se = (
                    _mean_nll_via(start_excluded_fn, test_pairs) if start_excluded_fn else None
                )

                online_nats: list[float] = []
                online_nats_se: list[float] | None = [] if start_excluded_fn else None
                l_test_by_ckpt: dict[int, float] = {}

                for i, pair in enumerate(train_pairs, start=1):
                    lp = learner.nll([pair])[0]
                    online_nats.append(lp.nats)
                    if start_excluded_fn is not None and online_nats_se is not None:
                        online_nats_se.append(start_excluded_fn([pair])[0].nats)
                    learner.update([pair], lr)
                    if i in active_ckpts:
                        l_test_by_ckpt[i] = _mean_nll(learner, test_pairs)

                if n_actual not in l_test_by_ckpt:
                    l_test_by_ckpt[n_actual] = _mean_nll(learner, test_pairs)
                l_test_n = l_test_by_ckpt[n_actual]
                l_test_n_se = (
                    _mean_nll_via(start_excluded_fn, test_pairs) if start_excluded_fn else None
                )

                edl_by_ckpt = {
                    cp: edl_of(online_nats[:cp], l_test_by_ckpt[cp])
                    for cp in sorted(l_test_by_ckpt)
                }
                edl_full = edl_by_ckpt[n_actual]
                edl_se = (
                    edl_of(online_nats_se, l_test_n_se)
                    if online_nats_se is not None and l_test_n_se is not None
                    else None
                )

                pass8_before: float | None = None
                pass8_after: float | None = None
                if pass8:
                    pass8_before = _pass_at_8(adapter, test_all, combo_seed)
                    pass8_after = _pass_at_8(learner.adapter(), test_all, combo_seed + 1)

                rows.append(
                    {
                        "class": class_name,
                        "model_identity": identity,
                        "lr": lr,
                        "order": order,
                        "n": n_actual,
                        "m": m_actual,
                        "seed": combo_seed,
                        "online_nats": _j(online_nats),
                        "L_test_0": l_test_0,
                        "L_test_n": l_test_n,
                        "L_test_by_checkpoint": _j(
                            {str(k): v for k, v in sorted(l_test_by_ckpt.items())}
                        ),
                        "EDL_by_checkpoint": _j(
                            {str(k): v for k, v in sorted(edl_by_ckpt.items())}
                        ),
                        "EDL": edl_full,
                        "EDL_per_example": edl_full / n_actual if n_actual else float("nan"),
                        "start_excluded_online_nats": _j(online_nats_se),
                        "L_test_0_start_excluded": l_test_0_se,
                        "L_test_n_start_excluded": l_test_n_se,
                        "EDL_start_excluded": edl_se,
                        "pass8_before": pass8_before,
                        "pass8_after": pass8_after,
                    }
                )
    return rows


# ---------------------------------------------------------------------------------- summarise_edl


def summarise_edl(
    rows: Sequence[Mapping[str, JSON]], *, B: int = 2000, seed: int = 0
) -> dict[str, JSON]:
    """Per (class, model identity): `EDL*` -- the full-`n` `EDL` at the learning rate with the
    smallest online code length (sum of `online_nats`, the MDL convention), plus `ln(k)` for `k`
    distinct learning rates swept (plan §B7's "+ ln 3" hyperparameter cost) -- its bootstrap CI
    over data orders (`nan` if fewer than 2 orders are present), and the mean `L_test` curve.

    The learning rate is chosen by code length, not by the smallest EDL: on classes the model
    already knows, a learning rate that damages the model raises the final test loss and so
    *lowers* EDL, which made min-over-lr EDL spuriously negative (the null control heldout_cap6
    gave about -5 nats/example that way, and about 0 with code-length selection).
    """
    groups: dict[tuple[str, str], list[Mapping[str, JSON]]] = {}
    for r in rows:
        ident_key = json.dumps(r["model_identity"], sort_keys=True)
        groups.setdefault((str(r["class"]), ident_key), []).append(r)

    rng = np.random.default_rng(seed)
    out: dict[str, JSON] = {}
    for (cls, ident_key), grp in groups.items():
        by_order: dict[int, list[Mapping[str, JSON]]] = {}
        for r in grp:
            by_order.setdefault(cast(int, r["order"]), []).append(r)
        n_lrs = len({cast(float, r["lr"]) for r in grp})
        correction = math.log(n_lrs) if n_lrs > 0 else 0.0

        order_stars: list[float] = []
        for _order, order_rows in sorted(by_order.items()):
            if order_rows:
                chosen = min(
                    order_rows,
                    key=lambda r: sum(float(x) for x in cast(list[float], r["online_nats"])),
                )
                order_stars.append(float(cast(float, chosen["EDL"])) + correction)
        star_mean = float(np.mean(order_stars)) if order_stars else float("nan")

        if len(order_stars) >= 2:
            arr = np.asarray(order_stars)
            idx = rng.integers(0, len(arr), size=(B, len(arr)))
            boot = arr[idx].mean(axis=1)
            ci: list[float] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        else:
            ci = [float("nan"), float("nan")]

        curve_sum: dict[str, float] = {}
        curve_count: dict[str, int] = {}
        for r in grp:
            for cp_str, val in cast(Mapping[str, float], r["L_test_by_checkpoint"]).items():
                curve_sum[cp_str] = curve_sum.get(cp_str, 0.0) + float(val)
                curve_count[cp_str] = curve_count.get(cp_str, 0) + 1
        curve_mean = {k: v / curve_count[k] for k, v in curve_sum.items()}

        out[f"{cls}||{ident_key}"] = _j(
            {
                "class": cls,
                "model_identity": json.loads(ident_key),
                "n_orders": len(order_stars),
                "EDL_star": star_mean,
                "EDL_star_ci": ci,
                "L_test_curve": curve_mean,
            }
        )
    return out
