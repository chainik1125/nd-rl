"""EDL for Dan Pandori's state-env models (`DanStateAdapter.learner`, `edl.build_classes`' Dan
options, `run_edl`'s progress/finite-check/resume hooks, `cli edl --classes/--resume`), on CPU
with tiny random Dan-format checkpoints (same fixture recipe as `test_interp_kl.py`/
`test_dan_interp.py`) and temp `Store`s. Nothing here reads a real checkpoint or the shared store.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Generator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

from code.ndbench.adapters.base import (
    Capability,
    Learner,
    ModelAdapter,
    NotSupported,
    SteeringTerm,
)
from code.ndbench.adapters.danstate import DanStateAdapter, _DanStateLearner, dan_grammar_error
from code.ndbench.cli import main
from code.ndbench.core.io import git_commit_sha, read_jsonl, sha256_str, utc_now_iso
from code.ndbench.core.types import (
    JSON,
    Attempt,
    LengthLabel,
    LogProb,
    Problem,
    ReferenceProof,
    SampleRequest,
    Sequent,
    SuiteManifest,
)
from code.ndbench.edl import build_classes, run_edl, summarise_edl
from code.ndbench.metrics import edl as edl_of
from code.ndbench.store import Store
from code.ndbench.vendor import load_danstate

# ----------------------------------------------------------------------------------- fixtures

_SIMPLE = ("THM ( P & Q ) SEQ P PRF", "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; QED")
_IMP = (
    "THM ( P & Q ) SEQ ( P > Q ) PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 | Q : ANDE2 N1 ; N4 ( P > Q ) : IMPI N2 N3 ; QED",
)
_OR_ELIM = (
    "THM ( P v Q ) SEQ ( Q v P ) PRF",
    "N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; "
    "N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED",
)
#: A box no IMPI/NEGI/ORE cites: outside the grammar (`logprob` gives NaN by default).
_UNCITED_BOX = (
    "THM ( P & Q ) SEQ P PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 P : ANDE1 N1 ; QED",
)


def _pair(prompt_body: tuple[str, str]) -> tuple[Sequent, str]:
    return Sequent.from_prompt(prompt_body[0]), prompt_body[1]


_GOOD = [_pair(_SIMPLE), _pair(_IMP), _pair(_OR_ELIM)]


def _save_tiny_dan_ckpt(path: Path, *, seed: int) -> None:
    ds = load_danstate()
    torch.manual_seed(seed)
    cfg = dict(vocab=111, n_layer=2, d=32, n_head=4, max_len=64, d_ff=64, arch="best")
    model = ds.best_model.ALiBiGPT(**cfg)
    torch.save(
        {"cfg": model.cfg, "state": model.state_dict(), "tok_mode": "lean_staten", "extra": {}},
        path,
    )


def _adapter(path: Path, *, name_base_max: int = 4) -> DanStateAdapter:
    return DanStateAdapter(
        ckpt=path, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=name_base_max
    )


@pytest.fixture
def tiny_ckpt(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_dan.pt"
    _save_tiny_dan_ckpt(path, seed=1)
    return path


@pytest.fixture
def dan(tiny_ckpt: Path) -> Generator[DanStateAdapter, None, None]:
    adapter = _adapter(tiny_ckpt)
    try:
        yield adapter
    finally:
        adapter.close()


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _batch_sizes(learner: _DanStateLearner) -> tuple[list[torch.Tensor], Any]:
    """Record every input tensor the learner's model sees (a forward pre-hook)."""
    seen: list[torch.Tensor] = []

    def hook(_module: Any, inputs: Any) -> None:
        seen.append(inputs[0].detach().clone())

    return seen, learner._model.register_forward_pre_hook(hook)


# ------------------------------------------------------------------------------- the learner


def test_dan_adapter_advertises_finetune(dan: DanStateAdapter) -> None:
    assert Capability.FINETUNE in dan.capabilities
    assert isinstance(dan.learner(seed=0), _DanStateLearner)


def test_nll_equals_minus_logprob_before_training_including_nan(dan: DanStateAdapter) -> None:
    pairs = [*_GOOD, _pair(_UNCITED_BOX)]
    lps = dan.logprob(pairs)
    nlls = dan.learner(seed=0).nll(pairs)
    for lp, nl in zip(lps[:3], nlls[:3], strict=True):
        assert nl.nats == -lp.nats  # exactly: same rows, same chunks, same arithmetic
        assert nl.nats > 0
        assert nl.n_tokens == lp.n_tokens
        assert nl.per_encoding == lp.per_encoding  # stays log-probabilities (LogProb docstring)
    assert math.isnan(lps[3].nats) and math.isnan(nlls[3].nats)
    learner = dan.learner(seed=0)
    learner.nll(pairs)
    assert set(learner.last_nll_errors) == {3}
    assert "uncited box" in learner.last_nll_errors[3]


def test_grammar_helper_matches_logprob(dan: DanStateAdapter) -> None:
    for seq, body in _GOOD:
        assert dan_grammar_error(seq, body) is None
    seq, body = _pair(_UNCITED_BOX)
    reason = dan_grammar_error(seq, body)
    assert reason is not None and "uncited box" in reason
    dan.logprob([(seq, body)])
    assert dan.last_logprob_errors[0] == reason


def test_nll_start_excluded_is_the_defining_names_excluded_logprob(dan: DanStateAdapter) -> None:
    learner = dan.learner(seed=0)
    excluded_lp = dan.logprob(_GOOD, exclude_defining_names=True)
    fresh = learner.nll_start_excluded(_GOOD)  # no preceding nll: computed directly
    full = learner.nll(_GOOD)
    cached = learner.nll_start_excluded(_GOOD)  # same pairs, same weights: reuses nll's forward
    for f, c, ex, nl in zip(fresh, cached, excluded_lp, full, strict=True):
        assert f.nats == c.nats == -ex.nats
        # dropping the (positive-NLL) defining-name tokens can only lower the NLL
        assert f.nats < nl.nats
    # a different pair list, or new weights, never gets the cached value
    assert learner.nll_start_excluded(_GOOD[:1])[0].nats == fresh[0].nats
    learner.update(_GOOD[:1], 1e-2)
    after = learner.nll_start_excluded(_GOOD)
    assert after[0].nats != fresh[0].nats


def test_update_lowers_the_loss_on_the_trained_example(dan: DanStateAdapter) -> None:
    learner = dan.learner(seed=0)
    pair = _pair(_OR_ELIM)
    before = learner.nll([pair])[0].nats
    losses = [learner.update([pair], 3e-3) for _ in range(10)]
    after = learner.nll([pair])[0].nats
    assert after < before - 1.0
    assert losses[-1] < losses[0]
    # the original adapter's weights never move
    assert dan.logprob([pair])[0].nats == -before


def test_lr_zero_leaves_weights_unchanged_and_returns_the_token_mean_nll(tiny_ckpt: Path) -> None:
    adapter = _adapter(tiny_ckpt, name_base_max=0)  # one base: the update loss is exactly logprob's
    learner = adapter.learner(seed=0)
    w0 = _state(learner._model)
    for seq, body in _GOOD:
        lp = adapter.logprob([(seq, body)])[0]
        loss = learner.update([(seq, body)], 0.0)
        assert loss == pytest.approx(-lp.nats / lp.n_tokens, rel=1e-5)
    assert _same_weights(w0, _state(learner._model))
    assert [x.nats for x in learner.nll(_GOOD)] == [-x.nats for x in adapter.logprob(_GOOD)]


def test_fresh_learners_start_from_the_original_weights_and_are_independent(
    dan: DanStateAdapter,
) -> None:
    original = _state(dan._model)
    first = dan.learner(seed=0)
    for _ in range(3):
        first.update(_GOOD, 1e-2)
    second = dan.learner(seed=0)
    assert _same_weights(original, _state(dan._model))
    assert _same_weights(original, _state(second._model))
    assert not _same_weights(original, _state(first._model))
    assert [x.nats for x in second.nll(_GOOD)] == [-x.nats for x in dan.logprob(_GOOD)]
    # no shared parameter storage, tokenizer or optimiser between learners and the adapter
    ptrs = {p.data_ptr() for p in dan._model.parameters()}
    assert not ptrs & {p.data_ptr() for p in first._model.parameters()}
    assert not ptrs & {p.data_ptr() for p in second._model.parameters()}
    assert first._tok is not dan._tok and first._opt is not second._opt


def test_update_shifts_copies_share_one_valid_base_per_proof_copy(dan: DanStateAdapter) -> None:
    learner = dan.learner(seed=0)
    learner.update_shifts = 5
    seq, body = _pair(_OR_ELIM)
    rows0 = learner._train_rows(seq, body)
    n_steps = int(rows0.x0.shape[0])
    seen, handle = _batch_sizes(learner)
    try:
        learner.update([(seq, body)], 1e-3)
    finally:
        handle.remove()
    assert len(seen) == 1
    x = seen[0]
    assert x.shape[0] == 5 * n_steps
    ref0 = int(learner._tok.ref0)
    x0 = rows0.x0[:, : x.shape[1]]
    bases = []
    for copy_idx in range(5):
        block = x[copy_idx * n_steps : (copy_idx + 1) * n_steps]
        matches = [
            b
            for b in rows0.valid_bases
            if torch.equal(block, torch.where(x0 >= ref0, x0 + b, x0)[:, : block.shape[1]])
        ]
        assert len(matches) == 1  # every step of this copy shifted by the same valid base
        bases.append(matches[0])
    assert set(bases) <= set(range(0, dan._name_base_max + 1))


def test_bases_are_seeded(dan: DanStateAdapter) -> None:
    def drawn(seed: int) -> list[torch.Tensor]:
        learner = dan.learner(seed=seed)
        learner.update_shifts = 8
        seen, handle = _batch_sizes(learner)
        try:
            learner.update(_GOOD, 0.0)
        finally:
            handle.remove()
        return seen

    assert torch.equal(drawn(3)[0], drawn(3)[0])
    assert not torch.equal(drawn(3)[0], drawn(4)[0])


def test_update_refuses_a_proof_outside_the_grammar(dan: DanStateAdapter) -> None:
    with pytest.raises(ValueError, match="outside Dan's state-env grammar"):
        dan.learner(seed=0).update([_pair(_UNCITED_BOX)], 1e-3)


def test_learner_adapter_samples_with_the_learners_live_weights(dan: DanStateAdapter) -> None:
    learner = dan.learner(seed=0)
    sampler = learner.adapter()
    assert isinstance(sampler, DanStateAdapter)
    assert sampler._model is learner._model and sampler._model is not dan._model
    assert sampler._tok is not dan._tok
    assert sampler.identity()["derived_from_learner"] is True
    assert "derived_from_learner" not in dan.identity()
    learner.update(_GOOD, 1e-2)
    # the sampler sees the update (same live model) and the original does not
    assert sampler.logprob(_GOOD)[0].nats == -learner.nll(_GOOD)[0].nats
    assert sampler.logprob(_GOOD)[0].nats != dan.logprob(_GOOD)[0].nats
    req = SampleRequest(
        pid="x", sequent=_GOOD[0][0], n=2, temperature=1.0, max_new_tokens=16, seed=0
    )
    out = sampler.sample([req])
    assert len(out) == 1 and len(out[0]) == 2 and all(isinstance(a, Attempt) for a in out[0])


def test_learner_refuses_a_steered_adapter(dan: DanStateAdapter) -> None:
    term = SteeringTerm(layer=0, vector=tuple([0.1] * 32), alpha=1.0, where="all")
    steered = dan.with_steering((term,))
    try:
        with pytest.raises(NotSupported):
            steered.learner(seed=0)
    finally:
        steered.close()


# ----------------------------------------------------------------------------- build_classes


def _manifest(suite: str, version: str, n: int) -> SuiteManifest:
    return SuiteManifest(
        suite=suite,
        version=version,
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=n,
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )


def _syn_problem(
    suite: str, i: int, tags: dict[str, JSON], *, n_prem: int, version: str = "v1"
) -> Problem:
    seq = Sequent(premises=tuple(f"SYN_p{j}" for j in range(n_prem)), conclusion=f"SYN_{suite}_{i}")
    ref = ReferenceProof(body="N1 P : PR ; QED", lines=1, pruned=1, source="constructed")
    return Problem(
        pid=f"{suite}:{i}",
        suite=suite,
        version=version,
        sequent=seq,
        length=LengthLabel(L=1, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
        ref=ref,
        tags=tags,
    )


class _StubExcluder:
    """Marks exactly the sequents whose conclusion is in `seen` as seen in training."""

    def __init__(self, seen: set[str]) -> None:
        self._seen = seen

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        return seq.conclusion in self._seen, ()


_MISSING = "/nonexistent/does-not-exist.jsonl"
_NO_FILES = {"gen_path": _MISSING, "heldout_cap6_path": _MISSING}


def _class_stats(stats: dict[str, JSON]) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], stats["class_stats"])


def test_max_premises_filters_before_the_split_and_is_reported(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    tags: dict[str, JSON] = {"family": "and_tower", "n": 8}
    items = [_syn_problem("families", i, tags, n_prem=1 + i % 3) for i in range(100)]
    items += [_syn_problem("families", 100 + i, tags, n_prem=4) for i in range(50)]
    store.write_suite(items, _manifest("families", "v1", len(items)))

    _, default_stats = build_classes(dict(_NO_FILES), store=store)
    assert default_stats["class_counts"] == {"and_tower_n8": {"n_train": 64, "n_test": 64}}
    assert _class_stats(default_stats)["and_tower_n8"]["n_dropped_premises"] == 0

    problems, stats = build_classes({**_NO_FILES, "max_premises": 3}, store=store)
    info = _class_stats(stats)["and_tower_n8"]
    assert info["n_dropped_premises"] == 50
    assert info["n_distinct"] == 100
    assert (info["n_train"], info["n_test"]) == (32, 32)  # refilled from the pool of 100
    assert set(info["premise_counts"]) <= {"1", "2", "3"}
    assert sum(info["premise_counts"].values()) == 64
    assert all(len(p.sequent.premises) <= 3 for p in problems)

    _, tight = build_classes({**_NO_FILES, "max_premises": 1}, store=store)
    assert "and_tower_n8" not in cast(dict[str, Any], tight["class_counts"])
    assert _class_stats(tight)["and_tower_n8"]["skipped"].startswith("only 34 distinct")


def test_max_premises_applies_to_file_backed_classes_too(tmp_path: Path) -> None:
    rows = [
        {
            "name": f"h{i}",
            "prompt": Sequent(
                premises=tuple(f"SYN_q{j}" for j in range(i % 5)), conclusion=f"SYN_ho_{i}"
            ).prompt(),
            "n_lines": 2,
            "proof": f"N1 SYN_ho_{i} : PR ; QED",
        }
        for i in range(200)
    ]
    path = tmp_path / "heldout_cap6.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    problems, stats = build_classes(
        {"gen_path": _MISSING, "heldout_cap6_path": str(path), "max_premises": 3},
        store=Store(tmp_path / "out"),
    )
    assert _class_stats(stats)["heldout_cap6"]["n_dropped_premises"] == 40
    assert all(len(p.sequent.premises) <= 3 for p in problems)


def test_classes_list_selects_and_validates(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    tags: dict[str, JSON] = {"family": "imp_nest", "n": 4}
    items = [_syn_problem("families", i, tags, n_prem=0) for i in range(64)]
    store.write_suite(items, _manifest("families", "v1", len(items)))
    _, stats = build_classes({**_NO_FILES, "classes": ["imp_nest_n4"]}, store=store)
    assert stats["n_classes_total"] == 1
    assert list(_class_stats(stats)) == ["imp_nest_n4"]
    _, stats2 = build_classes({**_NO_FILES, "classes": "imp_nest_n4,gen_L10"}, store=store)
    assert list(_class_stats(stats2)) == ["imp_nest_n4", "gen_L10"]
    with pytest.raises(ValueError, match="unknown class"):
        build_classes({**_NO_FILES, "classes": ["nope"]}, store=store)


def test_exclusion_family_drops_dan_seen_items_from_every_class(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    tags: dict[str, JSON] = {"schema": "contraposition", "s": 3}
    items = [_syn_problem("laws", i, tags, n_prem=1) for i in range(140)]
    store.write_suite(items, _manifest("laws", "v1", len(items)))
    seen = {f"SYN_laws_{i}" for i in range(20)}
    # ours (no exclusion_family): suite classes are flag-only, nothing dropped
    _, ours = build_classes(dict(_NO_FILES), store=store, excluder=_StubExcluder(seen))
    assert _class_stats(ours)["contraposition_s3"]["n_dropped_seen"] == 0
    # Dan's tiers apply to every class
    problems, stats = build_classes(
        {**_NO_FILES, "exclusion_family": "dan12"}, store=store, excluder=_StubExcluder(seen)
    )
    info = _class_stats(stats)["contraposition_s3"]
    assert info["n_dropped_seen"] == 20
    assert (info["n_train"], info["n_test"]) == (32, 32)  # 120 left
    assert not {p.sequent.conclusion for p in problems} & seen
    with pytest.raises(ValueError, match="unknown exclusion_family"):
        build_classes({**_NO_FILES, "exclusion_family": "dan99"}, store=store)


def test_dan_heldout_null_control_is_disjoint_from_dan_training(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    items = [_syn_problem("dan_heldout", i, {}, n_prem=i % 4) for i in range(150)]
    store.write_suite(items, _manifest("dan_heldout", "v1", len(items)))
    with pytest.raises(ValueError, match="needs exclusion_family"):
        build_classes({**_NO_FILES, "classes": ["dan_heldout"]}, store=store)
    seen = {f"SYN_dan_heldout_{i}" for i in range(0, 150, 10)}
    problems, stats = build_classes(
        {**_NO_FILES, "classes": ["dan_heldout"], "exclusion_family": "dan12"},
        store=store,
        excluder=_StubExcluder(seen),
    )
    info = _class_stats(stats)["dan_heldout"]
    assert info["n_dropped_seen"] == 15
    assert (info["n_train"], info["n_test"]) == (64, 64)
    assert {str(p.tags["class"]) for p in problems} == {"dan_heldout"}
    assert not {p.sequent.conclusion for p in problems} & seen
    # an explicit excluder without exclusion_family still makes it disjoint
    _, override = build_classes(
        {**_NO_FILES, "classes": ["dan_heldout"]}, store=store, excluder=_StubExcluder(seen)
    )
    assert _class_stats(override)["dan_heldout"]["n_dropped_seen"] == 15
    # not built unless asked for: the default class list is unchanged
    _, default = build_classes(dict(_NO_FILES), store=store)
    assert "dan_heldout" not in _class_stats(default)
    assert default["n_classes_total"] == 12


def test_require_grammar_drops_items_outside_dans_grammar(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    tags: dict[str, JSON] = {"family": "reorder_and", "n": 5}

    def real(i: int, prompt_body: tuple[str, str]) -> Problem:
        seq, body = _pair(prompt_body)
        ref = ReferenceProof(body=body, lines=2, pruned=2, source="constructed")
        return Problem(
            pid=f"families:{i}",
            suite="families",
            version="v1",
            sequent=seq,
            length=LengthLabel(L=2, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
            ref=ref,
            tags=tags,
        )

    items = [real(0, _SIMPLE), real(1, _IMP), *(real(2 + i, _UNCITED_BOX) for i in range(5))]
    store.write_suite(items, _manifest("families", "v1", len(items)))
    _, stats = build_classes({**_NO_FILES, "require_grammar": "danstate"}, store=store)
    info = _class_stats(stats)["reorder_and_n5"]
    assert info["n_dropped_grammar"] == 5
    assert info["n_distinct"] == 2
    _, ungated = build_classes(dict(_NO_FILES), store=store)
    assert _class_stats(ungated)["reorder_and_n5"]["n_dropped_grammar"] == 0
    with pytest.raises(ValueError, match="unknown require_grammar"):
        build_classes({**_NO_FILES, "require_grammar": "lean"}, store=store)


# ------------------------------------------------------------------------- run_edl end to end


def _or_intro_problems(n_items: int) -> list[Problem]:
    """`n_items` distinct (up to atom renaming) in-grammar theorems `P |- ( P v X )`, each with
    the two-line proof `PR ; ORI1`, tagged as the `and_tower_n8` family cell so `build_classes`
    picks them up from a fake `families` suite."""
    atoms = ["P", "Q", "R", "S"]
    level2 = [f"( {a} {op} {b} )" for op in ("&", "v", ">") for a in atoms for b in atoms]
    level2 += [f"( ~ {a} )" for a in atoms]
    formulas = level2 + [f"( ~ {f} )" for f in level2]
    formulas += [f"( {f} {op} {a} )" for op in ("&", ">") for f in level2 for a in ("P", "Q")]
    out: list[Problem] = []
    keys: set[str] = set()
    for x in formulas:
        seq = Sequent(premises=("P",), conclusion=f"( P v {x} )")
        if seq.key() in keys:
            continue
        keys.add(seq.key())
        body = f"N1 P : PR ; N2 ( P v {x} ) : ORI1 N1 ; QED"
        out.append(
            Problem(
                pid=f"families:or{len(out)}",
                suite="families",
                version="v1",
                sequent=seq,
                length=LengthLabel(L=2, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
                ref=ReferenceProof(body=body, lines=2, pruned=2, source="constructed"),
                tags={"family": "and_tower", "n": 8},
            )
        )
        if len(out) == n_items:
            break
    assert len(out) == n_items
    return out


def _dan_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "out")
    items = _or_intro_problems(70)
    store.write_suite(items, _manifest("families", "v1", len(items)))
    return store


_ROW_KEYS = {
    "class",
    "model_identity",
    "lr",
    "order",
    "n",
    "m",
    "seed",
    "online_nats",
    "L_test_0",
    "L_test_n",
    "L_test_by_checkpoint",
    "EDL_by_checkpoint",
    "EDL",
    "EDL_per_example",
    "start_excluded_online_nats",
    "L_test_0_start_excluded",
    "L_test_n_start_excluded",
    "EDL_start_excluded",
    "pass8_before",
    "pass8_after",
    "n_nonfinite_nll",
}


def test_run_edl_end_to_end_on_a_tiny_dan_model(
    tmp_path: Path, dan: DanStateAdapter, capfd: pytest.CaptureFixture[str]
) -> None:
    store = _dan_store(tmp_path)
    problems, stats = build_classes(
        {**_NO_FILES, "classes": ["and_tower_n8"], "require_grammar": "danstate"}, store=store
    )
    assert stats["class_counts"] == {"and_tower_n8": {"n_train": 32, "n_test": 32}}
    streamed: list[dict[str, JSON]] = []
    rows = run_edl(
        dan,
        problems,
        lrs=(1e-3, 0.0),
        orders=1,
        ckpts=(8, 16),
        shifts_per_update=2,
        on_row=streamed.append,
    )
    assert streamed == rows and len(rows) == 2
    test_pairs = [
        (p.sequent, cast(ReferenceProof, p.ref).body) for p in problems if p.tags["split"] == "test"
    ]
    l_test_0 = sum(-lp.nats for lp in dan.logprob(test_pairs)) / len(test_pairs)
    for row in rows:
        assert set(row) == _ROW_KEYS
        assert row["model_identity"] == dan.identity()
        assert (row["n"], row["m"], row["order"]) == (32, 32, 0)
        online = cast(list[float], row["online_nats"])
        assert len(online) == 32 and all(math.isfinite(x) and x > 0 for x in online)
        assert row["L_test_0"] == pytest.approx(l_test_0, rel=1e-12)
        assert set(cast(dict[str, float], row["L_test_by_checkpoint"])) == {"8", "16", "32"}
        assert row["EDL"] == edl_of(online, cast(float, row["L_test_n"]))
        se = cast(list[float], row["start_excluded_online_nats"])
        assert len(se) == 32 and all(a < b for a, b in zip(se, online, strict=True))
        assert row["EDL_start_excluded"] is not None
        assert row["n_nonfinite_nll"] == 0
    by_lr = {cast(float, r["lr"]): r for r in rows}
    # lr 0: weights never move, so the online losses are just the starting NLLs and L_test fixed
    assert by_lr[0.0]["L_test_n"] == by_lr[0.0]["L_test_0"]
    assert cast(float, by_lr[1e-3]["L_test_n"]) < cast(float, by_lr[1e-3]["L_test_0"])
    err = capfd.readouterr().err
    assert "[edl] 2 runs to do over 1 class(es)" in err
    assert "[edl] 1/2 and_tower_n8 lr=0.001 order=0" in err and "eta" in err
    assert "[edl] 2/2 and_tower_n8 lr=0 order=0" in err


def test_run_edl_raises_before_training_on_items_outside_the_grammar(
    dan: DanStateAdapter,
) -> None:
    problems = _or_intro_problems(8)
    seq, body = _pair(_UNCITED_BOX)
    bad = Problem(
        pid="bad:uncited",
        suite="families",
        version="v1",
        sequent=seq,
        length=LengthLabel(L=3, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
        ref=ReferenceProof(body=body, lines=3, pruned=3, source="constructed"),
        tags={},
    )
    streamed: list[dict[str, JSON]] = []
    with pytest.raises(ValueError, match=r"bad:uncited \(ValueError: uncited box"):
        run_edl(
            dan,
            [*problems[:4], bad, *problems[4:]],
            lrs=(1e-3,),
            orders=1,
            n=4,
            m=5,
            on_row=streamed.append,
        )
    assert streamed == []


class _DivergingLearner:
    """NLL 1.0 until an update at lr >= 1, NaN afterwards (a diverged run)."""

    def __init__(self) -> None:
        self.diverged = False

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        v = float("nan") if self.diverged else 1.0
        return [LogProb(nats=v, n_tokens=1, per_encoding=()) for _ in pairs]

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        self.diverged = self.diverged or lr >= 1.0
        return 0.0

    def adapter(self) -> ModelAdapter:
        raise NotSupported("test double")


class _DivergingAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.label = "diverging"
        self.format_id = "nd_v1"
        self.capabilities = Capability.FINETUNE

    def identity(self) -> dict[str, JSON]:
        return {"adapter": "diverging", "label": "diverging"}

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        return 1

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        raise NotSupported("test double")

    def learner(self, *, seed: int) -> Learner:
        return _DivergingLearner()


def test_divergence_during_training_is_counted_and_never_chosen_as_best_lr(
    capfd: pytest.CaptureFixture[str],
) -> None:
    problems = _or_intro_problems(8)
    # the diverging lr comes first, so a NaN-unsafe min() over code lengths would pick it
    rows = run_edl(_DivergingAdapter(), problems, lrs=(1.0, 0.1), orders=1, n=4, m=4)
    by_lr = {cast(float, r["lr"]): r for r in rows}
    assert cast(int, by_lr[1.0]["n_nonfinite_nll"]) > 0
    assert by_lr[0.1]["n_nonfinite_nll"] == 0
    assert "WARNING" in capfd.readouterr().err
    summary = cast(dict[str, dict[str, Any]], summarise_edl(rows))
    (entry,) = summary.values()
    assert entry["EDL_star"] == pytest.approx(math.log(2))  # lr 0.1: EDL 0, plus ln(2 lrs)


# ---------------------------------------------------------------------------------------- CLI


def test_cli_edl_with_a_danstate_checkpoint_classes_and_resume(
    tmp_path: Path, tiny_ckpt: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    store = _dan_store(tmp_path)
    cfg_path = tmp_path / "edl_dan.yaml"
    cfg_path.write_text(
        yaml.safe_dump({**_NO_FILES, "max_premises": 3, "require_grammar": "danstate"})
    )
    base_args = [
        "edl",
        "--model",
        f"danstate:{tiny_ckpt}",
        "--model-arg",
        "name_base_max=2",
        "--device",
        "cpu",
        "--config",
        str(cfg_path),
        "--classes",
        "and_tower_n8",
        "--orders",
        "1",
        "--n",
        "64",
        "--m",
        "64",
        "--run-id",
        "edl-tinydan-64-64",
        "--out",
        str(store.root),
    ]
    assert main([*base_args, "--lrs", "1e-3"]) == 0
    path = store.edl_dir / "edl-tinydan-64-64.jsonl"
    first = list(read_jsonl(path))
    assert [(r["class"], r["lr"]) for r in first] == [("and_tower_n8", 1e-3)]
    ident = cast(dict[str, JSON], first[0]["model_identity"])
    assert ident["adapter"] == "danstate" and ident["name_base_max"] == 2

    # --resume with one more lr: only the missing combination runs, earlier rows untouched
    assert main([*base_args, "--lrs", "1e-3,3e-4", "--resume"]) == 0
    second = list(read_jsonl(path))
    assert second[0] == first[0]
    assert [r["lr"] for r in second] == [1e-3, 3e-4]
    assert main([*base_args, "--lrs", "1e-3,3e-4", "--resume"]) == 0
    assert len(list(read_jsonl(path))) == 2

    # resuming into a run written by another model is refused (`main` turns the SystemExit into
    # a non-zero return code), and nothing is appended
    other = tmp_path / "other_dan.pt"
    _save_tiny_dan_ckpt(other, seed=7)
    other_args = [a if a != f"danstate:{tiny_ckpt}" else f"danstate:{other}" for a in base_args]
    capfd.readouterr()
    assert main([*other_args, "--lrs", "1e-3", "--resume"]) != 0
    assert "different model identity" in capfd.readouterr().err
    assert len(list(read_jsonl(path))) == 2


class _RecordingDan(DanStateAdapter):
    """Stands in for `DanStateAdapter` in `ADAPTERS`: writes the constructor kwargs `cmd_edl`
    passes to `$NDBENCH_TEST_RECORD` (a file: `cli` imports this class through `importlib`, which
    may load a second copy of this module), then stops before any checkpoint or device is
    touched."""

    def __init__(self, **kwargs: Any) -> None:
        Path(os.environ["NDBENCH_TEST_RECORD"]).write_text(json.dumps(kwargs))
        raise SystemExit("recorded")


def test_cli_edl_label_and_device_cuda_reach_the_dan_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--model dan12-fz-s0 --device cuda` resolves through `configs/models.yaml` (sha256-checked
    against the real checkpoint, a read) to `DanStateAdapter(ckpt=..., device="cuda")`. The CUDA
    guard is bypassed and the adapter replaced by a recorder, so no GPU is touched."""
    import code.ndbench.cli as cli

    models_yaml = Path(cli.DEFAULT_MODELS_CONFIG)
    entry = yaml.safe_load(models_yaml.read_text())["dan12-fz-s0"]
    if not Path(entry["ckpt"]).is_file():
        pytest.skip("real dan12-fz-s0 checkpoint not present")
    record = tmp_path / "record.json"
    monkeypatch.setenv("NDBENCH_TEST_RECORD", str(record))
    monkeypatch.setattr(cli, "_guard_cuda_device", lambda *_devices: None)
    monkeypatch.setitem(cli.ADAPTERS, "danstate", "code.tests.ndbench.test_dan_edl:_RecordingDan")
    cfg_path = tmp_path / "edl.yaml"
    cfg_path.write_text(yaml.safe_dump(dict(_NO_FILES)))
    rc = main(
        [
            "edl",
            "--model",
            "dan12-fz-s0",
            "--device",
            "cuda",
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc != 0  # the recorder's SystemExit
    assert json.loads(record.read_text()) == {"ckpt": entry["ckpt"], "device": "cuda"}


def test_n_and_m_cap_a_tagged_split_for_pilots() -> None:
    problems = [
        replace(p, tags={**p.tags, "class": "c", "split": "train" if i < 10 else "test"})
        for i, p in enumerate(_or_intro_problems(20))
    ]
    (row,) = run_edl(_DivergingAdapter(), problems, lrs=(0.1,), orders=1, n=4, m=3, ckpts=(2,))
    assert (row["n"], row["m"]) == (4, 3)
    (full,) = run_edl(_DivergingAdapter(), problems, lrs=(0.1,), orders=1)  # defaults 64/64
    assert (full["n"], full["m"]) == (10, 10)


def test_micro_batched_update_matches_one_batch(
    dan: DanStateAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same loss and same (accumulated, clipped) gradient as one batch. Gradients rather than the
    stepped weights are compared: Adam's first step is ~lr * sign(g), so float noise on a near-zero
    gradient component flips that component's step."""
    import code.ndbench.adapters.danstate as danstate

    def grads(learner: _DanStateLearner) -> dict[str, torch.Tensor]:
        return {n: cast(torch.Tensor, q.grad).clone() for n, q in learner._model.named_parameters()}

    whole = dan.learner(seed=5)
    loss_whole = whole.update(_GOOD, 0.0)
    monkeypatch.setattr(danstate, "_UPDATE_MAX_CELLS", 64)  # forces many micro-batches
    split = dan.learner(seed=5)
    seen, handle = _batch_sizes(split)
    try:
        loss_split = split.update(_GOOD, 0.0)
    finally:
        handle.remove()
    assert len(seen) > 3
    assert loss_split == pytest.approx(loss_whole, rel=1e-5)
    a, b = grads(whole), grads(split)
    assert all(torch.allclose(a[k], b[k], rtol=1e-4, atol=1e-7) for k in a)
