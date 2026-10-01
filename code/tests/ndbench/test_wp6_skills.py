"""WP6 acceptance test for `analysis/skills.py` (skill x length heat-map + GLM)."""

from __future__ import annotations

import math
from pathlib import Path

from code.ndbench.analysis import skills
from code.ndbench.dash import pages
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    problems = [
        make_problem("i1", "skill_pool", "v1", length=8, tags={"classical": True}),
        make_problem("i2", "skill_pool", "v1", length=8, tags={"classical": False}),
        make_problem("i3", "skill_pool", "v1", length=10, tags={"classical": True}),
    ]
    store.write_suite(problems, manifest("skill_pool", "v1", len(problems)))
    run_id = start_run(store, suite="skill_pool", version="v1", model="base", k=32)
    store.append_items(
        run_id,
        [
            make_item(run_id, "i1", 32, 32),
            make_item(run_id, "i2", 32, 0),
            make_item(run_id, "i3", 32, 16),
        ],
    )
    return store


def test_skills_view_nonempty_heatmap_and_glm(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    data = skills.view(store, {"skills": ["classical"]})
    assert data["empty"] is False
    base = data["models"]["base"]
    assert base["n_items"] == 3

    heat1 = base["heatmap_pass1"]["classical"]
    assert heat1["8.0"] == {"pass1": 1.0, "n_items": 1}
    assert heat1["10.0"] == {"pass1": 0.5, "n_items": 1}

    heat32 = base["heatmap_pass32"]["classical"]
    assert heat32["8.0"]["pass32"] == 1.0
    assert heat32["10.0"]["pass32"] == 1.0  # n - c = 16 < k = 32 -> pass@32 = 1.0

    glm = base["glm"]
    assert set(glm) == {"intercept", "L", "classical"}
    for coef in glm.values():
        assert math.isfinite(coef["coef"])
        assert math.isfinite(coef["se"])


def test_skills_tags_list_form(tmp_path: Path) -> None:
    """A `tags["skills"]` list form (instead of a per-skill boolean tag) is also honoured."""
    store = Store(tmp_path)
    problems = [
        make_problem("i1", "skill_pool", "v1", length=8, tags={"skills": ["case_split"]}),
        make_problem("i2", "skill_pool", "v1", length=8, tags={"skills": []}),
    ]
    store.write_suite(problems, manifest("skill_pool", "v1", len(problems)))
    run_id = start_run(store, suite="skill_pool", version="v1", model="base", k=32)
    store.append_items(run_id, [make_item(run_id, "i1", 32, 32), make_item(run_id, "i2", 32, 0)])
    data = skills.view(store, {"skills": ["case_split"]})
    heat1 = data["models"]["base"]["heatmap_pass1"]["case_split"]
    assert heat1 == {"8.0": {"pass1": 1.0, "n_items": 1}}


def test_skills_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = skills.view(store, {})
    assert data["empty"] is True


def _build_legacy_joined_store(tmp_path: Path) -> Store:
    """Fix list item 2: no run ever samples `skill_pool` itself -- the real results are legacy-
    imported runs under their own suite names/pids, joined back to `skill_pool` by theorem name.
    Mirrors the real store's shapes: `skill_pool` pids are `f"{source_set}:{name}"`
    (`suites/existing.py:skill_pool`), legacy pids are the bare `name`
    (`suites/existing.py:_import_monitor_run`/`_import_lengthexp_run`), `dev_L8-11` joins to the
    single suite `legacy_dev`, `lengthexp_std` joins to one legacy suite *per L bucket*
    (`legacy_lengthexp_std_L6`, `legacy_lengthexp_std_L7`, ...), legacy label `champion` -> `rl`
    and legacy labels `base`/`pretrain` both -> `base` (disjoint source sets, so a union not a
    conflict), and arm C's three lengthexp seeds (`C_s101`/`C_s102`/`C_s103`) are independent
    replications meant to be pooled (summed), not reruns of each other.
    """
    store = Store(tmp_path)
    problems = [
        make_problem(
            "dev_L8-11:t0",
            "skill_pool",
            "v1",
            length=8,
            tags={"source_set": "dev_L8-11", "classical": True},
        ),
        make_problem(
            "dev_L8-11:t1",
            "skill_pool",
            "v1",
            length=8,
            tags={"source_set": "dev_L8-11", "classical": False},
        ),
        make_problem(
            "lengthexp_std:t6",
            "skill_pool",
            "v1",
            length=6,
            tags={"source_set": "lengthexp_std", "classical": True},
        ),
    ]
    store.write_suite(problems, manifest("skill_pool", "v1", len(problems)))

    # dev_L8-11 -> legacy_dev, model labels base/champion.
    dev_base = start_run(
        store, suite="legacy_dev", version="k32_s1", model="base", k=32, status="legacy"
    )
    store.append_items(
        dev_base, [make_item(dev_base, "t0", 32, 32), make_item(dev_base, "t1", 32, 0)]
    )
    dev_champ = start_run(
        store, suite="legacy_dev", version="k32_s1", model="champion", k=32, status="legacy"
    )
    store.append_items(
        dev_champ, [make_item(dev_champ, "t0", 32, 16), make_item(dev_champ, "t1", 32, 16)]
    )

    # lengthexp_std -> legacy_lengthexp_std_L6, model labels pretrain (-> base) and three armC
    # seeds (pooled: 16+16+16 c out of 32+32+32 n).
    pre = start_run(
        store,
        suite="legacy_lengthexp_std_L6",
        version="pretrain",
        model="pretrain",
        k=32,
        status="legacy",
    )
    store.append_items(pre, [make_item(pre, "t6", 32, 8)])
    for seed in ("C_s101", "C_s102", "C_s103"):
        arm = start_run(
            store,
            suite="legacy_lengthexp_std_L6",
            version=seed,
            model="armC",
            k=32,
            status="legacy",
        )
        store.append_items(arm, [make_item(arm, "t6", 32, 16)])

    return store


def test_skills_joins_legacy_runs_to_skill_pool_by_pid(tmp_path: Path) -> None:
    store = _build_legacy_joined_store(tmp_path)
    data = skills.view(store, {"skills": ["classical"]})
    assert data["empty"] is False
    assert set(data["meta"]["models"]) == {"base", "rl", "armC"}
    assert data["meta"]["used_legacy_join"] is True

    base = data["models"]["base"]
    # base = union of legacy "base" (dev_L8-11) and legacy "pretrain" (lengthexp_std): all 3 items.
    assert base["n_items"] == 3
    rl = data["models"]["rl"]
    # rl only comes from legacy "champion" (dev_L8-11): 2 items.
    assert rl["n_items"] == 2
    armc = data["models"]["armC"]
    assert armc["n_items"] == 1
    # Three seeds at k=32 each, c=16 each, pooled by summing n and c (not averaged/picked-one).
    heat1 = armc["heatmap_pass1"]["classical"]
    assert heat1["6.0"] == {"pass1": 0.5, "n_items": 1}


def test_skills_legacy_sealed_prefers_highest_k_over_a_superseded_rerun(tmp_path: Path) -> None:
    """A (legacy suite, legacy model) pair run at two different k's (e.g. real sealed_L12-14's
    k64 then a k256 rerun) must keep only the higher-k run, not sum both (which would double-count
    a superseded rerun as if it were an independent replication like arm C's seeds)."""
    store = Store(tmp_path)
    problems = [
        make_problem(
            "sealed_L12-14:t0",
            "skill_pool",
            "v1",
            length=12,
            tags={"source_set": "sealed_L12-14", "classical": True},
        )
    ]
    store.write_suite(problems, manifest("skill_pool", "v1", len(problems)))
    low = start_run(
        store, suite="legacy_sealed_L12-14", version="k64_s0", model="base", k=64, status="legacy"
    )
    store.append_items(low, [make_item(low, "t0", 64, 10)])
    high = start_run(
        store, suite="legacy_sealed_L12-14", version="k256_s0", model="base", k=256, status="legacy"
    )
    store.append_items(high, [make_item(high, "t0", 256, 40)])

    data = skills.view(store, {"skills": ["classical"]})
    heat1 = data["models"]["base"]["heatmap_pass1"]["classical"]
    # pass@1 = c/n = 40/256, not pooled with the superseded k64 run.
    assert heat1["12.0"] == {"pass1": 40 / 256, "n_items": 1}


def test_render_skills_shows_the_legacy_join_note_and_cell_counts(tmp_path: Path) -> None:
    """Fix list item 2's "say so on the page": a reader must be told, on the page itself, that
    these are legacy-imported results joined by theorem name and how the model labels map, not
    just have the view compute `meta["legacy_note"]`/`meta["model_map"]` without displaying them.
    The heat-map must also show each cell's item count (plan §2 B3: "with cell counts"), not just
    the bare pass rate.
    """
    store = _build_legacy_joined_store(tmp_path)
    data = skills.view(store, {"skills": ["classical"]})
    figs = pages.render_skills(data)
    assert figs
    joined = "\n".join(figs)
    assert "http://" not in joined and "https://" not in joined
    assert "legacy-imported" in joined
    assert "champion" in joined and "rl" in joined  # the label mapping, spelled out
    assert "n=1" in joined or "n items" in joined  # a cell count shown somewhere (tooltip/table)
