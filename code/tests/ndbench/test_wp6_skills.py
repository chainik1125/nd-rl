"""WP6 acceptance test for `analysis/skills.py` (skill x length heat-map + GLM)."""

from __future__ import annotations

import math
from pathlib import Path

from code.ndbench.analysis import skills
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
