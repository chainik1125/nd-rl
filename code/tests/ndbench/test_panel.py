"""The benchmark panel (`code/ndbench/panel.yaml`) and `run_schedule.py`'s tier selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "code" / "ndbench" / "panel.yaml"
SCHEDULER = ROOT / "code" / "experiments" / "current" / "benchmarking_20260930" / "run_schedule.py"


def _scheduler() -> Any:
    spec = importlib.util.spec_from_file_location("run_schedule", SCHEDULER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jobs() -> list[dict[str, object]]:
    return yaml.safe_load(PANEL.read_text())["jobs"]


def test_every_panel_job_is_tagged() -> None:
    rs = _scheduler()
    names = [str(j["job"]) for j in _jobs()]
    assert len(names) == len(set(names))
    for j in _jobs():
        assert j["tier"] in rs.TIERS, j["job"]
        assert j["scope"] in ("any", "model", "pair"), j["job"]
        assert float(j["est_min"]) > 0, j["job"]  # type: ignore[arg-type]
        templated = "{model}" in str(j["job"])
        assert templated == (j["scope"] == "model"), j["job"]
        if j["tier"] == "cpu":
            assert "--device cuda" not in str(j["command"]), j["job"]
        if j["scope"] == "model":
            assert "--device cuda" in str(j["command"]), j["job"]


def test_needs_name_earlier_jobs() -> None:
    seen: set[str] = set()
    for j in _jobs():
        for need in j.get("needs", []) or []:  # type: ignore[union-attr]
            assert need in seen, (j["job"], need)
        seen.add(str(j["job"]))


def test_cheap_tier_selects_cpu_and_cheap_only() -> None:
    rs = _scheduler()
    sel = rs.select_jobs(_jobs(), want=None, max_tier="cheap", model="m", include_pair=False)
    tiers = {str(j["tier"]) for j in sel}
    assert tiers == {"cpu", "cheap"}
    assert all(j["scope"] != "pair" for j in sel)
    assert "prefix_m" in {j["job"] for j in sel}
    assert all("{model}" not in str(j["command"]) and "{out}" not in str(j["command"]) for j in sel)
    gpu_min = sum(float(j["est_min"]) for j in sel if j["tier"] != "cpu")  # type: ignore[arg-type]
    assert gpu_min <= 20


def test_cpu_tier_needs_no_model_and_no_gpu() -> None:
    rs = _scheduler()
    sel = rs.select_jobs(_jobs(), want=None, max_tier="cpu", model=None, include_pair=False)
    assert sel and all(j["tier"] == "cpu" for j in sel)
    assert all("--device cuda" not in str(j["command"]) for j in sel)


def test_pair_jobs_need_the_flag_and_jobs_narrow() -> None:
    rs = _scheduler()
    sel = rs.select_jobs(
        _jobs(), want={"interp_probes"}, max_tier="expensive", model="m", include_pair=True
    )
    assert [j["job"] for j in sel] == ["interp_probes"]
    none = rs.select_jobs(
        _jobs(), want={"interp_probes"}, max_tier="expensive", model="m", include_pair=False
    )
    assert none == []


def test_model_jobs_require_a_model() -> None:
    rs = _scheduler()
    with pytest.raises(SystemExit):
        rs.select_jobs(_jobs(), want=None, max_tier="cheap", model=None, include_pair=False)
