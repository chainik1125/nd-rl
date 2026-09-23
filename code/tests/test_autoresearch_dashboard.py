"""The development dashboard can read live results without sharing tracked directories."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def test_results_directory_can_be_a_live_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = Path(__file__).parents[1] / "experiments/current/autoresearch/dashboard/serve.py"
    spec = importlib.util.spec_from_file_location("research_dashboard", source)
    assert spec is not None and spec.loader is not None
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    live = tmp_path / "researcher/results"
    run = live / "015_test"
    run.mkdir(parents=True)
    (run / "params.json").write_text(
        json.dumps(
            {
                "id": "015",
                "slug": "test",
                "title": "test",
                "protocol": {
                    "budget": {"total_secs": 1200, "pretrain_secs": 300},
                    "rl": {"n_targets": 1500},
                    "eval": {"pool": {"n": 1108}},
                },
            }
        )
    )
    link = tmp_path / "live-results"
    link.symlink_to(live, target_is_directory=True)
    served = []

    class FakeServer:
        def __init__(self, *args: Any):
            pass

        def serve_forever(self):
            served.extend(dashboard.load_runs())

    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", FakeServer)
    dashboard.main(["--results-dir", str(link)])
    assert served[0]["status"] == "pending"
    (run / "results.json").write_text('{"metric": 149, "status": "keep"}')
    assert dashboard.load_runs()[0]["metric"] == 149


def test_reads_both_agents_from_a_git_ref(monkeypatch: pytest.MonkeyPatch):
    """Default mode: results come from the shared branch, not whichever checkout serves them."""
    source = Path(__file__).parents[1] / "experiments/current/autoresearch/dashboard/serve.py"
    spec = importlib.util.spec_from_file_location("research_dashboard", source)
    assert spec is not None and spec.loader is not None
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    monkeypatch.setattr(dashboard, "ref", "HEAD")
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 0.0)  # skip the network fetch
    runs = dashboard.load_runs()
    committed = sorted(p.parent.name for p in source.parents[1].glob("results/*/params.json"))
    assert [f"{r['id']}_{r['slug']}" for r in runs][:3] == committed[:3]
    assert runs[0]["agent"] is None  # 000 predates the parallel loops
