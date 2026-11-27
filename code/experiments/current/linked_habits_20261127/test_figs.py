"""The summary's figure scripts run on the current read-out and write their PDF / SVG / PNG triple; skipped where the
experiment's data folder (LH) is not on this machine.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_figs.py
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figcommon as FC  # noqa: E402

HAVE = (FC.LH / "gate" / "gate_v2.json").exists() and any((FC.LH / "read").glob("read_*.json"))
FIGS = [
    ("fig_gate", "fig1_gate"),
    ("fig_dpo", "fig2_dpo"),
    ("fig_accept", "fig3_accept"),
    ("fig_grpo", "fig4_grpo"),
    ("fig_perf", "fig5_perf"),
]


@pytest.mark.skipif(not HAVE, reason="no LH read-out on this machine")
@pytest.mark.parametrize(("module", "stem"), FIGS)
def test_figure_runs(module: str, stem: str, tmp_path: Path) -> None:
    if module == "fig_grpo" and not (FC.LH / "read" / "summary_extra.json").exists():
        pytest.skip("summary_extra.py has not been run")
    paths = importlib.import_module(module).main(["--out", str(tmp_path)])
    assert sorted(p.name for p in paths) == sorted(f"{stem}.{x}" for x in ("pdf", "svg", "png"))
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000


def test_est_and_fmt() -> None:
    e = {"point": 3.43, "ci": [-1.66, 8.88]}
    assert FC.est(e) == (3.43, -1.66, 8.88)
    assert FC.fmt(e) == "+3.4 [-1.7, +8.9]"
    assert FC.est(None) is None and FC.fmt({"point": None}) == "-"
