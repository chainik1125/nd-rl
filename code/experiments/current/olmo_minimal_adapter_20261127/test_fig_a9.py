"""fig_a9.py draws from a round-5 summary: complete cells, a cut cell, a cell not sampled."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
import fig_a9  # noqa: E402

HERE = Path(__file__).resolve().parent


def _c(v, lo, hi):
    return {"genuine_rate": {"value": v, "lo": lo, "hi": hi}}


def _summary():
    return {
        "cells": {
            "rank1_n64": _c(0.31, 0.22, 0.40),
            "full_n16": _c(0.26, 0.18, 0.35),
            "full_n4": _c(0.14, 0.07, 0.22),
            "ref": _c(0.365, 0.27, 0.46),
            "final_r4": _c(0.474, 0.367, 0.581),
            "onepass_r4": _c(0.349, 0.258, 0.443),
        },
        "unfinished_draws": {},
    }


def test_whisker_uses_the_bounds_for_a_cut_cell():
    res = _summary()
    assert fig_a9.whisker(res, "full_n16") == (0.26, 0.18, 0.35, False)
    res["unfinished_draws"] = {"full_n16": 40}
    res["bounds"] = {
        "unfinished_as_fail": {"cells": {"full_n16": _c(0.23, 0.15, 0.31)}},
        "unfinished_as_accepted_pass": {"cells": {"full_n16": _c(0.34, 0.25, 0.43)}},
    }
    assert fig_a9.whisker(res, "full_n16") == (0.26, 0.15, 0.43, True)
    assert fig_a9.whisker(res, "full_n4") == (0.14, 0.07, 0.22, False)
    assert fig_a9.reference_name(res["cells"]) == "ref"
    assert fig_a9.reference_name({**res["cells"], "ref_again": {}}) == "ref_again"


@pytest.mark.parametrize("drop", [None, "rank1_n64", "full_n16"])
def test_figure_renders_in_three_formats(tmp_path, drop):
    res = _summary()
    if drop:
        del res["cells"][drop]
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(res))
    out = tmp_path / "figs" / "a9"
    r = subprocess.run(
        [sys.executable, str(HERE / "fig_a9.py"), "--summary", str(src), "--out", str(out)],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"},
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    for ext in (".pdf", ".svg", ".png"):
        assert out.with_suffix(ext).stat().st_size > 2000
    svg = out.with_suffix(".svg").read_text()
    assert "<svg" in svg and svg.count("<path") > 20
