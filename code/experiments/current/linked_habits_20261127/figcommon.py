"""Shared pieces of the summary figures (`fig_*.py`): where the inputs are, the plain names and fixed encodings of
the corpora, and the paper style (`code/ndbench/figures/style.py`, loaded by path so these scripts also run as plain
files from this folder).

Inputs are only read-out and measurement summaries:
    LH/read/read_*.json             the read-out (lh_read.py); the newest by modification time unless --read is given
    LH/read/summary_extra.json      summary_extra.py's additions (GRPO lines against the base, the second bases'
                                    performance, the authors of the GRPO training samples)
    LH/gate/gate_v2.json            the stage-1 gate and the performance control (lh_gate.py)
    LH/arms/<run>/probe/probe.json  the cheap probe of a rated run's snapshots
Re-running a figure after new data arrive (a new read-out file) redraws it from the new numbers.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LH = Path(os.environ.get("LH", "/scratch/cowifornia/nd-rl/linked_habits"))
FIG_DIR = REPO / "experiment-summaries" / "2026-11-27-linked-habits" / "figures"


def _load_style() -> Any:
    name = "linked_habits_paper_style"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "code" / "ndbench" / "figures" / "style.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_style()
OI = S.OKABE_ITO

#: Plain names of the corpora (the code's labels are names only: c24 has 22 linked habits, c9 has 7).
NAME = {
    "c24": "all-linked (22 habits)",
    "c9": "seven-natural",
    "c4": "four-habit (Nov 18 set)",
    "c0": "no-links",
    "c0n": "no-links, no note lines",
    "authors_half": "Nov 14 base, 4 linked, no notes",
    "control_half": "Nov 14 base, none linked, no notes",
}
SHORT = {"c24": "all-linked", "c9": "seven-natural", "c4": "four-habit", "c0": "no-links"}
#: One colour and marker per corpus in every figure (Okabe-Ito; the three hues are the validated set of style.py).
CORPUS_STYLE = {
    "c24": {"color": OI["blue"], "marker": "o"},
    "c9": {"color": OI["bluish_green"], "marker": "^"},
    "c4": {"color": OI["vermillion"], "marker": "s"},
    "c0": {"color": OI["black"], "marker": "D"},
}


def newest_read(lh: Path = LH) -> Path:
    files = sorted((lh / "read").glob("read_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no read-out under {lh / 'read'}")
    return files[-1]


def parser(doc: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lh", default=str(LH), help="the experiment's data folder (LH)")
    ap.add_argument(
        "--read", default=None, help="read-out JSON (default: the newest LH/read/read_*.json)"
    )
    ap.add_argument("--out", default=str(FIG_DIR), help="folder for the figure files")
    return ap


def load_json(p: Path) -> Any:
    return json.loads(Path(p).read_text())


def load_read(a: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    p = Path(a.read) if a.read else newest_read(Path(a.lh))
    return load_json(p), p


def load_extra(lh: Path) -> dict[str, Any]:
    p = Path(lh) / "read" / "summary_extra.json"
    return load_json(p) if p.exists() else {}


def est(e: Any) -> tuple[float, float, float] | None:
    """(point, low, high) of a {point, ci} entry, or None."""
    if not isinstance(e, dict) or e.get("point") is None:
        return None
    ci = e.get("ci") or [None, None]
    lo, hi = ci if ci[0] is not None else (e["point"], e["point"])
    return float(e["point"]), float(lo), float(hi)


def fmt(e: Any) -> str:
    """As the read-out prints a {point, ci} entry (Python's rounding: 2.55 -> +2.5)."""
    v = est(e)
    return "-" if v is None else f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"


def corpus_handles(corpora: list[str], *, names: dict[str, str] | None = None) -> list[Any]:
    from matplotlib.lines import Line2D

    nm = names or NAME
    return [
        Line2D([], [], linestyle="none", label=nm[c], markersize=4.2, markeredgecolor=S.WHITE,
               markeredgewidth=0.5, markerfacecolor=CORPUS_STYLE[c]["color"], marker=CORPUS_STYLE[c]["marker"])
        for c in corpora
    ]  # fmt: skip


def save(fig: Any, out: Path, name: str) -> list[Path]:
    paths = S.save(fig, Path(out) / name)
    S.close(fig)
    for p in paths:
        print(f"wrote {p}")
    return paths
