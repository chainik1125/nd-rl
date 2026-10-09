"""The reused analysis modules, loaded by file path under unique names.

The gap-analysis and sealed-set modules use bare imports (`import common`, `import fastlen`),
and the fork has modules of the same names (`necessity`, `gen`), so importing them by bare name
picks the wrong file. Here each is executed once from its own path:

* `fastlen()`: test_21_30/fastlen.py (the fork's `minlen.Search` with a cross-budget memo);
* `gap_common()`: gap_analysis/common.py (theorem and proof features, box skeletons);
* `gap_necessity()`: gap_analysis/necessity.py (the restricted re-search).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    assert spec is not None and spec.loader is not None, rel
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def fastlen() -> ModuleType:
    return _load("h712_fastlen", "test_21_30/fastlen.py")


def gap_common() -> ModuleType:
    return _load("h712_gap_common", "gap_analysis/common.py")


def gap_necessity() -> ModuleType:
    return _load("h712_gap_necessity", "gap_analysis/necessity.py")
