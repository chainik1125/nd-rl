"""Loads the vendored fork modules (`vendor/danstate/`, Dan's `nd-takehome` at 4dcd0e4f) under
private names.

The fork's files use bare imports with generic names (`import record`, `from model import GPT`,
`from nd_verify.verify import ...`), several of them inside function bodies (`model.save_ckpt` ->
`record`, `model.load_ckpt` -> `tokenizer`, `lean_gate.Gate` -> `lean_prefilter` / `lean_judge`).
Putting the directory on `sys.path` would leak those names into the process and collide with the
other vendored trees here (ndbench's take-home and danstate copies, r1_cot's vendor on the test
path).  So, as `ndbench/vendor/danstate_loader.py` does:

* `load()` executes every file once under `sys.modules["thinking_dan_<name>"]`, aliasing the bare
  names of its dependencies only while it runs, then restores `sys.modules` and `sys.path` (the
  files insert their own directory);
* `bare()` installs the bare aliases for the duration of a `with` block, for calls that import
  lazily.  Every call into the fork that can reach a lazy import goes through it (`ckpt.save_ckpt`,
  `lean.gate`, ...).

Bytecode is never written into the vendored directory, and the registry's row file never defaults
into it (`REGISTRY_FALLBACK`).  `trace_dsl()` loads the experiment's validator against these same
modules.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

VENDOR = Path(__file__).resolve().parent / "vendor" / "danstate"
COMMIT = "4dcd0e4f"
TRACE_DSL = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "current"
    / "thinking_traces_20261005"
    / "trace_dsl.py"
)

#: Bare module names in a topological order of the module-level import graph
#: (see `vendor_danstate.FILES`).
ORDER: tuple[str, ...] = (
    "nd_verify",
    "nd2lean",
    "lean_tok",
    "model",
    "tokenizer",
    "best_model",
    "record",
    "sample",
    "state_env",
    "state_sample",
    "state_train",
    "state_train_best",
    "lean_prefilter",
    "lean_judge",
    "lean_gate",
    "step_check",
    "guided_sample",
    "guided_eval",
)
PREFIX = "thinking_dan_"
#: Where registry rows go when ND_REGISTRY_DIR is unset (the fork's default is
#: `<record.py dir>/artifacts/<run>/registry`, i.e. inside this package).
REGISTRY_FALLBACK = os.path.join(tempfile.gettempdir(), "thinking_registry")


class Dan:
    """Namespace of the loaded modules: `D.state_env.Env`, `D.best_model.ALiBiGPT`, ..."""

    nd_verify: Any
    nd2lean: Any
    lean_tok: Any
    model: Any
    tokenizer: Any
    best_model: Any
    record: Any
    sample: Any
    state_env: Any
    state_sample: Any
    state_train: Any
    state_train_best: Any
    lean_prefilter: Any
    lean_judge: Any
    lean_gate: Any
    step_check: Any
    guided_sample: Any
    guided_eval: Any

    def __init__(self, modules: dict[str, types.ModuleType]) -> None:
        self.modules = modules
        for k, v in modules.items():
            if "." not in k:
                setattr(self, k, v)


_LOADED: dict[str, Dan] = {}
_depth = [0]


@contextlib.contextmanager
def _isolated(extra: dict[str, types.ModuleType]) -> Iterator[None]:
    """Install `extra` as bare `sys.modules` entries; afterwards restore them and `sys.path`."""
    saved = {k: sys.modules.get(k) for k in extra}
    path = list(sys.path)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules.update(extra)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.path[:] = path
        sys.dont_write_bytecode = bytecode


def _exec(unique: str, path: Path, package: bool = False) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        unique, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[unique]
        raise
    return mod


def load() -> Dan:
    """Load every vendored module once (idempotent) and return the namespace."""
    if "D" in _LOADED:
        return _LOADED["D"]
    mods: dict[str, types.ModuleType] = {}
    before = set(sys.modules)
    with _isolated({}):
        for name in ORDER:
            with _isolated(mods):
                if name == "nd_verify":
                    pkg = PREFIX + "nd_verify"
                    # the package's `from .verify import ...` resolves to <pkg>.verify
                    mods["nd_verify"] = _exec(
                        pkg, VENDOR / "nd_verify" / "__init__.py", package=True
                    )
                    mods["nd_verify.verify"] = sys.modules[pkg + ".verify"]
                else:
                    mods[name] = _exec(PREFIX + name, VENDOR / f"{name}.py")
    leaked = [
        k
        for k in set(sys.modules) - before
        if not k.startswith(PREFIX)
        and str(getattr(sys.modules[k], "__file__", "")).startswith(str(VENDOR))
    ]
    for k in leaked:
        del sys.modules[k]
    if leaked:
        raise RuntimeError(
            f"vendored modules loaded under bare names (fix ORDER): {sorted(leaked)}"
        )
    _guard_registry(mods["record"])
    _LOADED["D"] = Dan(mods)
    return _LOADED["D"]


def _guard_registry(rec: Any) -> None:
    """Keep the fork's registry rows out of the code tree: when ND_REGISTRY_DIR is unset at the
    moment the row file is chosen (e.g. a compute block closed at exit after a test restored the
    environment), use REGISTRY_FALLBACK instead of `<vendor dir>/artifacts`."""
    orig = rec._file

    def _file() -> str:
        if rec._state["file"] is None and not os.environ.get("ND_REGISTRY_DIR"):
            os.environ["ND_REGISTRY_DIR"] = REGISTRY_FALLBACK
            try:
                return orig()
            finally:
                os.environ.pop("ND_REGISTRY_DIR", None)
        return orig()

    rec._file = _file


@contextlib.contextmanager
def bare() -> Iterator[Dan]:
    """The bare names (`record`, `lean_gate`, `nd_verify`, ...) resolve to the vendored modules
    inside the block.  Re-entrant; restores whatever `sys.modules` held before on exit."""
    d = load()
    if _depth[0]:
        _depth[0] += 1
        try:
            yield d
        finally:
            _depth[0] -= 1
        return
    _depth[0] = 1
    try:
        with _isolated(dict(d.modules)):
            yield d
    finally:
        _depth[0] = 0


_TRACE: list[Any] = []


def trace_dsl() -> Any:
    """The experiment's validator (`trace_dsl.py`), loaded once under `thinking_trace_dsl` with its
    bare imports bound to the vendored modules (never to a clone, whatever ND_TAKEHOME says).  Its
    validate(lean=True) imports `lean_gate` lazily: call it inside `bare()`."""
    if _TRACE:
        return _TRACE[0]
    with bare():
        env = os.environ.get("ND_TAKEHOME")
        os.environ["ND_TAKEHOME"] = str(VENDOR)
        try:
            mod = _exec("thinking_trace_dsl", TRACE_DSL)
        finally:
            if env is None:
                os.environ.pop("ND_TAKEHOME", None)
            else:
                os.environ["ND_TAKEHOME"] = env
    _TRACE.append(mod)
    return mod
