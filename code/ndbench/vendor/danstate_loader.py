"""`load_danstate()`: the namespaced loader for `vendor/danstate/` (Dan Pandori's `nd-takehome`
state-env modules, pinned at commit `0c492902`; see `vendor/danstate/PROVENANCE.md` for the file
list and `tools/vendor_danstate.py` for how they got there).

Modelled directly on `load_absgpt_module` (`vendor/__init__.py`): every vendored file is loaded
under a private `sys.modules["ndbench_danstate_<name>"]` entry via
`importlib.util.spec_from_file_location`, never under its own bare name (`model`, `tokenizer`,
`sample`, `record`, `best_model`, `nd2lean`, `lean_tok`, `state_env`, `state_sample`, `lean_gate`).
Those are exactly the generic names most likely to collide with an unrelated vendored tree
elsewhere in this monorepo -- the take-home directory ships its own `model.py`/`sample.py`, and
`vendor/absgpt/` its own `model.py`/`tokenizer.py`/`sample.py` (loaded the same way, under
`ndbench_absgpt_*`); `test_dan_vendor.py` checks both loaders can run in one process without
tripping over each other.

Two wrinkles beyond `load_absgpt_module`, both specific to this fork's files:

* Five of the ten (`nd2lean.py`, `lean_tok.py`, `state_env.py`, `state_sample.py`, `lean_gate.py`)
  do their own `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` at module level,
  so the fork's own scripts still work when run directly (`python3 lean_gate.py`). That is
  exactly the kind of global, process-wide side effect this loader must not leak:
  `_load_danstate_module` snapshots `sys.path` before `exec_module` and restores it verbatim in a
  `finally`, so none of those inserted entries survive the call, whether or not it succeeds.
* `nd_verify` is a real module-level dependency (`lean_tok.py`, `nd2lean.py`, `state_env.py` all
  do `from nd_verify...import ...` at module level) but is deliberately *not* namespaced the way
  `model`/`tokenizer`/etc. are: it is loaded once, under its own bare name, by this package's
  existing `load_nd_verify()` (the convention `load_absgpt_module`'s own callers already rely on).
  `nd_verify` is specific enough not to collide, and every vendor tree in this package is meant to
  see the identical copy (plan DANSTATE_PLAN.md §0: "Dan's `nd_verify` is byte-identical to our
  vendored copy"). `load_danstate()` calls `load_nd_verify()` first and leaves it bare afterwards,
  same as `load_absgpt_module`'s own callers do -- this is not a leftover to clean up.

**Forbidden calls** -- lazy, function-body-only bare imports that the module-level dependency scan
above cannot see (vendored, unmodified; see each file's own source for the exact line):

* `model.load_ckpt`: `from tokenizer import make_tokenizer`, and (when `cfg['arch'] == 'best'`)
  `from best_model import ALiBiGPT`.
* `tokenizer.make_tokenizer` (for any `lean*` mode) / `Tokenizer.selftest`: `from lean_tok import
  LeanTokenizer`; `from nd_verify import verify_text`.
* `sample.generate`'s `early='exact'`/`'goal'` and gated branches: `from nd2lean import
  parse_prompt`, `from lean_tok import ftoks`, `from lean_gate import Gate`.
* `state_sample.env_generate(..., gate=True)`: `from lean_gate import gate`.

None of these resolve against this loader's namespaced modules, because the bare names they ask
for (`tokenizer`, `best_model`, `lean_tok`, `nd_verify`-the-function-shaped-one, `lean_gate`) are
not in `sys.modules` after a normal `load_danstate()` call. **Do not call them.** Call the
vendored *classes* directly instead -- `ds.best_model.ALiBiGPT(**cfg)`, `ds.lean_tok.LeanTokenizer(mode)`,
`ds.state_env.Env(prompt, canon=True, base=b, assign=True)`, `ds.state_sample.env_generate(...,
gate=False)` -- see `load_dan_checkpoint` (`adapters/_danstate_load.py`) for a worked example, and
`adapters/absgpt.py`'s `_build_model_and_tokenizer` for the precedent: it deliberately does not
call absgpt's own `model.py:load_ckpt` for exactly this reason.

For a caller that genuinely needs one of the forbidden functions anyway, `DanState.lazy_imports()`
is a context manager that installs the real bare aliases (every name in `DanState._NAMES`) into
`sys.modules` for the duration of the `with` block only, restoring whatever was there before
(normally nothing) on exit -- `load_absgpt_module`'s `persist=True` idea, generalised to every
vendored name at once instead of one dependency at a time. It makes `model.load_ckpt` and
`tokenizer.make_tokenizer` resolve. It does **not** make `state_sample.env_generate(gate=True)` or
`sample.generate`'s gated branch work: those also need `lean_judge.py`/`lean_prefilter.py`, which
this work package (WP-D0) does not vendor (plan §7 WP-D2's job; see `vendor/danstate/PROVENANCE.md`
"Deliberately not vendored") -- calling them, even inside `lazy_imports()`, still raises
`ModuleNotFoundError: No module named 'lean_judge'`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

DANSTATE_DIR = Path(__file__).resolve().parent / "danstate"

#: Topological load order of the ten vendored files' module-level import graph (see this module's
#: docstring and `tools/vendor_danstate.py`'s own file-list docstring for how this was derived --
#: an AST scan of module-level `Import`/`ImportFrom` nodes, not a grep). Each entry's `bare_deps`
#: must already be loaded (checked by `_load_danstate_module`).
_MODULE_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nd2lean", ()),
    ("lean_tok", ("nd2lean",)),
    ("model", ()),
    ("tokenizer", ()),
    ("best_model", ("model",)),
    ("record", ()),
    ("sample", ("record", "model")),
    ("state_env", ("nd2lean", "lean_tok")),
    ("state_sample", ("record", "sample", "state_env")),
    ("lean_gate", ()),
)


def _unique(name: str) -> str:
    return f"ndbench_danstate_{name}"


def _load_danstate_module(name: str, *, bare_deps: tuple[str, ...] = ()) -> types.ModuleType:
    """Load `vendor/danstate/<name>.py` under `sys.modules["ndbench_danstate_<name>"]`.

    Same recipe as `load_absgpt_module` (see its docstring in `vendor/__init__.py`) for the
    `bare_deps` transient-alias mechanics, plus a `sys.path` snapshot/restore (module docstring)
    for the files that insert their own directory onto `sys.path` at module level. Idempotent: a
    second call with the same `name` returns the cached module without re-executing it.
    """
    unique_name = _unique(name)
    cached = sys.modules.get(unique_name)
    if cached is not None:
        return cached

    path = DANSTATE_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no vendored danstate module at {path}")

    path_snapshot = list(sys.path)
    installed_bare: list[str] = []
    try:
        for dep in bare_deps:
            dep_unique = _unique(dep)
            dep_mod = sys.modules.get(dep_unique)
            if dep_mod is None:
                raise RuntimeError(
                    f"_load_danstate_module({name!r}, bare_deps=...): dependency {dep!r} must be "
                    f"loaded (via _load_danstate_module({dep!r})) before {name!r}"
                )
            if dep not in sys.modules:
                sys.modules[dep] = dep_mod
                installed_bare.append(dep)

        spec = importlib.util.spec_from_file_location(unique_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create an import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[unique_name]
            raise
        return module
    finally:
        for dep in installed_bare:
            del sys.modules[dep]
        sys.path[:] = path_snapshot


class DanState:
    """Namespace of the loaded vendored `danstate` modules: `ds.model`, `ds.tokenizer`,
    `ds.best_model`, `ds.lean_tok`, `ds.nd2lean`, `ds.record`, `ds.sample`, `ds.state_env`,
    `ds.state_sample`, `ds.lean_gate`. See `danstate_loader`'s module docstring for what is and is
    not safe to call directly on these.
    """

    _NAMES: tuple[str, ...] = tuple(name for name, _ in _MODULE_ORDER)

    # Declared explicitly (set dynamically in __init__ via setattr) so static analysis knows
    # about them -- `pyright`'s `reportAttributeAccessIssue` cannot see through `setattr`.
    nd2lean: types.ModuleType
    lean_tok: types.ModuleType
    model: types.ModuleType
    tokenizer: types.ModuleType
    best_model: types.ModuleType
    record: types.ModuleType
    sample: types.ModuleType
    state_env: types.ModuleType
    state_sample: types.ModuleType
    lean_gate: types.ModuleType

    def __init__(self, modules: dict[str, types.ModuleType]) -> None:
        self._modules = modules
        for name, mod in modules.items():
            setattr(self, name, mod)

    @contextlib.contextmanager
    def lazy_imports(self) -> Any:
        """Install the real bare aliases (module docstring's "forbidden calls" section) for the
        duration of the `with` block, so a function with a lazy, module-escaping bare import
        (`model.load_ckpt`, `tokenizer.make_tokenizer`, ...) can resolve it against these
        namespaced modules. Restores whatever `sys.modules` held under each bare name beforehand
        (normally nothing) on exit, even if the block raises -- the same transient-alias
        discipline `_load_danstate_module`'s own `bare_deps` uses, just held open for the caller's
        own code instead of one vendored module's top-level statements.
        """
        saved: dict[str, types.ModuleType | None] = {}
        for name in self._modules:
            saved[name] = sys.modules.get(name)
            sys.modules[name] = self._modules[name]
        try:
            yield self
        finally:
            for name, prior in saved.items():
                if prior is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior


_CACHED: DanState | None = None


def load_danstate() -> DanState:
    """Load every vendored `danstate` module and return a cached `DanState` namespace.

    Idempotent and process-wide cached: a second call returns the same object without
    re-executing anything. See this module's docstring for the loading strategy and, in
    particular, the list of functions that are forbidden to call on the returned modules without
    `DanState.lazy_imports()`.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    from . import load_nd_verify  # local import: avoids relying on import order within __init__.py

    load_nd_verify()  # ensures the bare `nd_verify` module is importable (module docstring)

    modules: dict[str, types.ModuleType] = {}
    for name, bare_deps in _MODULE_ORDER:
        modules[name] = _load_danstate_module(name, bare_deps=bare_deps)

    _CACHED = DanState(modules)
    return _CACHED
