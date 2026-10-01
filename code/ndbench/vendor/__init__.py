"""ndbench vendor package.

Byte-identical copies of external/legacy source needed to build and judge ndbench suites
(nd_verify, take-home scripts, the frozen absgpt model/tokenizer, the lengthexp sampler/ndcheck/
common helpers, and lean_seq). See ``README.md`` and ``SHA256SUMS`` in this directory for the
provenance and integrity check of every vendored file. Nothing in this package is written by
ndbench; treat every file here (other than this ``__init__.py``) as frozen input.

Two loading strategies are used, matching the plan (§1: "Loading happens via the sys.path shim
or importlib unique names"):

* Take-home scripts (``vendor/takehome/*.py``) and the lengthexp helpers directly under
  ``vendor/`` (``ndcheck.py``, ``sampler.py``, ``common.py``) keep their own bare imports
  (``from nd_verify import verify_text``, ``from gen import fstr``, ``from prune import
  pruned_length``, ...). :func:`ensure_takehome_on_path` idempotently prepends this directory and
  ``vendor/takehome`` onto ``sys.path`` so those imports resolve to the vendored copies here.
* ``vendor/absgpt/{model,tokenizer}.py`` use generic names ("model", "tokenizer") that could
  collide with other, differently-vendored modules of the same name elsewhere in this monorepo.
  Instead of adding ``vendor/absgpt`` to ``sys.path``, :func:`load_absgpt_module` loads each file
  under a private ``sys.modules["ndbench_absgpt_<name>"]`` entry, only transiently aliasing a
  dependency's bare name while a sibling module that needs it (e.g. ``sample.py``'s
  ``from model import rope_cache``) is itself being loaded.
* ``vendor/danstate/`` (Dan Pandori's ``nd-takehome`` state-env modules) uses the same
  private-``sys.modules`` strategy as ``absgpt``, plus a ``sys.path`` snapshot/restore for the
  files that insert their own directory onto ``sys.path`` at module level. See
  :func:`load_danstate` and ``danstate_loader.py``'s own module docstring for the full loading
  strategy and the (documented, non-vendored) functions that are forbidden to call directly.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

from .danstate_loader import DanState, load_danstate

VENDOR_DIR = Path(__file__).resolve().parent
TAKEHOME_DIR = VENDOR_DIR / "takehome"
ABSGPT_DIR = VENDOR_DIR / "absgpt"
LEAN_SEQ_DIR = VENDOR_DIR / "lean_seq"
DANSTATE_DIR = VENDOR_DIR / "danstate"

__all__ = [
    "VENDOR_DIR",
    "TAKEHOME_DIR",
    "ABSGPT_DIR",
    "LEAN_SEQ_DIR",
    "DANSTATE_DIR",
    "ensure_takehome_on_path",
    "load_nd_verify",
    "load_ndcheck",
    "DanState",
    "load_danstate",
    "load_absgpt_module",
]


def ensure_takehome_on_path() -> None:
    """Idempotently prepend ``vendor/`` and ``vendor/takehome/`` onto ``sys.path``.

    Vendored take-home scripts (``minlen.py``, ``gen.py``, ...) and the lengthexp helpers
    (``ndcheck.py``, ``sampler.py``, ``common.py``) use bare imports such as
    ``from nd_verify import verify_text`` or ``from gen import fstr``. Both directories need to be
    on ``sys.path`` for those to resolve to the copies vendored here rather than to nothing (or,
    worse, to a same-named module from an unrelated part of this monorepo that happens to be on
    ``sys.path`` first).
    """
    for p in (str(VENDOR_DIR), str(TAKEHOME_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_nd_verify() -> types.ModuleType:
    """Import the vendored ``nd_verify`` package, ensuring ``vendor/`` is searched first.

    Safe to call repeatedly; ``importlib.import_module`` returns the cached module after the
    first call.
    """
    ensure_takehome_on_path()
    return importlib.import_module("nd_verify")


def load_ndcheck() -> types.ModuleType:
    """Import the vendored lengthexp ``ndcheck`` module.

    Imports :func:`load_nd_verify` first so ``nd_verify`` is already cached in ``sys.modules``
    from *this* vendor tree before ``ndcheck.py``'s own module-level
    ``sys.path.insert(0, '/scratch/.../ab3/frozen/lib')`` would otherwise resolve
    ``from nd_verify import ...`` against the live, read-only ``ab3`` tree instead. The two copies
    are byte-identical (see ``README.md``), but importing ours first keeps this hermetic.
    """
    load_nd_verify()
    ensure_takehome_on_path()
    return importlib.import_module("ndcheck")


def load_absgpt_module(
    name: str, *, bare_deps: tuple[str, ...] = (), persist: bool = False
) -> types.ModuleType:
    """Load ``vendor/absgpt/<name>.py`` under ``sys.modules["ndbench_absgpt_<name>"]``.

    Args:
        name: module basename under ``vendor/absgpt`` (e.g. ``"model"``, ``"tokenizer"``,
            ``"sample"``, ``"keys"``, ``"prune"``, ``"normalize"``).
        bare_deps: short names of other absgpt modules (already loaded via this function) that
            ``name``'s own top-level statements import bare, e.g. ``sample.py``'s
            ``from model import rope_cache`` needs ``bare_deps=("model",)``. Each dependency's
            bare name is installed in ``sys.modules`` only for the duration of this call (unless
            ``persist=True``) and removed afterwards if this call installed it.
        persist: keep the ``bare_deps`` aliases installed after this call returns. Needed only if
            the caller later invokes a *function* inside the loaded module (not a top-level
            statement) that itself does a bare import of a dependency, e.g. ``model.py``'s
            ``load_ckpt()`` (``from tokenizer import Tokenizer``) or ``tokenizer.py``'s
            ``selftest()`` (``from nd_verify import verify_text``); prefer calling the classes
            (``GPT``, ``Tokenizer``) directly instead, which needs no such alias.

    Returns:
        The loaded module (idempotent: a second call with the same ``name`` returns the cached
        module without re-executing it).
    """
    unique_name = f"ndbench_absgpt_{name}"
    cached = sys.modules.get(unique_name)
    if cached is not None:
        return cached

    path = ABSGPT_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no vendored absgpt module at {path}")

    installed_bare: list[str] = []
    try:
        for dep in bare_deps:
            dep_unique = f"ndbench_absgpt_{dep}"
            dep_mod = sys.modules.get(dep_unique)
            if dep_mod is None:
                raise RuntimeError(
                    f"load_absgpt_module({name!r}, bare_deps=...): dependency {dep!r} must be "
                    f"loaded (load_absgpt_module({dep!r})) before {name!r}"
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
        if not persist:
            for dep in installed_bare:
                del sys.modules[dep]
