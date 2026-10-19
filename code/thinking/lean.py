"""Lean 4 judging of finished proofs through the fork's `lean_gate`, with a configurable binary.

The fork hardcodes `LEAN = ~/.elan/bin/lean` (lean_gate.py:26@4dcd0e4f); `_run` reads the module
global at call time, so setting it here is enough.  Order of precedence: the `lean=` argument,
$ND_LEAN, ~/.elan/bin/lean (4.34.0 here; pods run 4.34.1).  The gate's log goes to $LEAN_GATE_LOG,
which must be set (`ckpt.run_env` puts it under the run's output directory; the fork's default is
`artifacts/lean_gate.jsonl` in the working directory).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .dan import bare, load

D = load()


def lean_path(lean: str | None = None) -> str:
    return lean or os.environ.get("ND_LEAN") or os.path.expanduser("~/.elan/bin/lean")


def _setup(lean: str | None) -> None:
    if not os.environ.get("LEAN_GATE_LOG"):
        raise RuntimeError(
            "set LEAN_GATE_LOG (ckpt.run_env does): the fork's default log path is cwd-relative"
        )
    D.lean_gate.LEAN = lean_path(lean)


def gate(
    tok: Any,
    prompts: Sequence[str],
    nd_proofs: Sequence[str],
    texts: Sequence[str | None],
    lean: str | None = None,
) -> list[str]:
    """fork `lean_gate.gate`: ND strings back, clean where Lean accepted, else marked 'LEANREJ ' or
    'LEANPARSE '."""
    if not prompts:
        return []
    _setup(lean)
    with bare():
        return list(D.lean_gate.gate(tok, list(prompts), list(nd_proofs), list(texts)))


def lean_check(items: Sequence[tuple[str, str]], lean: str | None = None) -> list[bool]:
    """(statement, tactic text) pairs -> Lean's verdicts (fork `lean_gate.lean_check`)."""
    _setup(lean)
    with bare():
        ok, _, _ = D.lean_gate.lean_check(list(items))
    return [bool(x) for x in ok]
