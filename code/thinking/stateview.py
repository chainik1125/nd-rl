"""State formats: how a proof state is rendered for the model.

    flat       the fork's `Env.state_tokens()`: every open frame's hypothesis lines, then `⊢` and
               the innermost frame's goal only --
               `<st> [frame-0 hyps] [frame-1 hyps] ... [frame-k hyps] ⊢ G_k <act>`
    goalstack  the same tokens with each enclosing frame's goal after that frame's hypotheses --
               `<st> [frame-0 hyps] ⊢ G0 <nl> [frame-1 hyps] ⊢ G1 <nl> ... [frame-k hyps]
               ⊢ G_k <act>`

A frame's goal is what it is proving: frame 0 the theorem's conclusion, an implication box its
consequent, a negation box `False`, an Or.elim branch the `have`'s formula.  With one frame the two
formats are identical.  Only existing tokens are used; the layout is unambiguous: hypothesis lines
start with a name and goal lines with `⊢`, every line but the last ends in `<nl>`, so a `⊢ ... <nl>`
line closes a frame and `⊢ ... <act>` is the focused goal (frame 0 may have no hypotheses; a box
always has its binder).

One format per process.  `set_state_format(fmt)` selects it and refuses a second, different one
(`StateFormatError`), so a checkpoint is never trained, validated or sampled in a format other than
its own.  For `goalstack` it also installs the renderer as `Env.state_tokens` on the vendored class
(the file is unchanged), so every caller picks it up: the sampler (`sample.think_generate`, through
`render`), the pair builders (`data.plain_pairs` via the fork's `decompose`, `data.trace_pairs`),
held-out validation (`data.heldout_records`), `rootplan` and the fork's own samplers.  Where the
format comes from:
  * datasets: `python -m code.thinking.stateview data --state-format goalstack -- <data args>`
    sets it, runs `code.thinking.data.main` unchanged and records `state_format` in the pickle and
    its report (`config.state_format`); a pickle without it is `flat`;
  * training: `train.run` sets it from the pickle before the held-out set is rendered and writes it
    into the checkpoint's extra;
  * checkpoints: `ckpt.load_ckpt` sets it from the extra (absent: `flat`) and `ckpt.save_ckpt`
    records the process's format when the extra has none, so eval, expert iteration and GRPO (whose
    round checkpoints carry the loaded extra) follow the checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .dan import load

D = load()
FORMATS = ("flat", "goalstack")
DEFAULT = "flat"
_FLAT = D.state_env.Env.state_tokens  # the fork's renderer, kept before any patch
_ftoks = D.state_env.ftoks
_state: dict[str, str | None] = {"fmt": None}  # None: never set in this process (renders flat)


class StateFormatError(RuntimeError):
    """A second, different state format in one process."""


def flat_tokens(env: Any) -> list[str]:
    """The fork's `Env.state_tokens` (unpatched)."""
    return _FLAT(env)


def goalstack_tokens(env: Any) -> list[str]:
    """`<st> [frame-0 hyps] ⊢ G0 <nl> ... [frame-k hyps] ⊢ G_k <act>` (see the module docstring)."""
    out = ["<st>"]
    last = len(env.frames) - 1
    for i, fr in enumerate(env.frames):
        for line in fr.htoks:
            out += line
        out += ["⊢", *_ftoks(fr.goal), "<nl>" if i < last else "<act>"]
    return out


RENDERERS = {"flat": flat_tokens, "goalstack": goalstack_tokens}


def _patched_state_tokens(self: Any) -> list[str]:
    return goalstack_tokens(self)


def check_format(fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"unknown state format {fmt!r} (one of {FORMATS})")
    return fmt


def current() -> str:
    """The process's state format (`flat` until one is set)."""
    return _state["fmt"] or DEFAULT


def is_set() -> bool:
    return _state["fmt"] is not None


def render(env: Any, fmt: str | None = None) -> list[str]:
    """The state tokens of `env` in `fmt` (default: the process's format)."""
    return RENDERERS[check_format(fmt or current())](env)


def _install(fmt: str) -> None:
    D.state_env.Env.state_tokens = _patched_state_tokens if fmt == "goalstack" else _FLAT


def set_state_format(fmt: str) -> str:
    """Select the process's state format; a different one already set raises StateFormatError."""
    check_format(fmt)
    cur = _state["fmt"]
    if cur is not None and cur != fmt:
        raise StateFormatError(
            f"state format {cur!r} is already in use in this process; refusing {fmt!r} "
            "(one format per process: train, validate and sample a checkpoint in its own format)"
        )
    _state["fmt"] = fmt
    _install(fmt)
    return fmt


def reset_state_format() -> None:
    """Back to 'never set' (flat, unpatched).  Tests only: production code never switches."""
    _state["fmt"] = None
    _install(DEFAULT)


@contextlib.contextmanager
def state_format(fmt: str) -> Iterator[str]:
    """Tests: run a block under `fmt` from a clean slate, then restore what was there."""
    saved = _state["fmt"]
    reset_state_format()
    try:
        yield set_state_format(fmt)
    finally:
        reset_state_format()
        if saved is not None:
            set_state_format(saved)


def data_format(data: Mapping[str, Any]) -> str:
    """A pair pickle's state format (`stateview data` records it; absent: flat)."""
    fmt = data.get("state_format") or data.get("config", {}).get("state_format") or DEFAULT
    return check_format(str(fmt))


def ckpt_format(extra: Mapping[str, Any]) -> str:
    """A checkpoint extra's state format (absent: flat)."""
    return check_format(str(extra.get("state_format") or DEFAULT))


def annotate(data: dict[str, Any], fmt: str) -> dict[str, Any]:
    """The pickle dict with its state format recorded (top level and in the config, so the report
    JSON shows it)."""
    return {**data, "state_format": fmt, "config": {**data["config"], "state_format": fmt}}


# ------------------------------------------------------------------------------------- the wrapper
def run_data(fmt: str, argv: list[str]) -> int:
    """`code.thinking.data.main(argv)` under `fmt`, the pickle annotated with it."""
    from . import data as dmod

    set_state_format(fmt)
    orig = dmod.save

    def save(d: dict[str, Any], out: str | Path) -> Path:
        return orig(annotate(d, fmt), out)

    dmod.save = save
    try:
        rc = dmod.main(argv)
    finally:
        dmod.save = orig
    if rc:
        return rc
    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--out")
    path = Path(out.parse_known_args(argv)[0].out)
    rp = path.with_suffix(path.suffix + ".report.json")
    if json.loads(rp.read_text())["config"].get("state_format") != fmt:  # data.main saved otherwise
        orig(annotate(dmod.load_pairs(path), fmt), path)
    print(f"state format {fmt} recorded in {path} and {rp}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    head, rest = argv, []
    if "--" in argv:
        k = argv.index("--")
        head, rest = argv[:k], argv[k + 1 :]
    ap = argparse.ArgumentParser(
        prog="python -m code.thinking.stateview",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="usage: stateview data --state-format goalstack -- <code.thinking.data arguments>",
    )
    ap.add_argument("command", choices=["data"])
    ap.add_argument("--state-format", dest="state_format", choices=FORMATS, required=True)
    a = ap.parse_args(head)
    return run_data(a.state_format, rest)


if __name__ == "__main__":
    sys.exit(main())
