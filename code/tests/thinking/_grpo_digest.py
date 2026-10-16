"""Digests of tiny-model GRPO runs, to pin the default path (no `--auto_levels`) to a commit.

`run_digest(tmp, monkeypatch, config)` runs `grpo.run` as `test_grpo_guard.Go` does (a tiny effort
model, budgets 0 and 16, an <eos>-biased init, the content-hash judge) with one of `CONFIGS`, and
returns a sha256 per part: every sampler result (`rollouts`), the metrics rows without their
timings (`rows`), the printed lines with the timings masked (`lines`), the weights of every
checkpoint written (`weights`), and args.json without the options added since (`NEW_ARGS`) and
its `_meta` (`args`). Paths are written relative to the run's directory. Torch runs
single-threaded meanwhile, so the floats do not depend on the thread count.

The golden values in test_grpo_auto.py were computed with this module against a clean export of
e027354 (`git archive e027354 code`), loaded by path as `code.tests.thinking._grpo_digest`, so it
only uses what e027354 already had:

    PYTHONPATH=EXPORT python DRIVER   (DRIVER: importlib.util.spec_from_file_location on this file)
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo

from ._pipeline_helpers import bias_logits, isolate
from .test_grpo_guard import Go, _crc_judge

NEW_ARGS = ("auto_levels", "choice_lambda")  # options added after e027354 (with their defaults)
TIMING = ("secs", "secs_sample", "attempts_per_s")
TIMED_LINE = re.compile(r"\d+s \(\d+ att/s sampling\)")


def _guarded(iso: Path) -> list[str]:
    """The launcher's safeguards on: canary, KL anchor, monitor, rollbacks with the restore check
    (the <eos>-biased tiny model raises the think ban alarm every iteration), --overlong fail."""
    ref = iso / "g" / "ckpts" / "g_it0.pt"  # the effort expansion of the init, written first
    return [
        *("--iters", "3", "--monitor", "--overlong", "fail"),
        *("--canary", str(iso / "canary.jsonl"), "--canary_n", "4"),
        *("--kl_anchor_coef", "0.5", "--kl_anchor_ref", str(ref)),
        *("--rollback_on_alarm", "--save_rolling", "1", "--ban_restore_check", "--warmup", "1"),
    ]


CONFIGS: dict[str, Callable[[Path], list[str]]] = {
    "guarded": _guarded,
    "verifier_stop": lambda iso: [
        *("--iters", "2", "--arm", "logical", "--max_rej", "0", "--rej_penalty", "0.1"),
        "--monitor",
    ],
    "verifier_redraw": lambda iso: [
        *("--iters", "2", "--arm", "logical", "--max_rej", "3", "--rej_penalty", "0.25"),
        "--monitor",
    ],
}


def _hex(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    if isinstance(x, dict):
        return {str(k): _hex(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_hex(v) for v in x]
    return x


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def setup_go(tmp: Path, mp: pytest.MonkeyPatch) -> Go:
    """test_grpo_guard's `go` fixture as a function: an isolated `Go` on an <eos>-biased init
    with the content-hash judge."""
    iso = isolate(tmp, mp)
    real_load = grpo.load_ckpt

    def eos_biased_load(path: Any, device: str = "cpu") -> Any:
        m, tok, x = real_load(path, device)
        b = torch.zeros(m.cfg["vocab"])
        b[tok.eos] = 3.0
        return bias_logits(m, b), tok, x

    mp.setattr(grpo, "judge", _crc_judge)
    mp.setattr(grpo, "load_ckpt", eos_biased_load)
    return Go(iso)


def run_digest(tmp: Path, mp: pytest.MonkeyPatch, config: str) -> dict[str, str]:
    go = setup_go(tmp, mp)
    iso = go.iso
    real_gen = grpo.think_generate
    rolls: list[list[dict[str, Any]]] = []

    def spy(*args: Any, **kw: Any) -> Any:
        res = real_gen(*args, **kw)
        rolls.append(res)
        return res

    mp.setattr(grpo, "think_generate", spy)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _, out = go(CONFIGS[config](iso))
    finally:
        torch.set_num_threads(threads)
    root = str(iso)
    rows = []
    for line in (out / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        for k in TIMING:
            r.pop(k, None)
        if isinstance(r.get("canary"), dict):
            r["canary"].pop("secs", None)
        rows.append(r)
    args = json.loads((out / "args.json").read_text())
    for k in (*NEW_ARGS, "_meta"):  # _meta: the time, host and commit of the run
        args.pop(k, None)
    weights = hashlib.sha256()
    for p in sorted((out / "ckpts").glob("*.pt")):
        weights.update(p.name.encode())
        st = torch.load(str(p), map_location="cpu", weights_only=False)["state"]
        for k in sorted(st):
            weights.update(k.encode())
            weights.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    lines = TIMED_LINE.sub("<T>", buf.getvalue()).replace(root, "<ISO>")
    return {
        "rollouts": _sha(json.dumps(_hex(rolls), sort_keys=True)),
        "rows": _sha(json.dumps(rows, sort_keys=True).replace(root, "<ISO>")),
        "lines": _sha(lines),
        "weights": weights.hexdigest(),
        "args": _sha(json.dumps(args, sort_keys=True).replace(root, "<ISO>")),
    }
