"""`throughput(models, sets, batches) -> json`: samples/s and Learner-steps/s for a grid of
(checkpoint, probe set, batch size) (plan §1/§3 WP1). Wired to the WP0 CLI's `throughput`
subcommand (`cli.py:cmd_throughput`, which resolves `"code.ndbench.tools.throughput:throughput"`
and forwards its three comma-separated `--models`/`--sets`/`--batches` arguments positionally).

`models` entries are `"label=ckpt_path"` (or a bare path, labelled by its filename stem).

`sets` entries take three forms, tried in this order (fix list item 9 added the third):

1. a path to an existing JSONL file with a `"prompt"` field per row (its first few rows become the
   probe prompts, at the caller's `max_new_tokens`);
2. a bare integer string, interpreted as an approximate target proof length `L`: prompt length
   barely affects sampling throughput (the same autoregressive decoding loop runs regardless of how
   many premises a sequent has), so a tiny fixed two-premise sequent is used as the prompt and
   `max_new_tokens` is instead scaled to roughly `6*L` tokens (a rough, documented-as-approximate
   tokens-per-line budget; see plan §0's token-budget note for real per-family figures);
3. the name of a suite already built into `store` (e.g. `textbook_dev`, `gen_ctrl`, `families`):
   `store.load_suite(spec, suite_version)`'s first few `Problem`s become the probe prompts (their
   real `Sequent.prompt()`s), and `max_new_tokens` is scaled the same `6*L` way off those problems'
   own certified/reference lengths instead of a literal integer.

Every form is "good enough for a *throughput* probe, whose only job is to control how many decoding
steps `sample()` actually takes, not what gets generated or whether it is correct" -- form 3 adds
realistic prompt shapes (real premises/atoms) without needing suite-specific plumbing beyond the
`Store` `run`/`edl` already depend on. Forms 1 and 2 keep `throughput()` usable with no `Store` at
all (`store=None`), e.g. before any suite builder (WP2/WP3) has landed, which is why form 3 is tried
last and only when `store` is given.

For each (model, set, batch) cell: builds one `AbsGptAdapter` per model (reused across its sets and
batches, mutating `._batch` between cells to avoid reloading the checkpoint each time), times one
`sample()` call for `n_samples` draws per probe prompt, and -- if at least one sampled attempt's
body is a real (non-`UNPARSEABLE`) proof body -- also times a few `Learner.update` steps on that
attempt as a (`Sequent`, `body`) pair, reusing it purely to exercise the fine-tuning code path at
realistic sequence lengths (its correctness as a proof is irrelevant to a throughput measurement).
`learner_steps_per_s` is `null` when no usable attempt was available to time (e.g. every sample was
`UNPARSEABLE`, plausible only for a very undertrained/random model).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..adapters.absgpt import AbsGptAdapter
from ..core.types import JSON, Problem, SampleRequest, Sequent

if TYPE_CHECKING:
    from ..store import Store

#: Rows read from a `sets` JSONL probe file, or `Problem`s sampled from a built suite, capped at
#: this many prompts per set (throughput does not need more than a handful to get a stable
#: samples/s estimate).
_MAX_PROBE_PROMPTS = 4

#: Fallback target length (fix list item 9) for a suite-name probe set whose sampled `Problem`s
#: somehow carry no length information at all (defensive; every certified suite item has one).
_DEFAULT_PROBE_LENGTH = 10.0


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """`"label=ckpt_path"` -> `(label, ckpt_path)`; a bare path is labelled by its filename stem."""
    label, sep, path = spec.partition("=")
    if not sep:
        return Path(spec).stem, spec
    return label, path


def _approx_length(p: Problem) -> float:
    """A plotting/scaling-friendly length estimate for `p` (exact `L` if certified, else the
    tightest available bound, else the module default) -- used only to scale a throughput probe's
    `max_new_tokens`, never to report or plot an actual length."""
    length = p.length
    if length.L is not None:
        return float(length.L)
    if length.L_ub is not None:
        return float(length.L_ub)
    if length.L_lb is not None:
        return float(length.L_lb)
    return _DEFAULT_PROBE_LENGTH


def _probe(
    spec: str, max_new_tokens: int, *, store: Store | None = None, suite_version: str = "v1"
) -> tuple[list[str], int]:
    """One `sets` entry -> `(prompts, max_new_tokens)` for that probe (see module docstring's three
    forms)."""
    path = Path(spec)
    if path.is_file():
        prompts: list[str] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line)["prompt"])
                if len(prompts) >= _MAX_PROBE_PROMPTS:
                    break
        if not prompts:
            raise ValueError(
                f"throughput: probe set file {spec!r} has no rows with a 'prompt' field"
            )
        return prompts, max_new_tokens
    try:
        length = int(spec)
    except ValueError:
        length = None
    if length is not None:
        seq = Sequent(premises=("P", "Q"), conclusion="P")
        return [seq.prompt()], max(16, min(max_new_tokens, 6 * length))
    if store is None:
        raise ValueError(
            f"throughput: probe set {spec!r} is not an existing file or an integer target "
            "length, and no `store=` was given to look it up as a built suite name"
        )
    try:
        problems = store.load_suite(spec, suite_version)
    except FileNotFoundError as e:
        raise ValueError(
            f"throughput: probe set {spec!r} is not a file, an integer length, or a suite built "
            f"at version {suite_version!r} under {store.suites_dir}: {e}"
        ) from e
    if not problems:
        raise ValueError(f"throughput: suite {spec!r}/{suite_version!r} has no problems")
    sample = problems[:_MAX_PROBE_PROMPTS]
    approx_l = max(_approx_length(p) for p in sample)
    return [p.sequent.prompt() for p in sample], max(16, min(max_new_tokens, int(6 * approx_l)))


def _re_encodable(adapter: AbsGptAdapter, body: str) -> bool:
    """True iff `body` (a possibly-garbage sampled `Attempt.body`) can be fed back through the
    tokenizer's `encode_proof`. An undertrained/random model's raw output, after
    `core.proof.renumber`'s uniform shift, can contain an out-of-vocabulary `N<k>` token (`k <= 0`
    or `k > MAXN`, e.g. `"N-4"`) if its own N-tokens were not consistently offset to begin with;
    `NDJudge` handles that gracefully (`ndcheck.classify` just fails to parse the line), but
    re-encoding it as a `Learner.update` training pair would raise `KeyError`/`ValueError`. This is
    only used to pick a body to *time* `Learner.update` on (never to judge correctness), so
    skipping an unencodable one and falling back to `learner_steps_per_s: null` is harmless.
    """
    try:
        adapter._tok.encode_proof(body)
    except (KeyError, ValueError):
        return False
    return True


def throughput(
    models: Sequence[str],
    sets: Sequence[str],
    batches: Sequence[int],
    *,
    device: str | None = None,
    n_samples: int = 32,
    max_new_tokens: int = 128,
    update_pairs: int = 4,
    update_steps: int = 5,
    store: Store | None = None,
    suite_version: str = "v1",
) -> dict[str, JSON]:
    """Measure `sample()` throughput (and, opportunistically, `Learner.update` throughput) for
    every (model, set, batch) combination. `device` defaults to `"cuda"` if available else `"cpu"`
    (auto-detected at call time, so the real scheduled run -- launched with `CUDA_VISIBLE_DEVICES=4`
    already set in the environment before the process starts, and inside `runner.gpu.lease`, which
    now *requires* that rather than setting it itself (fix list item 8) -- picks up GPU 4 as plain
    `"cuda"`); pass `device="cpu"` explicitly to keep a test or a dry run off the GPU regardless of
    what else is visible on a shared host. `store`/`suite_version` (fix list item 9) let a `sets`
    entry name a suite already built into `store` instead of only a JSONL file or an integer length
    (see the module docstring's three probe-set forms).
    """
    dev = device
    if dev is None:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"

    results: list[dict[str, JSON]] = []
    for model_spec in models:
        label, ckpt_path = _parse_model_spec(model_spec)
        adapter = AbsGptAdapter(ckpt=ckpt_path, device=dev, batch=batches[0] if batches else 1024)
        try:
            for set_spec in sets:
                prompts, probe_max_new = _probe(
                    set_spec, max_new_tokens, store=store, suite_version=suite_version
                )
                seqs = [Sequent.from_prompt(p) for p in prompts]
                for batch in batches:
                    adapter._batch = batch  # same package/author as adapters/absgpt.py (WP1)
                    reqs = [
                        SampleRequest(
                            pid=f"probe{i}",
                            sequent=s,
                            n=n_samples,
                            temperature=0.8,
                            max_new_tokens=probe_max_new,
                            seed=0,
                        )
                        for i, s in enumerate(seqs)
                    ]
                    t0 = time.monotonic()
                    attempts = adapter.sample(reqs)
                    elapsed = max(time.monotonic() - t0, 1e-9)
                    n_total = sum(len(a) for a in attempts)
                    record: dict[str, JSON] = {
                        "model": label,
                        "ckpt": str(ckpt_path),
                        "set": set_spec,
                        "batch": batch,
                        "device": dev,
                        "n_samples": n_total,
                        "max_new_tokens": probe_max_new,
                        "elapsed_s": elapsed,
                        "samples_per_s": n_total / elapsed,
                        "learner_steps_per_s": None,
                    }
                    pairs = [
                        (seqs[0], a.body)
                        for a in attempts[0][:update_pairs]
                        if a.body
                        and not a.body.startswith("UNPARSEABLE")
                        and _re_encodable(adapter, a.body)
                    ]
                    if pairs:
                        learner = adapter.learner(seed=0)
                        t1 = time.monotonic()
                        for _ in range(update_steps):
                            learner.update(pairs, lr=1e-4)
                        elapsed_learn = max(time.monotonic() - t1, 1e-9)
                        record["learner_steps_per_s"] = update_steps / elapsed_learn
                    results.append(record)
        finally:
            adapter.close()
    return {"device": dev, "results": cast(list[JSON], results)}
