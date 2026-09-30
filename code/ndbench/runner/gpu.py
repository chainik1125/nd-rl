"""GPU leasing and budget guarding for the ndbench benchmark run (plan §1/§3 WP1, §A2, §4).

Only GPU 4 may ever be used by any ndbench job (GPUs 0-3 belong to another user on this shared
host and must never be touched); `lease` enforces this both before and after talking to the shared
`gpulock.py` (`/scratch/cowifornia/nd-rl/ab2/frozen/gpulock.py`, read-only, called as a subprocess
per plan §3 WP1 -- never imported directly, since it is a script rather than a package, and calling
it out-of-process is also what keeps its detached FIFO-queue holder process working exactly as
`ab3/lengthexp/evaluate.py`'s own `Lease` wrapper relies on).

**Ordering constraint / the only supported way to launch a GPU job (fix list item 8):**
`CUDA_VISIBLE_DEVICES` only takes effect for a CUDA context initialised *after* it is set; once any
CUDA call has happened in this process (`torch.cuda.is_available()` counts as a *check*, not an
init, but e.g. moving a tensor `.cuda()` does), changing the environment variable no longer does
anything. Because of that, `lease` does **not** set `CUDA_VISIBLE_DEVICES` itself -- it used to, but
"after acquiring the lock" can easily be after some other import has already touched torch/CUDA in
this process, at which point setting the variable silently does nothing and the process could end
up seeing every GPU on the host. Instead, `lease` *requires* `CUDA_VISIBLE_DEVICES` to already equal
`"4"` in the environment when it is entered, and refuses (`RuntimeError`) otherwise, before ever
calling `gpulock`. The only supported way to run a GPU job is therefore to set it for the whole
process from the start, e.g. `CUDA_VISIBLE_DEVICES=4 python -m code.ndbench run --device cuda ...`
(`cli.py`'s `_guard_cuda_device` enforces the same thing at the CLI layer, before an adapter module
that imports torch is ever resolved) -- never to import torch and then hope `lease` fixes up the
environment in time. `lease` still warns (`RuntimeWarning`, not a hard failure -- see `lease`'s own
docstring for why) if it detects `torch.cuda` was already initialised in this process, as a defence
in depth against a context having been created against some other, unintended device set of
visible devices before this env-var precondition was satisfied.

`AB_GPULOCK_DIR` (read directly by `gpulock.py`, not by this module) overrides the lock directory;
tests set it to a temporary directory so they never touch the real, shared lock files.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator

from ..core.io import utc_now_iso
from ..store import Store

#: The vendored FIFO GPU lock, called as a subprocess (see module docstring). Read-only; not
#: vendored into `code/ndbench/vendor` since WP1 calls it as an external tool, not a library.
GPULOCK_SCRIPT = "/scratch/cowifornia/nd-rl/ab2/frozen/gpulock.py"

#: The only GPU any ndbench job may lease (plan §A2: "GPUs 0-4 belong to another user: never touch
#: them" other than this one).
LEASED_GPU = 4


class BudgetExceeded(Exception):
    """Raised by `budget_guard` when leasing would exceed the GPU-hour cap (plan §4)."""


def budget_guard(store: Store, est_s: float, cap_h: float = 9.5) -> float:
    """Refuse (raise `BudgetExceeded`) a job estimated at `est_s` seconds if `spent + 1.5*est`
    would exceed `cap_h` hours of ledgered GPU wall time (plan §4's stop rule: "continue only if
    spent + 1.5*estimate <= 9.5h"). Returns the remaining budget in hours if the job is allowed.

    Independent of `lease`: a caller is expected to call `budget_guard` for its next job *before*
    entering `lease` for that job (plan §4's schedule loop), not the other way around -- `lease`
    itself does not call this, since a caller may want to check the budget without necessarily
    leasing immediately (e.g. `python -m code.ndbench budget --est-h ...`, WP0's `cmd_budget`).
    """
    spent_h = store.ledger_total_s() / 3600.0
    est_h = est_s / 3600.0
    if spent_h + 1.5 * est_h > cap_h:
        raise BudgetExceeded(
            f"budget_guard: spent {spent_h:.3f}h + 1.5*est {est_h:.3f}h "
            f"would exceed cap {cap_h:.3f}h"
        )
    return cap_h - spent_h


@contextlib.contextmanager
def lease(
    job: str,
    est_s: float,
    store: Store,
    *,
    gpu: int = LEASED_GPU,
    wait_s: float = 0.0,
    gpulock_script: str = GPULOCK_SCRIPT,
) -> Iterator[int]:
    """Context manager: verify `CUDA_VISIBLE_DEVICES` is already `"4"` in the environment, acquire
    `gpu` (must be `LEASED_GPU`, i.e. 4) through `gpulock.py`, yield the acquired GPU index, then
    release and append one ledger entry to `store` (plan §1/§3 WP1) regardless of how the `with`
    block exits.

        # CUDA_VISIBLE_DEVICES=4 python -m code.ndbench ...  (set for the whole process, first)
        with gpu.lease("G0_throughput", est_s=480, store=store) as gpu_index:
            assert gpu_index == 4
            ...  # import torch only here, or in a subprocess launched from here

    Raises `ValueError` immediately (before touching any lock file) if `gpu != LEASED_GPU`, and
    `RuntimeError` if `CUDA_VISIBLE_DEVICES` is not already `"4"` (fix list item 8: `lease` no
    longer sets it itself -- see the module docstring) or if the `gpulock acquire` subprocess
    fails/times out. Raises `AssertionError` if `gpulock` itself somehow reports acquiring a
    different GPU -- both this and the `ValueError` above are defence in depth for the hard "GPU 4
    only" invariant, not expected to normally trigger. If this process has already initialised a
    CUDA context (see the module docstring's ordering constraint), emits a `RuntimeWarning` rather
    than raising: by this point `CUDA_VISIBLE_DEVICES` was already required to be `"4"` before any
    CUDA init could plausibly have happened correctly, but this is a best-effort, not fully
    reliable check -- e.g. `torch.optim.Optimizer.step()` (observed with `AdamW`, this torch build,
    this host's 4 visible CUDA devices) lazily initialises a CUDA context to probe device
    capabilities even for an optimizer stepping purely CPU tensors, so a process that merely
    trained something on CPU before leasing would otherwise be flagged despite never having
    touched a GPU.
    """
    if gpu != LEASED_GPU:
        raise ValueError(
            f"gpu.lease: only GPU {LEASED_GPU} may be leased (got gpu={gpu}); "
            "GPUs 0-3 belong to another user and must never be touched"
        )
    current_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current_env != str(LEASED_GPU):
        raise RuntimeError(
            f"gpu.lease: CUDA_VISIBLE_DEVICES must already be {str(LEASED_GPU)!r} when lease() is "
            f"entered (got {current_env!r}); set it for the whole process from the start (e.g. "
            f"`CUDA_VISIBLE_DEVICES={LEASED_GPU} python -m code.ndbench ...`), before torch (or "
            "anything that imports it) is imported -- lease() no longer sets it itself, since "
            "doing so after the lock is acquired can already be too late for it to take effect "
            "(see this module's docstring)."
        )
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(torch_mod, "cuda", None) is not None:
        try:
            already_initialised = bool(torch_mod.cuda.is_initialized())
        except Exception:  # pragma: no cover - defensive only
            already_initialised = False
        if already_initialised:
            warnings.warn(
                "gpu.lease: torch's CUDA context is already initialised in this process; "
                "CUDA_VISIBLE_DEVICES must be set before any CUDA call to take effect, ideally "
                "before `import torch` at all (see gpu.py's module docstring). This may be a false "
                "alarm (e.g. a CPU-only optimizer step can trigger it) but if this process really "
                "did touch a GPU already, prefer running this lease in a fresh process.",
                RuntimeWarning,
                stacklevel=2,
            )

    owner = os.getpid()
    proc = subprocess.run(
        [
            sys.executable,
            gpulock_script,
            "acquire",
            "--gpus",
            str(gpu),
            "--owner",
            str(owner),
            "--run",
            job,
            "--wait",
            str(wait_s),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gpu.lease: gpulock acquire failed for job {job!r}: "
            f"{proc.stderr.strip() or proc.stdout.strip() or f'exit {proc.returncode}'}"
        )
    acquired = int(proc.stdout.strip())
    assert acquired == LEASED_GPU, (
        f"gpu.lease: expected GPU {LEASED_GPU}, gpulock returned {acquired}"
    )

    # Deliberately does not set `os.environ["CUDA_VISIBLE_DEVICES"]` here (fix list item 8): the
    # precondition check above already requires it to be "4" for the whole process, which is the
    # only way this is reliable (see the module docstring's ordering constraint).
    t0 = time.monotonic()
    start_utc = utc_now_iso()
    status = "ok"
    try:
        yield acquired
    except BaseException:
        status = "error"
        raise
    finally:
        wall_s = time.monotonic() - t0
        rel = subprocess.run(
            [sys.executable, gpulock_script, "release", str(acquired)],
            capture_output=True,
            text=True,
            check=False,
        )
        store.ledger_append(
            {
                "job": job,
                "gpu": acquired,
                "start_utc": start_utc,
                "end_utc": utc_now_iso(),
                "gpu_s": wall_s,
                "est_s": est_s,
                "status": status,
                "released_ok": rel.returncode == 0,
            }
        )
