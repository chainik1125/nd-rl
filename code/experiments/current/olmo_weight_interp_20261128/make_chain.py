#!/usr/bin/env python3
"""Write the pod chains of the weight-interpolation run (format of metrics 2's pod/m2_chain.sh).

    python3 make_chain.py --pod-start-utc 2026-11-28T12:00:00Z [--cap-usd 21] [--mode lanes]
        -> <out>/chain_prep.txt and <out>/chain_cells.txt  (chain_interp.txt with --mode sequential)

One job per line: `<job> <timeout-s> <continue-codes> [--nogpu] -- <command>`. A chain stops at the
first job whose exit code is not in its list; a job whose state/DONE exists is skipped on a re-run.

lanes (default; audit finding 2): two chains run side by side.
  prep   (CPU and disk only)  both downloads with sha256, the header census and config check, then
                              the three writes with their checks, in the order alpha 0.5, 0.75, 0.25
  cells  (the GPU)            the alpha 1.0 cell on the stored s2300 weights, then for alpha 0.5,
                              0.75, 0.25: sample, delete. Each cell first waits for the prep job it
                              needs (pod/wi_cell.sh), so the GPU is never idle during a write.
  Up to three merged checkpoints can exist at once: 129 + 3 x 64.5 = 322 GB on the 500 GB volume.
  The merged checkpoints live under this run's own folder (/workspace/wi/models); the two downloads
  are where Dan's fetch script puts them (/workspace/models).
sequential: one chain, one merged checkpoint at a time (write, sample, delete, next); every write
  is behind the ledger gate too, because the GPU waits while it runs.

Stop rules written into the continue lists:
  downloads, preflight, writes, the alpha 1.0 cell   only 0 continues
  a merged cell                                      0, and 124 (cut by its own 90-minute timeout:
                                                     rows kept and graded, the chain goes on)
  a delete step                                      0, and 11 (cell not complete: checkpoint kept)
Every cell carries the ledger gate with all four numbers spelled out: it is skipped (exit 9, the
chain stops) if the pod's cost so far plus a 90-minute cell would pass the cap.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ORDER = ("a100", "a050", "a075", "a025")
RATE = 5.29
CELL_TIMEOUT = 5400  # the sampler's own timeout (PREREG: 90 minutes)
WAIT_S = 3600  # longest a cell waits for its write (lanes)
WRITE_TIMEOUT = 5400
WRITE_USD = 1.32  # sequential mode: 15 minutes of idle GPU per write and check
LINE_RE = re.compile(r"^[A-Za-z0-9_.-]+\s+[0-9]+\s+[0-9]+(,[0-9]+)*(\s+--nogpu)?\s+--\s+\S")


def _ledger_env(start: str, cap: float, prior: float, cell_timeout: int, prep_chain: str) -> str:
    cell_usd = round(cell_timeout / 3600 * RATE, 2)
    return (
        f"env WI_POD_START_UTC={start} WI_CAP_USD={cap:g} WI_RATE={RATE:g} WI_PRIOR_USD={prior:g} "
        f"WI_CELL_USD={cell_usd:g} WI_CELL_TIMEOUT={cell_timeout} WI_PREP_CHAIN={prep_chain} "
        f"WI_WAIT_S={WAIT_S}"
    )


def chains(
    pod_start_utc: str,
    cap_usd: float = 21.0,
    prior_usd: float = 0.0,
    mode: str = "lanes",
    cell_timeout: int = CELL_TIMEOUT,
    threads: int | None = None,
) -> dict[str, list[str]]:
    """{chain name: lines}. lanes -> prep and cells; sequential -> interp."""
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", pod_start_utc):
        raise SystemExit("--pod-start-utc must look like 2026-11-28T12:00:00Z")
    if mode not in ("lanes", "sequential"):
        raise SystemExit("--mode is lanes or sequential")
    threads = threads or (8 if mode == "lanes" else 16)
    prep_name = "prep" if mode == "lanes" else "interp"
    env = _ledger_env(pod_start_utc, cap_usd, prior_usd, cell_timeout, prep_name)
    job_to = cell_timeout + 900 + (WAIT_S if mode == "lanes" else 0)
    # records whether the folder existed before this run, then metrics 2's download with sha256
    dl = "bash pod/wi_download.sh"
    gate = (
        f"python3 tools/ledger_ok.py --start-utc {pod_start_utc} --rate {RATE:g} "
        f"--prior {prior_usd:g} --cap {cap_usd:g} "
        f"--cell-usd {round(cell_timeout / 3600 * RATE + WRITE_USD, 2):g} && "
    )
    prep = [
        f"download_s2300 3600 0 --nogpu -- {dl} s2300",
        f"download_dpo 3600 0 --nogpu -- {dl} dpo",
        f"preflight 900 0 --nogpu -- python3 interp_write.py preflight --threads {threads}",
    ]
    writes, cells = {}, {}
    for i, tag in enumerate(ORDER):
        codes = "0" if tag == "a100" else "0,124"
        cells[tag] = [f"cell_{tag} {job_to} {codes} -- {env} bash pod/wi_cell.sh {tag}"]
        if tag == "a100":
            continue
        # the passes that do not depend on alpha (bf16 steps, alpha 1 identity) run once, with the
        # last write: nothing waits for that one, while the alpha 0.5 cell waits for the first
        census = "" if i == len(ORDER) - 1 else " --no-census"
        cmd = f"python3 interp_write.py write --tag {tag} --threads {threads}{census}"
        if mode == "sequential":
            cmd = gate + cmd
        writes[tag] = f"write_{tag} {WRITE_TIMEOUT} 0 --nogpu -- {cmd}"
        cells[tag].append(f"rm_{tag} 900 0,11 -- bash pod/wi_rm.sh {tag}")
    if mode == "lanes":
        out = {
            "prep": prep + [writes[t] for t in ORDER[1:]],
            "cells": [ln for t in ORDER for ln in cells[t]],
        }
    else:
        seq = prep + cells["a100"]
        for t in ORDER[1:]:
            seq += [writes[t], *cells[t]]
        out = {"interp": seq}
    assert all(LINE_RE.match(ln) for lines in out.values() for ln in lines)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pod-start-utc", required=True, help="the pod's creation time (ledger gate)")
    ap.add_argument("--cap-usd", type=float, default=21.0)
    ap.add_argument("--prior-usd", type=float, default=0.0)
    ap.add_argument("--mode", choices=["lanes", "sequential"], default="lanes")
    ap.add_argument("--out", default="/scratch/cowifornia/nd-rl/olmo_elicit/interp/chains")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, lines in chains(a.pod_start_utc, a.cap_usd, a.prior_usd, a.mode).items():
        head = (
            f"# Weight-interpolation chain `{name}` (make_chain.py --mode {a.mode}; PREREG.md). "
            f"pod start {a.pod_start_utc}, ask before ${a.cap_usd:g}."
        )
        f = out / f"chain_{name}.txt"
        f.write_text("\n".join([head, *lines]) + "\n")
        print(f"# ---- {f}")
        print(f.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
