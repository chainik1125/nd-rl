"""Shared paths and Dan's sigma classifier for the knockout / dose replication with GRPO.

Dan's create-or-elicit code (`coe/coe_sigma.py`, `coe/coe_assemble.py`, `coe/coe_dose.py`,
`patterns.py`, `gen.py`) is used unchanged from the read-only clone of `dan-pandori/nd-takehome`
in `/scratch/cowifornia/nd-rl/ext/nd-takehome-ps` (branch `dan_proof-strategies` at `e6bfe0ea`,
which carries create-or-elicit's `coe/` tree; env `ND_PS_FORK`). Its modules use bare imports
(`patterns`, `gen`), so `dan()` puts the clone on `sys.path` once, in this process only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FORK = Path(os.environ.get("ND_PS_FORK", "/scratch/cowifornia/nd-rl/ext/nd-takehome-ps"))
FORK_COMMIT = "e6bfe0ea"
OUT = Path(os.environ.get("KO_OUT", "/scratch/cowifornia/nd-rl/knockout_grpo"))
THINK = Path("/scratch/cowifornia/nd-rl/think")
RUNS = THINK / "runs"
TRACES = THINK / "traces/search/full"
K12 = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
A1 = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_depth3_f0_a1.jsonl")
DAN_DATA = OUT / "dan_data"  # read-only copies of Dan's bucket files (fetch_dan.sh)
COE = FORK / "data/coe"
STRICT106 = COE / "sigma_heldout_strict.txt"
TRACK832 = COE / "transfer_track832.jsonl"


def dan() -> Any:
    """Dan's modules: returns a namespace with coe_sigma, coe_assemble, patterns, gen."""
    if str(FORK) not in sys.path:
        sys.path.insert(0, str(FORK))
    import types

    import gen  # type: ignore[import-not-found]
    import patterns  # type: ignore[import-not-found]
    from coe import coe_assemble, coe_sigma  # type: ignore[import-not-found]

    return types.SimpleNamespace(
        coe_sigma=coe_sigma, coe_assemble=coe_assemble, patterns=patterns, gen=gen
    )


def opener(fn: str | Path) -> Any:
    fn = str(fn)
    return gzip.open(fn, "rt") if fn.endswith(".gz") else open(fn)


def jsonl(fn: str | Path) -> Iterator[dict[str, Any]]:
    with opener(fn) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256(fn: str | Path) -> str:
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def label(proof: str) -> dict[str, Any]:
    """Dan's flags for one ND proof string: sigma (his dose predicate: `sigma_strict` in the
    written or pruned form, coe_dose.py), sigma_any (`coe_sigma.has_sigma`: strict or chain),
    sigma_plus (`coe_assemble.knock`: the knockout predicate, DN citing a derived line, in either
    form) and lemma_route (`coe_assemble.lemma_route`, written form)."""
    D = dan()
    fl = D.coe_sigma.flags(proof)
    if fl is None:
        return {"parse": False}
    drop, _ = D.coe_assemble.knock(proof)
    lines = D.patterns.parse(proof)
    return {
        "parse": True,
        "sigma": bool(fl["w"]["sigma_strict"] or fl["p"]["sigma_strict"]),
        "sigma_any": bool(D.coe_sigma.has_sigma(fl)),
        "sigma_plus": bool(drop),
        "lemma_route": bool(D.coe_assemble.lemma_route(lines)),
        "dn_any": bool(fl["w"]["dn_any"]),
    }
