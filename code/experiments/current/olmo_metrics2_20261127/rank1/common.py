# ruff: noqa: E501
"""Shared pieces of the M2 (rank1) tooling: tensor names, seeds, shard readers, graded-row readers, bootstrap.

Design: discourse/rank1.md Turn 5 section B. CPU-testable on tiny shapes; the pod runs the same code on the real shards.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np

# 7 matrix types per layer (the 448 = 64 x 7) and the (out, in) shapes from config.json (hidden 5120, 40/8 heads of 128,
# MLP 27648). Used only for names and the real-model shape table; tests pass their own sizes.
MATRIX_TYPES = [
    ("self_attn.q_proj", "q"),
    ("self_attn.k_proj", "k"),
    ("self_attn.v_proj", "v"),
    ("self_attn.o_proj", "o"),
    ("mlp.gate_proj", "gate"),
    ("mlp.up_proj", "up"),
    ("mlp.down_proj", "down"),
]
REAL_SHAPES = {
    "self_attn.q_proj": (5120, 5120),
    "self_attn.k_proj": (1024, 5120),
    "self_attn.v_proj": (1024, 5120),
    "self_attn.o_proj": (5120, 5120),
    "mlp.gate_proj": (27648, 5120),
    "mlp.up_proj": (27648, 5120),
    "mlp.down_proj": (5120, 27648),
}
N_LAYERS = 64
GAMMAS_REPORT = (3.17, 3.5, 4.6)
SEEDS = {
    "A1": 710001,
    "A1u": 710002,
    "A2": 710003,
    "X_dev_g46": 710010,
    "X_dev_g35": 710011,
    "X_held": 710012,
}
DELTAS = {"d100": ("dpo", "s0100"), "d500": ("dpo", "s0500"), "d2300": ("dpo", "s2300")}


def matrix_names(n_layers: int = N_LAYERS) -> list[str]:
    """HF tensor names of the matrices, layer-major (e.g. model.layers.0.self_attn.q_proj.weight)."""
    return [f"model.layers.{i}.{t}.weight" for i in range(n_layers) for t, _ in MATRIX_TYPES]


def module_of(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")]


def peft_key(tensor_name: str, which: str) -> str:
    """PEFT key of a matrix's LoRA factor: which in {A, B}."""
    return f"base_model.model.{module_of(tensor_name)}.lora_{which}.weight"


def tensor_seed(global_seed: int, name: str) -> int:
    """Per-tensor seed: global seed + crc32 of the tensor name (stable across runs and machines)."""
    return (global_seed + zlib.crc32(name.encode())) % (2**63)


def random_pair(m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A2 (Turn 6 D8): g then h from ONE numpy default_rng(seed) stream, fp64 standard normals, each normalised (uniform on
    the sphere). seed = 710003 + crc32(tensor name)."""
    rng = np.random.default_rng(seed)
    g, h = rng.standard_normal(m), rng.standard_normal(n)
    return g / np.linalg.norm(g), h / np.linalg.norm(h)


def sha256_file(p, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------- safetensors checkpoints
class Checkpoint:
    """A directory with model.safetensors.index.json (or a single model.safetensors): get(name) -> torch tensor."""

    def __init__(self, d):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.dir = Path(d)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.index = json.loads(idx.read_text())
            self.wmap = self.index["weight_map"]
        else:
            self.index = None
            self.wmap = None
        self._open: dict[str, object] = {}

    def _f(self, shard: str):
        if shard not in self._open:
            self._open[shard] = self._safe_open(str(self.dir / shard), framework="pt", device="cpu")
        return self._open[shard]

    def names(self) -> list[str]:
        if self.wmap is not None:
            return list(self.wmap)
        return list(self._f("model.safetensors").keys())

    def shard_of(self, name: str) -> str:
        return self.wmap[name] if self.wmap is not None else "model.safetensors"

    def get(self, name: str):
        return self._f(self.shard_of(name)).get_tensor(name)


# --------------------------------------------------------------------------------------- graded rows and references
def read_jsonl(p) -> list[dict]:
    op = gzip.open if str(p).endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for ln in f:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    break  # a killed job may leave a truncated last line
    return out


def read_pids(p) -> list[str]:
    return [ln.strip() for ln in open(p) if ln.strip()]


def graded_by_pid(rows: list[dict], pids: list[str] | None = None) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for r in rows:
        if pids is None or r["pid"] in pids:
            d.setdefault(r["pid"], []).append(r)
    return d


def prompt_rates(
    rows: list[dict], pids: list[str], planned_per_prompt: int | None = None
) -> dict[str, float]:
    """Per-prompt strict pass share. Without planned_per_prompt: over the prompt's rows. With it (a cell of ours, one draw
    per request): passes / planned draws, so killed (missing) draws score as fails; the all-pass bound is in missing_bound()."""
    by = graded_by_pid(rows, set(pids))  # type: ignore[arg-type]
    if planned_per_prompt:
        return {p: sum(bool(r["strict"]) for r in by.get(p, [])) / planned_per_prompt for p in pids}
    return {p: float(np.mean([bool(r["strict"]) for r in v])) for p, v in by.items() if v}


def missing_bound(rows: list[dict], pids: list[str], planned_per_prompt: int) -> dict[str, float]:
    """Upper bound of the per-prompt rate if every missing (killed) draw had passed."""
    by = graded_by_pid(rows, set(pids))  # type: ignore[arg-type]
    out = {}
    for p in pids:
        v = by.get(p, [])
        miss = max(planned_per_prompt - len(v), 0)
        out[p] = (sum(bool(r["strict"]) for r in v) + miss) / planned_per_prompt
    return out


def collapse_stats(rows: list[dict], planned_draws: int) -> dict:
    """Watchdog bookkeeping (Turn 6 D1): with --chunk 1 a row is one draw. A draw is 'bad' if it has no row (killed in
    flight), hit the cap (finish == 'length'), or has no end-of-thinking token; collapsed = |union| / planned > 5 %."""
    n = len(rows)
    missing = max(planned_draws - n, 0)
    capped = sum(1 for r in rows if r.get("finish") == "length")
    no_end = sum(1 for r in rows if not r.get("think_end", True))
    union_rows = sum(1 for r in rows if r.get("finish") == "length" or not r.get("think_end", True))
    bad = missing + union_rows
    frac = bad / planned_draws if planned_draws else float("nan")
    return {
        "planned": planned_draws,
        "rows": n,
        "missing": missing,
        "capped": capped,
        "no_end": no_end,
        "bad_union": bad,
        "frac": frac,
        "collapsed": bool(frac > 0.05),
    }


def boot_ci(x: np.ndarray, lo: float = 2.5, hi: float = 97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def nan_safe_ratio(num: float, den: float) -> float:
    return num / den if den and not math.isclose(den, 0.0) else float("nan")
