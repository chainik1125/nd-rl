"""Shared paths, Dan's classifier loader and the Lean -> ND conversion for this experiment.

Dan's proof-strategy classifier (taxonomy v1.2) is used unchanged from a read-only clone of his
fork, `dan-pandori/nd-takehome` branch `dan_proof-strategies` at `e6bfe0ea` (env `ND_PS_FORK`).
Its files use bare imports (`patterns`, `nd_verify`), so `dan_strategies()` puts the clone on
`sys.path` once, in this process only; the Lean -> ND step uses the vendored fork copy in
`code/thinking/vendor/danstate` (`code.thinking.dan.load()`, private module names), so the two
trees never share a bare module name.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FORK = Path(os.environ.get("ND_PS_FORK", "/scratch/cowifornia/nd-rl/ext/nd-takehome-ps"))
FORK_COMMIT = "e6bfe0ea"
OUT = Path(os.environ.get("ND_PS_OUT", "/scratch/cowifornia/nd-rl/proof_strategies_grpo_20261101"))
RUNS = Path("/scratch/cowifornia/nd-rl/think/runs")
V2 = RUNS / "eval/v2"
TRACES = Path("/scratch/cowifornia/nd-rl/think/traces/search/full")
K12 = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
COMPUTE = REPO / "code/experiments/current/high_rl_passk_20261030/lowrl_compute.tsv"
CHARLES_CLEAN = RUNS / "pools/charles_1002/charles_clean.jsonl"
TB72 = [
    Path(
        "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72/textbook_dev.jsonl"
    ),
    Path(
        "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72/textbook_train.jsonl"
    ),
]

#: Checkpoints read on charles149 (k1024) and t1320s250 (k512), seed 0 unless noted, with their
#: eval stem. Order: increasing R (the RL / Stage-1 FLOP ratio of `lowrl_compute.tsv`).
CHAIN = [
    "s1g3",
    "grpo_think_vs_mini20_kp16_L2st1",
    "grpo_think_vs_mini20_kp16_L2st2",
    "grpo_think_vs_mini20_kp16_L2st4",
    "grpo_think_vs_mini20_kp16_L2st8",
    "grpo_think_vs_mini20_kp16_L2st16",
    "grpo_think_vs_mini20_kdense_L2st1",
    "grpo_think_vs_mini20_kp16_L2st32",
    "grpo_think_vs_mini20_kdense_L2st2",
    "grpo_think_vs_mini20_kdense_L2st4",
    "grpo_think_vs_mini20_L2it2",
    "grpo_think_vs_mini20_kdense_L2st8",
    "grpo_think_vs_mini20_L2it4",
    "grpo_think_vs_mini20_L2it8",
    "grpo_think_vs_L2",
    "grpo_think_vs_mb_kkl0",
    "grpo_think_vs_mb_ki78",
]
END = "grpo_think_vs_mb_kkl0"  # the recipe's L3 end
SHORT = {
    "s1g3": "base",
    "grpo_think_vs_L2": "L2 end",
    "grpo_think_vs_mb_kkl0": "L3 end",
    "grpo_think_vs_mb_ki78": "L3 x2 (ki78)",
}


def short(stem: str) -> str:
    if stem in SHORT:
        return SHORT[stem]
    return stem.replace("grpo_think_vs_", "")


def read_path(st: str, stem: str, seed: int, k: int) -> Path:
    return V2 / f"{st}_{stem}_s{seed}_plain_k{k}" / "rows_plain_B64.jsonl"


def compute_R() -> dict[str, float]:
    """eval stem -> R (RL FLOPs / Stage-1 FLOPs); the base is 0."""
    out = {"s1g3": 0.0}
    lines = COMPUTE.read_text().splitlines()
    head = lines[0].split("\t")
    for ln in lines[1:]:
        r = dict(zip(head, ln.split("\t"), strict=True))
        out[r["checkpoint"]] = float(r["R"])
    return out


def names(path: Path) -> set[str]:
    with open(path) as f:
        return {json.loads(x)["name"] for x in f if x.strip()}


def opener(path: Path | str, mode: str = "rt") -> Any:
    p = str(path)
    return gzip.open(p, mode) if p.endswith(".gz") else open(p, mode)


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------ Dan's classifier
_DAN: dict[str, Any] = {}


def dan_strategies() -> Any:
    """Dan's `proof_strategies/strategies.py` (module), imported from the clone."""
    if "s" not in _DAN:
        if not (FORK / "proof_strategies/strategies.py").exists():
            raise FileNotFoundError(f"Dan's fork not found at {FORK} (set ND_PS_FORK)")
        sys.dont_write_bytecode = True
        for p in (str(FORK / "proof_strategies"), str(FORK)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import strategies  # type: ignore[import-not-found]

        _DAN["s"] = strategies
    return _DAN["s"]


def dan_theorem_level() -> Any:
    """Dan's G4ip decision procedure (`intuit.prove`) and `theorem_level` helpers."""
    if "t" not in _DAN:
        dan_strategies()
        import theorem_level  # type: ignore[import-not-found]

        _DAN["t"] = theorem_level
    return _DAN["t"]


def reviewer_cls() -> Any:
    """The reviewer's blind reimplementation (`review_ps/cls.py`)."""
    if "r" not in _DAN:
        sys.dont_write_bytecode = True
        p = str(FORK / "review_ps")
        if p not in sys.path:
            sys.path.insert(0, p)
        import cls  # type: ignore[import-not-found]

        _DAN["r"] = cls
    return _DAN["r"]


# ------------------------------------------------------------------------------ Lean -> ND
GLUED = re.compile(r"^([nh]\d+)(\.1|\.2|\.elim)$")
_ASCII = {
    "~": "¬",
    "&": "∧",
    "v": "∨",
    ">": "→",
    "F": "False",
    "<|": "⟨",
    "|>": "⟩",
    "|-": "⊢",
}
_VEND: dict[str, Any] = {}


def _vendor() -> Any:
    if "D" not in _VEND:
        from code.thinking.dan import load

        _VEND["D"] = load()
    return _VEND["D"]


def lean_tokens(text: str) -> list[str]:
    """Lean text of a proof (unicode, or the traces' ASCII surface) -> fork tokens."""
    out = []
    for raw in text.split():
        m = GLUED.match(raw)
        parts = [m.group(1), m.group(2)] if m else [raw]
        out += [_ASCII.get(p, p) for p in parts]
    return out


def lean_to_nd(text: str) -> str:
    """The fork's `lean_tok.inverse`: Lean text -> ND body 'N1 ... ; QED'. Raises on a parse
    failure (`ParseFail`)."""
    return str(_vendor().lean_tok.inverse(lean_tokens(text)))


def replay_actions(prompt: str, actions: list[str]) -> str:
    """Replay a trace's actions in the fork's state environment -> the assembled Lean text."""
    D = _vendor()
    env = D.state_env.Env(prompt, canon=True, base=0, assign=False)
    for a in actions:
        ok, why = env.apply(lean_tokens(a))
        if not ok:
            raise ValueError(f"replay failed: {why}")
    if not env.done:
        raise ValueError("actions do not finish the proof")
    return str(D.lean_tok.LeanTokenizer("lean_staten").text(env.text))


def nd_to_lean_text(nd: str) -> str:
    """ND body -> the Lean tactic text the models write (`lean_tok.proof_tokens`, lean_seq
    names)."""
    D = _vendor()
    order: dict[int, int] = {}
    toks = []
    for t in D.lean_tok.proof_tokens(nd):
        if isinstance(t, tuple):
            toks.append(f"n{order.setdefault(t[1], len(order) + 1)}")
        else:
            toks.append(t)
    return str(D.lean_tok.LeanTokenizer("lean_staten").text(toks))


def lean_statement(prompt: str) -> str:
    D = _vendor()
    return str(D.lean_tok.LeanTokenizer("lean_staten").statement(prompt))


# ------------------------------------------------------------------------------ reads
def iter_rows(path: Path, keep: set[str] | None = None) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if keep is None or r["name"] in keep:
                yield r


def merged_accepted(row: dict[str, Any]) -> tuple[dict[str, int], int]:
    """A read row's accepted {Lean text: count} -> ({ND body: count}, n unparsable). Texts that
    differ only in hypothesis names become the same ND body."""
    out: collections.Counter[str] = collections.Counter()
    bad = 0
    for text, c in row["accepted"].items():
        try:
            out[lean_to_nd(text)] += int(c)
        except Exception:
            bad += int(c)
    return dict(out), bad
