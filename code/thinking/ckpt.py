"""Checkpoints: load / save in the fork's format, the run environment, and the cold-start vocabulary
expansion.

Saving goes through the fork's `model.save_ckpt` (same file format, so `load_ckpt` and every fork
loader can read a `lean_staten` checkpoint written here), but **always offline**: `save_ckpt`
uploads to Dan's HF bucket unless `ND_OFFLINE=1` (fork `model.py:333-339@4dcd0e4f`,
`record.publish_ckpt`).  `run_env(out)` forces that, and points the registry rows and the Lean gate
log under `out` (the fork's defaults are `<record.py dir>/artifacts/...` and
`artifacts/lean_gate.jsonl` relative to the working directory, i.e. a code directory).

Cold start (`expand_vocab`).  A `lean_staten` checkpoint has exactly two vocabulary-sized tensors,
`emb.weight` and `head.weight` (111 x d; no weight tying, the head has no bias, the MTP head is not
saved and reuses `emb` / `head`). Old rows move to their new ids (`tok.remap_ids`: ids < 39 stay,
h1.. and n1.. move up by 3).  The three new rows:
  * embedding: a draw from the diagonal Gaussian fitted to the old embedding rows (per-dimension
    mean and std), so the new inputs look like typical tokens to the first block and are distinct
    from one another;
  * head: the mean of the old head rows, so each new token's logit is the average old logit and the
    initial next-token distribution over old tokens is only renormalised (Hewitt 2021, "Initializing
    new word embeddings for pretrained language models").
On any input made of old tokens, the logits over old tokens are then exactly the old model's
(test_cold_start).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .dan import bare, load
from .tok import ThinkTokenizer, make_tokenizer, remap_ids

D = load()
VOCAB_TENSORS = ("emb.weight", "head.weight")


def ensure_offline() -> None:
    """Never upload: the fork's save_ckpt / registry sync go to Dan's bucket unless ND_OFFLINE=1."""
    os.environ["ND_OFFLINE"] = "1"
    os.environ["ND_REGISTRY_SYNC"] = "0"


def run_env(out: str | Path) -> dict[str, str]:
    """Process environment for an entry point writing under `out`: offline, registry rows in
    `out/registry`, Lean gate log in `out/lean_gate.jsonl`.  Returns what was set."""
    ensure_offline()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {
        "ND_OFFLINE": "1",
        "ND_REGISTRY_SYNC": "0",
        "ND_REGISTRY_DIR": str(out / "registry"),
        "LEAN_GATE_LOG": str(out / "lean_gate.jsonl"),
    }
    os.environ.update(env)
    D.record._state["file"] = None  # the row file is chosen on the first row, from ND_REGISTRY_DIR
    return env


def save_ckpt(
    path: str | Path, model: Any, tok_mode: str, extra: Mapping[str, Any] | None = None
) -> Any:
    """The fork's save_ckpt (same format; a `ckpt_saved` registry row), forced offline."""
    ensure_offline()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with bare():
        assert D.record.offline()
        return D.model.save_ckpt(str(path), model, tok_mode, extra=dict(extra or {}))


def build_model(cfg: Mapping[str, Any]) -> Any:
    cfg = dict(cfg)
    if cfg.get("arch") == "best":
        return D.best_model.ALiBiGPT(**cfg)
    return D.model.GPT(**cfg)


def load_ckpt(path: str | Path, device: str = "cpu") -> tuple[Any, Any, dict[str, Any]]:
    """fork `model.load_ckpt` (model.py:342-354@4dcd0e4f) with this package's tokenizer dispatch, so
    `lean_thinkn` checkpoints load too.  -> (model in eval mode, tokenizer, extra)."""
    ck = torch.load(str(path), map_location=device)
    tok = make_tokenizer(ck["tok_mode"])
    m = build_model(ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, tok, ck.get("extra", {})


def vocab_tensors(model: Any) -> dict[str, torch.Tensor]:
    """Every tensor with a vocabulary-sized leading dimension; checks expand_vocab's layout."""
    V = model.cfg["vocab"]
    sd = model.state_dict()
    found = {k: v for k, v in sd.items() if v.ndim >= 1 and v.shape[0] == V}
    assert set(found) == set(VOCAB_TENSORS), f"unexpected vocabulary-sized tensors: {sorted(found)}"
    assert model.emb.weight.data_ptr() != model.head.weight.data_ptr(), (
        "tied embeddings: not handled"
    )
    assert model.head.bias is None
    return found


@torch.no_grad()
def expand_vocab(model: Any, old_tok: Any, new_tok: Any = None, seed: int = 0) -> tuple[Any, Any]:
    """A copy of `model` (vocabulary `old_tok`) for `new_tok` (default lean_thinkn); see the module
    docstring."""
    new_tok = ThinkTokenizer() if new_tok is None else new_tok
    old = vocab_tensors(model)
    remap = remap_ids(old_tok, new_tok)
    added = [i for i in range(new_tok.vocab_size) if i not in set(remap)]
    cfg = dict(model.cfg, vocab=new_tok.vocab_size)
    out = build_model(cfg)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(seed)
    for k, w in old.items():
        w = w.float()
        nw = torch.empty((new_tok.vocab_size, *w.shape[1:]), dtype=w.dtype)
        nw[remap] = w
        if k == "emb.weight":
            mu, sd_ = w.mean(0), w.std(0)
            nw[added] = mu + sd_ * torch.randn((len(added), *w.shape[1:]), generator=g)
        else:
            nw[added] = w.mean(0)
        sd[k] = nw.to(model.state_dict()[k].dtype)
    out.load_state_dict(sd)
    out.to(next(model.parameters()).device)
    out.eval()
    return out, new_tok


def cold_start(
    path: str | Path, device: str = "cpu", seed: int = 0
) -> tuple[Any, Any, dict[str, Any]]:
    """A `lean_staten` Stage-1 checkpoint expanded to `lean_thinkn` -> (model, tok, info)."""
    m, tok, extra = load_ckpt(path, device)
    assert tok.mode == "lean_staten", f"cold start expects a lean_staten checkpoint, got {tok.mode}"
    new, ntok = expand_vocab(m, tok, seed=seed)
    info = {
        "init": str(path),
        "init_mode": tok.mode,
        "init_extra_steps": extra.get("steps"),
        "init_seed": seed,
    }
    return new, ntok, info
