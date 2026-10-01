"""`load_dan_checkpoint`: strict load of one of Dan Pandori's `best-state` checkpoints (plan
`DANSTATE_PLAN.md` §1/§7 WP-D0), for `adapters/danstate.py` (WP-D1, not yet written) to build on.

Mirrors `adapters/absgpt.py`'s `_build_model_and_tokenizer`: it deliberately never calls the
vendored `model.py:load_ckpt`, whose own lazy `from tokenizer import make_tokenizer` / (when
`cfg['arch'] == 'best'`) `from best_model import ALiBiGPT` bare imports are exactly what
`vendor.load_danstate`'s namespacing is designed to avoid needing -- see its module docstring
("Forbidden calls").

**Checkpoint format** (verified directly on a real file, CPU, `weights_only=True`:
`/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/stage1/stage1_best12_s0_b1200.pt`; see plan
§0): a plain dict `{"cfg": ..., "state": ..., "tok_mode": ..., "extra": ...}`, written by the
vendored `model.py:save_ckpt`.

- `cfg` is exactly `ALiBiGPT.__init__`'s own kwargs (`vocab`, `n_layer`, `d`, `n_head`, `max_len`,
  `d_ff`, `arch`). The real checkpoints all carry
  `{"vocab": 111, "n_layer": 6, "d": 384, "n_head": 8, "max_len": 1024, "d_ff": 1280, "arch":
  "best"}`, matching `best_model.py`'s own `N_LAYER, D, N_HEAD, D_FF = 6, 384, 8, 1280` defaults.
- `state` has no `_orig_mod.`/`module.` (`torch.compile`/DDP) key prefix (checked directly on the
  real checkpoint above) -- unlike `adapters/absgpt.py`'s checkpoints, no prefix stripping is
  needed here, and `load_state_dict(strict=True)` is used as-is.
- `tok_mode` is `"lean_staten"` for every real `best-state` checkpoint (arm SN: canonical,
  environment-assigned hypothesis names). `LeanTokenizer(tok_mode).canon` is `True` and
  `.vocab_size == 111` for this mode (`lean_tok.py`'s own vocabulary construction: 2 pad/eos + 11
  formula symbols + 6 prompt symbols + 16 tactic symbols + 4 state symbols (state modes only) + 8
  premise-name tokens (`MAXH`) + 64 hypothesis-name tokens (`MAXN`) = 111).
- `extra` is the training run's own metadata dict (args, loss curve, `n_params`, ...); passed
  through verbatim in `meta["extra"]`, not interpreted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core.io import sha256_file
from ..vendor import load_danstate


def load_dan_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Strictly load one `best-state` checkpoint.

    Returns `(model, tokenizer, cfg, meta)`:

    - `model`: a vendored `ALiBiGPT(**cfg)` (`best_model.py`), `load_state_dict(strict=True)`,
      `.to(device)`, `.eval()`.
    - `tokenizer`: a vendored `LeanTokenizer(tok_mode)` (`lean_tok.py`). Asserted (module
      docstring): `tok_mode == "lean_staten"`, `.canon` is `True`, `.vocab_size == 111`.
    - `cfg`: the checkpoint's own `cfg` dict (the exact kwargs `ALiBiGPT(**cfg)` was built from).
      Asserted `cfg["arch"] == "best"`.
    - `meta`: `{"tok_mode", "sha256", "n_params", "extra"}` -- `n_params` from `model.n_params()`
      (9,560,832 for every real `best-state` checkpoint, plan §0), `sha256` from
      `core.io.sha256_file`, `extra` the checkpoint's own training-run metadata dict.

    Never calls the vendored `model.py:load_ckpt`/`tokenizer.py:make_tokenizer` (module
    docstring); does its own `torch.load(..., weights_only=True)` and strict state-dict load, the
    way `adapters/absgpt.py:_build_model_and_tokenizer` does for the absgpt checkpoints.
    """
    ds = load_danstate()
    path = Path(path)
    ck = torch.load(str(path), map_location=device, weights_only=True)

    tok_mode = ck.get("tok_mode")
    if tok_mode != "lean_staten":
        raise ValueError(
            f"load_dan_checkpoint({path}): expected tok_mode 'lean_staten', got {tok_mode!r}"
        )

    tok = ds.lean_tok.LeanTokenizer(tok_mode)
    if not tok.canon:
        raise ValueError(f"load_dan_checkpoint({path}): LeanTokenizer({tok_mode!r}).canon is False")
    if tok.vocab_size != 111:
        raise ValueError(
            f"load_dan_checkpoint({path}): expected vocab_size 111, got {tok.vocab_size}"
        )

    cfg = dict(ck["cfg"])
    if cfg.get("arch") != "best":
        raise ValueError(
            f"load_dan_checkpoint({path}): expected cfg['arch'] == 'best', got {cfg.get('arch')!r}"
        )
    cfg.setdefault("vocab", tok.vocab_size)

    model = ds.best_model.ALiBiGPT(**cfg)
    model.load_state_dict(ck["state"], strict=True)
    model.to(device)
    model.eval()

    meta = {
        "tok_mode": tok_mode,
        "sha256": sha256_file(path),
        "n_params": model.n_params(),
        "extra": ck.get("extra", {}),
    }
    return model, tok, cfg, meta
