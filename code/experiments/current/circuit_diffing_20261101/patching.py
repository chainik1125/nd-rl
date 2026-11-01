"""Crosscoder patching inside `sample.think_generate`: a model wrapper that runs a driver model
(whose logits are sampled) and, on the same tokens, the other model of a crosscoder pair up to
the crosscoder's layer, then edits the driver's residual stream after that layer.

The sampler makes one KV-cache dict per entry of `model.blocks` and compacts every tensor in
those dicts when rows finish. The wrapper's `blocks` is the driver's blocks followed by the other
model's first `layer` blocks, so both models' caches live in the sampler's list and are compacted
together. Chunked prefill is switched off on both models (an instance `PREFILL_ELEMS`), so a
forward's rows always line up; use a batch of at most about 256 rows.

Edits, in raw units of the driver's residual stream (crosscoder units are divided by the
model's scale s_m; f is the crosscoder's inference-time encoding of both models' activations):
  ablate          driver RL:   h - sum_{j in S} f_j d_j^RL / s_RL
  rescale         driver base: h + sum_{j in S} f_j (r_j - 1) d_j^base / s_base, with
                  r_j = (||d_j^RL|| / s_RL) / (||d_j^base|| / s_base): shared latents set to the RL
                  model's strength along the base's own directions
  inject          driver base: h + sum_{j in S} f_j d_j^RL / s_RL
  replace         driver base: h <- the RL model's residual at that layer (the positive control)
  recon           driver RL:   h <- the crosscoder reconstruction (no error term)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .common import capture
from .crosscoder import Crosscoder

MODES = ("ablate", "rescale", "inject", "replace", "recon")


def load_cc(d: Path, device: str) -> tuple[Crosscoder, dict[str, Any]]:
    ck = torch.load(Path(d) / "crosscoder.pt", map_location=device, weights_only=False)
    meta = ck["meta"]
    n, m, dd = ck["state"]["W_dec"].shape
    cc = Crosscoder(dd, n, meta["k"], m).to(device)
    cc.load_state_dict(ck["state"])
    cc.eval()
    for p in cc.parameters():
        p.requires_grad_(False)
    return cc, meta


class PairPatched(torch.nn.Module):
    def __init__(
        self,
        base: Any,
        rl: Any,
        cc: Crosscoder,
        meta: dict[str, Any],
        mode: str,
        latents: torch.Tensor | None,
    ) -> None:
        super().__init__()
        assert mode in MODES, mode
        self.mode, self.layer = mode, int(meta["layer"])
        drv_rl = mode in ("ablate", "recon")
        self.driver, self.other = (rl, base) if drv_rl else (base, rl)
        self.drv_idx = 1 if drv_rl else 0  # crosscoder model index of the driver
        self.cc = cc
        dev = next(rl.parameters()).device
        self.scale = torch.tensor(meta["scale"], device=dev, dtype=torch.float32)
        n = cc.W_dec.shape[0]
        sel = torch.zeros(n, dtype=torch.bool, device=dev)
        if latents is not None:
            sel[latents.to(dev)] = True
        self.sel = sel
        norms = cc.dec_norms()
        self.r = (norms[:, 1] / self.scale[1]) / (norms[:, 0] / self.scale[0]).clamp_min(1e-12)
        for m in (base, rl):
            m.PREFILL_ELEMS = 2**62
        self.blocks = list(self.driver.blocks) + list(self.other.blocks[: self.layer])
        self.n_drv = len(self.driver.blocks)
        self.cfg = self.driver.cfg
        self.think_format = getattr(self.driver, "think_format", None)
        self.state_format = getattr(self.driver, "state_format", None)
        self.stats = {"tokens": 0, "edit_norm": 0.0, "h_norm": 0.0, "active_sel": 0.0}

    def parameters(self, recurse: bool = True):  # the driver's device
        return self.driver.parameters(recurse)

    def _other_hidden(self, idx, pos, mask, caches) -> torch.Tensor:
        full = self.other.blocks
        self.other.blocks = full[: self.layer]
        try:
            with capture(self.other, [self.layer]) as cap:
                self.other.features(idx, pos, mask, caches)
            return cap[self.layer]
        finally:
            self.other.blocks = full

    def forward(self, idx, pos=None, mask=None, caches=None):
        c_d = None if caches is None else caches[: self.n_drv]
        c_o = None if caches is None else caches[self.n_drv :]
        h_o = self._other_hidden(idx, pos, mask, c_o)

        def hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
            return self.edit(out, h_o)

        h = self.driver.blocks[self.layer - 1].register_forward_hook(hook)
        try:
            return self.driver(idx, pos=pos, mask=mask, caches=c_d)
        finally:
            h.remove()

    @torch.no_grad()
    def edit(self, h_d: torch.Tensor, h_o: torch.Tensor) -> torch.Tensor:
        B, T, d = h_d.shape
        hd, ho = h_d.float(), h_o.float()
        pair = (ho, hd) if self.drv_idx == 1 else (hd, ho)  # (base, rl)
        x = torch.stack(pair, 2).view(B * T, 2, d) * self.scale[None, :, None]
        if self.mode == "replace":
            new = ho
        else:
            f, _ = self.cc.encode(x)
            D = self.cc.W_dec
            fs = f * self.sel
            if self.mode == "ablate":
                new = hd - (fs @ D[:, 1]).view(B, T, d) / self.scale[1]
            elif self.mode == "inject":
                new = hd + (fs @ D[:, 1]).view(B, T, d) / self.scale[1]
            elif self.mode == "rescale":
                new = hd + ((fs * (self.r - 1)) @ D[:, 0]).view(B, T, d) / self.scale[0]
            else:  # recon
                xh = self.cc.decode(f)
                new = (xh[:, 1] / self.scale[1]).view(B, T, d)
            self.stats["active_sel"] += float((fs > 0).sum())
        self.stats["tokens"] += B * T
        self.stats["edit_norm"] += float((new - hd).norm(dim=-1).sum())
        self.stats["h_norm"] += float(hd.norm(dim=-1).sum())
        return new.to(h_d.dtype)
