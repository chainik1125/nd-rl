"""`AbsGptAdapter`: the `ModelAdapter` for the frozen absgpt decoder-only GPT (plan §1/§3 WP1).

Built the way `/scratch/cowifornia/nd-rl/ab3/lengthexp/evaluate.py:load_model` does: default arch
config when the checkpoint carries none, a `_orig_mod.`/`module.` (compile/DDP) prefix stripped
from every state-dict key, `strict=True` load. Deliberately does **not** call the vendored
`vendor/absgpt/model.py:load_ckpt` (plan §3 WP1's explicit instruction) -- that function does its
own bare `from tokenizer import Tokenizer` import, which is exactly what `vendor/__init__.py`'s
`load_absgpt_module` architecture is designed to avoid needing.

Sampling uses the vendored *compacting* sampler's `_gen` (`vendor/sampler.py`, ~1.35x faster than
`vendor/absgpt/sample.py:generate`) directly on explicit token-id lists, not `sampler.sample_ids`
(which only accepts prompt strings and cannot express a forced prefix or a forced start index).
One call to `_gen` always samples `n` copies of *one* `SampleRequest`'s exact input ids (prompt,
plus any forced prefix), chunked to `batch` rows per `_gen` call and sharing one seeded
`torch.Generator` across chunks -- the same chunking/seeding discipline `sampler.sample_ids` uses
for its (differently-shaped) multi-prompt batches.

Start-index handling (plan §3 WP1, and see `ab3/lengthexp/evaluate.py`/`vendor/ndcheck.py`'s
`start`/`index_overflow` handling, which is what a caller needs `Verdict.start` for downstream):
the `abs` tokenizer's training-time random-shift augmentation means a freshly sampled proof can
legitimately start at any `N<k>`, so `sample()` always decodes the *raw* model output into
`Attempt.native` first (preserving the original start index verbatim) and then canonicalises
`Attempt.body` to start at `N1` via `core.proof.renumber` (a no-op if it already does).
`options={"start_index": 1}` instead *forces* the very first generated token to be the `N1` index
token (by prepending that one synthetic id to the input, exactly like a one-token forced prefix),
which avoids the N64 overflow entirely for a long proof (plan §2 B2) rather than merely relabelling
it after the fact. A `SampleRequest.prefix` takes priority over `start_index` when both are given.

`logprob`/`Learner.nll` marginalise over every start-shift encoding by log-sum-exp, matching
`/scratch/cowifornia/nd-rl/external/nd-takehome/novelty.py`'s `score()` (T=1, proof tokens
including `QED`, conditioned on the prompt); `Learner.update` instead trains on a small number of
*randomly* shifted copies per example (cheap Monte-Carlo estimate of the same objective, the same
way the checkpoints being evaluated were themselves trained with `Tokenizer.shift_abs`).

Interp hooks (plan §A3): `activations` captures the residual stream after each requested block via
a `register_forward_hook`; `with_steering` returns a new adapter (over a deep-copied model, so the
original adapter's forward passes are never affected) whose blocks carry additional hooks that add
`alpha * vector` to the residual stream at the token positions `where` selects. Since a `_gen`
generation step only ever sees the newly produced tokens (the rest lives in the KV cache), the
hook recovers each call's absolute token positions from that block's own cache dict (`cache['n']`
is the running length *after* this call's tokens were appended, updated identically by every
block, so `cache['n'] - T .. cache['n']` is exactly this call's token range) rather than needing
`_gen`'s otherwise-private `pos` tensor; the non-cached forward passes used by `logprob`/`nll`/
`activations`/`Learner.update` instead use plain `arange(T)` positions. Either way, "prompt" vs.
"proof" is decided against `self._steer_prompt_len`, an attribute the calling method sets just
before invoking the model and clears just after.
"""

from __future__ import annotations

import copy
import importlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.io import sha256_file, sha256_str
from ..core.proof import renumber
from ..core.types import JSON, Attempt, LogProb, SampleRequest, Sequent
from ..vendor import ensure_takehome_on_path, load_absgpt_module
from .base import Capability, ModelAdapter, SteeringSpec

# ---------------------------------------------------------------- vendor loading (module-load time)

_MODEL_MOD = load_absgpt_module("model")
_TOK_MOD = load_absgpt_module("tokenizer")
GPT = _MODEL_MOD.GPT
Tokenizer = _TOK_MOD.Tokenizer
MAXN: int = _TOK_MOD.MAXN

# `vendor/sampler.py` (the lengthexp "compacting" sampler) lives directly under `vendor/`, not
# under `vendor/absgpt/`, and keeps its own bare imports (`import torch, ...`; no bare
# `from model import ...`/`from tokenizer import ...`), so it is loaded via the plain
# `ensure_takehome_on_path` + `importlib` shim (strategy 1 of `vendor/README.md#loading`), the same
# way `ndcheck.py`/`common.py` are -- not via `load_absgpt_module`'s per-module namespacing (that
# strategy exists specifically for `model.py`/`tokenizer.py`'s generic names).
ensure_takehome_on_path()
_SAMPLER_MOD = importlib.import_module("sampler")
_GEN = _SAMPLER_MOD._gen


def _build_model_and_tokenizer(
    ckpt_path: str | Path, device: torch.device
) -> tuple[Any, Any, dict[str, int]]:
    """Build `(model, tokenizer, arch_cfg)` from a checkpoint the way `evaluate.py:load_model`
    does (see module docstring): default arch config, `_orig_mod.`/`module.` prefix stripped,
    `strict=True`. Never calls the vendored `model.py:load_ckpt`."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    tok = Tokenizer(ck.get("tok_mode", "abs"))
    cfg = dict(ck.get("cfg") or dict(vocab=tok.vocab_size, n_layer=4, d=256, n_head=8))
    cfg.setdefault("vocab", tok.vocab_size)
    model = GPT(**cfg)
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in ck["state"].items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tok, cfg


def _shift_marginal_logprob(
    model: Any,
    tok: Any,
    device: torch.device,
    seq: Sequent,
    body: str,
    owner: Any,
    maxn: int,
    *,
    start_excluded: bool = False,
) -> LogProb:
    """`log sum_s p(shift_s(body) | prompt)` at temperature 1, over every shift `s` the `abs`
    tokenizer's `MAXN` allows (matches `external/nd-takehome/novelty.py`'s `score()`); shared by
    `AbsGptAdapter.logprob` and `_AbsGptLearner.nll`/`nll_start_excluded` (plan §3 WP1: "nll uses
    the shift-marginal loss ..."). Returns a log-*probability* (`LogProb.nats` <= 0, typically);
    `_AbsGptLearner.nll` negates it to get a positive NLL (see `core.types.LogProb`'s docstring for
    the sign convention every caller relies on). `owner._steer_prompt_len` is set/cleared around
    each forward call so a steered adapter's hooks (if any) see the right prompt/proof split;
    unused (and harmless) otherwise.

    `start_excluded`: drop the very first proof token's log-probability from every shift's summed
    total before marginalising (plan §B7's start-token-excluded sensitivity variant; matches
    `test_wp5_fixtures.score_all_shifts(..., start_excluded=True)`), and report `n_tokens` one
    shorter to match.
    """
    pid = tok.encode_prompt(seq.prompt())
    qid = tok.encode_proof(body)
    ref0 = tok.ref0
    mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
    max_shift = max(maxn - mx, 0)
    per_shift: list[float] = []
    model.eval()
    mask_start = len(pid) if start_excluded else len(pid) - 1
    with torch.no_grad():
        for s in range(0, max_shift + 1):
            ids = pid + [x + s if x >= ref0 else x for x in qid]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            owner._steer_prompt_len = len(pid)
            try:
                logits = model(x[:, :-1]).float()
            finally:
                owner._steer_prompt_len = 0
            tgt = x[:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            mask[:, mask_start:] = True
            per_shift.append(float((logp * mask).sum().item()))
    nats = (
        float(torch.logsumexp(torch.tensor(per_shift), dim=0).item())
        if per_shift
        else float("-inf")
    )
    n_tokens = len(qid) - 1 if start_excluded else len(qid)
    return LogProb(nats=nats, n_tokens=n_tokens, per_encoding=tuple(per_shift))


def _shift_marginal_nll(
    model: Any,
    tok: Any,
    device: torch.device,
    seq: Sequent,
    body: str,
    owner: Any,
    maxn: int,
    *,
    start_excluded: bool = False,
) -> LogProb:
    """`-log sum_s p(shift_s(body) | prompt)`: the positive-NLL counterpart of
    `_shift_marginal_logprob`, for `_AbsGptLearner.nll`/`nll_start_excluded` (`Learner.nll`'s
    contract is a negative log-likelihood, the opposite sign from `ModelAdapter.logprob`; see
    `core.types.LogProb`). Only `.nats` is negated -- `.per_encoding` stays the per-shift
    log-*probabilities* `_shift_marginal_logprob` computed, matching
    `test_wp5_fixtures.TinyGptLearner.nll`'s convention."""
    lp = _shift_marginal_logprob(
        model, tok, device, seq, body, owner, maxn, start_excluded=start_excluded
    )
    return LogProb(nats=-lp.nats, n_tokens=lp.n_tokens, per_encoding=lp.per_encoding)


def _make_capture_hook(captured: dict[int, torch.Tensor], layer: int) -> Any:
    """A `register_forward_hook` callback that stores `output` into `captured[layer]`. A plain
    function factory (rather than a closure defined inline inside the `for layer in layers` loop
    in `AbsGptAdapter.activations`) so `layer`/`captured` are bound by ordinary argument passing,
    not by closing over a loop variable."""

    def _hook(module: Any, inputs: Any, output: torch.Tensor) -> None:
        captured[layer] = output.detach()

    return _hook


def _resolve_steering_vector(vector: tuple[float, ...] | str) -> list[float]:
    if isinstance(vector, str):
        arr = np.load(vector)
        return [float(v) for v in arr.tolist()]
    return list(vector)


def _steering_spec_hash(spec: SteeringSpec) -> str:
    """A stable fingerprint of `spec`, included in a steered adapter's `identity()` (plan §A3)."""
    parts = []
    for t in spec:
        if isinstance(t.vector, str):
            vec_repr = f"path:{t.vector}"
        else:
            vec_repr = ",".join(f"{v:.10g}" for v in t.vector)
        parts.append(f"{t.layer}|{vec_repr}|{t.alpha:.10g}|{t.where}")
    return sha256_str("||".join(parts))


class AbsGptAdapter(ModelAdapter):
    """`ModelAdapter` for the frozen absgpt GPT: SAMPLE, PREFIX, LOGPROB, FINETUNE, ACTIVATIONS,
    STEER (plan §1/§3 WP1, §A3)."""

    def __init__(
        self,
        ckpt: str | Path,
        device: str = "cpu",
        batch: int = 1024,
        *,
        label: str | None = None,
        compact_every: int = 16,
        compact_below: float = 0.8,
        update_shifts: int = 8,
    ) -> None:
        dev = torch.device(device)
        model, tok, cfg = _build_model_and_tokenizer(ckpt, dev)
        ckpt_path = Path(ckpt)
        ckpt_sha256 = sha256_file(ckpt_path)
        self._setup(
            model=model,
            tok=tok,
            cfg=cfg,
            device=dev,
            batch=batch,
            ckpt_path=ckpt_path,
            ckpt_sha256=ckpt_sha256,
            label=label,
            compact_every=compact_every,
            compact_below=compact_below,
            update_shifts=update_shifts,
            steering_spec=None,
            from_learner=False,
        )

    @classmethod
    def _from_model(
        cls,
        *,
        model: Any,
        tok: Any,
        cfg: dict[str, int],
        device: torch.device,
        batch: int,
        ckpt_path: Path,
        ckpt_sha256: str,
        label: str | None,
        compact_every: int,
        compact_below: float,
        update_shifts: int,
        steering_spec: SteeringSpec | None,
        from_learner: bool,
    ) -> AbsGptAdapter:
        """Build an adapter around an already-in-memory `model`/`tok` (no checkpoint load): used
        by `with_steering` (a deep-copied model with new hooks) and `_AbsGptLearner.adapter`
        (the learner's own, live, mutable model)."""
        obj = cls.__new__(cls)
        obj._setup(
            model=model,
            tok=tok,
            cfg=cfg,
            device=device,
            batch=batch,
            ckpt_path=ckpt_path,
            ckpt_sha256=ckpt_sha256,
            label=label,
            compact_every=compact_every,
            compact_below=compact_below,
            update_shifts=update_shifts,
            steering_spec=steering_spec,
            from_learner=from_learner,
        )
        return obj

    def _setup(
        self,
        *,
        model: Any,
        tok: Any,
        cfg: dict[str, int],
        device: torch.device,
        batch: int,
        ckpt_path: Path,
        ckpt_sha256: str,
        label: str | None,
        compact_every: int,
        compact_below: float,
        update_shifts: int,
        steering_spec: SteeringSpec | None,
        from_learner: bool,
    ) -> None:
        self._model = model
        self._tok = tok
        self._cfg = cfg
        self._device = device
        self._batch = batch
        self._ckpt_path = ckpt_path
        self._ckpt_sha256 = ckpt_sha256
        self._compact_every = compact_every
        self._compact_below = compact_below
        self._update_shifts = update_shifts
        self._steering_spec: SteeringSpec = steering_spec or ()
        self._from_learner = from_learner
        self._steer_prompt_len = 0

        default_label = f"absgpt:{ckpt_path.stem}"
        if from_learner:
            default_label += ":learner"
        if self._steering_spec:
            default_label += f"+steer{len(self._steering_spec)}"
        self.label = label or default_label
        self.format_id = "nd_v1"
        self.capabilities = (
            Capability.SAMPLE
            | Capability.PREFIX
            | Capability.LOGPROB
            | Capability.FINETUNE
            | Capability.ACTIVATIONS
            | Capability.STEER
        )
        self._steer_handles: list[Any] = (
            self._install_steering_hooks(self._steering_spec) if self._steering_spec else []
        )

    # ------------------------------------------------------------------- ModelAdapter contract

    def identity(self) -> dict[str, JSON]:
        d: dict[str, JSON] = {
            "adapter": "absgpt",
            "ckpt": str(self._ckpt_path),
            "ckpt_sha256": self._ckpt_sha256,
            "arch": dict(self._cfg),
            "tokenizer_mode": self._tok.mode,
            "tokenizer_vocab_sha256": sha256_str(",".join(self._tok.itos))[:16],
            "tokenizer_maxn": MAXN,
        }
        if self._from_learner:
            d["derived_from_learner"] = True
        if self._steering_spec:
            d["steering_sha256"] = _steering_spec_hash(self._steering_spec)
            d["steering_terms"] = len(self._steering_spec)
        return d

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(self._tok.encode_prompt(seq.prompt()))
        if body:
            n += len(self._tok.encode_proof(body))
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        out: list[list[Attempt]] = []
        for req in reqs:
            prompt_ids = self._tok.encode_prompt(req.sequent.prompt())
            forced_ids: list[int] = []
            if req.prefix is not None:
                forced_ids = self._tok.encode_proof(req.prefix.body)
            elif req.options.get("start_index") == 1:
                forced_ids = [self._tok._ref(1)]
            input_ids = prompt_ids + forced_ids

            continuations = self._gen_many(
                input_ids, req.n, req.temperature, req.max_new_tokens, req.seed
            )
            attempts: list[Attempt] = []
            for i, cont in enumerate(continuations):
                finished = bool(cont) and cont[-1] == self._tok.eos
                full_ids = forced_ids + cont
                native = self._tok.decode(full_ids)
                body = renumber(native, new_start=1)
                attempts.append(
                    Attempt(
                        pid=req.pid,
                        i=i,
                        body=body,
                        native=native,
                        n_tokens=len(cont),
                        finished=finished,
                    )
                )
            out.append(attempts)
        return out

    def _gen_many(
        self, input_ids: list[int], n: int, temperature: float, max_new: int, seed: int
    ) -> list[list[int]]:
        """`n` continuations of `input_ids`, chunked to `self._batch` rows per `_gen` call and
        sharing one seeded generator across chunks (matches `sampler.sample_ids`'s discipline)."""
        gen = torch.Generator(device=self._device)
        gen.manual_seed(seed)
        results: list[list[int]] = []
        remaining = n
        while remaining > 0:
            this_n = min(self._batch, remaining)
            batch_ids = [input_ids] * this_n
            cuda_enabled = self._device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda_enabled):
                outs = _GEN(
                    self._model,
                    self._tok,
                    batch_ids,
                    temperature,
                    max_new,
                    gen,
                    self._compact_every,
                    self._compact_below,
                )
            results.extend(outs)
            remaining -= this_n
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
        return results

    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        return [
            _shift_marginal_logprob(self._model, self._tok, self._device, seq, body, self, MAXN)
            for seq, body in pairs
        ]

    def learner(self, *, seed: int) -> _AbsGptLearner:
        """Narrower than `ModelAdapter.learner`'s declared `-> Learner` (a legal covariant
        override, matching `test_wp5_fixtures.TinyGptAdapter.learner`'s own pattern): lets a
        caller that holds an `AbsGptAdapter` reach `_AbsGptLearner`'s duck-typed extensions
        (`update_shifts`, `nll_start_excluded`) directly and type-checked, without `edl.run_edl`'s
        `getattr`/`setattr` fallback (needed only when the concrete `Learner` type isn't known)."""
        model_copy = copy.deepcopy(self._model)
        opt = torch.optim.AdamW(
            model_copy.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0
        )
        return _AbsGptLearner(
            model=model_copy,
            tok=self._tok,
            device=self._device,
            opt=opt,
            update_shifts=self._update_shifts,
            seed=seed,
            base_adapter=self,
        )

    def activations(
        self, pairs: Sequence[tuple[Sequent, str]], layers: Sequence[int]
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for seq, body in pairs:
            ids = self._tok.encode_prompt(seq.prompt()) + self._tok.encode_proof(body)
            x = torch.tensor([ids], dtype=torch.long, device=self._device)
            captured: dict[int, torch.Tensor] = {}
            handles = []
            for layer in layers:
                block = self._model.blocks[layer]
                handles.append(block.register_forward_hook(_make_capture_hook(captured, layer)))
            try:
                with torch.no_grad():
                    self._model(x)
            finally:
                for h in handles:
                    h.remove()
            arr = np.stack([captured[layer][0].cpu().numpy() for layer in layers], axis=0)
            out.append(arr)
        return out

    def with_steering(self, spec: SteeringSpec) -> ModelAdapter:
        model_copy = copy.deepcopy(self._model)
        return AbsGptAdapter._from_model(
            model=model_copy,
            tok=self._tok,
            cfg=self._cfg,
            device=self._device,
            batch=self._batch,
            ckpt_path=self._ckpt_path,
            ckpt_sha256=self._ckpt_sha256,
            label=None,
            compact_every=self._compact_every,
            compact_below=self._compact_below,
            update_shifts=self._update_shifts,
            steering_spec=spec,
            from_learner=self._from_learner,
        )

    def close(self) -> None:
        for h in getattr(self, "_steer_handles", ()):
            h.remove()
        self._steer_handles = []

    # ------------------------------------------------------------------------------- steering

    def _install_steering_hooks(self, spec: SteeringSpec) -> list[Any]:
        handles: list[Any] = []
        for term in spec:
            vec = _resolve_steering_vector(term.vector)
            delta = torch.tensor(vec, dtype=torch.float32, device=self._device)
            block = self._model.blocks[term.layer]
            handles.append(block.register_forward_hook(_make_steering_hook(self, term, delta)))
        return handles


def _make_steering_hook(owner: AbsGptAdapter, term: Any, delta: torch.Tensor) -> Any:
    """A `register_forward_hook` callback adding `term.alpha * delta` to the residual stream at
    the positions `term.where` selects (plan §A3). A plain function factory (rather than a closure
    defined inline inside `AbsGptAdapter._install_steering_hooks`'s `for term in spec` loop) so
    `term`/`delta` are bound by ordinary argument passing, not by closing over a loop variable.

    Reads `owner._steer_prompt_len` (set/cleared by whichever adapter method is currently driving
    the model) to split "prompt" from "proof" positions; see the module docstring for how the
    absolute token positions of one call are recovered from the block's own KV-cache dict during
    `_gen`-driven sampling, versus a plain `arange(T)` for a non-cached forward pass.
    """

    def _hook(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        cache = inputs[4] if len(inputs) > 4 else None
        _b, t, _d = output.shape
        if isinstance(cache, dict) and "n" in cache:
            end = int(cache["n"])
            abs_pos = torch.arange(end - t, end, device=output.device)
        else:
            abs_pos = torch.arange(t, device=output.device)
        prompt_len = owner._steer_prompt_len
        if term.where == "all":
            pos_mask = torch.ones(t, dtype=torch.bool, device=output.device)
        elif term.where == "prompt":
            pos_mask = abs_pos < prompt_len
        else:  # "proof"
            pos_mask = abs_pos >= prompt_len
        if not bool(pos_mask.any()):
            return output
        updated = output.clone()
        addition = term.alpha * delta.to(output.dtype)
        updated[:, pos_mask, :] = updated[:, pos_mask, :] + addition
        return updated

    return _hook


class _AbsGptLearner:
    """`Learner` for `AbsGptAdapter` (plan §1): AdamW (betas 0.9/0.95, wd 0, grad-clip 1.0) over a
    deep-copied model, trained on `update_shifts` random start-shift copies per example per call.
    """

    def __init__(
        self,
        *,
        model: Any,
        tok: Any,
        device: torch.device,
        opt: torch.optim.Optimizer,
        update_shifts: int,
        seed: int,
        base_adapter: AbsGptAdapter,
    ) -> None:
        self._model = model
        self._tok = tok
        self._device = device
        self._opt = opt
        #: Public (not `_`-prefixed): the number of independent random-shift copies `update()`
        #: makes per training example. `edl.py`'s `run_edl` reads/sets this duck-typed attribute
        #: (`getattr`/`setattr`, no `Learner`-protocol change) to make sure exactly one layer does
        #: the shift-copying -- it passes each pair to `update()` once, relying on this attribute,
        #: rather than also repeating the pair `shifts_per_update` times itself (plan §B7's "8
        #: random-shift copies per update", not 8*8).
        self.update_shifts = update_shifts
        self._rng = random.Random(seed)
        self._base_adapter = base_adapter
        self._steer_prompt_len = 0

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Shift-marginal *negative* log-likelihood at the learner's current weights (plan §3 WP1:
        "nll uses the shift-marginal loss over proof tokens including QED, conditioned on the
        prompt"; `Learner.nll`'s contract per `core.types.LogProb` is positive nats, the opposite
        sign from `ModelAdapter.logprob`)."""
        return [
            _shift_marginal_nll(self._model, self._tok, self._device, seq, body, self, MAXN)
            for seq, body in pairs
        ]

    def nll_start_excluded(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Optional sensitivity-check variant of `nll` (plan §B7; `edl.py`'s `run_edl` looks this
        up via `getattr`): the start-index token's contribution to each shift's log-probability is
        dropped before marginalising, matching `test_wp5_fixtures.TinyGptLearner.
        nll_start_excluded`."""
        return [
            _shift_marginal_nll(
                self._model, self._tok, self._device, seq, body, self, MAXN, start_excluded=True
            )
            for seq, body in pairs
        ]

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        self._model.train()
        for g in self._opt.param_groups:
            g["lr"] = lr
        ref0 = self._tok.ref0
        rows: list[list[int]] = []
        prompt_lens: list[int] = []
        for seq, body in pairs:
            pid = self._tok.encode_prompt(seq.prompt())
            qid = self._tok.encode_proof(body)
            mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
            max_shift = max(MAXN - mx, 0)
            for _ in range(self.update_shifts):
                s = self._rng.randint(0, max_shift) if max_shift > 0 else 0
                ids = pid + [x + s if x >= ref0 else x for x in qid]
                rows.append(ids)
                prompt_lens.append(len(pid))
        if not rows:
            self._model.eval()
            return 0.0
        max_len = max(len(r) for r in rows)
        x = torch.full((len(rows), max_len), self._tok.pad, dtype=torch.long, device=self._device)
        for i, r in enumerate(rows):
            x[i, : len(r)] = torch.tensor(r, device=self._device)
        logits = self._model(x[:, :-1]).float()
        tgt = x[:, 1:]
        logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
        mask = torch.zeros_like(tgt, dtype=torch.bool)
        for i, (r, lp) in enumerate(zip(rows, prompt_lens, strict=True)):
            mask[i, lp - 1 : len(r) - 1] = True
        n_tok = mask.sum().clamp(min=1)
        loss = -(logp * mask).sum() / n_tok
        pre_update_loss = float(loss.item())
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._opt.step()
        self._model.eval()
        return pre_update_loss

    def adapter(self) -> ModelAdapter:
        base = self._base_adapter
        return AbsGptAdapter._from_model(
            model=self._model,
            tok=self._tok,
            cfg=base._cfg,
            device=self._device,
            batch=base._batch,
            ckpt_path=base._ckpt_path,
            ckpt_sha256=base._ckpt_sha256,
            label=f"{base.label}:learner",
            compact_every=base._compact_every,
            compact_below=base._compact_below,
            update_shifts=base._update_shifts,
            steering_spec=None,
            from_learner=True,
        )
