"""`DanStateAdapter`: the `ModelAdapter` for Dan Pandori's `best-state` state-env checkpoints
(plan `experiments/current/benchmarking_20260930/DANSTATE_PLAN.md` §1/§7 WP-D1).

Where this differs from a "normal" `ModelAdapter` (`adapters/absgpt.py`'s own docstring is the
reference point throughout): the model is a *step* policy, not a whole-proof one. It reads one
tactic state (`<st> h1 : F <nl> ... ⊢ G <act>`) and writes one `lean_seq` action; an **environment**
(the vendored `vendor/danstate/state_env.py:Env`) applies that action syntactically, renders the
assembled `lean_seq` text, and (via `lean_tok.inverse`) derives the ND proof the text denotes.
Sampling a `SampleRequest` is therefore a *rollout*: repeatedly observe the current state, sample
one action from the model, apply it, until the env finishes (`exact` closes the top-level goal),
fails syntactically, hits `max_steps`, or an action itself never emits `<eos>` within `max_action`
tokens (truncated).

**`_rollout` is a line-for-line port of `vendor/danstate/state_sample.py:env_generate` (lines
~35-98)**, with exactly three changes: (1) no `lean_gate`/`record` calls (this work package does
not judge Lean -- that is WP-D2's `LeanJudge`; `record.*` is Dan's results-registry instrumentation,
irrelevant here); (2) it returns a richer `_RolloutOut` per attempt (`nd`, `lean_text`, `steps`,
`end`, `n_tokens`, `base`) instead of only the gated ND string; (3) a KV-memory guard (see below)
that can split one wave's `generate_ids_fast` call into several smaller sub-calls. Everything else
-- the worklist refill loop, `Env(prompt, canon=True, base=b, assign=True)`, `bf16` autocast on
CUDA only, `generate_ids_fast(..., early='eos', compact=True)`, the per-row end-of-attempt
dispositions (`truncated` / `syntax:<why>` / `done` / `step_cap`) -- matches `env_generate` exactly.
`test_dan_adapter.py`'s V3 parity test checks `_rollout` against the vendored `env_generate(gate=
False)` directly, same prompts/seed/batch, with Dan's own per-attempt base rule plugged in via
`_rollout`'s `make_env` callback (see below) rather than our own base convention.

**The `native["lean"]` contract (load-bearing for WP-D2's `LeanJudge`).** `Attempt.native` is a
JSON-encoded string (matching `core.types.Attempt.native`'s `str | None` field type -- "JSON" here
means "a json.dumps string", not that the field itself holds a dict) of
`{"lean": <text or null>, "base": b, "steps": n, "end": "done"|"syntax:<why>"|"truncated"|
"step_cap"}`. **`native["lean"]` is exactly `texts[i]`, the value `env_generate(..., gate=True)`
would pass as the `texts` argument to `vendor/danstate/lean_gate.py:gate(tok, prompts, nd_proofs,
texts)` for that attempt** -- i.e. `tok.text(e.text) if e.done and not nd.startswith('LEANPARSE')
else None` (`state_sample.env_generate`'s own `finish()` closure, `res_tx[i]`). `gate()` then checks
exactly `tok.statement(prompt) + " " + native["lean"]` in Lean (`lean_gate.Gate._dispatch`). This is
`None` whenever the env did not finish, or finished but `lean_tok.inverse` itself failed on the
assembled text (`nd.startswith("LEANPARSE")`) -- in both cases there is no literal Lean text worth
judging. `Attempt.body` instead uses the base-class convention (`core/types.py`'s docstring): the ND
proof `inverse(...)` denotes, or the literal string `"UNPARSEABLE <why>"` (Dan's own `"LEANPARSE
<why>"` marker with the prefix swapped, never the native "LEANPARSE" spelling -- `core.types.
Attempt`'s own contract is `"UNPARSEABLE ..."`). `finished` is `False` only for `end` in
`("truncated", "step_cap")` (plan §1): a syntactically-rejected action is treated as the attempt
having *finished* writing something (just an invalid something), matching the plan's literal
wording, even though the resulting `body` is then `"UNPARSEABLE ..."`.

**The KV-memory guard.** `generate_ids_fast` (`vendor/danstate/sample.py`) preallocates one shared
KV cache per block of shape `(rows, n_head, cache_max, head_dim)` where `cache_max = L + max_action`
(`L` = this call's own longest row, after padding -- `sample.py:146-153`), so one
`generate_ids_fast` call's total KV memory is `rows * (L + max_action) * bytes_per_token` where
`bytes_per_token = 2 * n_layer * d * dtype_bytes` (`K` and `V`, `n_layer` blocks, model width `d`,
`dtype_bytes` 2 for the
`bf16` autocast this rollout uses on CUDA, else 4 for the fp32 this CPU-only worktree always uses --
see `_kv_bytes_per_token`'s own docstring: for the real checkpoint (`d=384`, `n_layer=6`) on GPU
bf16 this works out to 9,216 B, confirming -- not just trusting -- the plan's own figure).
`_rollout` therefore does not call `generate_ids_fast` once per wave on the *whole* worklist; it
first splits the wave's rows (via `_kv_safe_chunks`, in their original worklist order -- never
re-sorted, so when nothing needs to split, behaviour is byte-identical to Dan's own un-split call)
into sub-calls whose own `rows * (chunk_max_len + max_action) * bytes_per_token` stays within
`kv_budget_gb` (default 14 GiB). When a wave fits in one chunk (every test in this file, and every
realistic CPU run at this adapter's default `batch`), the sub-call's seed is exactly Dan's own
`seed * 100003 + wave` -- byte-identical to `env_generate`. A wave that genuinely needs several
sub-calls (large GPU batches of long-state suites) uses a distinguishable, still-deterministic
seed per sub-call (`seed * 100003 + wave + chunk_index * 7919 + 1`); this is a new mechanism
`env_generate` itself has no equivalent of, so there is nothing for it to match.

**Name bases.** Every attempt gets its own canonical-name base `b` (arm SN: `Env(..., canon=True,
base=b, assign=True)` makes the *environment*, not the model, choose the name each `have` action
introduces, as `max name in scope + b + 1`). This adapter's own convention (used by `sample()`) is
`b = random.Random(f"{req.seed}:{i}").randint(0, name_base_max)` for request `req`'s `i`-th draw --
*not* Dan's own `random.Random(seed * 1000003 + nxt).randint(0, NAME_BASE_MAX)` (`nxt` = that
prompt's position in one `env_generate` call's flattened `prompts` list), which has no
per-`SampleRequest` structure to key off. `_rollout` itself does not hard-code either rule: it takes
a `make_env(pos)` callback that builds (and, for a forced `STATEPREFIX`, partially replays) the
`Env` for worklist position `pos`; `test_dan_adapter.py`'s V3 parity test passes a `make_env` that
reproduces Dan's own base rule exactly, so `_rollout`'s output matches vendored `env_generate(gate=
False)` byte for byte on the same prompts/seed/batch.

**Prefix (`STATEPREFIX`).** `sample()` only honours a `SampleRequest.prefix` whose `body` matches
`"STATEPREFIX m=<m> | <full reference ND body>"` (built by WP-D3's `suites.danstate.prefix_state`);
any other `PrefixSpec.body` raises `NotSupported`. The first `m` canonical (base-0) actions of the
reference body are recovered via `state_env.canonicalise` + `split_actions`, every name token in
them is shifted by `+b` (`_shift_name_tokens`; "canonical names at base `b` are exactly base-0 names
`+ b`", plan §1's `logprob` section -- the same convention justifies replay here), and replayed into
a fresh `Env(prompt, canon=True, base=b, assign=True)` via `env.apply`; `env.renamed == 0` is
asserted afterwards (the shifted actions must need no further renaming -- a consistency check that
the shift matches the environment's own canonical numbering at base `b`). Sampling then continues
from that replayed state exactly as an unprefixed attempt would, so `Attempt.body` includes the
forced prefix automatically (it is already in `env.text`), and `native["steps"]`/`max_steps`
naturally count the replayed actions too (`env.apply` always increments `env.steps`, prefix or not).

**`logprob`.** Marginalises over every canonical-name base `0..name_base_max` the proof's own
largest name index allows (`mx + b <= 64`; `mx` from replaying the proof once, canon=True, base 0).
Each step `(state, action)` of `state_env.decompose(prompt, body, canon=True)` becomes one row
`pre + post` where `pre = tok.encode_toks(state_tokens)`, `post = tok.encode_toks(action_tokens) +
[tok.eos]` -- the exact `state_train.py:load()`/`batch()` row shape (verified against the clone,
`state_train.py:39-53`: `pre`/`post` built the same way, next-token CE on `post` only, i.e. the mask
starts at `len(pre) - 1` when predicting `x[:, 1:]`, matching `adapters.absgpt`'s own `mask_start =
len(pid) - 1` convention) -- and the proof's score at base `b` is the *sum* of that step's masked
log-probability over every step (the policy is state-Markov: the joint log-probability of the whole
action sequence is the sum of each step's own conditional log-probability given its own state,
which already encodes everything earlier steps did). `per_encoding[b] = (sum log p at base b) -
log(name_base_max + 1)` (the training-time base draw is uniform over `0..name_base_max`, including
bases this particular proof cannot realise, which therefore silently contribute zero mass -- the
fixed divisor, not a renormalised one over only the valid bases, is what keeps this the correct
marginal); `nats = logsumexp(per_encoding)`. A proof outside the grammar (PR lines not a declared
prefix in order; more than 64 names; a box cited other than by its last line; an uncited box, which
`proof_tokens` silently drops so re-encoding would score a *different*, shorter proof under the
pretense of scoring the given one -- rejected whenever `allow_reencode=False`, the default) returns
`LogProb(nan, 0, ())`, with the reason recorded in `self.last_logprob_errors[pair_idx]`; callers
*must* filter NaN results themselves (`ModelAdapter.logprob`'s contract says nothing about this --
it is this adapter's own, format-specific failure mode). Known gap: "an ORE branch ending in False"
(plan §1) is not separately detected as its own case; any such malformed proof is expected to
already fail inside `state_env.decompose`/`proof_tokens` for an unrelated structural reason, but
this has not been exercised by a dedicated test (see the module's test file and this WP's final
report for what *is* covered).

**Activations / steering (plan §A3).** `activations` teacher-forces one row per `(pair, step)` (not
one row per whole proof -- see below) at base 0, hooks `model.blocks[l]`, and returns `(n_layers,
n_positions, d)` per pair, `n_positions` either one vector per step (`positions="act"`, the `<act>`
token immediately preceding the sampled action -- the policy's actual decision point) or every
action token including `<eos>` (`positions="action"`), concatenated across the proof's steps in
order. `with_steering` returns a new adapter over a deep-copied model whose blocks carry additional
forward hooks adding `alpha * vector` to the residual stream at the positions `where` selects
(`SteeringTerm`/`SteeringSpec`, `adapters/base.py` -- the same dataclasses `absgpt.py` uses; "state"
and "action" below are literally `where="prompt"` and `where="proof"`, there is no separate
vocabulary). Two regimes, both inside one hook (`_make_steering_hook`):

* **Sampling** (a cached `generate_ids_fast` block call): a step's state is *never* split across
  the same call as (part of) its action -- every `generate_ids_fast` call is either that step's
  state **prefill** (`T` = the padded state length, always `> 1` in practice) or one of its action's
  single-token **decode** steps (`T == 1`). The plan's "T > 1 = state/prompt, T == 1 = action/proof"
  rule is therefore *exactly* `is_prompt = (T > 1)` for every row and position in that call, with no
  per-row bookkeeping needed (there is exactly one state-vs-action boundary per `generate_ids_fast`
  call, not one per row as in `absgpt.py`'s cross-request batching, because this rollout never
  batches a forced prefix into the *same* call as fresh decoding -- a `STATEPREFIX` is replayed
  purely in Python against the environment, never forced through the model). This is simpler than
  `absgpt.AbsGptAdapter`'s own hook, which must recover each row's absolute position from the
  block's cumulative `keep` mask (its own cross-request batching can mix rows of different real
  prompt lengths in one call); nothing here needs that.
* **`logprob`/`activations`** (a plain, non-cached forward over one `pre + post` row): classified
  the same way `absgpt.py` does, via a per-row prompt-length threshold against `owner.
  _steer_prompt_len` (a plain `int` or a `(B,)` tensor) compared to `arange(T)` -- each row here is
  exactly one step, so this is the single-pair case of `absgpt`'s own mechanism, not its
  multi-pair-per-chunk generalisation.

`identity()` includes `_steering_spec_hash(spec)` whenever steering is active, so a steered adapter
is just another model to the `Store` (plan §A3, `adapters/README.md`).

**Everything this module is not.** `STEP_ENV` is *not* advertised: the single-step transition is
already fully exercised inside `sample()`/`_rollout` (via `Env.apply`), but there is no standalone
"observe one state, apply one externally-chosen action" method a caller could drive directly --
that would be new surface area plan §1 does not ask WP-D1 to build. `FINETUNE` is not implemented
either (no `Learner` for a step-conditioned policy is specified here; plan §3 B7/EDL explicitly
defers a state-format `Learner` as "needs a state-format Learner", out of scope for WP-D1).
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.io import sha256_str
from ..core.types import JSON, Attempt, LogProb, PrefixSpec, SampleRequest, Sequent
from ..vendor import load_danstate, load_nd_verify
from ._danstate_load import load_dan_checkpoint
from .base import Capability, ModelAdapter, NotSupported, SteeringSpec

#: Commit of Dan Pandori's `nd-takehome` fork these vendored modules were copied from (plan §1:
#: "vendor commit" in `identity()`), matching `vendor/danstate/PROVENANCE.md`/`tools/
#: vendor_danstate.py`'s `DEFAULT_COMMIT`.
VENDOR_COMMIT = "0c492902"

_NAME_TOKEN_RE = re.compile(r"^n(\d+)$")
_STATEPREFIX_RE = re.compile(r"^STATEPREFIX m=(\d+) \| (.*)$", re.DOTALL)

# Row-batching caps for `logprob`/`activations` (same two-cap discipline as `adapters.absgpt`'s
# `_LOGPROB_MAX_BATCH_ROWS`/`_LOGPROB_MAX_BATCH_CELLS`; kept as independent constants here rather
# than importing absgpt's, since the two adapters otherwise share no code or vendor tree).
_LOGPROB_MAX_ROWS = 4096
_LOGPROB_MAX_CELLS = 4096 * 64
_ACT_MAX_ROWS = 4096
_ACT_MAX_CELLS = 4096 * 64


# --------------------------------------------------------------------------------- small helpers


def _cross_request_seed(reqs: Sequence[SampleRequest]) -> int:
    """One `sample()` call's base seed: a hash of every request's own `seed` field, in `reqs`
    order. Independent copy of `adapters.absgpt._cross_request_seed`'s exact recipe (first 8 bytes
    of a sha256 of the colon-joined seeds) -- duplicated rather than imported, so this module never
    pulls in `adapters.absgpt`'s own (unrelated, heavier) vendor tree."""
    payload = ":".join(str(req.seed) for req in reqs)
    return int(sha256_str(payload)[:16], 16)


def _coerce_int(value: Any, name: str) -> int:
    """Kwarg coercion (plan §1): `--model-arg`/`configs/models.yaml` values arrive as plain
    strings (`cli._resolve_model` passes them straight through to `adapter_cls(**kwargs)`); every
    numeric constructor kwarg below is coerced explicitly rather than relying on Python to do it."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DanStateAdapter: {name}={value!r} is not an int") from exc


def _coerce_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DanStateAdapter: {name}={value!r} is not a float") from exc


def _length_bucketed_chunks(
    order: Sequence[int], lengths: Sequence[int], max_rows: int, max_cells: int
) -> list[list[int]]:
    """Copied from `adapters.absgpt._length_bucketed_chunks` (plan §1: "length-bucketed batches
    (copy `absgpt._length_bucketed_chunks`)") -- see that function's own docstring for why both
    caps (row count and total padded cell count) matter independently."""
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_max_len = 0
    for idx in order:
        length = lengths[idx]
        candidate_max = max(cur_max_len, length)
        if cur and (len(cur) + 1 > max_rows or (len(cur) + 1) * candidate_max > max_cells):
            chunks.append(cur)
            cur = []
            candidate_max = length
        cur.append(idx)
        cur_max_len = candidate_max
    if cur:
        chunks.append(cur)
    return chunks


def _shift_name_tokens(toks: Sequence[str], b: int) -> list[str]:
    """Shift every `n<k>` token in `toks` by `+b`; every other token (formula symbols, `h<k>`
    premise tokens, `<st>`/`<nl>`/`⊢`/`<act>`, ...) is left untouched. "Canonical names at base `b`
    are exactly base-0 names `+ b`" (plan §1's `logprob` section; also the basis for `STATEPREFIX`
    replay below)."""
    out = []
    for t in toks:
        m = _NAME_TOKEN_RE.match(t)
        out.append(f"n{int(m.group(1)) + b}" if m else t)
    return out


def _kv_bytes_per_token(cfg: Mapping[str, Any], device: torch.device) -> int:
    """KV-cache bytes per cached token per `generate_ids_fast` call, derived from the model's own
    dims (plan §1: "derive bytes/token from the model dims rather than trusting the plan's 9,216
    B") rather than hard-coded: `n_layer` blocks each cache K and V of total width `d` (`best_model.
    PeriLNBlock.attn`'s `cache['k']`/`cache['v']`, shape `(rows, n_head, cache_max, d // n_head)`
    each -- `n_head * (d // n_head) == d`) at `dtype_bytes` per element -- `bf16` (2 B) under the
    `torch.autocast('cuda', ...)` this adapter's rollout wraps `generate_ids_fast` in exactly when
    `device.type == 'cuda'` (matching `state_sample.env_generate`'s own autocast), else the model's
    own fp32 parameters (4 B), which is what every CPU-only call in this worktree actually uses.
    `2 * d * n_layer * dtype_bytes`: for the real checkpoint (`d=384`, `n_layer=6`) on GPU bf16 this
    comes out to 9,216 B -- confirming, not just trusting, the plan's own figure; GPU is never
    exercised by this worktree's own tests or runs (the CPU-only rule), so only the fp32 branch is
    covered."""
    dtype_bytes = 2 if device.type == "cuda" else 4
    return 2 * int(cfg["d"]) * int(cfg["n_layer"]) * dtype_bytes


def _kv_safe_chunks(
    lengths: Sequence[int], max_action: int, kv_budget_gb: float, bytes_per_token: int
) -> list[list[int]]:
    """Split one wave's row indices (`0..len(lengths)-1`, in their original worklist order --
    *never* re-sorted, so a single-chunk result is byte-identical in effect to Dan's own un-split
    `generate_ids_fast` call; see the module docstring's KV-guard section) into groups whose own
    `generate_ids_fast` call stays within `kv_budget_gb`: one shared KV-cache allocation sized
    `rows * (that group's own max row length + max_action) * bytes_per_token` (`sample.py`'s
    `cache['max'] = L + max_new`, one allocation for the whole call, not per row). A single row
    that alone would already exceed the budget still gets its own one-row chunk -- nothing more can
    be done for it, and refusing to sample it at all would be worse."""
    budget = kv_budget_gb * (1 << 30)
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx, length in enumerate(lengths):
        candidate_max = max(cur_max, length)
        if cur and (len(cur) + 1) * (candidate_max + max_action) * bytes_per_token > budget:
            chunks.append(cur)
            cur = []
            candidate_max = length
        cur.append(idx)
        cur_max = candidate_max
    if cur:
        chunks.append(cur)
    return chunks


def _defining_name_positions(action: Sequence[str]) -> set[int]:
    """Token position(s) within one `lean_seq` action list that a canon (`assign=True`)
    environment *overrides* regardless of what the model wrote there (`state_env.Env.
    _assign_names`'s own index arithmetic, reproduced here for `logprob`'s `exclude_defining_names`
    sensitivity variant): position 1 (the `have`'s own defining name) always, plus the box binder's
    name when the action opens an implication/negation box (`'(' 'fun' '(' name ...`, index `i+3`
    after the `':='` at index `i-1`) or an `Or.elim` (`'Or.elim' <name> '(' 'fun' '(' name ...`,
    index `i+5`). An `exact` action never defines a name."""
    if not action or action[0] != "have":
        return set()
    try:
        j = action.index(":=")
    except ValueError:
        return {1}
    positions = {1}
    i = j + 1
    if i < len(action) and action[i] == "(" and i + 3 < len(action):
        positions.add(i + 3)
    elif i < len(action) and action[i] == "Or.elim" and i + 5 < len(action):
        positions.add(i + 5)
    return positions


def _uncited_box_lines(lines: Sequence[Mapping[str, Any]]) -> list[int]:
    """Line indices of every `AS` (box-opening) line that no later `IMPI`/`NEGI`/`ORE` rule ever
    cites as its box's `(start, end)` span. `proof_tokens` (`vendor/danstate/lean_tok.py`) silently
    *drops* such a box's lines from the rendered `lean_seq` text (its `box()` helper is only called
    from an `IMPI`/`NEGI`/`ORE` term, never standalone), so re-encoding such a proof denotes a
    strictly shorter, different proof than the one asked about -- `logprob`'s `allow_reencode=
    False` default treats this as outside the grammar (plan §1) rather than silently scoring the
    wrong proof."""
    starts = {ln["idx"] for ln in lines if ln["rule"] == "AS"}
    cited: set[int] = set()
    for ln in lines:
        rule, refs = ln["rule"], ln["refs"]
        if rule in ("IMPI", "NEGI") and len(refs) == 2:
            cited.add(refs[0])
        elif rule == "ORE" and len(refs) == 5:
            cited.add(refs[1])
            cited.add(refs[3])
    return sorted(starts - cited)


def _make_capture_hook(captured: dict[int, torch.Tensor], layer: int) -> Any:
    """A `register_forward_hook` callback storing `output` into `captured[layer]`. A plain function
    factory (not a closure inside the `for layer in layers` loop), matching `adapters.absgpt`'s own
    `_make_capture_hook`."""

    def _hook(module: Any, inputs: Any, output: torch.Tensor) -> None:
        captured[layer] = output.detach()

    return _hook


def _resolve_steering_vector(vector: tuple[float, ...] | str) -> list[float]:
    if isinstance(vector, str):
        arr = np.load(vector)
        return [float(v) for v in arr.tolist()]
    return list(vector)


def _steering_spec_hash(spec: SteeringSpec) -> str:
    """A stable fingerprint of `spec`, included in a steered adapter's `identity()` (plan §A3) --
    same recipe as `adapters.absgpt._steering_spec_hash`."""
    parts = []
    for t in spec:
        vec_repr = (
            f"path:{t.vector}"
            if isinstance(t.vector, str)
            else ",".join(f"{v:.10g}" for v in t.vector)
        )
        parts.append(f"{t.layer}|{vec_repr}|{t.alpha:.10g}|{t.where}")
    return sha256_str("||".join(parts))


def _make_steering_hook(owner: DanStateAdapter, term: Any, delta: torch.Tensor) -> Any:
    """A `register_forward_hook` callback adding `term.alpha * delta` to the residual stream at the
    positions `term.where` selects (plan §A3). See the module docstring's "Activations / steering"
    section for the two regimes this distinguishes (a cached, `generate_ids_fast`-driven call during
    `sample()`, classified purely by `T` -- `T > 1` is always this step's whole state, `T == 1` is
    always one decode step of this step's action -- versus a plain, non-cached call during
    `logprob`/`activations`, classified the same way `adapters.absgpt`'s own hook is, via
    `owner._steer_prompt_len` compared against `arange(T)`)."""

    def _hook(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        cache = inputs[4] if len(inputs) > 4 else None
        b, t, _d = output.shape
        cached = isinstance(cache, dict) and "n" in cache
        if cached:
            is_prompt = t > 1
            if term.where == "all":
                pos_mask = torch.ones((b, t), dtype=torch.bool, device=output.device)
            elif term.where == "prompt":
                pos_mask = torch.full((b, t), is_prompt, dtype=torch.bool, device=output.device)
            else:  # "proof"
                pos_mask = torch.full((b, t), not is_prompt, dtype=torch.bool, device=output.device)
        else:
            abs_pos = torch.arange(t, device=output.device).unsqueeze(0).expand(b, t)
            prompt_len = owner._steer_prompt_len
            if isinstance(prompt_len, torch.Tensor):
                prompt_len = (
                    prompt_len.to(device=abs_pos.device).view(-1, 1)
                    if prompt_len.numel() == b
                    else 0
                )
            if term.where == "all":
                pos_mask = torch.ones_like(abs_pos, dtype=torch.bool)
            elif term.where == "prompt":
                pos_mask = abs_pos < prompt_len
            else:  # "proof"
                pos_mask = abs_pos >= prompt_len
        if not bool(pos_mask.any()):
            return output
        updated = output.clone()
        updated[pos_mask] = updated[pos_mask] + term.alpha * delta.to(output.dtype)
        return updated

    return _hook


# --------------------------------------------------------------------------------- the rollout


@dataclass(frozen=True)
class _RolloutOut:
    """One attempt's outcome from `_rollout` (module docstring)."""

    nd: str
    lean_text: str | None
    steps: int
    end: str
    n_tokens: int
    base: int


def _rollout(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    make_env: Callable[[int], Any],
    *,
    greedy: bool,
    temperature: float,
    max_action: int,
    max_steps: int,
    batch: int,
    seed: int,
    kv_budget_gb: float,
    bytes_per_kv_token: int,
) -> list[_RolloutOut]:
    """Line-for-line port of `vendor/danstate/state_sample.py:env_generate` (module docstring).
    `make_env(pos)` builds (and, for a forced prefix, replays into) the `Env` for worklist position
    `pos` (`0 <= pos < len(prompts)`); it is called exactly once per position, in increasing order,
    the same moment `env_generate`'s own `base = random.Random(...); e = Env(...)` line would run.
    """
    ds = load_danstate()
    generate_ids_fast = ds.sample.generate_ids_fast
    prompt_ids = ds.state_sample.prompt_ids

    dev = next(model.parameters()).device
    results: list[_RolloutOut | None] = [None] * len(prompts)
    n_tok = [0] * len(prompts)
    live: list[tuple[int, Any]] = []
    nxt = 0
    wave = 0

    def finish(i: int, e: Any, end: str) -> None:
        nd = e.nd()
        lean_text = tok.text(e.text) if e.done and not nd.startswith("LEANPARSE") else None
        results[i] = _RolloutOut(
            nd=nd, lean_text=lean_text, steps=e.steps, end=end, n_tokens=n_tok[i], base=e.base
        )

    while nxt < len(prompts) or live:
        while len(live) < batch and nxt < len(prompts):
            e = make_env(nxt)
            # `make_env` can hand back an `Env` that is already finished (a `STATEPREFIX` whose
            # `m` replays the reference proof's *entire* action list -- plan §1's explicit "replay
            # all actions of a reference reproduces it") or already at/over `max_steps` (a forced
            # prefix at least that long): `Env.apply`/`Env.done` give the env no memory of how it
            # got into this state, so nothing downstream would otherwise notice and the model
            # would be asked for one more action from an already-closed proof. `env_generate`
            # itself never needs this check (every `Env` it creates starts fresh, steps=0,
            # done=False), so it is not "missing" from the vendored port -- it is new behaviour
            # this adapter's prefix feature requires.
            if e.done:
                finish(nxt, e, "done")
            elif e.steps >= max_steps:
                e.failed = "step cap"
                finish(nxt, e, "step_cap")
            else:
                live.append((nxt, e))
            nxt += 1
        pids = [prompt_ids(tok, e) for _, e in live]
        lengths = [len(p) for p in pids]
        chunks = _kv_safe_chunks(lengths, max_action, kv_budget_gb, bytes_per_kv_token)

        outs: list[list[int] | None] = [None] * len(live)
        for c_idx, chunk in enumerate(chunks):
            chunk_pids = [pids[j] for j in chunk]
            wave_seed = (
                seed * 100003 + wave
                if len(chunks) == 1
                else seed * 100003 + wave + c_idx * 7919 + 1
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                chunk_outs = generate_ids_fast(
                    model,
                    tok,
                    chunk_pids,
                    goals=None,
                    greedy=greedy,
                    temperature=temperature,
                    max_new=max_action,
                    seed=wave_seed,
                    early="eos",
                    compact=True,
                    stats=None,
                )
            for j, o in zip(chunk, chunk_outs, strict=True):
                outs[j] = o
        wave += 1

        keep: list[tuple[int, Any]] = []
        for (i, e), o in zip(live, outs, strict=True):
            assert o is not None
            atoks, ended = tok.decode_action(o)
            n_tok[i] += len(atoks) + (1 if ended else 0)
            if not ended:
                e.failed = "action truncated"
                finish(i, e, "truncated")
                continue
            ok, why = e.apply(atoks)
            if not ok:
                finish(i, e, f"syntax:{why}")
                continue
            if e.done:
                finish(i, e, "done")
                continue
            if e.steps >= max_steps:
                e.failed = "step cap"
                finish(i, e, "step_cap")
                continue
            keep.append((i, e))
        live = keep

    assert all(r is not None for r in results)
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------------- prefix replay


def _parse_state_prefix(body: str) -> tuple[int, str]:
    """`"STATEPREFIX m=<m> | <full reference ND body>"` -> `(m, reference_body)`. Raises
    `NotSupported` for any other `PrefixSpec.body` (plan §1)."""
    m = _STATEPREFIX_RE.match(body)
    if not m:
        raise NotSupported(
            "DanStateAdapter.sample: unsupported PrefixSpec body "
            f"{body!r} (expected 'STATEPREFIX m=<m> | <full reference ND body>')"
        )
    return int(m.group(1)), m.group(2)


def _replay_prefix_env(ds: Any, prompt: str, ref_body: str, m: int, b: int) -> Any:
    """Build an `Env(prompt, canon=True, base=b, assign=True)` with its first `m` canonical
    actions of `ref_body` already applied (module docstring's "Prefix" section)."""
    toks = ds.state_env.canonicalise(prompt, ref_body)
    acts = ds.state_env.split_actions(toks)
    if m > len(acts):
        raise ValueError(f"STATEPREFIX m={m} exceeds the reference proof's {len(acts)} actions")
    env = ds.state_env.Env(prompt, canon=True, base=b, assign=True)
    for act in acts[:m]:
        shifted = _shift_name_tokens(act, b)
        ok, why = env.apply(shifted)
        if not ok:
            raise ValueError(f"STATEPREFIX replay failed at base {b}: {why}")
    if env.renamed:
        raise ValueError(
            f"STATEPREFIX replay: env renamed {env.renamed} defining name(s) at base {b} -- the "
            "shifted prefix does not match the environment's own canonical numbering"
        )
    return env


def _attempt_fields(out: _RolloutOut) -> tuple[str, dict[str, JSON], bool]:
    """`_RolloutOut` -> `(Attempt.body, native dict, Attempt.finished)` (module docstring)."""
    nd = out.nd
    body = "UNPARSEABLE" + nd[len("LEANPARSE") :] if nd.startswith("LEANPARSE") else nd
    native: dict[str, JSON] = {
        "lean": out.lean_text,
        "base": out.base,
        "steps": out.steps,
        "end": out.end,
    }
    finished = out.end not in ("truncated", "step_cap")
    return body, native, finished


@dataclass(frozen=True)
class MaskedLogProbs:
    """Full next-token log-probabilities (natural-log `log_softmax`, every vocab entry) at every
    *scored* position of one teacher-forced `(sequent, body)` pair at a single, caller-fixed
    canonical-name base -- the per-model half of `interp/kl.py`'s exact token-level KL divergence
    between two `DanStateAdapter`s that share one tokenizer/vocab (`DanStateAdapter.
    masked_next_logprobs`'s own docstring). "Scored" is exactly `logprob`'s own per-base mask
    (every action token, including the trailing `<eos>`, minus `exclude_defining_names`'s
    positions when that flag is set) -- *not* marginalised over bases, unlike `logprob` itself.

    `step_of[k]` is which step (0-based, in `state_env.decompose`'s own order -- the same order
    `interp.danstate_support.classify_steps` labels) scored position `k` belongs to; `n_steps` is
    the item's total step count, including the proof-closing `exact` (`danstate_support.
    classify_steps`'s own "FINAL" step)."""

    log_probs: np.ndarray  # (n_scored, vocab), float32
    target_ids: np.ndarray  # (n_scored,), int64 -- the actual next token at that position
    step_of: np.ndarray  # (n_scored,), int64
    n_steps: int
    n_tokens: int  # == len(target_ids)


# --------------------------------------------------------------------------------- the adapter


class DanStateAdapter(ModelAdapter):
    """`ModelAdapter` for Dan Pandori's `best-state` checkpoints: SAMPLE, PREFIX, LOGPROB,
    ACTIVATIONS, STEER (plan §1/§7 WP-D1; see the module docstring for the full contract)."""

    def __init__(
        self,
        ckpt: str | Path,
        device: str = "cpu",
        batch: int | str = 1024,
        max_action: int | str = 512,
        max_steps: int | str = 96,
        name_base_max: int | str = 32,
        kv_budget_gb: float | str = 14.0,
        *,
        label: str | None = None,
        steering_spec: SteeringSpec | None = None,
    ) -> None:
        dev = torch.device(device)
        model, tok, cfg, meta = load_dan_checkpoint(ckpt, device=str(dev))
        self._setup(
            model=model,
            tok=tok,
            cfg=cfg,
            meta=meta,
            device=dev,
            ckpt_path=Path(ckpt),
            batch=_coerce_int(batch, "batch"),
            max_action=_coerce_int(max_action, "max_action"),
            max_steps=_coerce_int(max_steps, "max_steps"),
            name_base_max=_coerce_int(name_base_max, "name_base_max"),
            kv_budget_gb=_coerce_float(kv_budget_gb, "kv_budget_gb"),
            label=label,
            steering_spec=steering_spec,
        )

    @classmethod
    def _from_model(
        cls,
        *,
        model: Any,
        tok: Any,
        cfg: dict[str, Any],
        meta: dict[str, Any],
        device: torch.device,
        ckpt_path: Path,
        batch: int,
        max_action: int,
        max_steps: int,
        name_base_max: int,
        kv_budget_gb: float,
        label: str | None,
        steering_spec: SteeringSpec | None,
    ) -> DanStateAdapter:
        """Build an adapter around an already-in-memory `model` (used by `with_steering`'s
        deep-copied model)."""
        obj = cls.__new__(cls)
        obj._setup(
            model=model,
            tok=tok,
            cfg=cfg,
            meta=meta,
            device=device,
            ckpt_path=ckpt_path,
            batch=batch,
            max_action=max_action,
            max_steps=max_steps,
            name_base_max=name_base_max,
            kv_budget_gb=kv_budget_gb,
            label=label,
            steering_spec=steering_spec,
        )
        return obj

    def _setup(
        self,
        *,
        model: Any,
        tok: Any,
        cfg: dict[str, Any],
        meta: dict[str, Any],
        device: torch.device,
        ckpt_path: Path,
        batch: int,
        max_action: int,
        max_steps: int,
        name_base_max: int,
        kv_budget_gb: float,
        label: str | None,
        steering_spec: SteeringSpec | None,
    ) -> None:
        self._model = model
        self._tok = tok
        self._cfg = cfg
        self._meta = meta
        self._device = device
        self._ckpt_path = ckpt_path
        self._batch = batch
        self._max_action = max_action
        self._max_steps = max_steps
        self._name_base_max = name_base_max
        self._kv_budget_gb = kv_budget_gb
        self._steering_spec: SteeringSpec = steering_spec or ()
        self._steer_prompt_len: int | torch.Tensor = 0
        #: `pair_idx -> reason` for the most recent `logprob()` call's NaN results (module
        #: docstring's `logprob` section). Replaced wholesale on every call.
        self.last_logprob_errors: dict[int, str] = {}

        default_label = f"danstate:{ckpt_path.stem}"
        if self._steering_spec:
            default_label += f"+steer{len(self._steering_spec)}"
        self.label = label or default_label
        self.format_id = "nd_v1"
        self.capabilities = (
            Capability.SAMPLE
            | Capability.PREFIX
            | Capability.LOGPROB
            | Capability.ACTIVATIONS
            | Capability.STEER
        )
        self._steer_handles: list[Any] = (
            self._install_steering_hooks(self._steering_spec) if self._steering_spec else []
        )

    # ---------------------------------------------------------------------- ModelAdapter contract

    def identity(self) -> dict[str, JSON]:
        d: dict[str, JSON] = {
            "adapter": "danstate",
            "ckpt": str(self._ckpt_path),
            "ckpt_sha256": self._meta["sha256"],
            "arch": dict(self._cfg),
            "tokenizer_mode": self._meta["tok_mode"],
            "tokenizer_vocab_sha256": sha256_str(",".join(self._tok.itos))[:16],
            "vendor_commit": VENDOR_COMMIT,
            "rollout": "danstate_v1",
            "batch": self._batch,
            "max_action": self._max_action,
            "max_steps": self._max_steps,
            "name_base_max": self._name_base_max,
            # Not in the plan's own explicit field list, but a deliberate addition: a different KV
            # budget can change *which* wave-splitting (and therefore which noise draws) a large
            # batch gets, so two adapters that would otherwise be "the same model" to the `Store`
            # could silently sample differently. Harmless for every CPU test/run in this worktree
            # (never splits), but keeps the `Store` run key honest if that ever changes.
            "kv_budget_gb": self._kv_budget_gb,
        }
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
        """Draw every request's `n` attempts via `_rollout` (module docstring). Requests are
        flattened into one pool of attempts (one per `(req, draw index)`), each with its own name
        base (`random.Random(f"{req.seed}:{i}").randint(0, name_base_max)`) and its own, possibly
        `None`, `PrefixSpec`; the pool is then grouped by `req.temperature` (`0.0` means greedy --
        plan §1) since `generate_ids_fast`'s `greedy`/`temperature` apply uniformly to one whole
        call. `req.max_new_tokens` is **not used**: unlike a whole-proof adapter, this rollout's
        length is governed entirely by the adapter's own `max_action`/`max_steps`, uniformly across
        every request (the field still has to be supplied -- `SampleRequest` has no default -- it
        is just not consulted by this adapter).

        Every `PrefixSpec` is validated up front (so an unsupported body raises `NotSupported`
        before any model call, not partway through a shared rollout); one `torch.Generator`-free
        `seed` is derived per temperature group from a hash of every request's own seed
        (`_cross_request_seed`) combined with that group's position, so two calls with an
        identical `reqs` reproduce byte-identical attempts, and different temperature groups within
        one call do not draw from identical noise.
        """
        if not reqs:
            return []
        ds = load_danstate()

        flat_prompts: list[str] = []
        flat_bases: list[int] = []
        flat_prefix: list[PrefixSpec | None] = []
        flat_req_draw: list[tuple[int, int]] = []
        flat_temperature: list[float] = []
        for r_idx, req in enumerate(reqs):
            prompt = req.sequent.prompt()
            for i in range(req.n):
                b = random.Random(f"{req.seed}:{i}").randint(0, self._name_base_max)
                flat_prompts.append(prompt)
                flat_bases.append(b)
                flat_prefix.append(req.prefix)
                flat_req_draw.append((r_idx, i))
                flat_temperature.append(req.temperature)

        # Fail fast on an unsupported prefix body before any model call.
        for prefix in flat_prefix:
            if prefix is not None:
                _parse_state_prefix(prefix.body)

        def make_env(pos: int) -> Any:
            prompt = flat_prompts[pos]
            b = flat_bases[pos]
            prefix = flat_prefix[pos]
            if prefix is None:
                return ds.state_env.Env(prompt, canon=True, base=b, assign=True)
            m, ref_body = _parse_state_prefix(prefix.body)
            return _replay_prefix_env(ds, prompt, ref_body, m, b)

        call_seed = _cross_request_seed(reqs)
        groups: dict[float, list[int]] = {}
        for idx, temp in enumerate(flat_temperature):
            groups.setdefault(temp, []).append(idx)

        bytes_per_token = _kv_bytes_per_token(self._cfg, self._device)
        rollout_results: list[_RolloutOut | None] = [None] * len(flat_prompts)
        for group_no, (temperature, idxs) in enumerate(groups.items()):
            group_seed = (call_seed * 1_000_003 + group_no) % (2**63 - 1)
            group_prompts = [flat_prompts[i] for i in idxs]

            def group_make_env(pos: int, _idxs: list[int] = idxs) -> Any:
                return make_env(_idxs[pos])

            group_out = _rollout(
                self._model,
                self._tok,
                group_prompts,
                group_make_env,
                greedy=(temperature == 0),
                temperature=(temperature if temperature else 1.0),
                max_action=self._max_action,
                max_steps=self._max_steps,
                batch=self._batch,
                seed=group_seed,
                kv_budget_gb=self._kv_budget_gb,
                bytes_per_kv_token=bytes_per_token,
            )
            for pos, global_idx in enumerate(idxs):
                rollout_results[global_idx] = group_out[pos]

        out: list[list[Attempt]] = [[] for _ in reqs]
        for flat_idx, (r_idx, i) in enumerate(flat_req_draw):
            rollout_out = rollout_results[flat_idx]
            assert rollout_out is not None
            body, native_obj, finished = _attempt_fields(rollout_out)
            out[r_idx].append(
                Attempt(
                    pid=reqs[r_idx].pid,
                    i=i,
                    body=body,
                    native=json.dumps(native_obj),
                    n_tokens=rollout_out.n_tokens,
                    finished=finished,
                )
            )
        return out

    def canonical_actions(self, seq: Sequent, body: str) -> tuple[tuple[str, ...], ...]:
        """Base-0 canonical `lean_seq` action sequence for `(seq, body)`, as a hashable tuple of
        tuples of token strings -- a base-independent ordering-dedup key (plan §3's I1 likelihood
        benchmark: "deduplicate orderings by action sequence"). Raises the same exceptions
        `state_env.canonicalise`/`split_actions` would for a proof outside the grammar; a caller
        that wants the NaN-on-bad-grammar convention instead should go through `logprob`."""
        ds = load_danstate()
        toks = ds.state_env.canonicalise(seq.prompt(), body)
        acts = ds.state_env.split_actions(toks)
        return tuple(tuple(a) for a in acts)

    def logprob(
        self,
        pairs: Sequence[tuple[Sequent, str]],
        *,
        allow_reencode: bool = False,
        exclude_defining_names: bool = False,
    ) -> list[LogProb]:
        """Marginal log-probability over canonical-name bases (module docstring). `allow_reencode`
        and `exclude_defining_names` are this adapter's own extensions beyond the base
        `ModelAdapter.logprob(pairs)` signature (plan §1's "sensitivity variant"); ordinary callers
        leave both at their defaults."""
        self.last_logprob_errors = {}
        if not pairs:
            return []
        ds = load_danstate()
        nd_verify = load_nd_verify()

        results: list[LogProb | None] = [None] * len(pairs)
        n_tokens_by_pair: list[int] = [0] * len(pairs)
        valid_bases_by_pair: list[list[int]] = [[] for _ in pairs]

        row_ids: list[list[int]] = []
        row_mask_start: list[int] = []
        row_prompt_len: list[int] = []
        row_exclude: list[set[int]] = []
        row_key: list[tuple[int, int]] = []  # (pair_idx, base)

        for pair_idx, (seq, body) in enumerate(pairs):
            prompt = seq.prompt()
            try:
                if not allow_reencode:
                    lines = nd_verify.parse_proof_tokens(body.split())
                    uncited = _uncited_box_lines(lines)
                    if uncited:
                        raise ValueError(
                            f"uncited box(es) at line(s) {uncited}: re-encoding would silently "
                            "drop them, denoting a different proof (allow_reencode=False)"
                        )
                steps, _toks, env = ds.state_env.decompose(prompt, body, canon=True)
            except (
                ds.lean_tok.ParseFail,
                nd_verify.ParseError,
                AssertionError,
                ValueError,
                IndexError,
                KeyError,
            ) as exc:
                self.last_logprob_errors[pair_idx] = f"{type(exc).__name__}: {exc}"
                results[pair_idx] = LogProb(nats=float("nan"), n_tokens=0, per_encoding=())
                continue

            n_tokens_by_pair[pair_idx] = sum(len(action) + 1 for _state, action, _hist in steps)
            mx = env.maxname
            valid_bases = [b for b in range(0, self._name_base_max + 1) if mx + b <= 64]
            valid_bases_by_pair[pair_idx] = valid_bases
            for b in valid_bases:
                for state_toks, action_toks, _hist in steps:
                    pre = self._tok.encode_toks(_shift_name_tokens(state_toks, b))
                    post = self._tok.encode_toks(_shift_name_tokens(action_toks, b)) + [
                        self._tok.eos
                    ]
                    row_ids.append(pre + post)
                    row_mask_start.append(len(pre) - 1)
                    row_prompt_len.append(len(pre))
                    if exclude_defining_names:
                        defining = _defining_name_positions(action_toks)
                        row_exclude.append({len(pre) + p for p in defining})
                    else:
                        row_exclude.append(set())
                    row_key.append((pair_idx, b))

        n_rows = len(row_ids)
        per_pair_base_logp: dict[tuple[int, int], float] = {}
        if n_rows:
            row_lens = [len(r) for r in row_ids]
            order = sorted(range(n_rows), key=lambda i: row_lens[i])
            chunks = _length_bucketed_chunks(order, row_lens, _LOGPROB_MAX_ROWS, _LOGPROB_MAX_CELLS)
            per_row_logp = [0.0] * n_rows

            self._model.eval()
            with torch.no_grad():
                for chunk in chunks:
                    chunk_ids = [row_ids[i] for i in chunk]
                    b_ = len(chunk_ids)
                    length = max(len(r) for r in chunk_ids)
                    x = torch.full(
                        (b_, length), self._tok.pad, dtype=torch.long, device=self._device
                    )
                    for bi, r in enumerate(chunk_ids):
                        x[bi, : len(r)] = torch.tensor(r, device=self._device)
                    lengths_t = torch.tensor([len(row_ids[i]) for i in chunk], device=self._device)
                    mask_starts = torch.tensor(
                        [row_mask_start[i] for i in chunk], device=self._device
                    )
                    prompt_lens = torch.tensor(
                        [row_prompt_len[i] for i in chunk], dtype=torch.long, device=self._device
                    )

                    self._steer_prompt_len = prompt_lens
                    try:
                        logits = self._model(x[:, :-1]).float()
                    finally:
                        self._steer_prompt_len = 0

                    tgt = x[:, 1:]
                    logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
                    col = torch.arange(logp.shape[1], device=self._device).unsqueeze(0)
                    valid_end = (lengths_t - 1).unsqueeze(1)
                    row_mask = (col >= mask_starts.unsqueeze(1)) & (col < valid_end)
                    for local_i, global_i in enumerate(chunk):
                        for pos in row_exclude[global_i]:
                            c = pos - 1  # logp/row_mask column c predicts x[:, c + 1]
                            if 0 <= c < row_mask.shape[1]:
                                row_mask[local_i, c] = False
                    row_sums = (logp * row_mask).sum(dim=1)
                    for local_i, global_i in enumerate(chunk):
                        per_row_logp[global_i] = float(row_sums[local_i].item())

            for row_idx in range(n_rows):
                key = row_key[row_idx]
                per_pair_base_logp[key] = per_pair_base_logp.get(key, 0.0) + per_row_logp[row_idx]

        log_n = math.log(self._name_base_max + 1)
        for pair_idx in range(len(pairs)):
            if results[pair_idx] is not None:
                continue
            valid_bases = valid_bases_by_pair[pair_idx]
            per_encoding = tuple(per_pair_base_logp[(pair_idx, b)] - log_n for b in valid_bases)
            nats = (
                float(torch.logsumexp(torch.tensor(per_encoding), dim=0).item())
                if per_encoding
                else float("-inf")
            )
            results[pair_idx] = LogProb(
                nats=nats, n_tokens=n_tokens_by_pair[pair_idx], per_encoding=per_encoding
            )
        return [r for r in results if r is not None]

    def masked_next_logprobs(
        self,
        items: Sequence[tuple[Sequent, str, int]],
        *,
        exclude_defining_names: bool = False,
        allow_reencode: bool = False,
    ) -> list[MaskedLogProbs | None]:
        """Full next-token log-probabilities at every scored position of each `(seq, body, base)`
        triple, teacher-forced at exactly that caller-given `base` (module docstring's `logprob`
        section: "canonical names at base b are exactly base-0 names + b") -- `interp/kl.py`'s
        "use the sample's own name base, not a marginal over bases" requirement, unlike `logprob`
        itself (which marginalises over every `valid_bases` entry). Masking/exclusion is identical
        to one of `logprob`'s own per-base rows (same `_shift_name_tokens`/
        `_defining_name_positions` helpers, same `mask_start`/`valid_end` column range, same
        length-bucketed batching) -- it just returns the full `log_softmax` row at each scored
        column instead of gathering-and-summing at the single target token, so
        `sum(r.log_probs[k, r.target_ids[k]] for k in range(r.n_tokens))` reproduces exactly one
        `valid_bases` term of `logprob([(seq, body)]).per_encoding` (modulo the uniform `- log(
        name_base_max + 1)` prior offset `logprob` adds and this method does not).

        Returns `None` for an item whose body is outside the ND/state-env grammar: the same
        exception set `logprob` catches, plus the same uncited-box check when `allow_reencode=
        False` -- callers count these themselves (`interp/kl.py`'s "skip ... (count them)").
        """
        ds = load_danstate()
        nd_verify = load_nd_verify()
        vocab = int(self._tok.vocab_size)

        item_ok = [False] * len(items)
        item_n_steps = [0] * len(items)

        row_ids: list[list[int]] = []
        row_mask_start: list[int] = []
        row_prompt_len: list[int] = []
        row_exclude: list[set[int]] = []
        row_item: list[int] = []
        row_step: list[int] = []

        for item_idx, (seq, body, b) in enumerate(items):
            prompt = seq.prompt()
            try:
                if not allow_reencode:
                    lines = nd_verify.parse_proof_tokens(body.split())
                    uncited = _uncited_box_lines(lines)
                    if uncited:
                        raise ValueError(
                            f"uncited box(es) at line(s) {uncited}: re-encoding would silently "
                            "drop them, denoting a different proof (allow_reencode=False)"
                        )
                steps, _toks, _env = ds.state_env.decompose(prompt, body, canon=True)
            except (
                ds.lean_tok.ParseFail,
                nd_verify.ParseError,
                AssertionError,
                ValueError,
                IndexError,
                KeyError,
            ):
                continue
            item_ok[item_idx] = True
            item_n_steps[item_idx] = len(steps)
            for step_idx, (state_toks, action_toks, _hist) in enumerate(steps):
                pre = self._tok.encode_toks(_shift_name_tokens(state_toks, b))
                post = self._tok.encode_toks(_shift_name_tokens(action_toks, b)) + [self._tok.eos]
                row_ids.append(pre + post)
                row_mask_start.append(len(pre) - 1)
                row_prompt_len.append(len(pre))
                if exclude_defining_names:
                    defining = _defining_name_positions(action_toks)
                    row_exclude.append({len(pre) + p for p in defining})
                else:
                    row_exclude.append(set())
                row_item.append(item_idx)
                row_step.append(step_idx)

        n_rows = len(row_ids)
        per_row_logprobs: list[np.ndarray | None] = [None] * n_rows
        per_row_targets: list[np.ndarray | None] = [None] * n_rows

        if n_rows:
            row_lens = [len(r) for r in row_ids]
            order = sorted(range(n_rows), key=lambda i: row_lens[i])
            chunks = _length_bucketed_chunks(order, row_lens, _LOGPROB_MAX_ROWS, _LOGPROB_MAX_CELLS)

            self._model.eval()
            with torch.no_grad():
                for chunk in chunks:
                    chunk_ids = [row_ids[i] for i in chunk]
                    b_ = len(chunk_ids)
                    length = max(len(r) for r in chunk_ids)
                    x = torch.full(
                        (b_, length), self._tok.pad, dtype=torch.long, device=self._device
                    )
                    for bi, r in enumerate(chunk_ids):
                        x[bi, : len(r)] = torch.tensor(r, device=self._device)
                    lengths_t = torch.tensor([len(row_ids[i]) for i in chunk], device=self._device)
                    mask_starts = torch.tensor(
                        [row_mask_start[i] for i in chunk], device=self._device
                    )
                    prompt_lens = torch.tensor(
                        [row_prompt_len[i] for i in chunk], dtype=torch.long, device=self._device
                    )

                    self._steer_prompt_len = prompt_lens
                    try:
                        logits = self._model(x[:, :-1]).float()
                    finally:
                        self._steer_prompt_len = 0

                    logp = F.log_softmax(logits, dim=-1)
                    tgt = x[:, 1:]
                    col = torch.arange(logp.shape[1], device=self._device).unsqueeze(0)
                    valid_end = (lengths_t - 1).unsqueeze(1)
                    row_mask = (col >= mask_starts.unsqueeze(1)) & (col < valid_end)
                    for local_i, global_i in enumerate(chunk):
                        for pos in row_exclude[global_i]:
                            c = pos - 1  # logp/row_mask column c predicts x[:, c + 1]
                            if 0 <= c < row_mask.shape[1]:
                                row_mask[local_i, c] = False
                    for local_i, global_i in enumerate(chunk):
                        idxs = row_mask[local_i].nonzero(as_tuple=True)[0]
                        per_row_logprobs[global_i] = logp[local_i, idxs].cpu().numpy()
                        per_row_targets[global_i] = tgt[local_i, idxs].cpu().numpy()

        per_item_rows: list[list[int]] = [[] for _ in items]
        for row_idx in range(n_rows):
            per_item_rows[row_item[row_idx]].append(row_idx)

        results: list[MaskedLogProbs | None] = [None] * len(items)
        for item_idx in range(len(items)):
            if not item_ok[item_idx]:
                continue
            rows_for_item = per_item_rows[item_idx]
            if rows_for_item:
                row_logprobs: list[np.ndarray] = []
                row_targets: list[np.ndarray] = []
                row_steps: list[np.ndarray] = []
                for r in rows_for_item:
                    lp, tg = per_row_logprobs[r], per_row_targets[r]
                    assert lp is not None and tg is not None
                    row_logprobs.append(lp)
                    row_targets.append(tg)
                    row_steps.append(np.full(lp.shape[0], row_step[r], dtype=np.int64))
                log_probs = np.concatenate(row_logprobs, axis=0)
                target_ids = np.concatenate(row_targets, axis=0)
                step_of = np.concatenate(row_steps)
            else:
                log_probs = np.zeros((0, vocab), dtype=np.float32)
                target_ids = np.zeros((0,), dtype=np.int64)
                step_of = np.zeros((0,), dtype=np.int64)
            results[item_idx] = MaskedLogProbs(
                log_probs=log_probs.astype(np.float32, copy=False),
                target_ids=target_ids.astype(np.int64, copy=False),
                step_of=step_of.astype(np.int64, copy=False),
                n_steps=item_n_steps[item_idx],
                n_tokens=int(target_ids.shape[0]),
            )
        return results

    def activations(
        self,
        pairs: Sequence[tuple[Sequent, str]],
        layers: Sequence[int],
        *,
        positions: str = "act",
    ) -> list[np.ndarray]:
        """Teacher-forced residual-stream activations at base 0, one row per `(pair, step)` (module
        docstring: this keeps one state/action boundary per row, unlike a single row per whole
        proof, so steering's per-row prompt-length threshold stays correct). `positions="act"`:
        one vector per step, at the `<act>` token itself (the decision point); `positions="action"`:
        every action token including `<eos>`. Returns `(n_layers, n_positions, d)` per pair,
        `n_positions`'s vectors concatenated across the proof's steps in order."""
        if positions not in ("act", "action"):
            raise ValueError(f"activations: positions must be 'act' or 'action', got {positions!r}")
        ds = load_danstate()

        row_ids: list[list[int]] = []
        row_prompt_len: list[int] = []
        row_pair: list[int] = []
        row_select: list[list[int]] = []
        for pair_idx, (seq, body) in enumerate(pairs):
            steps, _toks, _env = ds.state_env.decompose(seq.prompt(), body, canon=True)
            for state_toks, action_toks, _hist in steps:
                pre = self._tok.encode_toks(state_toks)
                post = self._tok.encode_toks(action_toks) + [self._tok.eos]
                row_ids.append(pre + post)
                row_prompt_len.append(len(pre))
                row_pair.append(pair_idx)
                if positions == "act":
                    row_select.append([len(pre) - 1])
                else:
                    row_select.append(list(range(len(pre), len(pre) + len(post))))

        n_rows = len(row_ids)
        per_row_acts: list[np.ndarray | None] = [None] * n_rows
        if n_rows:
            row_lens = [len(r) for r in row_ids]
            order = sorted(range(n_rows), key=lambda i: row_lens[i])
            chunks = _length_bucketed_chunks(order, row_lens, _ACT_MAX_ROWS, _ACT_MAX_CELLS)
            for chunk in chunks:
                chunk_ids = [row_ids[i] for i in chunk]
                b_ = len(chunk_ids)
                length = max(len(r) for r in chunk_ids)
                x = torch.full((b_, length), self._tok.pad, dtype=torch.long, device=self._device)
                for bi, r in enumerate(chunk_ids):
                    x[bi, : len(r)] = torch.tensor(r, device=self._device)
                prompt_lens_t = torch.tensor(
                    [row_prompt_len[i] for i in chunk], dtype=torch.long, device=self._device
                )
                captured: dict[int, torch.Tensor] = {}
                handles = [
                    self._model.blocks[layer].register_forward_hook(
                        _make_capture_hook(captured, layer)
                    )
                    for layer in layers
                ]
                self._steer_prompt_len = prompt_lens_t
                try:
                    with torch.no_grad():
                        self._model(x)
                finally:
                    for h in handles:
                        h.remove()
                    self._steer_prompt_len = 0
                for local_i, row_idx in enumerate(chunk):
                    sel = row_select[row_idx]
                    per_row_acts[row_idx] = np.stack(
                        [captured[layer][local_i, sel].cpu().numpy() for layer in layers], axis=0
                    )

        per_pair_rows: list[list[np.ndarray]] = [[] for _ in pairs]
        for row_idx in range(n_rows):
            acts = per_row_acts[row_idx]
            assert acts is not None
            per_pair_rows[row_pair[row_idx]].append(acts)
        d = int(self._cfg["d"])
        return [
            np.concatenate(acts, axis=1) if acts else np.zeros((len(layers), 0, d))
            for acts in per_pair_rows
        ]

    def with_steering(self, spec: SteeringSpec) -> DanStateAdapter:
        """Narrower than `ModelAdapter.with_steering`'s declared `-> ModelAdapter` (same rationale
        as `adapters.absgpt.AbsGptAdapter.with_steering`'s own narrowing): the returned adapter is
        always in fact another `DanStateAdapter` (over a deep-copied, hook-carrying model)."""
        model_copy = copy.deepcopy(self._model)
        return DanStateAdapter._from_model(
            model=model_copy,
            tok=self._tok,
            cfg=self._cfg,
            meta=self._meta,
            device=self._device,
            ckpt_path=self._ckpt_path,
            batch=self._batch,
            max_action=self._max_action,
            max_steps=self._max_steps,
            name_base_max=self._name_base_max,
            kv_budget_gb=self._kv_budget_gb,
            label=None,
            steering_spec=spec,
        )

    def close(self) -> None:
        for h in getattr(self, "_steer_handles", ()):
            h.remove()
        self._steer_handles = []

    # --------------------------------------------------------------------------------- steering

    def _install_steering_hooks(self, spec: SteeringSpec) -> list[Any]:
        handles: list[Any] = []
        for term in spec:
            vec = _resolve_steering_vector(term.vector)
            delta = torch.tensor(vec, dtype=torch.float32, device=self._device)
            block = self._model.blocks[term.layer]
            handles.append(block.register_forward_hook(_make_steering_hook(self, term, delta)))
        return handles
