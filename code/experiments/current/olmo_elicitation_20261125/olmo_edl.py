# ruff: noqa: E501
"""Excess description length (EDL) of one Olmo 3 / 3.1 Think checkpoint on the s2300 response pool (SPEC.md, Stage 1).

    olmo_edl.py run   --model <dir|hub id> [--revision R] --ckpt <name> --pool <dir> --out <dir>
                      [--lrs 3e-5,1e-4,3e-4] [--orders 2] [--n 64] [--m 32] [--evals 0,8,16,32,64] [--reach-evals 0,64]
                      [--seed 0] [--max-tok CAP] [--save-adapter best|all|none]
    olmo_edl.py bench --model <dir|hub id> --pool <dir> [--n 4] [--out <dir>]

Definition (Donoway et al. 2026, arXiv 2601.04728; `code/ndbench/edl.py`): for a data order x_1..x_n,
    EDL(n') = sum_{i<=n'} l(theta_{i-1}; x_i) - n' * L_test(theta_n'),
l = NLL in nats of the response tokens given the templated prompt, scored before the update on x_i. This script
writes the raw per-example losses; `analyze_edl.py` computes EDL, EDL* and intervals from them.

Protocol per (lr, order), with the base model loaded once per process:
  reset the LoRA adapter (A: PEFT's kaiming init from a seed fixed by (seed, order); B = 0, so the adapter is exactly
  inert) and a fresh AdamW; for i = 1..n: score x_i in eval mode under torch.inference_mode (batch 1), then one
  training step on x_i (mean token cross-entropy over the response tokens, as Dan's lora.py; grad clip 1.0; constant
  lr); after step n' in --evals score the test set, after n' in --reach-evals the reach set.
  Step 0 (the train stream, test and reach under the untouched checkpoint) does not depend on lr or order: it is
  computed once into <out>/step0.json and reused (the i = 1 online score of every run must equal its step-0 value:
  checked and recorded as `reset_check`).
Encoding (Dan's olmo_ifladder/lora.py, unchanged): prompt ids = olmo_thinker chat template, add_generation_prompt
(ends "<|im_start|>assistant\\n<think>"), pad ids dropped; response = the stored ids (thinking, `</think>`, answer, EOS).
Loss is over exactly those response ids. Segments: `think` = ids[:think_end] (through the `>` token closing
`</think>`), `answer` = ids[think_end:] (EOS included).
Learner: PEFT LoRA r 16, alpha 32, dropout 0, Dan's target modules, bias none; bf16 base weights, fp32 LoRA weights;
AdamW (0.9, 0.999, eps 1e-8, weight decay 0); batch 1; gradient checkpointing (non-reentrant) during training steps.

Outputs in --out: edl_rows.jsonl (one JSON line per (ckpt, lr, order), appended atomically), step0.json, heartbeat.json
(rewritten at least every --heartbeat-secs, default 60, with the last-progress time), DONE (when every requested row
exists), ERROR.json (fatal error), adapters/ (--save-adapter). A restart skips (lr, order) rows already present.
Exit 0 only if every requested row exists; 2 if some are missing; 3 on a fatal error (out of memory, pool mismatch).
A non-finite loss stops that row (status "nonfinite", recorded with what it had) and the next row starts.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmo_template import TEMPLATE_SHA256, prompt_ids  # noqa: E402

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]  # Dan's lora.py
RECIPE = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.0,
    "targets": TARGETS,
    "bias": "none",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "batch": 1,
    "steps_per_example": 1,
    "schedule": "constant",
    "loss": "mean token cross-entropy over the response tokens (train); per-example sum in nats (score)",
    "base_dtype": "bfloat16",
    "lora_dtype": "float32",
    "grad_checkpointing": "non-reentrant, training steps only",
    "lora_init": "PEFT default (A kaiming-uniform from torch seed, B zeros), seed per (seed, order)",
    "encoding": "olmo_thinker template, add_generation_prompt, stored response ids (Dan's lora.py)",
}
SCHEMA = "olmo_edl_row/1"
CONV_SCHEMA = "olmo_edl_conv/1"  # --conv-rows best: the extra-epochs result of one row, written after the grid
SETS = ("train", "test", "reach")


# --------------------------------------------------------------------------------------------- data


@dataclass
class Example:
    pid: str
    prompt: list[int]
    resp: list[int]
    think_end: int
    label: str

    @property
    def n_tok(self) -> int:
        return len(self.resp)

    @property
    def total(self) -> int:
        return len(self.prompt) + len(self.resp)


def b64_ids(s: str) -> list[int]:
    return np.frombuffer(base64.b64decode(s), dtype=np.int32).tolist()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pool(
    pool: Path,
    tok,
    *,
    n: int,
    m: int,
    max_tok: int | None,
    n_reach: int | None = None,
    allow_short: bool = False,
) -> tuple[dict[str, list[Example]], dict]:
    """The pool's sets in file order, rows with n_tok > max_tok dropped (never truncated), train[:n], test[:m],
    reach[:n_reach]. Prompt ids are re-made with `tok` and must equal the stored ones (Dan's encoding)."""
    man_path = pool / "MANIFEST.json"
    info = {
        "pool": str(pool),
        "manifest_sha256": sha256_file(man_path) if man_path.exists() else None,
        "max_tok_filter": max_tok,
        "dropped_over_cap": {},
    }
    out: dict[str, list[Example]] = {}
    for name, limit in (("train", n), ("test", m), ("reach", n_reach)):
        rows = [
            json.loads(ln) for ln in (pool / f"{name}.jsonl").read_text().splitlines() if ln.strip()
        ]
        kept = [r for r in rows if max_tok is None or r["n_tok"] <= max_tok]
        info["dropped_over_cap"][name] = len(rows) - len(kept)
        if limit is not None:
            if len(kept) < limit and not allow_short:
                raise SystemExit(
                    f"pool {name}: {len(kept)} rows after the cap filter, {limit} requested"
                )
            kept = kept[:limit]
        exs = []
        for r in kept:
            p = prompt_ids(tok, r["prompt"])
            if "prompt_ids" in r and b64_ids(r["prompt_ids"]) != p:
                raise SystemExit(
                    f"{name}/{r['pid']}: this tokenizer's prompt ids differ from the pool's (Dan's encoding)"
                )
            resp = b64_ids(r["ids"])
            assert len(resp) == r["n_tok"] and 0 < r["think_end"] <= len(resp), r["pid"]
            exs.append(Example(r["pid"], p, resp, int(r["think_end"]), r.get("label", "")))
        out[name] = exs
    return out, info


# -------------------------------------------------------------------------------------------- model


def load_model(
    name: str,
    revision: str | None,
    *,
    dtype: str = "bf16",
    device: str = "cuda",
    attn: str = "sdpa",
):
    import torch
    from transformers import AutoModelForCausalLM

    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    kw = {"revision": revision} if revision else {}
    m = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dt,
        device_map={"": 0} if device == "cuda" else None,
        attn_implementation=attn,
        **kw,
    )
    m.config.use_cache = False
    return m


def wrap_lora(model, *, grad_ckpt: bool = True, r: int = 16, alpha: int = 32):
    import torch
    from peft import LoraConfig, get_peft_model

    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(
        model, cfg
    )  # autocast_adapter_dtype=True: LoRA weights in fp32 on a bf16 base
    for nme, p in pm.named_parameters():
        if p.requires_grad:
            assert "lora_" in nme and p.dtype == torch.float32, (nme, p.dtype)
    return pm


def lora_layers(model):
    from peft.tuners.lora import LoraLayer

    return [mod for mod in model.modules() if isinstance(mod, LoraLayer)]


def reset_lora(model, seed: int) -> None:
    """Fresh adapter: PEFT's own init (A kaiming-uniform under torch seed `seed`, B = 0). With B = 0 every LoRA layer adds
    exact zeros, so the model scores exactly as the untouched checkpoint."""
    import torch

    torch.manual_seed(seed)
    layers = lora_layers(model)
    assert layers, "no LoRA layers"
    with torch.no_grad():
        for mod in layers:
            mod.reset_lora_parameters("default", True)
            assert not mod.lora_B["default"].weight.any()
    for p in model.parameters():
        p.grad = None


def trainable(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def _parts(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_decoder(), base.get_output_embeddings()


TOPK = 20
OPTS = {"ce_checkpoint": False, "loss_chunk": 2048}  # --ce-checkpoint / --loss-chunk (memory knobs)


def _head_ce(head, h, tgt):
    import torch.nn.functional as F

    return F.cross_entropy(head(h).float(), tgt, reduction="none")


def token_stats(logits, tgt, topk: int = TOPK) -> dict:
    """Per position: rank of the stored token (0 = top-1; count of strictly larger logits), entropy of the next-token
    distribution (nats), top-k ids and log-probs. `logits` fp32 (T, V)."""
    import torch

    logp = torch.log_softmax(logits, dim=-1)
    lt = logp.gather(1, tgt[:, None])
    rank = (logp > lt).sum(dim=1)
    ent = -(logp.exp() * logp).sum(dim=1)
    tv, ti = logp.topk(topk, dim=1)
    return {
        "nll": (-lt[:, 0]).float().cpu().numpy(),
        "rank": rank.int().cpu().numpy(),
        "entropy": ent.half().cpu().numpy(),
        "topk_ids": ti.int().cpu().numpy(),
        "topk_logp": tv.half().cpu().numpy(),
    }


def token_nll(
    model,
    seqs: list[tuple[list[int], list[int]]],
    *,
    pad_id: int = 0,
    chunk: int | None = None,
    stats: list | None = None,
) -> list:
    """Per-token NLL (fp32) of each response given its prompt: right-padded batch, hidden states from the decoder, the
    LM head applied to the response positions only (bf16 logits upcast to fp32, as HF's causal-LM loss does).
    stats (a list): also append, per sequence, `token_stats` of the same logits (numpy, concatenated over chunks)."""
    import torch
    import torch.nn.functional as F

    decoder, head = _parts(model)
    dev = head.weight.device
    chunk = chunk or OPTS["loss_chunk"]
    lens = [len(p) + len(r) for p, r in seqs]
    L = max(lens)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=dev)
    for b, (p, r) in enumerate(seqs):
        ids[b, : lens[b]] = torch.tensor(p + r, dtype=torch.long, device=dev)
    am = None
    if len(seqs) > 1:
        am = (torch.arange(L, device=dev)[None, :] < torch.tensor(lens, device=dev)[:, None]).long()
    h = decoder(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    out = []
    for b, (p, r) in enumerate(seqs):
        hs = h[b, len(p) - 1 : len(p) + len(r) - 1]
        tgt = torch.tensor(r, dtype=torch.long, device=dev)
        parts, st = [], []
        for s in range(0, len(r), chunk):
            if OPTS["ce_checkpoint"] and torch.is_grad_enabled() and stats is None:
                from torch.utils.checkpoint import checkpoint

                parts.append(
                    checkpoint(
                        _head_ce, head, hs[s : s + chunk], tgt[s : s + chunk], use_reentrant=False
                    )
                )
                continue
            lg = head(hs[s : s + chunk]).float()
            parts.append(F.cross_entropy(lg, tgt[s : s + chunk], reduction="none"))
            if stats is not None:
                st.append(token_stats(lg, tgt[s : s + chunk]))
            del lg
        out.append(torch.cat(parts))
        if stats is not None:
            stats.append({k: np.concatenate([x[k] for x in st]) for k in st[0]})
    return out


def seg_sums(nll, think_end: int) -> tuple[float, float]:
    v = nll.detach().double().cpu()
    return float(v[:think_end].sum()), float(v[think_end:].sum())


class NonFinite(Exception):
    pass


def score(
    model, exs: list[Example], *, batch_tokens: int = 0, pad_id: int = 0, stats: dict | None = None
) -> dict:
    """Per-example (think, answer) nats in the order of `exs`. batch_tokens > 0: length-sorted batches of at most that many
    (padded) tokens; 0: batch 1. stats (a dict): filled with pid -> per-position `token_stats`."""
    import torch

    model.eval()
    order = sorted(range(len(exs)), key=lambda k: -exs[k].total)
    batches, cur = [], []
    for k in order:
        if cur and (batch_tokens <= 0 or (len(cur) + 1) * exs[cur[0]].total > batch_tokens):
            batches.append(cur)
            cur = []
        cur.append(k)
    if cur:
        batches.append(cur)
    th = [0.0] * len(exs)
    an = [0.0] * len(exs)
    with torch.inference_mode():
        for bt in batches:
            stl: list | None = [] if stats is not None else None
            nl = token_nll(
                model, [(exs[k].prompt, exs[k].resp) for k in bt], pad_id=pad_id, stats=stl
            )
            if stats is not None and stl is not None:
                for k, d in zip(bt, stl, strict=True):
                    stats[exs[k].pid] = d
            for k, v in zip(bt, nl, strict=True):
                th[k], an[k] = seg_sums(v, exs[k].think_end)
                if not (math.isfinite(th[k]) and math.isfinite(an[k])):
                    raise NonFinite(f"score {exs[k].pid}: think {th[k]} answer {an[k]}")
    return {"think_nats": th, "answer_nats": an}


def train_step(
    model, opt, params, ex: Example, *, grad_clip: float, pad_id: int = 0
) -> tuple[float, float]:
    import torch

    model.train()
    nll = token_nll(model, [(ex.prompt, ex.resp)], pad_id=pad_id)[0]
    loss = nll.mean()
    if not torch.isfinite(loss):
        raise NonFinite(f"train loss {float(loss)} on {ex.pid}")
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
    if not math.isfinite(gn):
        opt.zero_grad(set_to_none=True)
        raise NonFinite(f"grad norm {gn} on {ex.pid}")
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach()), gn


def set_block(exs: list[Example]) -> dict:
    return {
        "pids": [e.pid for e in exs],
        "n_think": [e.think_end for e in exs],
        "n_answer": [e.n_tok - e.think_end for e in exs],
    }


# ------------------------------------------------------------------------------------- bookkeeping


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seed_of(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big"
    ) % (2**31)


def order_perm(n: int, seed: int, order: int) -> list[int]:
    perm = list(range(n))
    random.Random(f"{seed}:order:{order}").shuffle(perm)
    return perm


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_row(path: Path, row: dict) -> None:
    data = (json.dumps(row) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        n = os.write(fd, data)
        assert n == len(data), "short write"
        os.fsync(fd)
    finally:
        os.close(fd)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue  # a torn last line from a killed process: that row is redone
    return rows


class Heartbeat:
    """Rewrites <out>/heartbeat.json every `secs` from a daemon thread; `progress()` marks real progress, so a watchdog can
    tell a live-but-stuck process (old last_progress_utc) from a dead one (old file)."""

    def __init__(self, out: Path, secs: float, **ident):
        self.path = out / "heartbeat.json"
        self.secs = secs
        self.state = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **ident,
            "phase": "start",
            "started_utc": utc(),
            "last_progress_utc": utc(),
        }
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _write(self) -> None:
        with self._lock:
            st = dict(self.state, utc=utc())
        with contextlib.suppress(OSError):
            atomic_write(self.path, json.dumps(st) + "\n")

    def _loop(self) -> None:
        while not self._stop.wait(self.secs):
            self._write()

    def start(self) -> Heartbeat:
        self._write()
        self._t.start()
        return self

    def progress(self, **kw) -> None:
        with self._lock:
            self.state.update(kw, last_progress_utc=utc())
        self._write()

    def stop(self, phase: str) -> None:
        self.progress(phase=phase)
        self._stop.set()


def env_info() -> dict:
    import peft
    import torch
    import transformers

    d = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "host": socket.gethostname(),
    }
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
        d["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return d


def model_identity(model_name: str, revision: str | None, model, commit: str | None) -> dict:
    d = {
        "model": model_name,
        "revision": revision,
        "commit": commit or getattr(model.config, "_commit_hash", None),
        "template_sha256": TEMPLATE_SHA256,
    }
    p = Path(model_name)
    if p.is_dir():
        for f in ("config.json", "model.safetensors.index.json"):
            if (p / f).exists():
                d[f"sha256_{f}"] = sha256_file(p / f)
    return d


def peak_gb() -> float | None:
    import torch

    return (
        round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    )


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------------------------- core


def save_token_stats(path: Path, sets: dict[str, list[Example]], stats: dict[str, dict]) -> dict:
    """One .npz: per example (pid, set, think_end, n_tok, offset into the flat arrays) and per response position the
    stored token id, its NLL (fp32), its rank under the checkpoint (int32, 0 = top-1), the entropy of the next-token
    distribution (fp16) and the top-20 ids (int32) / log-probs (fp16)."""
    exs = [(nm, e) for nm in SETS for e in sets[nm]]
    off = np.zeros(len(exs) + 1, dtype=np.int64)
    for j, (_nm, e) in enumerate(exs):
        off[j + 1] = off[j] + e.n_tok
    cat = {
        k: np.concatenate([stats[e.pid][k] for _nm, e in exs])
        for k in ("nll", "rank", "entropy", "topk_ids", "topk_logp")
    }
    arrays = {
        "pid": np.array([e.pid for _nm, e in exs]),
        "set": np.array([nm for nm, _e in exs]),
        "think_end": np.array([e.think_end for _nm, e in exs], dtype=np.int32),
        "n_tok": np.array([e.n_tok for _nm, e in exs], dtype=np.int32),
        "offset": off,
        "token": np.concatenate([np.asarray(e.resp, dtype=np.int32) for _nm, e in exs]),
        **cat,
    }
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return {
        "file": path.name,
        "positions": int(off[-1]),
        "examples": len(exs),
        "topk": TOPK,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_token_stats(path: Path) -> dict[str, dict]:
    """pid -> {set, think_end, n_tok, token, nll, rank, entropy, topk_ids, topk_logp} (per-position arrays)."""
    z = np.load(path)
    out = {}
    off = z["offset"]
    for j, pid in enumerate(z["pid"].tolist()):
        a, b = int(off[j]), int(off[j + 1])
        d = {k: z[k][a:b] for k in ("token", "nll", "rank", "entropy", "topk_ids", "topk_logp")}
        d.update(set=str(z["set"][j]), think_end=int(z["think_end"][j]), n_tok=int(z["n_tok"][j]))
        out[pid] = d
    return out


def compute_step0(
    model,
    sets: dict[str, list[Example]],
    *,
    batch_tokens: int,
    pad_id: int,
    hb: Heartbeat | None = None,
    stats_path: Path | None = None,
) -> dict:
    t0 = time.time()
    reset_peak()
    out: dict = {}
    stats: dict | None = {} if stats_path is not None else None
    for name in SETS:
        if hb:
            hb.progress(phase=f"step0 {name}")
        t = time.time()
        sc = score(
            model,
            sets[name],
            batch_tokens=0 if name == "train" else batch_tokens,
            pad_id=pad_id,
            stats=stats,
        )
        out[name] = {
            **set_block(sets[name]),
            **sc,
            "secs": round(time.time() - t, 1),
            "batched": name != "train",
        }
    if stats is not None and stats_path is not None:
        t = time.time()
        out["token_stats"] = save_token_stats(stats_path, sets, stats)
        out["token_stats"]["save_secs"] = round(time.time() - t, 1)
    out["secs"] = round(time.time() - t0, 1)
    out["peak_gpu_gb"] = peak_gb()
    return out


def run_one(
    model,
    sets,
    step0: dict,
    *,
    lr: float,
    order: int,
    seed: int,
    evals: list[int],
    reach_evals: list[int],
    batch_tokens: int,
    pad_id: int,
    grad_clip: float = 1.0,
    hb: Heartbeat | None = None,
    ckpt: str = "",
    extra_epochs: int = 0,
    on_online_done=None,
) -> tuple[dict, object]:
    """One (lr, order) run. Returns (row fields, model). Raises nothing for a non-finite loss: the row says so.
    `on_online_done(model, code_length)` is called after the online pass (theta_n, before any extra epoch).
    extra_epochs > 0: then E more epochs over the same n examples (same adapter, optimizer and constant lr; epoch e in
    the order random.Random(f"{seed}:order:{order}:epoch:{e}")), and the test and reach sets scored once more
    (row["conv"]: the residual after training towards convergence, as in "Bits That Count")."""
    import torch

    train, test, reach = sets["train"], sets["test"], sets["reach"]
    n = len(train)
    perm = order_perm(n, seed, order)
    init_seed = seed_of(seed, "init", order)
    reset_lora(model, init_seed)
    params = trainable(model)
    opt = torch.optim.AdamW(
        params,
        lr=lr,
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["eps"],
        weight_decay=RECIPE["weight_decay"],
    )
    reset_peak()
    row: dict = {
        "lr": lr,
        "order": order,
        "perm": perm,
        "init_seed": init_seed,
        "status": "ok",
        "error": None,
    }
    on = {
        "pids": [],
        "think_nats": [],
        "answer_nats": [],
        "n_think": [],
        "n_answer": [],
        "train_loss": [],
        "grad_norm": [],
        "score_secs": [],
        "train_secs": [],
    }
    ev: dict = {
        "0": {
            "test": {k: step0["test"][k] for k in ("think_nats", "answer_nats")},
            "from": "step0.json",
        }
    }
    if 0 in reach_evals:
        ev["0"]["reach"] = {k: step0["reach"][k] for k in ("think_nats", "answer_nats")}
    t_run = time.time()
    tm = {"score_online": 0.0, "train": 0.0, "eval_test": 0.0, "eval_reach": 0.0}
    try:
        for i, k in enumerate(perm, start=1):
            ex = train[k]
            t = time.time()
            sc = score(model, [ex], batch_tokens=0, pad_id=pad_id)
            th, an = sc["think_nats"][0], sc["answer_nats"][0]
            dt_s = time.time() - t
            on["pids"].append(ex.pid)
            on["think_nats"].append(th)
            on["answer_nats"].append(an)
            on["n_think"].append(ex.think_end)
            on["n_answer"].append(ex.n_tok - ex.think_end)
            on["score_secs"].append(round(dt_s, 3))
            tm["score_online"] += dt_s
            if i == 1:
                ref = step0["train"]["think_nats"][k] + step0["train"]["answer_nats"][k]
                row["reset_check"] = {
                    "pid": ex.pid,
                    "step0_nats": ref,
                    "online_nats": th + an,
                    "abs_diff": abs(th + an - ref),
                }
            t = time.time()
            loss, gn = train_step(model, opt, params, ex, grad_clip=grad_clip, pad_id=pad_id)
            dt_t = time.time() - t
            on["train_loss"].append(loss)
            on["grad_norm"].append(gn)
            on["train_secs"].append(round(dt_t, 3))
            tm["train"] += dt_t
            if hb:
                hb.progress(phase="train", lr=lr, order=order, i=i)
            if i in evals or i in reach_evals:
                e: dict = {}
                if i in evals:
                    t = time.time()
                    e["test"] = score(model, test, batch_tokens=batch_tokens, pad_id=pad_id)
                    tm["eval_test"] += time.time() - t
                if i in reach_evals:
                    t = time.time()
                    e["reach"] = score(model, reach, batch_tokens=batch_tokens, pad_id=pad_id)
                    tm["eval_reach"] += time.time() - t
                ev[str(i)] = e
                if hb:
                    hb.progress(phase="eval", lr=lr, order=order, i=i)
            print(
                f"[{utc()}] {ckpt} lr={lr:g} order={order} i={i}/{n} {ex.pid} nats={th + an:.1f} tok={ex.n_tok} loss={loss:.4f} gn={gn:.3f} "
                f"score={dt_s:.1f}s step={dt_t:.1f}s"
                + (
                    f" | L_test={np.mean(np.add(ev[str(i)]['test']['think_nats'], ev[str(i)]['test']['answer_nats'])):.2f}"
                    if str(i) in ev and "test" in ev[str(i)]
                    else ""
                ),
                flush=True,
            )
    except NonFinite as exc:
        row["status"] = "nonfinite"
        row["error"] = str(exc)
        print(f"NONFINITE {ckpt} lr={lr:g} order={order}: {exc}", flush=True)
    if row["status"] == "ok" and on_online_done is not None:
        on_online_done(model, sum(on["think_nats"]) + sum(on["answer_nats"]), opt)
    if row["status"] == "ok" and extra_epochs > 0:
        row["conv"], t_conv = extra_epochs_pass(
            model,
            opt,
            params,
            sets,
            lr=lr,
            order=order,
            seed=seed,
            epochs=extra_epochs,
            batch_tokens=batch_tokens,
            pad_id=pad_id,
            grad_clip=grad_clip,
            hb=hb,
            ckpt=ckpt,
        )
        row["conv"]["mode"] = (
            "all: continued in place after this row's online pass (same adapter and AdamW state)"
        )
        tm.update(t_conv)
    del opt
    row["online"] = on
    row["evals"] = ev
    row["times"] = {
        **{k: round(v, 1) for k, v in tm.items()},
        "total": round(time.time() - t_run, 1),
    }
    row["peak_gpu_gb"] = peak_gb()
    return row, model


def extra_epochs_pass(
    model,
    opt,
    params,
    sets,
    *,
    lr: float,
    order: int,
    seed: int,
    epochs: int,
    batch_tokens: int,
    pad_id: int,
    grad_clip: float = 1.0,
    hb: Heartbeat | None = None,
    ckpt: str = "",
) -> tuple[dict, dict]:
    """`epochs` more epochs over the n training examples with the given adapter and optimizer (constant lr; epoch e in the
    order random.Random(f"{seed}:order:{order}:epoch:{e}")), then the test and reach sets scored once. Returns (conv
    block, times)."""
    train, test, reach = sets["train"], sets["test"], sets["reach"]
    n = len(train)
    conv: dict = {
        "epochs": epochs,
        "lr": lr,
        "order": order,
        "status": "ok",
        "error": None,
        "train_loss": [],
        "grad_norm": [],
        "perm": [],
    }
    tm: dict = {}
    t = time.time()
    try:
        for ep in range(1, epochs + 1):
            pe = list(range(n))
            random.Random(f"{seed}:order:{order}:epoch:{ep}").shuffle(pe)
            conv["perm"].append(pe)
            ls, gs = [], []
            for j, k in enumerate(pe, start=1):
                loss, gn = train_step(
                    model, opt, params, train[k], grad_clip=grad_clip, pad_id=pad_id
                )
                ls.append(loss)
                gs.append(gn)
                if hb:
                    hb.progress(phase="extra_epochs", lr=lr, order=order, epoch=ep, i=j)
            conv["train_loss"].append(ls)
            conv["grad_norm"].append(gs)
            print(
                f"[{utc()}] {ckpt} lr={lr:g} order={order} extra epoch {ep}/{epochs}: mean loss {np.mean(ls):.4f}",
                flush=True,
            )
        tm["extra_train"] = round(time.time() - t, 1)
        t = time.time()
        conv["test"] = score(model, test, batch_tokens=batch_tokens, pad_id=pad_id)
        conv["reach"] = score(model, reach, batch_tokens=batch_tokens, pad_id=pad_id)
        tm["extra_eval"] = round(time.time() - t, 1)
    except NonFinite as exc:
        conv["status"] = "nonfinite"
        conv["error"] = str(exc)
        print(f"NONFINITE (extra epochs) {ckpt} lr={lr:g} order={order}: {exc}", flush=True)
    return conv, tm


def load_state(model, params, src: Path, lr: float):
    """Adapter weights and AdamW state saved by `save_adapter(..., opt=...)`, loaded into the wrapped model in place."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    dev = params[0].device
    res = set_peft_model_state_dict(
        model, load_file(str(src / "adapter_model.safetensors"), device=str(dev))
    )
    bad = [k for k in getattr(res, "unexpected_keys", []) if "lora_" in k]
    assert not bad, f"unexpected adapter keys: {bad[:5]}"
    opt = torch.optim.AdamW(
        params,
        lr=lr,
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["eps"],
        weight_decay=RECIPE["weight_decay"],
    )
    opt.load_state_dict(torch.load(src / "optimizer.pt", map_location=dev))
    for g in opt.param_groups:
        assert g["lr"] == lr, (g["lr"], lr)
    return opt


def save_adapter(model, dest: Path, opt=None) -> None:
    import shutil

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    if opt is not None:
        import torch

        torch.save(opt.state_dict(), tmp / "optimizer.pt")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)


def run_edl(
    model,
    tok,
    sets: dict[str, list[Example]],
    pool_info: dict,
    out: Path,
    *,
    ckpt: str,
    ident: dict,
    lrs: list[float],
    orders: int,
    seed: int,
    evals: list[int],
    reach_evals: list[int],
    batch_tokens: int = 16384,
    save: str = "none",
    heartbeat_secs: float = 60.0,
    recipe_extra: dict | None = None,
    hb: Heartbeat | None = None,
    extra_epochs: int = 0,
    token_stats_on: bool = True,
    conv_rows: str = "all",
) -> int:
    """The whole grid for one checkpoint on an already-wrapped model. Returns the exit code."""
    out.mkdir(parents=True, exist_ok=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    n = len(sets["train"])
    if max(evals + reach_evals) > n or min(evals + reach_evals) < 0:
        raise SystemExit(f"eval points {evals} / {reach_evals} must lie in [0, n={n}]")
    recipe = {**RECIPE, **(recipe_extra or {})}
    key = {
        "ckpt": ckpt,
        "pool_manifest_sha256": pool_info["manifest_sha256"],
        "n": n,
        "m": len(sets["test"]),
        "n_reach": len(sets["reach"]),
        "train_pids": [e.pid for e in sets["train"]],
        "seed": seed,
    }
    step0_key = dict(key)
    assert conv_rows in ("all", "best"), conv_rows
    conv_best = conv_rows == "best" and extra_epochs > 0
    key["extra_epochs"] = (
        extra_epochs if conv_rows == "all" else 0
    )  # epochs run inside every online row
    rows_path = out / "edl_rows.jsonl"
    if hb is None:
        hb = Heartbeat(out, heartbeat_secs, ckpt=ckpt).start()
    try:
        done = {}
        for r in read_rows(rows_path):
            if r.get("schema") != SCHEMA:
                continue
            mism = [
                k
                for k in (
                    "ckpt",
                    "pool_manifest_sha256",
                    "n",
                    "m",
                    "n_reach",
                    "seed",
                    "extra_epochs",
                )
                if r.get(k, 0 if k == "extra_epochs" else None) != key[k]
            ]
            if mism:
                raise SystemExit(
                    f"{rows_path} holds rows of another setup (differs in {mism}); use a fresh --out"
                )
            if r.get("status") in ("ok", "nonfinite"):
                done[(float(r["lr"]), int(r["order"]))] = r
        todo = [(lr, o) for o in range(orders) for lr in lrs if (float(lr), o) not in done]
        (out / "DONE").unlink(
            missing_ok=True
        )  # re-validated (and rewritten) at the end of this call
        print(
            f"[{utc()}] {ckpt}: {len(todo)} runs to do, {len(done)} already in {rows_path}",
            flush=True,
        )

        s0_path = out / "step0.json"
        step0 = None
        if s0_path.exists():
            s = json.loads(s0_path.read_text())
            if all(s.get(k) == step0_key[k] for k in step0_key) and (
                not token_stats_on or (out / "step0_tokens.npz").exists()
            ):
                step0 = s
        if step0 is None:
            hb.progress(phase="step0")
            s = compute_step0(
                model,
                sets,
                batch_tokens=batch_tokens,
                pad_id=pad_id,
                hb=hb,
                stats_path=out / "step0_tokens.npz" if token_stats_on else None,
            )
            step0 = {
                **step0_key,
                "identity": ident,
                "env": env_info(),
                "pool": pool_info,
                "utc": utc(),
                **s,
            }
            atomic_write(s0_path, json.dumps(step0) + "\n")
            print(
                f"[{utc()}] step0 done in {s['secs']}s: "
                + ", ".join(
                    f"{nm} {sum(s[nm]['think_nats']) + sum(s[nm]['answer_nats']):.0f} nats / {sum(s[nm]['n_think']) + sum(s[nm]['n_answer'])} tok"
                    for nm in SETS
                ),
                flush=True,
            )

        req = {float(x) for x in lrs}
        best_dir = out / "adapters" / "best_order0"
        best_path = out / "adapters" / "BEST.json"

        def code_of(r: dict) -> float:
            return sum(r["online"]["think_nats"]) + sum(r["online"]["answer_nats"])

        def saved_lr() -> float | None:
            """lr of the saved best_order0 state (adapter, and optimizer.pt when conv_best needs it), else None."""
            if not best_path.exists() or not (best_dir / "adapter_model.safetensors").exists():
                return None
            if conv_best and not (best_dir / "optimizer.pt").exists():
                return None
            return float(json.loads(best_path.read_text())["lr"])

        def grid_best(rows: dict) -> tuple[float, float] | None:
            """(lr, code length) of the shortest-code-length order-0 ok row among the REQUESTED lrs."""
            c = {
                k[0]: code_of(v)
                for k, v in rows.items()
                if k[1] == 0 and v["status"] == "ok" and k[0] in req
            }
            if not c:
                return None
            b = min(c, key=c.get)
            return b, c[b]

        # the incumbent among the requested lrs comes from the rows on file, not from BEST.json (a grid shift can leave
        # BEST.json pointing at an lr outside the grid)
        gb = grid_best(done)
        best = {"lr": gb[0], "code_length": gb[1]} if gb else None
        for lr, order in todo:
            hb.progress(phase="run", lr=lr, order=order, i=0)
            t0 = utc()

            def online_done(model_, code, opt_, lr=lr, order=order):
                nonlocal best
                if save == "all":
                    save_adapter(model_, out / "adapters" / f"lr{lr:g}_o{order}")
                if (
                    (save == "best" or conv_best)
                    and order == 0
                    and (best is None or code < best["code_length"])
                ):
                    save_adapter(model_, best_dir, opt=opt_ if conv_best else None)
                    best = {"lr": lr, "code_length": code}
                    atomic_write(
                        best_path,
                        json.dumps(
                            {
                                "lr": lr,
                                "order": 0,
                                "code_length": code,
                                "grid_lrs": sorted(req),
                                "rule": "smallest online code length among order-0 rows of the requested lrs; theta_n after the online pass (before extra epochs)",
                                "utc": utc(),
                            }
                        )
                        + "\n",
                    )

            fields, model = run_one(
                model,
                sets,
                step0,
                lr=lr,
                order=order,
                seed=seed,
                evals=evals,
                reach_evals=reach_evals,
                batch_tokens=batch_tokens,
                pad_id=pad_id,
                grad_clip=RECIPE["grad_clip"],
                hb=hb,
                ckpt=ckpt,
                extra_epochs=key["extra_epochs"],
                on_online_done=online_done if (save != "none" or conv_best) else None,
            )
            row = {
                "schema": SCHEMA,
                **key,
                "identity": ident,
                "recipe": {**recipe, "lr": lr},
                "env": env_info(),
                "pool": pool_info,
                "evals_requested": evals,
                "reach_evals_requested": reach_evals,
                "test": set_block(sets["test"]),
                "reach": set_block(sets["reach"]),
                "utc_start": t0,
                "utc_end": utc(),
                **fields,
            }
            append_row(rows_path, row)
            code = code_of(fields)
            print(
                f"[{utc()}] ROW {ckpt} lr={lr:g} order={order} status={row['status']} code_length={code:.1f} nats "
                f"reset_diff={row.get('reset_check', {}).get('abs_diff')} times={row['times']} peak={row['peak_gpu_gb']}GB",
                flush=True,
            )
        file_rows = read_rows(rows_path)
        all_rows = {
            (float(r["lr"]), int(r["order"])): r
            for r in file_rows
            if r.get("schema") == SCHEMA and r.get("status") in ("ok", "nonfinite")
        }
        grid_rows = {k: v for k, v in all_rows.items() if k[0] in req}
        missing: list = [
            (lr, o) for o in range(orders) for lr in lrs if (float(lr), o) not in all_rows
        ]
        gb = grid_best(all_rows)
        if conv_best and not missing and gb is not None:
            blr = gb[0]
            conv_on_file = [
                r
                for r in file_rows
                if r.get("schema") == CONV_SCHEMA and r.get("epochs") == extra_epochs
            ]
            have = [
                r for r in conv_on_file if float(r["lr"]) == blr and int(r.get("order", 0)) == 0
            ]
            if have:
                print(
                    f"[{utc()}] conv-best: the conv row for the grid's best lr {blr:g} is already on file",
                    flush=True,
                )
            else:
                hb.progress(phase="conv_best", lr=blr, order=0)
                t0 = utc()
                ref_row = all_rows[(blr, 0)]
                if saved_lr() == blr:
                    params = trainable(model)
                    opt = load_state(model, params, best_dir, blr)
                    ref = ref_row["evals"][str(n)]["test"]
                    chk = score(model, sets["test"], batch_tokens=batch_tokens, pad_id=pad_id)
                    diff = max(
                        abs(a - b)
                        for a, b in zip(
                            chk["think_nats"] + chk["answer_nats"],
                            ref["think_nats"] + ref["answer_nats"],
                            strict=True,
                        )
                    )
                    conv, t_conv = extra_epochs_pass(
                        model,
                        opt,
                        params,
                        sets,
                        lr=blr,
                        order=0,
                        seed=seed,
                        epochs=extra_epochs,
                        batch_tokens=batch_tokens,
                        pad_id=pad_id,
                        grad_clip=RECIPE["grad_clip"],
                        hb=hb,
                        ckpt=ckpt,
                    )
                    del opt
                    how = {
                        "mode": "best: after all online rows, the saved theta_n adapter and AdamW state of the grid's "
                        "shortest-code-length order-0 row were loaded (exact continuation, no re-run of its online pass)",
                        "load_check_max_abs_diff_test_nats": diff,
                    }
                else:
                    # no saved state for this lr (the grid moved away from the lr whose state was saved): re-run the
                    # row's online pass (same seeds), save its theta_n as the best adapter, and continue in place
                    def regen(model_, code, opt_, blr=blr):
                        save_adapter(model_, best_dir, opt=opt_)
                        atomic_write(
                            best_path,
                            json.dumps(
                                {
                                    "lr": blr,
                                    "order": 0,
                                    "code_length": code,
                                    "grid_lrs": sorted(req),
                                    "rule": "regenerated by re-running the grid's best order-0 row",
                                    "utc": utc(),
                                }
                            )
                            + "\n",
                        )

                    fields, model = run_one(
                        model,
                        sets,
                        step0,
                        lr=blr,
                        order=0,
                        seed=seed,
                        evals=[n],
                        reach_evals=[],
                        batch_tokens=batch_tokens,
                        pad_id=pad_id,
                        grad_clip=RECIPE["grad_clip"],
                        hb=hb,
                        ckpt=ckpt,
                        extra_epochs=extra_epochs,
                        on_online_done=regen,
                    )
                    conv, t_conv = (
                        fields.get(
                            "conv", {"status": fields["status"], "error": fields.get("error")}
                        ),
                        fields["times"],
                    )
                    how = {
                        "mode": "best: no saved state for the grid's best lr, so its online pass was re-run (same seeds) "
                        "and continued in place",
                        "rerun_code_length_diff": code_of(fields) - code_of(ref_row),
                    }
                crow = {
                    "schema": CONV_SCHEMA,
                    **{k: v for k, v in key.items() if k != "extra_epochs"},
                    "lr": blr,
                    "order": 0,
                    "epochs": extra_epochs,
                    "grid_lrs": sorted(req),
                    "current": True,
                    "supersedes": [
                        {"lr": float(r["lr"]), "utc_end": r.get("utc_end")} for r in conv_on_file
                    ],
                    **how,
                    "code_length": gb[1],
                    "identity": ident,
                    "env": env_info(),
                    "utc_start": t0,
                    "utc_end": utc(),
                    "times": t_conv,
                    "peak_gpu_gb": peak_gb(),
                    "conv": conv,
                }
                append_row(rows_path, crow)
                print(
                    f"[{utc()}] CONV ROW {ckpt} lr={blr:g} order=0 status={conv.get('status')} {how} times={t_conv}",
                    flush=True,
                )
        if conv_best and not missing and gb is not None:
            has_conv = any(
                r.get("schema") == CONV_SCHEMA
                and r.get("epochs") == extra_epochs
                and float(r["lr"]) == gb[0]
                for r in read_rows(rows_path)
            )
            if not has_conv:
                missing.append(("conv", gb[0]))
        if missing:
            hb.stop("incomplete")
            print(f"INCOMPLETE {ckpt}: missing {missing}", flush=True)
            return 2
        atomic_write(
            out / "DONE",
            json.dumps(
                {
                    "ckpt": ckpt,
                    "utc": utc(),
                    "lrs": sorted(req),
                    "rows": len(grid_rows),
                    "status": {f"{k[0]:g}/{k[1]}": v["status"] for k, v in grid_rows.items()},
                    "off_grid_rows_on_file": sorted(
                        f"{k[0]:g}/{k[1]}" for k in all_rows if k[0] not in req
                    ),
                    "conv_best_lr": gb[0] if (conv_best and gb) else None,
                }
            )
            + "\n",
        )
        hb.stop("done")
        print(f"[{utc()}] DONE {ckpt}: {len(grid_rows)} rows of the requested lrs", flush=True)
        return 0
    except BaseException:
        hb.stop("error")
        raise


# ---------------------------------------------------------------------------------------------- cli


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def setup(a):
    import torch
    from transformers import AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(
        a.tokenizer or a.model,
        **({"revision": a.revision} if a.revision and not a.tokenizer else {}),
    )
    t0 = time.time()
    model = load_model(a.model, a.revision, dtype=a.dtype, device=device, attn=a.attn)
    t_load = time.time() - t0
    OPTS["ce_checkpoint"] = bool(getattr(a, "ce_checkpoint", False))
    OPTS["loss_chunk"] = int(getattr(a, "loss_chunk", 2048) or 2048)
    if torch.cuda.is_available() and getattr(a, "max_memory_gb", None):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_memory_gb * 2**30 / total), 0)
        print(
            f"[{utc()}] CUDA memory capped at {a.max_memory_gb} GB of {total / 2**30:.1f}",
            flush=True,
        )
    print(f"[{utc()}] loaded {a.model} ({a.revision}) in {t_load:.0f}s on {device}", flush=True)
    model = wrap_lora(model, grad_ckpt=not a.no_grad_ckpt)
    return tok, model, t_load


def cmd_run(a) -> int:
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(out, a.heartbeat_secs, ckpt=a.ckpt).start()
    hb.progress(phase="load")
    try:
        tok, model, t_load = setup(a)
        sets, info = load_pool(
            Path(a.pool),
            tok,
            n=a.n,
            m=a.m,
            max_tok=a.max_tok,
            n_reach=a.n_reach,
            allow_short=a.allow_short,
        )
        ident = {
            **model_identity(a.model, a.revision, model, a.commit),
            "load_secs": round(t_load, 1),
        }
        evals = sorted(set(parse_ints(a.evals)))
        reach_evals = sorted(set(parse_ints(a.reach_evals)))
        return run_edl(
            model,
            tok,
            sets,
            info,
            out,
            ckpt=a.ckpt,
            ident=ident,
            lrs=parse_floats(a.lrs),
            orders=a.orders,
            seed=a.seed,
            evals=evals,
            reach_evals=reach_evals,
            batch_tokens=a.eval_batch_tokens,
            save=a.save_adapter,
            heartbeat_secs=a.heartbeat_secs,
            recipe_extra={
                "grad_checkpointing": not a.no_grad_ckpt,
                "eval_batch_tokens": a.eval_batch_tokens,
                "extra_epochs": a.extra_epochs,
                "conv_rows": a.conv_rows,
                "ce_checkpoint": a.ce_checkpoint,
                "loss_chunk": a.loss_chunk,
                "max_memory_gb": a.max_memory_gb,
            },
            hb=hb,
            extra_epochs=a.extra_epochs,
            token_stats_on=not a.no_token_stats,
            conv_rows=a.conv_rows,
        )
    except torch.cuda.OutOfMemoryError as e:
        hb.stop("error")
        atomic_write(
            out / "ERROR.json",
            json.dumps({"utc": utc(), "error": "CUDA out of memory", "detail": str(e)[:2000]})
            + "\n",
        )
        print(
            f"FATAL out of memory (no truncation is done; lower --max-tok or use a bigger GPU): {e}",
            flush=True,
        )
        return 3
    except SystemExit as e:
        hb.stop("error")
        atomic_write(out / "ERROR.json", json.dumps({"utc": utc(), "error": str(e)}) + "\n")
        print(f"FATAL {e}", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        hb.stop("error")
        atomic_write(
            out / "ERROR.json",
            json.dumps(
                {"utc": utc(), "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
            )
            + "\n",
        )
        traceback.print_exc()
        return 3


def cmd_bench(a) -> int:
    """Forward and training-step time and peak memory at the median and longest pool sequence, and a projection of the grid."""
    import torch

    tok, model, t_load = setup(a)
    sets, info = load_pool(
        Path(a.pool),
        tok,
        n=a.grid_n,
        m=a.grid_m,
        max_tok=a.max_tok,
        n_reach=None,
        allow_short=a.allow_short,
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    allx = sorted(sets["train"] + sets["test"] + sets["reach"], key=lambda e: e.total)
    picks = {"median": allx[len(allx) // 2], "longest": allx[-1]}
    if a.also_short:
        picks["shortest"] = allx[0]
    res: dict = {
        "env": env_info(),
        "load_secs": round(t_load, 1),
        "pool": info,
        "reps": a.n,
        "grad_ckpt": not a.no_grad_ckpt,
        "seqs": {},
    }
    reset_lora(model, 0)
    params = trainable(model)
    res["trainable_params"] = sum(p.numel() for p in params)
    res["footprint_gb"] = (
        round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    )
    res["ce_checkpoint"] = OPTS["ce_checkpoint"]
    res["loss_chunk"] = OPTS["loss_chunk"]
    res["max_memory_gb"] = a.max_memory_gb
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for name, ex in picks.items():
        r = {"pid": ex.pid, "prompt_tok": len(ex.prompt), "resp_tok": ex.n_tok}
        for what in ("forward", "forward_stats", "train_step"):
            ts = []
            reset_peak()
            opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
            for k in range(a.n + 1):
                sync()
                t = time.time()
                if what == "forward":
                    score(model, [ex], batch_tokens=0, pad_id=pad_id)
                elif what == "forward_stats":
                    score(model, [ex], batch_tokens=0, pad_id=pad_id, stats={})
                else:
                    train_step(model, opt, params, ex, grad_clip=1.0, pad_id=pad_id)
                sync()
                if k > 0:  # the first call is a warm-up
                    ts.append(time.time() - t)
            del opt
            r[what] = {
                "secs_median": round(float(np.median(ts)), 3),
                "secs_all": [round(x, 3) for x in ts],
                "peak_gpu_gb": peak_gb(),
            }
            reset_lora(model, 0)
            print(
                f"[{utc()}] bench {name} ({ex.total} tok) {what}: {r[what]['secs_median']:.2f}s peak {r[what]['peak_gpu_gb']} GB",
                flush=True,
            )
        res["seqs"][name] = r
    sync()
    t = time.time()
    reset_peak()
    score(model, sets["test"], batch_tokens=a.eval_batch_tokens, pad_id=pad_id)
    sync()
    res["test_set_batched"] = {
        "secs": round(time.time() - t, 2),
        "examples": len(sets["test"]),
        "eval_batch_tokens": a.eval_batch_tokens,
        "peak_gpu_gb": peak_gb(),
    }
    t = time.time()
    score(model, sets["test"], batch_tokens=0, pad_id=pad_id)
    sync()
    res["test_set_batch1_secs"] = round(time.time() - t, 2)
    print(
        f"[{utc()}] bench test set ({len(sets['test'])} ex): batched {res['test_set_batched']['secs']}s, batch-1 {res['test_set_batch1_secs']}s",
        flush=True,
    )

    # projection: seconds linear in sequence length through the two measured points
    a_, b_ = picks["median"], picks["longest"]

    def lin(what):
        y1, y2 = (
            res["seqs"]["median"][what]["secs_median"],
            res["seqs"]["longest"][what]["secs_median"],
        )
        x1, x2 = a_.total, b_.total
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else y2 / max(x2, 1)
        return lambda L: max(0.0, y1 + slope * (L - x1))

    fwd, stp, fws = lin("forward"), lin("train_step"), lin("forward_stats")
    eval_test = min(res["test_set_batched"]["secs"], res["test_set_batch1_secs"])
    reach_fwd = sum(fwd(e.total) for e in sets["reach"])
    n_ev = len([x for x in parse_ints(a.evals) if x > 0])
    n_rev = len([x for x in parse_ints(a.reach_evals) if x > 0])
    per_run = (
        sum(fwd(e.total) + stp(e.total) for e in sets["train"])
        + n_ev * eval_test
        + n_rev * reach_fwd
    )
    step0 = sum(
        fws(e.total) for e in sets["train"] + sets["test"] + sets["reach"]
    )  # with token stats, batch 1
    n_lrs = len(parse_floats(a.lrs))
    E = a.extra_epochs
    extra = E * sum(stp(e.total) for e in sets["train"]) + eval_test + reach_fwd if E > 0 else 0.0
    proj = {
        "per_run_min": round(per_run / 60, 1),
        "extra_epochs": E,
        "extra_per_run_min": round(extra / 60, 1),
        "conv_best_extra_per_ckpt_min": round(
            extra / 60, 1
        ),  # --conv-rows best: the extra epochs once per checkpoint
        "per_ckpt_min_1_order_with_extra": round(
            (t_load + step0 + n_lrs * (per_run + extra)) / 60, 1
        ),
        "per_ckpt_min_2_orders_with_extra": round(
            (t_load + step0 + 2 * n_lrs * (per_run + extra)) / 60, 1
        ),
        "step0_min": round(step0 / 60, 1),
        "load_min": round(t_load / 60, 1),
        "per_ckpt_min_1_order": round((t_load + step0 + n_lrs * per_run) / 60, 1),
        "per_ckpt_min_2_orders": round((t_load + step0 + 2 * n_lrs * per_run) / 60, 1),
        "assumes": f"n={len(sets['train'])}, m={len(sets['test'])}, reach={len(sets['reach'])}, {n_lrs} lrs, {n_ev} test evals and {n_rev} reach evals after step 0; "
        "time linear in sequence length through the median and longest measurements; excludes the weight download",
    }
    res["projection"] = proj
    print("PROJECTION " + json.dumps(proj), flush=True)
    lg = res["seqs"]["longest"]
    peak_all = max((lg[w]["peak_gpu_gb"] or 0) for w in ("forward", "forward_stats", "train_step"))
    res["memory"] = {
        "footprint_gb": res["footprint_gb"],
        "peak_train_step_longest_gb": lg["train_step"]["peak_gpu_gb"],
        "peak_forward_stats_longest_gb": lg["forward_stats"]["peak_gpu_gb"],
        "peak_test_batched_gb": res["test_set_batched"]["peak_gpu_gb"],
        "longest_tokens": picks["longest"].total,
        "fits_78gb": bool(
            peak_all and max(peak_all, res["test_set_batched"]["peak_gpu_gb"] or 0) < 78
        ),
    }
    print(
        "MEMORY "
        + json.dumps(res["memory"])
        + "  (knobs if over 78 GB: --ce-checkpoint --loss-chunk 512, --eval-batch-tokens 8192 or 0; check with --max-memory-gb 78)",
        flush=True,
    )
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        atomic_write(Path(a.out) / "bench.json", json.dumps(res, indent=1) + "\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["run", "bench"])
    ap.add_argument("--model", required=True, help="local dir or hub id")
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--commit", default=None, help="commit the weights came from (recorded; for a local dir)"
    )
    ap.add_argument("--tokenizer", default=None, help="default: --model")
    ap.add_argument("--ckpt", default="model", help="checkpoint label (dpo, s0100, ...)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lrs", default="3e-5,1e-4,3e-4")
    ap.add_argument("--orders", type=int, default=2)
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="run: train examples (default 64); bench: timed repetitions per measurement (default 4)",
    )
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument(
        "--n-reach", type=int, default=None, help="use only the first N reach rows (smoke tests)"
    )
    ap.add_argument("--evals", default="0,8,16,32,64")
    ap.add_argument("--reach-evals", default="0,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-tok",
        type=int,
        default=None,
        help="drop pool rows with more response tokens (never truncates)",
    )
    ap.add_argument(
        "--allow-short",
        action="store_true",
        help="run with fewer rows than requested after --max-tok",
    )
    ap.add_argument("--save-adapter", choices=["best", "all", "none"], default="best")
    ap.add_argument(
        "--eval-batch-tokens",
        type=int,
        default=16384,
        help="padded tokens per test/reach scoring batch (0 = batch 1)",
    )
    ap.add_argument("--heartbeat-secs", type=float, default=60.0)
    ap.add_argument(
        "--conv-rows",
        choices=["all", "best"],
        default="all",
        help="with --extra-epochs: in every row (all) or once, after the grid, for the shortest-code-length order-0 row (best)",
    )
    ap.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="cap this process's CUDA memory (torch per-process fraction), e.g. 78 to emulate an 80 GB card on an H200",
    )
    ap.add_argument(
        "--ce-checkpoint",
        action="store_true",
        help="recompute the LM head + cross-entropy in backward (chunk-wise) instead of keeping the fp32 logits: less memory, same numbers",
    )
    ap.add_argument(
        "--loss-chunk",
        type=int,
        default=2048,
        help="response positions per LM-head/cross-entropy chunk (memory knob; with --ce-checkpoint the logits memory is about chunk x 1.4 MB)",
    )
    ap.add_argument(
        "--no-token-stats",
        action="store_true",
        help="skip step0_tokens.npz (per-position NLL, rank, entropy, top-20 under the untouched checkpoint)",
    )
    ap.add_argument(
        "--extra-epochs",
        type=int,
        default=0,
        help="after the online pass, E more epochs over the n examples, then test/reach once more (row['conv'])",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument(
        "--grid-n", type=int, default=64, help="bench: train examples assumed by the projection"
    )
    ap.add_argument(
        "--grid-m", type=int, default=32, help="bench: test examples assumed by the projection"
    )
    ap.add_argument(
        "--also-short", action="store_true", help="bench: also time the shortest sequence"
    )
    a = ap.parse_args()
    if a.n is None:
        a.n = 64 if a.cmd == "run" else 4
    if a.cmd == "run":
        if not a.out:
            ap.error("run needs --out")
        sys.exit(cmd_run(a))
    sys.exit(cmd_bench(a))


if __name__ == "__main__":
    main()
