# ruff: noqa: E501
"""Minimal-adapter trainer: one cell = one adapter family on one pool, an lr grid, keep the (lr, step) with the lowest dev loss.

    mini_train.py run   --model DIR --ckpt NAME --plan plan.json --root POOLS_DIR --out DIR [--ce-checkpoint] [--max-memory-gb 78]
                        [--loss-chunk N] [--heartbeat-secs 60] [--select-on total|answer]
    mini_train.py bench --model DIR --ckpt NAME --plan plan.json --root POOLS_DIR [--out DIR]

Families (spec strings; alpha = 2 r, dropout 0, bias none, fp32 adapter on a bf16 base):
    lora:r=16:targets=all            LoRA on q,k,v,o,gate,up,down (or a comma list of module names)
    mask:k=4096:seed=0               rank-1 all-module LoRA, A frozen at its seeded init, B trainable on a fixed seeded 0/1 mask
                                     of exactly k entries over the concatenation of all B entries (sorted parameter names)
Cell: {"cell", "pool", "family", "n", "lrs" | "lr_from", "steps": 64, "eval_steps", "edge_extend"}; `pool` is relative to --root.
Training run (one lr): adapter reset (A from the family's seed, the same for every lr; B = 0), fresh AdamW (0.9, 0.999, 1e-8, wd 0),
constant lr, batch 1, grad clip 1, loss = mean token CE (olmo_edl.train_step), `steps` optimizer steps cycling over the first n pool
examples (epoch 0 in pool order, later epochs seeded shuffles). Each example is scored in eval mode before its FIRST update (online
code, MDL). Dev loss at eval_steps. Per-token dev loss = pooled nats / pooled tokens (total, or answer segment with --select-on answer).
Every cell first scores the step-0 nats of its first n training examples (cached per pool in <out>/step0, also written to
<out>/step0_train/ as by `step0-train`); cell.json then carries the difficulty-matched EDL blocks (edl_selected_corr,
edl_fixed_step_corr, edl_by_step_corr; `edl_corrected`) next to the raw ones. The corrected EDL is the primary capacity number; the
raw one is kept. edl_fixed_step takes the lr with the lowest dev loss at the step among the lrs whose run REACHED it.
Outputs per cell in <out>/cells/<cell>/: runs.jsonl, cell.json, adapter/, gate.npz, DONE. A cell without DONE is redone from scratch.
Exit: 0 all cells done, 2 some failed, 3 fatal (OOM, pool mismatch).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "olmo_elicitation_20261125")):
    if _p not in sys.path:
        sys.path.append(_p)
sys.path.insert(0, str(HERE))
import olmo_edl as oe  # noqa: E402

ALL_TARGETS = list(oe.TARGETS)
LN2 = math.log(2)
SCHEMA = "olmo_mini_cell/1"


# ------------------------------------------------------------------------------------------ families


class Family:
    def __init__(self, spec: str):
        self.spec = spec
        parts = spec.split(":")
        self.kind = parts[0]
        kv = dict(p.split("=", 1) for p in parts[1:])
        if self.kind == "lora":
            self.r = int(kv["r"])
            t = kv.get("targets", "all")
            self.targets = ALL_TARGETS if t == "all" else t.split(",")
            self.k = None
            self.mask_seed = None
        elif self.kind == "mask":
            self.r = 1
            # targets= restricts BOTH the LoRA's modules (cleaner export, fewer zero tensors) and the module set the
            # k entries are drawn from; the adapter stays rank 1.
            t = kv.get("targets", "all")
            self.targets = ALL_TARGETS if t == "all" else t.split(",")
            self.k = int(kv["k"])
            self.mask_seed = int(kv.get("seed", 0))
        else:
            raise ValueError(f"unknown family {spec!r}")
        self.alpha = 2 * self.r

    @property
    def init_seed(self) -> int:
        return oe.seed_of("A-init", self.spec)


class Adapter:
    """A PEFT-wrapped model for one family: the trained-entry count P, the formula check, the mask and the hooks."""

    def __init__(self, base, fam: Family, grad_ckpt: bool = True):
        import torch
        from peft import LoraConfig, get_peft_model

        self.fam = fam
        if grad_ckpt:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            base.enable_input_require_grads()
        cfg = LoraConfig(
            r=fam.r,
            lora_alpha=fam.alpha,
            lora_dropout=0.0,
            target_modules=fam.targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, cfg)
        self.formula = sum(
            fam.r * (m.base_layer.in_features + m.base_layer.out_features)
            for m in oe.lora_layers(self.model)
        )
        self.masks: dict[str, torch.Tensor] = {}
        self.a_snapshot: dict[str, torch.Tensor] = {}
        if fam.kind == "mask":
            self._setup_mask()
            self.P = int(sum(int(m.sum()) for m in self.masks.values()))
            self.formula = fam.k
        else:
            self.P = sum(p.numel() for p in oe.trainable(self.model))
        for n_, p in self.model.named_parameters():
            if "lora_" in n_:
                assert p.dtype == torch.float32, (n_, p.dtype)
        self.n_modules = len(oe.lora_layers(self.model))

    def _setup_mask(self) -> None:
        import torch

        named = sorted((n_, p) for n_, p in self.model.named_parameters() if "lora_B" in n_)
        for n_, p in self.model.named_parameters():
            if "lora_A" in n_:
                p.requires_grad_(False)
        total = sum(p.numel() for _, p in named)
        k = self.fam.k
        if k > total:
            raise SystemExit(f"mask k={k} exceeds the {total} B entries")
        idx = np.random.default_rng(oe.seed_of("mask", self.fam.mask_seed)).choice(
            total, size=k, replace=False
        )
        flat = np.zeros(total, dtype=bool)
        flat[idx] = True
        off = 0
        for n_, p in named:
            m = torch.from_numpy(flat[off : off + p.numel()].reshape(p.shape).copy()).to(p.device)
            off += p.numel()
            self.masks[n_] = m
            p.register_hook(lambda g, m=m: g.masked_fill(~m, 0.0))
        self.total_B = total

    def reset(self, seed: int) -> None:
        oe.reset_lora(self.model, seed)
        if self.fam.kind == "mask":
            self.a_snapshot = {
                n_: p.detach().clone() for n_, p in self.model.named_parameters() if "lora_A" in n_
            }

    def check_mask(self) -> dict:
        """Masked B entries exactly 0 and A unchanged since reset (mask family); nothing to check for lora."""
        if self.fam.kind != "mask":
            return {"kind": "lora"}
        import torch

        params = dict(self.model.named_parameters())
        for n_, m in self.masks.items():
            assert not params[n_].detach()[~m].any(), f"masked entries of {n_} moved"
        for n_, a in self.a_snapshot.items():
            assert torch.equal(params[n_].detach(), a), f"A changed: {n_}"
        return {"kind": "mask", "masked_zero": True, "A_unchanged": True, "n_trained": self.P}

    def release(self):
        """Back to the bare base model (adapter modules removed)."""
        base = self.model.unload()
        for attr in ("peft_config",):
            if hasattr(base, attr):
                delattr(base, attr)
        self.model = None
        return base


# ------------------------------------------------------------------------------------------- helpers


def lr_neighbor(x: float, up: bool) -> float:
    """Next value of the 1-3-10 sequence above/below x (x on the sequence)."""
    e = math.floor(math.log10(x) + 1e-9)
    j = 0 if x / 10**e < 2 else 1
    q = 2 * e + j + (1 if up else -1)
    return float(f"{[1, 3][q % 2]}e{q // 2}")


def tokens_of(exs, think_end=False):
    return np.array([e.n_tok for e in exs], dtype=float)


def dev_loss(sc: dict, exs, select_on: str = "total") -> float:
    th, an = np.asarray(sc["think_nats"]), np.asarray(sc["answer_nats"])
    ans_tok = np.array([e.n_tok - e.think_end for e in exs], dtype=float)
    if select_on == "answer":
        return float(an.sum() / ans_tok.sum())
    return float((th + an).sum() / np.array([e.n_tok for e in exs], dtype=float).sum())


def default_eval_steps(n: int, steps: int) -> list[int]:
    base = [1, 2, 4, 8, 16, 32, 64] if n <= 4 else ([16, 32, 64] if n >= 64 else [4, 8, 16, 32, 64])
    es = sorted({s for s in base if s <= steps} | {steps})
    return es


def epoch_order(n: int, seed_key: str, e: int) -> list[int]:
    perm = list(range(n))
    if e > 0:
        random.Random(f"{seed_key}:epoch:{e}").shuffle(perm)
    return perm


def step_sequence(n: int, steps: int, seed_key: str) -> list[int]:
    seq: list[int] = []
    e = 0
    while len(seq) < steps:
        seq += epoch_order(n, seed_key, e)
        e += 1
    return seq[:steps]


def pool_sha(pool: Path) -> str:
    mp = pool / "MANIFEST.json"
    return oe.sha256_file(mp) if mp.exists() else oe.sha256_file(pool / "test.jsonl")


class Step0:
    """Per-pool cache of untouched-checkpoint scores: the dev set, and train examples on demand."""

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {"dev": None, "train": {}}

    def save(self) -> None:
        oe.atomic_write(self.path, json.dumps(self.d) + "\n")

    def dev(self, model, exs, *, bt, pad_id) -> dict:
        if self.d["dev"] is None or self.d["dev"]["pids"] != [e.pid for e in exs]:
            sc = oe.score(model, exs, batch_tokens=bt, pad_id=pad_id)
            self.d["dev"] = {"pids": [e.pid for e in exs], **sc}
            self.save()
        return self.d["dev"]

    def train(self, model, ex, *, pad_id) -> float:
        if ex.pid not in self.d["train"]:
            sc = oe.score(model, [ex], batch_tokens=0, pad_id=pad_id)
            self.d["train"][ex.pid] = [sc["think_nats"][0], sc["answer_nats"][0]]
            self.save()
        return sum(self.d["train"][ex.pid])


# ------------------------------------------------------------------------------------------ one cell


class LrRun:
    """State of one lr's training run (kept across the stage-a / stage-b split)."""

    def __init__(self, lr: float, row: dict, opt, params, seq: list[int]):
        self.lr, self.row, self.opt, self.params, self.seq = lr, row, opt, params, seq
        self.t = 0
        self.seen: set[int] = set()
        self.best_L: float | None = None
        self.bad = 0
        self.early: int | None = None
        self.secs = 0.0


def edl_at(row: dict, t: int, n: int, dev_exs, P: int) -> dict:
    """EDL after step t of one run: m = min(t, n) examples were scored (nats_i, tokens_i) before their first update;
    EDL_tok = sum nats - sum tokens * L_dev(theta_t) [nats/token, total]; EDL_ex = sum nats - m * mean per-example dev nats."""
    on = row["online"]
    m = min(t, n, len(on["pids"]))
    nats = float(sum(on["think_nats"][:m]) + sum(on["answer_nats"][:m]))
    toks = int(sum(on["n_tok"][:m]))
    d = row["dev"][str(t)]
    tot = np.add(d["think_nats"], d["answer_nats"])
    if dev_exs is None:  # derive the dev token total from the row's own pooled loss
        dev_tok = float(tot.sum() / d["loss_total"])
    elif isinstance(dev_exs, int | float):
        dev_tok = float(dev_exs)
    else:
        dev_tok = float(sum(e.n_tok for e in dev_exs))
    L_tot = float(tot.sum() / dev_tok)
    tok_n = nats - toks * L_tot
    ex_n = nats - m * float(tot.mean())
    return {
        "t": t,
        "m": m,
        "mdl_nats": nats,
        "mdl_tokens": toks,
        "dev_nats_per_tok_total": L_tot,
        "edl_tok_nats": tok_n,
        "edl_ex_nats": ex_n,
        "edl_tok_bits": tok_n / LN2,
        "edl_ex_bits": ex_n / LN2,
        "edl_tok_bits_per_trained_weight": tok_n / LN2 / P,
        "edl_ex_bits_per_trained_weight": ex_n / LN2 / P,
    }


def fixed_step_block(rows: dict, FIX: int, n: int, dev_tok, P: int) -> dict | None:
    """EDL at step FIX for the lr with the lowest dev loss at FIX among the lrs whose run REACHED FIX: FIX was evaluated and the run
    was not early-stopped before it (early_stopped_at is None or >= FIX). `rows` = latest record per lr (an extended a+b run carries
    the whole history); dev_tok = dev token total (or None: derived from the row)."""
    per_lr = {
        f"{lr:g}": {
            **edl_at(r, FIX, n, dev_tok, P),
            "dev_loss_criterion": r["dev"][str(FIX)]["loss"],
            "early_stopped_at": r.get("early_stopped_at"),
        }
        for lr, r in rows.items()
        if str(FIX) in r["dev"]
        and (r.get("early_stopped_at") is None or r["early_stopped_at"] >= FIX)
    }
    if not per_lr:
        return None
    lr_best = min(per_lr, key=lambda k: per_lr[k]["dev_loss_criterion"])
    return {"step": FIX, "lr": float(lr_best), **per_lr[lr_best], "per_lr": per_lr}


class CellRunner:
    def __init__(self, ad: Adapter, tok, cell: dict, ctx: dict):
        self.ad, self.tok, self.ctx = ad, tok, ctx
        self.recipe = None
        if cell.get("recipe_from"):
            rp = cell.get("recipe_cell_json") or str(
                Path(ctx["out"]) / "cells" / cell["recipe_from"] / "cell.json"
            )
            sel = json.loads(Path(rp).read_text())["selected"]
            self.recipe = {"lr": float(sel["lr"]), "step": int(sel["step"]), "path": str(rp)}
            cell = {**cell, "lrs": [self.recipe["lr"]], "steps": self.recipe["step"], "fixed": True}
            cell.pop("lr_from", None)
        self.init_adapter = cell.get("init_adapter")
        if self.init_adapter:
            cell = {**cell, "fixed": True, "steps": 0, "lrs": [], "n": cell["n"]}
            cell = {k: v for k, v in cell.items() if k not in ("extend", "edge_extend", "lr_from")}
        self.fixed = bool(cell.get("fixed"))
        if self.fixed and not self.init_adapter:
            assert len(cell.get("lrs", [])) == 1, "a fixed cell needs exactly one lr"
            cell = {k: v for k, v in cell.items() if k not in ("extend", "edge_extend")}
        self.cell = cell
        self.fam = ad.fam
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.cdir = Path(ctx["out"]) / "cells" / cell["cell"]
        self.pool = (
            Path(cell["pool"]) if os.path.isabs(cell["pool"]) else Path(ctx["root"]) / cell["pool"]
        )
        self.n = int(cell["n"])
        self.select_on = ctx.get("select_on", "total")
        self.seed = int(cell["seed"]) if "seed" in cell else self.fam.init_seed
        ext = cell.get("extend")
        self.ext = ext
        steps = int(cell.get("steps", ext["to"] if ext else 64))
        if ext:
            self.after, self.total = int(ext["after"]), int(ext["to"])
            assert self.after < self.total, "extend.after must be below extend.to"
            self.evals_a = sorted(
                {
                    s
                    for s in (cell.get("eval_steps") or default_eval_steps(self.n, self.after))
                    if s <= self.after
                }
                | {self.after}
            )
            self.evals_b = sorted(
                {s for s in ext.get("eval_steps", []) if self.after < s <= self.total}
                | {self.total}
            )
        else:
            self.after = self.total = steps
            self.evals_a = sorted(
                {
                    s
                    for s in (cell.get("eval_steps") or default_eval_steps(self.n, steps))
                    if s <= steps
                }
                | {steps}
            )
            self.evals_b = []
        self.steps = self.total
        self.best: dict | None = None  # across all lrs of the cell
        self.lruns: dict[float, LrRun] = {}
        self.rows: dict[float, dict] = {}  # latest snapshot per lr

    # data
    def load(self) -> None:
        sets, info = oe.load_pool(self.pool, self.tok, n=self.n, m=None, max_tok=None, n_reach=0)
        self.train, self.dev, self.info = sets["train"], sets["test"], info
        self.dev_tok = sum(e.n_tok for e in self.dev)
        assert len(self.train) == self.n, (len(self.train), self.n)
        # step-0 nats of the first n training examples (untouched checkpoint, adapter inert), cached per pool in the out dir
        self.ad.reset(self.seed)
        self.s0 = step0_train(
            self.ad.model,
            self.tok,
            Path(self.ctx["root"]),
            Path(self.ctx["out"]),
            self.ctx["ckpt"],
            self.cell["pool"],
            self.n,
            self.ctx["eval_bt"],
            self.ctx.get("hb"),
        )
        step0_dir = Path(self.ctx["out"]) / "step0"
        step0_dir.mkdir(parents=True, exist_ok=True)
        self.step0 = Step0(step0_dir / f"{self.ctx['ckpt']}_{pool_sha(self.pool)[:16]}.json")
        man = self.pool / "MANIFEST.json"
        self.manifest = json.loads(man.read_text()) if man.exists() else {}
        self.seq = step_sequence(self.n, self.total, f"{self.seed}:{self.cell['cell']}")

    def resolve_lrs(self) -> list[float]:
        if "lrs" in self.cell:
            return [float(x) for x in self.cell["lrs"]]
        src = Path(self.ctx["out"]) / "cells" / self.cell["lr_from"]
        if not (src / "DONE").exists():
            raise RuntimeError(f"lr_from cell {self.cell['lr_from']} is not done")
        return [float(json.loads((src / "cell.json").read_text())["selected"]["lr"])]

    # one lr
    def new_run(self, lr: float) -> LrRun:
        import torch

        ad, model = self.ad, self.ad.model
        ad.reset(self.seed)
        params = oe.trainable(model)
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(oe.RECIPE["betas"]),
            eps=oe.RECIPE["eps"],
            weight_decay=oe.RECIPE["weight_decay"],
        )
        oe.reset_peak()
        dev0 = self.step0.dev(model, self.dev, bt=self.ctx["eval_bt"], pad_id=self.pad_id)
        row: dict = {
            "cell": self.cell["cell"],
            "lr": lr,
            "status": "ok",
            "error": None,
            "init_seed": self.seed,
            "seq": self.seq,
            "n_params_trained": ad.P,
            "steps_log": [],
            "online": {
                k: []
                for k in (
                    "pids",
                    "think_nats",
                    "answer_nats",
                    "n_tok",
                    "n_think",
                    "n_answer",
                    "score_secs",
                )
            },
            "dev": {
                "0": {
                    "think_nats": dev0["think_nats"],
                    "answer_nats": dev0["answer_nats"],
                    "loss": dev_loss(dev0, self.dev, self.select_on),
                    "loss_total": dev_loss(dev0, self.dev, "total"),
                    "loss_answer": dev_loss(dev0, self.dev, "answer"),
                    "from": "step0 cache",
                }
            },
        }
        return LrRun(lr, row, opt, params, self.seq)

    def advance(self, run: LrRun, to_step: int, evals: list[int]) -> None:
        """Train `run` from its current step to `to_step` (or to an early stop / non-finite loss)."""
        model, row, hb, cid = self.ad.model, run.row, self.ctx.get("hb"), self.cell["cell"]
        t_start = time.time()
        try:
            for t in range(run.t + 1, to_step + 1):
                k = run.seq[t - 1]
                ex = self.train[k]
                if k not in run.seen:
                    run.seen.add(k)
                    t0 = time.time()
                    sc = oe.score(model, [ex], batch_tokens=0, pad_id=self.pad_id)
                    on = row["online"]
                    th, an = sc["think_nats"][0], sc["answer_nats"][0]
                    on["pids"].append(ex.pid)
                    on["think_nats"].append(th)
                    on["answer_nats"].append(an)
                    on["n_tok"].append(ex.n_tok)
                    on["n_think"].append(ex.think_end)
                    on["n_answer"].append(ex.n_tok - ex.think_end)
                    on["score_secs"].append(round(time.time() - t0, 3))
                    if len(run.seen) == 1:
                        ref = self.step0.train(model, ex, pad_id=self.pad_id)
                        row["reset_check"] = {
                            "pid": ex.pid,
                            "step0_nats": ref,
                            "online_nats": th + an,
                            "abs_diff": abs(th + an - ref),
                        }
                t0 = time.time()
                loss, gn = oe.train_step(
                    model, run.opt, run.params, ex, grad_clip=1.0, pad_id=self.pad_id
                )
                row["steps_log"].append(
                    {
                        "step": t,
                        "k": k,
                        "pid": ex.pid,
                        "loss": loss,
                        "grad_norm": gn,
                        "secs": round(time.time() - t0, 3),
                    }
                )
                run.t = t
                if hb:
                    hb.progress(phase="train", cell=cid, lr=run.lr, step=t)
                if t in evals:
                    t1 = time.time()
                    sc = oe.score(
                        model, self.dev, batch_tokens=self.ctx["eval_bt"], pad_id=self.pad_id
                    )
                    L = dev_loss(sc, self.dev, self.select_on)
                    row["dev"][str(t)] = {
                        **sc,
                        "loss": L,
                        "loss_total": dev_loss(sc, self.dev, "total"),
                        "loss_answer": dev_loss(sc, self.dev, "answer"),
                        "secs": round(time.time() - t1, 2),
                    }
                    print(
                        f"[{oe.utc()}] {cid} lr={run.lr:g} step={t}/{self.total} train_loss={loss:.4f} dev_loss/tok={L:.5f} (step0 {row['dev']['0']['loss']:.5f})",
                        flush=True,
                    )
                    if (self.fixed and t == self.total) or (
                        not self.fixed and (self.best is None or self.best["loss"] > L)
                    ):
                        self.save_best(run.lr, t, L, sc)
                    if hb:
                        hb.progress(phase="eval", cell=cid, lr=run.lr, step=t)
                    # early stop: above this lr's own running best at two consecutive evals
                    if run.best_L is not None and run.best_L < L:
                        run.bad += 1
                    else:
                        run.bad = 0
                    run.best_L = L if run.best_L is None else min(run.best_L, L)
                    if run.bad >= 2 and not self.fixed:
                        run.early = t
                        print(
                            f"[{oe.utc()}] {cid} lr={run.lr:g} early stop at step {t}", flush=True
                        )
                        break
        except oe.NonFinite as exc:
            row["status"] = "nonfinite"
            row["error"] = str(exc)
            print(f"NONFINITE {cid} lr={run.lr:g}: {exc}", flush=True)
        run.secs += time.time() - t_start

    def b_stats(self) -> dict:
        mx, nz = 0.0, 0
        for n_, p in self.ad.model.named_parameters():
            if "lora_B" in n_:
                a = p.detach().abs()
                mx = max(mx, float(a.max()))
                nz += int((a > 0).sum())
        return {"max_abs_B": mx, "n_B_nonzero": nz}

    def snapshot(self, run: LrRun, stage: str) -> dict:
        row = copy.deepcopy(run.row)
        row["stage"] = stage
        row["steps_completed"] = run.t
        row["early_stopped_at"] = run.early
        if row["status"] == "ok":
            row["adapter_check"] = self.ad.check_mask()
        row["b_stats"] = self.b_stats()
        row["times"] = {"total": round(run.secs, 1)}
        row["peak_gpu_gb"] = oe.peak_gb()
        self.rows[run.lr] = row
        oe.append_row(self.cdir / "runs.jsonl", row)
        return row

    def save_best(self, lr: float, step: int, L: float, sc: dict) -> None:
        self.cdir.mkdir(parents=True, exist_ok=True)
        oe.save_adapter(self.ad.model, self.cdir / "adapter")
        self.best = {
            "lr": lr,
            "step": step,
            "loss": L,
            "think_nats": list(sc["think_nats"]),
            "answer_nats": list(sc["answer_nats"]),
        }

    def state_dir(self, lr: float) -> Path:
        return self.cdir / "state" / f"lr_{lr:g}"

    def stage_a(self, lr: float) -> None:
        run = self.new_run(lr)
        self.advance(run, self.after, self.evals_a)
        self.lruns[lr] = run
        if self.ext and run.early is None and run.row["status"] == "ok" and run.t >= self.after:
            oe.save_adapter(self.ad.model, self.state_dir(lr), opt=run.opt)
        self.snapshot(run, "a")

    def winner_lr(self) -> float | None:
        """Extension winner: lowest dev loss at step `after` among lrs that reached it without an early stop (with no
        extension: the cell's best lr so far)."""
        if not self.ext:
            return self.best["lr"] if self.best else None
        ok = {
            lr: r.row["dev"][str(self.after)]["loss"]
            for lr, r in self.lruns.items()
            if r.early is None and r.row["status"] == "ok" and str(self.after) in r.row["dev"]
        }
        return min(ok, key=ok.get) if ok else None

    def stage_b(self, lr: float) -> None:
        run = self.lruns[lr]
        run.opt = oe.load_state(self.ad.model, run.params, self.state_dir(lr), lr)
        run.early = None
        self.advance(run, self.total, self.evals_b)
        self.snapshot(run, "a+b")

    # a ready-made adapter, no training
    def run_init(self) -> dict:
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ad, model, fam = self.ad, self.ad.model, self.fam
        src = Path(self.init_adapter)
        cfg = json.loads((src / "adapter_config.json").read_text())
        if (
            int(cfg["r"]) != fam.r
            or float(cfg["lora_alpha"]) != fam.alpha
            or set(cfg["target_modules"]) != set(fam.targets)
        ):
            raise RuntimeError(
                f"init_adapter config (r={cfg['r']}, alpha={cfg['lora_alpha']}, targets={sorted(cfg['target_modules'])}) does not match family {fam.spec}"
            )
        dev0 = self.step0.dev(model, self.dev, bt=self.ctx["eval_bt"], pad_id=self.pad_id)
        ad.reset(self.seed)  # inert (B = 0)
        trn = oe.trainable(model)
        sd = load_file(str(src / "adapter_model.safetensors"), device=str(trn[0].device))
        want = {n_.replace(".default", "") for n_, _ in model.named_parameters() if "lora_" in n_}
        missing, unexpected = sorted(want - set(sd)), sorted(set(sd) - want)
        if missing or unexpected:
            raise RuntimeError(
                f"init_adapter tensor names: {len(missing)} missing {missing[:3]}, {len(unexpected)} unexpected {unexpected[:3]}"
            )
        set_peft_model_state_dict(model, sd)
        sc = oe.score(model, self.dev, batch_tokens=self.ctx["eval_bt"], pad_id=self.pad_id)
        L = dev_loss(sc, self.dev, self.select_on)
        self.best = {
            "lr": None,
            "step": 0,
            "loss": L,
            "think_nats": list(sc["think_nats"]),
            "answer_nats": list(sc["answer_nats"]),
        }
        oe.save_adapter(model, self.cdir / "adapter")
        out_sd = load_file(str(self.cdir / "adapter" / "adapter_model.safetensors"), device="cpu")
        exact = set(out_sd) == set(sd) and all(torch.equal(out_sd[k], sd[k].cpu()) for k in sd)
        reexport_diff = max(float((out_sd[k] - sd[k].cpu()).abs().max()) for k in sd)
        reload_diff = self.reload_and_gate(self.best)
        L0 = dev_loss(dev0, self.dev, self.select_on)
        dev_tot = np.add(sc["think_nats"], sc["answer_nats"])
        dtok = np.array([e.n_tok for e in self.dev], dtype=float)
        cell_json = {
            "schema": SCHEMA,
            "cell": self.cell["cell"],
            "family": fam.spec,
            "P": ad.P,
            "P_formula": ad.formula,
            "P_matches_formula": ad.formula == ad.P,
            "n": self.n,
            "steps": 0,
            "source": self.cell.get("source", "final"),
            "ckpt": self.ctx["ckpt"],
            "pool": str(self.pool),
            "pool_manifest_sha256": pool_sha(self.pool),
            "lora_r": fam.r,
            "target_modules": list(fam.targets),
            "select_on": self.select_on,
            "selection": "init adapter, no training",
            "init_adapter": str(src),
            "init_adapter_config": cfg,
            "init_adapter_tensors": len(sd),
            "init_adapter_missing": missing,
            "init_adapter_unexpected": unexpected,
            "reexport_equal": bool(exact),
            "reexport_max_abs_diff": reexport_diff,
            "lrs_tried": [],
            "selected": {
                "lr": None,
                "step": 0,
                "dev_nats_per_tok_total": float(dev_tot.sum() / dtok.sum()),
                "dev_nats_per_tok_answer": float(
                    np.sum(sc["answer_nats"]) / sum(e.n_tok - e.think_end for e in self.dev)
                ),
                "criterion_loss": L,
            },
            "step0_dev_nats_per_tok_total": dev_loss(dev0, self.dev, "total"),
            "step0_dev_nats_per_tok_answer": dev_loss(dev0, self.dev, "answer"),
            "did_not_train": bool(not L < L0),
            "unconverged": None,
            "mdl_nats": None,
            "mdl_tokens": None,
            "n_online_examples": 0,
            "edl_tok_nats": None,
            "edl_ex_nats": None,
            "edl_tok_bits": None,
            "edl_ex_bits": None,
            "edl_tok_bits_per_trained_weight": None,
            "edl_ex_bits_per_trained_weight": None,
            "edl_selected": None,
            "edl_fixed_step": None,
            "edl_by_step": {},
            "edl_selected_corr": None,
            "edl_fixed_step_corr": None,
            "edl_by_step_corr": {},
            "step0_train_nats_per_tok": self.s0["mean_nats_per_tok_total"],
            "delta_nats_per_tok": None,
            "reload_check_max_abs_diff": reload_diff,
            "written_utc": oe.utc(),
        }
        oe.atomic_write(self.cdir / "cell.json", json.dumps(cell_json, indent=1) + "\n")
        (self.cdir / "DONE").write_text(oe.utc() + "\n")
        return cell_json

    # whole cell
    def run(self) -> dict:
        if self.cdir.exists():
            shutil.rmtree(self.cdir)
        self.cdir.mkdir(parents=True)
        self.load()
        if self.init_adapter:
            return self.run_init()
        lrs = self.resolve_lrs()
        tried: list[float] = []
        for lr in lrs:
            self.stage_a(lr)
            tried.append(lr)
        if self.cell.get("edge_extend") and self.best is not None and lrs:
            sel = self.winner_lr()
            if sel is not None:
                top = math.isclose(sel, max(lrs), rel_tol=1e-9)
                if top or math.isclose(sel, min(lrs), rel_tol=1e-9):
                    nxt = lr_neighbor(sel, up=top)
                    if not any(math.isclose(nxt, x, rel_tol=1e-9) for x in tried):
                        self.stage_a(nxt)
                        self.rows[nxt]["edge_extension"] = True
                        tried.append(nxt)
        if self.ext:
            w = self.winner_lr()
            if w is not None:
                self.stage_b(w)
        shutil.rmtree(self.cdir / "state", ignore_errors=True)
        return self.finalize(tried)

    def reload_and_gate(self, b: dict) -> float:
        """Scramble, reload the saved adapter, re-score the dev set (max abs diff vs the recorded nats), write gate.npz."""
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ad, model = self.ad, self.ad.model
        ad.reset(self.seed + 12345)
        trn = oe.trainable(model)
        sd = load_file(
            str(self.cdir / "adapter" / "adapter_model.safetensors"),
            device=str(trn[0].device) if trn else "cpu",
        )
        set_peft_model_state_dict(model, sd)
        sc = oe.score(model, self.dev, batch_tokens=self.ctx["eval_bt"], pad_id=self.pad_id)
        tot = np.add(sc["think_nats"], sc["answer_nats"])
        rec = np.add(b["think_nats"], b["answer_nats"])
        reload_diff = float(np.max(np.abs(tot - rec)))
        # gate arrays
        n_tr = 1 if self.n == 1 else 2
        gate_ex = [("train", e) for e in self.train[:n_tr]] + [
            ("dev", e) for e in self.dev[: 4 - n_tr]
        ]
        gate = {
            "kind": np.array([k for k, _ in gate_ex]),
            "pid": np.array([e.pid for _, e in gate_ex]),
        }
        with torch.inference_mode():
            model.eval()
            for j, (_, e) in enumerate(gate_ex):
                gate[f"prompt_ids_{j}"] = np.asarray(e.prompt, dtype=np.int32)
                gate[f"resp_ids_{j}"] = np.asarray(e.resp, dtype=np.int32)
                gate[f"adapter_nll_{j}"] = (
                    oe.token_nll(model, [(e.prompt, e.resp)], pad_id=self.pad_id)[0]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                with model.disable_adapter():
                    gate[f"base_nll_{j}"] = (
                        oe.token_nll(model, [(e.prompt, e.resp)], pad_id=self.pad_id)[0]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
        np.savez(self.cdir / "gate.npz", **gate)
        return reload_diff

    def finalize(self, tried: list[float]) -> dict:
        ad = self.ad
        cid = self.cell["cell"]
        dev0 = self.step0.d["dev"]
        L0 = dev_loss(dev0, self.dev, self.select_on)
        L0_total = dev_loss(dev0, self.dev, "total")
        L0_answer = dev_loss(dev0, self.dev, "answer")
        if self.best is None:
            raise RuntimeError(f"{cid}: no run produced a dev evaluation")
        b = self.best
        run = self.rows[b["lr"]]
        reload_diff = self.reload_and_gate(b)
        # EDL at the selected step, at fixed step 64 per lr, and by eval step
        dev_tot = np.add(b["think_nats"], b["answer_nats"])
        dev_tok = np.array([e.n_tok for e in self.dev], dtype=float)
        dev_ans_tok = np.array([e.n_tok - e.think_end for e in self.dev], dtype=float)
        L_tot = float(dev_tot.sum() / dev_tok.sum())
        L_ans = float(np.sum(b["answer_nats"]) / dev_ans_tok.sum())
        sel_edl = edl_at(run, b["step"], self.n, self.dev, ad.P)
        FIX = int(self.cell.get("edl_fixed_step", 64))
        edl_fixed = fixed_step_block(self.rows, FIX, self.n, self.dev_tok, ad.P)
        edl_by_step = {
            s: edl_at(run, int(s), self.n, self.dev, ad.P)
            for s in run["dev"]
            if s != "0" and int(s) <= self.n
        }
        # unconverged: the selected lr's last dev loss still falling by more than 1 % of (L0 - L_best)
        evs = sorted(int(s) for s in run["dev"] if s != "0")
        unconverged = None
        if len(evs) >= 2:
            prev, last = run["dev"][str(evs[-2])]["loss"], run["dev"][str(evs[-1])]["loss"]
            unconverged = bool(prev - last > 0.01 * (L0 - b["loss"]))
        did_not_train = not any(
            dev_loss(r["dev"][s], self.dev, self.select_on) < L0
            for r in self.rows.values()
            for s in r["dev"]
            if s != "0"
        )
        lr_summary = {
            f"{lr:g}": {
                "status": r["status"],
                "stage": r["stage"],
                "steps_completed": r["steps_completed"],
                "early_stopped_at": r["early_stopped_at"],
                "dev_loss_by_step": {s: v["loss"] for s, v in r["dev"].items()},
                "dev_loss_answer_by_step": {s: v["loss_answer"] for s, v in r["dev"].items()},
                **r["b_stats"],
            }
            for lr, r in self.rows.items()
        }
        cell_json = {
            "schema": SCHEMA,
            "cell": cid,
            "family": self.fam.spec,
            "P": ad.P,
            "P_formula": ad.formula,
            "P_matches_formula": ad.formula == ad.P,
            "n": self.n,
            "steps": self.total,
            "schedule": {
                "evals_a": self.evals_a,
                "after": self.after,
                "evals_b": self.evals_b,
                "to": self.total,
                "extend": self.ext,
            },
            "source": self.manifest.get("source", self.pool.name),
            "ckpt": self.ctx["ckpt"],
            "pool": str(self.pool),
            "pool_manifest_sha256": pool_sha(self.pool),
            "lora_r": self.fam.r,
            "target_modules": list(self.fam.targets),
            "select_on": self.select_on,
            "lrs_tried": tried,
            "run_status": {r["lr"]: r["status"] for r in self.rows.values()},
            "lr_summary": lr_summary,
            "selected": {
                "lr": b["lr"],
                "step": b["step"],
                "dev_nats_per_tok_total": L_tot,
                "dev_nats_per_tok_answer": L_ans,
                "criterion_loss": b["loss"],
            },
            "step0_dev_nats_per_tok_total": L0_total,
            "step0_dev_nats_per_tok_answer": L0_answer,
            "did_not_train": bool(did_not_train),
            "unconverged": unconverged,
            "early_stopped_at": run["early_stopped_at"],
            "mdl_nats": sel_edl["mdl_nats"],
            "mdl_tokens": sel_edl["mdl_tokens"],
            "selection": "fixed recipe" if self.fixed else "dev loss",
            "recipe_from": self.cell.get("recipe_from"),
            "recipe": self.recipe,
            "n_online_examples": sel_edl["m"],
            "edl_tok_nats": sel_edl["edl_tok_nats"],
            "edl_ex_nats": sel_edl["edl_ex_nats"],
            "edl_tok_bits": sel_edl["edl_tok_bits"],
            "edl_ex_bits": sel_edl["edl_ex_bits"],
            "edl_tok_bits_per_trained_weight": sel_edl["edl_tok_bits_per_trained_weight"],
            "edl_ex_bits_per_trained_weight": sel_edl["edl_ex_bits_per_trained_weight"],
            "edl_selected": sel_edl,
            "edl_fixed_step": edl_fixed,
            "edl_by_step": edl_by_step,
            "edl_note": "step i scores example i before its first update; at step t, m = min(t, n): EDL_tok = sum_{i<=m} nats_i - (sum_{i<=m} tokens_i) * L_dev(theta_t) [nats/token, total]; EDL_ex = sum_{i<=m} nats_i - m * mean per-example dev nats(theta_t). edl_fixed_step: among the lrs whose run REACHED the step (evaluated there, early_stopped_at None or >= step), the lr with the lowest dev loss at that step. The *_corr blocks are difficulty-matched (edl_corrected, step-0 nats of the same first m training examples) and are the primary capacity numbers; the raw ones are kept.",
            "b_stats": run["b_stats"],
            "reload_check_max_abs_diff": reload_diff,
            "reset_check_abs_diff": run.get("reset_check", {}).get("abs_diff"),
            "adapter_check": run.get("adapter_check"),
            "written_utc": oe.utc(),
        }
        corr = edl_corrected(cell_json, self.s0)
        cell_json.update(
            edl_selected_corr=corr["edl_selected"],
            edl_fixed_step_corr=corr["edl_fixed_step"],
            edl_by_step_corr=corr["edl_by_step"],
            step0_train_nats_per_tok=self.s0["mean_nats_per_tok_total"],
            delta_nats_per_tok=corr["edl_selected"]["delta_nats_per_tok"],
        )
        oe.atomic_write(self.cdir / "cell.json", json.dumps(cell_json, indent=1) + "\n")
        (self.cdir / "DONE").write_text(oe.utc() + "\n")
        return cell_json


# -------------------------------------------------------------------------------------------- driver


def load_plan(path: Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    cells = d["cells"] if isinstance(d, dict) else d
    ids = [c["cell"] for c in cells]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    return cells


def group_by_family(cells: list[dict]) -> list[dict]:
    first: dict[str, int] = {}
    for i, c in enumerate(cells):
        first.setdefault(c["family"], i)
    return sorted(cells, key=lambda c: (first[c["family"]], cells.index(c)))


def select_cells(
    cells: list[dict], *, conditional: bool = False, only: list[str] | None = None
) -> list[dict]:
    """Drop template cells always, conditional cells unless asked for, and keep only `only` ids when given."""
    out = [c for c in cells if not c.get("template")]
    if only:
        return [c for c in out if c["cell"] in only]
    return [c for c in out if conditional or not c.get("conditional")]


def run_cells(
    model, tok, cells: list[dict], ctx: dict, *, grad_ckpt: bool = True, group: bool = False
) -> int:
    out = Path(ctx["out"])
    hb = ctx.get("hb")
    ad: Adapter | None = None
    fam_spec = None
    failed = 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for cell in group_by_family(cells) if group else cells:
        cdir = out / "cells" / cell["cell"]
        if (cdir / "DONE").exists():
            print(f"skip {cell['cell']} (DONE)", flush=True)
            continue
        if cell["family"] != fam_spec:
            base = ad.release() if ad is not None else model
            if ad is not None:
                ctx["base_checks"].append(base_check(base, tok, cell, ctx, pad_id))
            ad = Adapter(base, Family(cell["family"]), grad_ckpt=grad_ckpt)
            fam_spec = cell["family"]
            print(
                f"[{oe.utc()}] wrapped family {fam_spec}: P={ad.P} (formula {ad.formula}) over {ad.n_modules} modules",
                flush=True,
            )
        if hb:
            hb.progress(phase="cell", cell=cell["cell"])
        try:
            cj = CellRunner(ad, tok, cell, ctx).run()
            print(
                f"[{oe.utc()}] DONE {cell['cell']}: selected {cj['selected']} step0 {cj['step0_dev_nats_per_tok_total']:.4f} did_not_train={cj['did_not_train']} reload_diff={cj['reload_check_max_abs_diff']:.2e}",
                flush=True,
            )
        except (oe.NonFinite, RuntimeError, AssertionError) as e:
            failed += 1
            cdir.mkdir(parents=True, exist_ok=True)
            oe.atomic_write(
                cdir / "FAILED.json",
                json.dumps(
                    {"utc": oe.utc(), "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
                )
                + "\n",
            )
            print(f"FAILED {cell['cell']}: {e!r}", flush=True)
    return 2 if failed else 0


def base_check(base, tok, cell, ctx, pad_id) -> dict:
    """After unloading: the bare base scores one dev example as the step-0 cache (adapter inert) did."""
    pool = Path(ctx["root"]) / cell["pool"]
    sets, _ = oe.load_pool(pool, tok, n=1, m=1, max_tok=None, n_reach=0)
    ex = sets["test"][0]
    sc = oe.score(base, [ex], batch_tokens=0, pad_id=pad_id)
    step0 = Step0(Path(ctx["out"]) / "step0" / f"{ctx['ckpt']}_{pool_sha(pool)[:16]}.json")
    res = {"pid": ex.pid, "nats": sc["think_nats"][0] + sc["answer_nats"][0], "abs_diff": None}
    if step0.d["dev"] and ex.pid in step0.d["dev"]["pids"]:
        k = step0.d["dev"]["pids"].index(ex.pid)
        ref = step0.d["dev"]["think_nats"][k] + step0.d["dev"]["answer_nats"][k]
        res["abs_diff"] = abs(res["nats"] - ref)
        if res["abs_diff"] > 1e-2:
            raise SystemExit(f"base after unload scores {res['nats']} vs step-0 {ref}")
    print(f"base check after unload: {res}", flush=True)
    return res


def setup(a):
    import torch
    from transformers import AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    t0 = time.time()
    model = oe.load_model(a.model, None, dtype=a.dtype, device=device, attn=a.attn)
    oe.OPTS["ce_checkpoint"] = bool(a.ce_checkpoint)
    oe.OPTS["loss_chunk"] = int(a.loss_chunk or 2048)
    if torch.cuda.is_available() and a.max_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_memory_gb * 2**30 / total), 0)
    print(f"[{oe.utc()}] loaded {a.model} in {time.time() - t0:.0f}s on {device}", flush=True)
    return tok, model, time.time() - t0


def cmd_run(a) -> int:
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = oe.Heartbeat(out, a.heartbeat_secs, ckpt=a.ckpt).start()
    hb.progress(phase="load")
    try:
        tok, model, t_load = setup(a)
        cells = select_cells(
            load_plan(Path(a.plan)),
            conditional=a.conditional,
            only=a.only.split(",") if a.only else None,
        )
        ctx = {
            "out": out,
            "root": a.root,
            "ckpt": a.ckpt,
            "hb": hb,
            "eval_bt": a.eval_batch_tokens,
            "select_on": a.select_on,
            "base_checks": [],
        }
        oe.atomic_write(
            out / "run_info.json",
            json.dumps(
                {
                    "env": oe.env_info(),
                    "identity": oe.model_identity(a.model, None, model, None),
                    "load_secs": round(t_load, 1),
                    "plan": str(a.plan),
                    "recipe": oe.RECIPE,
                }
            )
            + "\n",
        )
        rc = run_cells(model, tok, cells, ctx, grad_ckpt=not a.no_grad_ckpt, group=a.group_families)
        if rc == 0:
            (out / "DONE").write_text(oe.utc() + "\n")
        hb.stop("done" if rc == 0 else "partial")
        return rc
    except torch.cuda.OutOfMemoryError as e:
        hb.stop("error")
        oe.atomic_write(
            out / "ERROR.json",
            json.dumps({"utc": oe.utc(), "error": "CUDA out of memory", "detail": str(e)[:2000]})
            + "\n",
        )
        print(f"FATAL out of memory: {e}", flush=True)
        return 3
    except SystemExit as e:
        hb.stop("error")
        oe.atomic_write(out / "ERROR.json", json.dumps({"utc": oe.utc(), "error": str(e)}) + "\n")
        print(f"FATAL {e}", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        hb.stop("error")
        oe.atomic_write(
            out / "ERROR.json",
            json.dumps(
                {"utc": oe.utc(), "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
            )
            + "\n",
        )
        traceback.print_exc()
        return 3


def step0_train(
    model, tok, root: Path, out: Path, ckpt: str, pool: str, n: int, bt: int, hb=None
) -> dict:
    """Step-0 (untouched checkpoint) scores of the first n training examples and the dev set of one pool, via the same Step0 cache
    the cells use (batch 1 for train, so each example equals a cell's online step-0 score)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    pdir = Path(root) / pool
    sets, _ = oe.load_pool(pdir, tok, n=n, m=None, max_tok=None, n_reach=0)
    (out / "step0").mkdir(parents=True, exist_ok=True)
    cache = Step0(out / "step0" / f"{ckpt}_{pool_sha(pdir)[:16]}.json")
    dev0 = cache.dev(model, sets["test"], bt=bt, pad_id=pad_id)
    th, an = [], []
    for i, ex in enumerate(sets["train"]):
        cache.train(model, ex, pad_id=pad_id)
        t_, a_ = cache.d["train"][ex.pid]
        th.append(t_)
        an.append(a_)
        if hb:
            hb.progress(phase="step0-train", pool=pool, i=i + 1)
    tr = sets["train"]
    ntok = [e.n_tok for e in tr]
    nans = [e.n_tok - e.think_end for e in tr]
    dtok = sum(e.n_tok for e in sets["test"])
    dans = sum(e.n_tok - e.think_end for e in sets["test"])
    res = {
        "ckpt": ckpt,
        "pool": pool,
        "n": n,
        "pids": [e.pid for e in tr],
        "think_nats": th,
        "answer_nats": an,
        "n_tok": ntok,
        "n_answer": nans,
        "sum_nats": sum(th) + sum(an),
        "sum_tokens": sum(ntok),
        "mean_nats_per_tok_total": (sum(th) + sum(an)) / sum(ntok),
        "mean_nats_per_tok_answer": sum(an) / sum(nans),
        "L0_dev_total": (sum(dev0["think_nats"]) + sum(dev0["answer_nats"])) / dtok,
        "L0_dev_answer": sum(dev0["answer_nats"]) / dans,
        "pool_manifest_sha256": pool_sha(pdir),
        "written_utc": oe.utc(),
    }
    d = out / "step0_train"
    d.mkdir(parents=True, exist_ok=True)
    oe.atomic_write(
        d / f"{ckpt}__{pool.replace('/', '_')}__n{n}.json", json.dumps(res, indent=1) + "\n"
    )
    return res


def edl_corrected(cell_json: dict, s0: dict, t: int | None = None) -> dict:
    """Difficulty-matched EDL_tok: EDL_corr = sum nats_i - tokens_m * (L_dev(theta_t) + delta_m) = edl_tok - tokens_m * delta_m,
    delta_m = (sum_{i<=m} step-0 nats_i) / (sum_{i<=m} tokens_i) - L0_dev over the SAME first m examples. Pure json.
    Returns the corrected blocks for edl_selected, edl_fixed_step (and each of its per_lr entries) and edl_by_step (only step t if given)."""
    P = cell_json["P"]
    nat0 = np.add(s0["think_nats"], s0["answer_nats"])

    def corr(b: dict) -> dict:
        m = int(b["m"])
        delta = float(nat0[:m].sum() / sum(s0["n_tok"][:m]) - s0["L0_dev_total"])
        v = b["edl_tok_nats"] - b["mdl_tokens"] * delta
        return {
            "t": b["t"],
            "m": m,
            "delta_nats_per_tok": delta,
            "edl_tok_nats": b["edl_tok_nats"],
            "edl_tok_corr_nats": v,
            "edl_tok_corr_bits": v / LN2,
            "edl_tok_corr_bits_per_trained_weight": v / LN2 / P,
        }

    out: dict = {
        "edl_selected": corr(cell_json["edl_selected"]) if cell_json.get("edl_selected") else None
    }
    f = cell_json.get("edl_fixed_step")
    out["edl_fixed_step"] = (
        {**corr(f), "lr": f["lr"], "per_lr": {k: corr(v) for k, v in f["per_lr"].items()}}
        if f
        else None
    )
    out["edl_by_step"] = {
        k: corr(v) for k, v in cell_json.get("edl_by_step", {}).items() if t is None or int(k) == t
    }
    return out


def cmd_step0_train(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = oe.Heartbeat(out, a.heartbeat_secs, ckpt=a.ckpt).start()
    hb.progress(phase="load")
    try:
        tok, model, _ = setup(a)
        for pool in a.pools.split(","):
            r = step0_train(
                model, tok, Path(a.root), out, a.ckpt, pool, a.n, a.eval_batch_tokens, hb
            )
            print(
                f"[{oe.utc()}] {pool}: {a.n} examples, {r['mean_nats_per_tok_total']:.4f} nats/tok train vs dev {r['L0_dev_total']:.4f}",
                flush=True,
            )
        hb.stop("done")
        return 0
    except Exception as e:  # noqa: BLE001
        hb.stop("error")
        oe.atomic_write(
            out / "ERROR.json",
            json.dumps(
                {"utc": oe.utc(), "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
            )
            + "\n",
        )
        traceback.print_exc()
        return 3


def cmd_bench(a) -> int:
    """Time one train step and one forward at the longest pool sequence per family; project the plan's cost (linear in tokens)."""
    import torch

    tok, model, _ = setup(a)
    cells = select_cells(
        load_plan(Path(a.plan)),
        conditional=a.conditional,
        only=a.only.split(",") if a.only else None,
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    per_tok: dict[str, tuple[float, float]] = {}
    ad = None
    for fam_spec in dict.fromkeys(c["family"] for c in cells):
        base = ad.release() if ad is not None else model
        ad = Adapter(base, Family(fam_spec), grad_ckpt=not a.no_grad_ckpt)
        c0 = next(c for c in cells if c["family"] == fam_spec)
        sets, _ = oe.load_pool(
            Path(a.root) / c0["pool"], tok, n=None, m=None, max_tok=None, n_reach=0
        )
        ex = max(sets["train"] + sets["test"], key=lambda e: e.total)
        ad.reset(0)
        params = oe.trainable(ad.model)
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        oe.reset_peak()
        ts, fs = [], []
        for _ in range(a.reps + 1):
            sync()
            t = time.time()
            oe.train_step(ad.model, opt, params, ex, grad_clip=1.0, pad_id=pad_id)
            sync()
            ts.append(time.time() - t)
            t = time.time()
            oe.score(ad.model, [ex], batch_tokens=0, pad_id=pad_id)
            sync()
            fs.append(time.time() - t)
        tt, ff = float(np.mean(ts[1:])), float(np.mean(fs[1:]))
        per_tok[fam_spec] = (tt / ex.total, ff / ex.total)
        print(
            f"{fam_spec}: P={ad.P} longest {ex.total} tok: train step {tt:.2f}s, forward {ff:.2f}s, peak {oe.peak_gb()} GB",
            flush=True,
        )
    tot = 0.0
    for c in cells:
        sets, _ = oe.load_pool(
            Path(a.root) / c["pool"], tok, n=None, m=None, max_tok=None, n_reach=0
        )
        n = int(c["n"])
        nl_ = nl = len(c["lrs"]) if "lrs" in c else 1
        ext = c.get("extend")
        steps = int(ext["to"] if ext else c.get("steps", 64))
        seq = step_sequence(n, steps, "bench")
        after = int(ext["after"]) if ext else steps
        tr = nl_ * sum(sets["train"][k].total for k in seq[:after]) + sum(
            sets["train"][k].total for k in seq[after:]
        )
        on = sum(e.total for e in sets["train"][:n])
        dv = sum(e.total for e in sets["test"])
        n_ev = len(c.get("eval_steps") or default_eval_steps(n, after)) + 1
        pt, pf = per_tok[c["family"]]
        secs = tr * pt + nl * (on + n_ev * dv) * pf
        tot += secs
        print(f"  {c['cell']}: {nl} lr(s) -> {secs / 60:.1f} min")
    print(f"projected total {tot / 3600:.2f} h (edge extensions and unload/rewrap not included)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["run", "bench", "step0-train"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument(
        "--pools",
        default=None,
        help="step0-train: comma list of pools under --root, e.g. dpo/final,dpo/own,dpo/unsel",
    )
    ap.add_argument("--n", type=int, default=64, help="step0-train: training examples per pool")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--select-on", choices=["total", "answer"], default="total")
    ap.add_argument("--eval-batch-tokens", type=int, default=16384)
    ap.add_argument("--heartbeat-secs", type=float, default=60.0)
    ap.add_argument("--max-memory-gb", type=float, default=None)
    ap.add_argument("--ce-checkpoint", action="store_true")
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument(
        "--conditional",
        action="store_true",
        help="also run cells marked conditional (template cells never run)",
    )
    ap.add_argument(
        "--only", default=None, help="comma list of cell ids to run (overrides --conditional)"
    )
    ap.add_argument(
        "--group-families",
        action="store_true",
        help="run cells grouped by family (default: plan order)",
    )
    ap.add_argument("--reps", type=int, default=2, help="bench: timed repetitions")
    a = ap.parse_args()
    if a.cmd == "step0-train":
        if not (a.out and a.pools):
            ap.error("step0-train needs --out and --pools")
        sys.exit(cmd_step0_train(a))
    if not a.plan:
        ap.error("run and bench need --plan")
    if a.cmd == "run":
        if not a.out:
            ap.error("run needs --out")
        sys.exit(cmd_run(a))
    sys.exit(cmd_bench(a))


if __name__ == "__main__":
    main()
