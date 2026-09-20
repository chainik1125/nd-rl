#!/usr/bin/env python3
"""Exposure-matched cap-six retention-only control for the four LM arms.

Each outer update samples the exact main-trainer retention RNG schedule and
reuses that batch for four optimizer epochs. There is no rollout, outcome
gradient, entropy bonus, or critic. It controls retention exposure, not compute.
Supports exact resume and arbitrary outer-update checkpoint endpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import signal
import time

import torch

import common
import learning as L
import train_lm as T
from nd_environment import episode

VERSION = "review-retention-exposure-control-v1"


def retention_epoch(model, tok, rows, optimizer, *, weight=.1, microbatch=16,
                    precision="fp32", grad_clip=1., ledger=None):
    """Match the main update's mean-example token NLL and clipping exactly."""
    if not rows or weight <= 0:
        raise ValueError("Retention control needs positive retention weight and examples")
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    nll = 0.
    device = next(model.parameters()).device
    for offset in range(0, len(rows), microbatch):
        batch = L.collate(rows[offset:offset + microbatch], tok.pad_id, device)
        selected = L.logprobs(model, batch, precision=precision)
        per_example = -(selected * batch["mask"]).sum(-1) / batch["mask"].sum(-1)
        loss = per_example.sum() / len(rows)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite retention NLL")
        (weight * loss).backward()
        nll += float(loss.detach())
        if ledger is not None:
            L.account(ledger, model, batch, "retention_training", True)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite retention gradient")
    optimizer.step()
    return {"retention_loss": nll, "weighted_loss": weight * nll, "grad_norm": float(norm)}


def identity(args):
    ceilings = {"resume", "max_updates", "max_seconds", "device", "cpu_threads", "checkpoint_every", "checkpoint_steps", "log_every"}
    return {"version": VERSION, "config": {k: v for k, v in vars(args).items() if k not in ceilings},
            "inputs": {k: common.sha(getattr(args, k)) for k in ("checkpoint", "retention")},
            "code": {str(Path(__file__).resolve()): common.sha(__file__), **T.code_identity()}}


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ("manifest.json", "updates.jsonl", "final.pt")):
        raise ValueError("Existing run requires explicit resume")
    ident = identity(args)
    ident_sha = common.digest(ident)
    records = T.training_sources(args.retention)
    model, tok, origin = L.load_model(args.resume or args.checkpoint, args.device)
    rows = [episode(r, 0, tok) for r in records]
    if max(len(r["prompt_ids"]) + len(r["response_ids"]) for r in rows) > model.config.max_seq_len:
        raise ValueError("Retention certificate exceeds model context")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(.9, .95), fused=next(model.parameters()).is_cuda)
    rng = random.Random(args.seed + 1000003)
    state = {"step": 0, "optimizer_steps": 0, "retention_examples": 0, "retention_response_tokens": 0,
             "elapsed_seconds": 0., "ledger": {}, "retention_schedule_digest": "", "identity_sha256": ident_sha}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = payload["training_state"]
        if previous["identity_sha256"] != ident_sha:
            raise ValueError("Resume protocol/data/code identity mismatch")
        optimizer.load_state_dict(previous.pop("optimizer"))
        torch.set_rng_state(previous.pop("torch_rng_state"))
        cuda_states = previous.pop("cuda_rng_states")
        if cuda_states and args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(cuda_states)
        rng.setstate(previous.pop("retention_rng_state"))
        state = previous
        recovery = T.recover_logs(output, state["step"])
        T.append_jsonl(output / "resume_events.jsonl", [{"checkpoint": str(Path(args.resume).resolve()),
            "checkpoint_sha256": common.sha(args.resume), "step": state["step"], "discarded_tail_rows": recovery,
            "requested_config": vars(args), "time_unix": time.time()}])
    else:
        common.write_json(output / "manifest.json", {"identity": ident, "identity_sha256": ident_sha,
            "config": vars(args), "retention_records": len(records), "source_cap": 6,
            "origin_metadata": origin, "rollout_or_outcome_gradient": False, "critic": False,
            "matching": "Same main retention RNG seed+1000003, batch, epochs, optimizer, temperature-1 proof-mean NLL, coefficient and gradient clip; exposure matched, compute not matched.",
            "interpretation": "Bounds continued short-proof fitting as a source of absolute gain; does not subtract retention effects additively from RL.",
            "time_unix": time.time()})
    ledger = state["ledger"]
    elapsed_before = state["elapsed_seconds"]
    start = time.monotonic()
    stop_signal = {"value": None}
    old_handlers = {s: signal.signal(s, lambda number, frame: stop_signal.update(value=number))
                    for s in (signal.SIGINT, signal.SIGTERM)}
    def elapsed():
        return elapsed_before + time.monotonic() - start
    def save(path):
        state["elapsed_seconds"] = elapsed()
        state["retention_rng_state"] = rng.getstate()
        T.atomic_checkpoint(path, model, None, tok, optimizer, state,
                            {"step": state["step"], "objective": "retention_only", "source_cap": 6,
                             "identity_sha256": ident_sha, "review_control": VERSION})
        common.write_json(output / "latest_checkpoint.json", {"path": str(path.resolve()), "step": state["step"]})
    stop_reason = "outer_update_ceiling"
    try:
        for step in range(state["step"] + 1, args.max_updates + 1):
            if stop_signal["value"] is not None:
                stop_reason = f"signal_{stop_signal['value']}"
                break
            if elapsed() >= args.max_seconds:
                stop_reason = "wall_time_ceiling"
                break
            selected = [rows[rng.randrange(len(rows))] for _ in range(args.retention_batch)]
            state["retention_schedule_digest"] = common.digest([state["retention_schedule_digest"], [r["source_id"] for r in selected]])
            records_epoch = [retention_epoch(model, tok, selected, optimizer, weight=args.retention_weight,
                microbatch=args.microbatch, precision=args.precision, grad_clip=args.grad_clip, ledger=ledger)
                for _ in range(args.epochs)]
            state["step"] = step
            state["optimizer_steps"] += args.epochs
            state["retention_examples"] += len(selected) * args.epochs
            state["retention_response_tokens"] += sum(len(r["response_ids"]) for r in selected) * args.epochs
            event = {"step": step, "optimizer_steps_total": state["optimizer_steps"], "epochs": records_epoch,
                     "retention_source_ids": [r["source_id"] for r in selected],
                     "retention_schedule_digest": state["retention_schedule_digest"],
                     "retention_examples_total": state["retention_examples"],
                     "retention_response_tokens_total": state["retention_response_tokens"],
                     "cumulative_flop_proxy": L.total_flop_proxy(ledger), "elapsed_seconds": elapsed()}
            T.append_jsonl(output / "updates.jsonl", [event])
            common.write_json(output / "progress.json", event)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(event), flush=True)
            if step in args.checkpoint_steps or (args.checkpoint_every and step % args.checkpoint_every == 0):
                save(output / f"step_{step:06d}.pt")
        save(output / "final.pt")
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    result = {"status": "completed", "objective": "retention_only", "step": state["step"],
              "optimizer_steps": state["optimizer_steps"], "retention_examples": state["retention_examples"],
              "retention_response_tokens": state["retention_response_tokens"], "generated_tokens": 0,
              "ledger": ledger, "cumulative_flop_proxy": L.total_flop_proxy(ledger),
              "elapsed_seconds": elapsed(), "stop_reason": stop_reason,
              "retention_schedule_digest": state["retention_schedule_digest"],
              "identity_sha256": ident_sha, "final_checkpoint_sha256": common.sha(output / "final.pt")}
    common.write_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    for key in ("checkpoint", "retention", "output", "resume"):
        p.add_argument("--" + key)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    for key, value in (("seed", 20260912), ("cpu_threads", 4), ("max_updates", 5000),
                       ("retention_batch", 16), ("epochs", 4), ("microbatch", 16),
                       ("checkpoint_every", 500), ("log_every", 100)):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    for key, value in (("lr", 5e-5), ("weight_decay", 0.), ("retention_weight", .1),
                       ("grad_clip", 1.), ("max_seconds", 3600.)):
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    p.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        cfg = json.loads(Path(pre.config).read_text())
        unknown = set(cfg) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"Unknown control configuration keys: {sorted(unknown)}")
        p.set_defaults(**cfg)
    args = p.parse_args(argv)
    del args.config
    for key in ("checkpoint", "retention", "output"):
        if not getattr(args, key):
            p.error(f"--{key} is required")
    for key in ("checkpoint", "retention", "output", "resume"):
        if getattr(args, key):
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
    for key in ("cpu_threads", "max_updates", "retention_batch", "epochs", "microbatch", "log_every",
                "lr", "retention_weight", "grad_clip", "max_seconds"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.weight_decay < 0 or args.checkpoint_every < 0 or any(x <= 0 for x in args.checkpoint_steps):
        p.error("Invalid nonnegative weight decay/checkpoint schedule")
    return args


if __name__ == "__main__":
    run(arguments())
