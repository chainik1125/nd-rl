"""Capture frozen proof-state attempts without exposing verifier feedback."""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from lean_gate import check_sources  # noqa: E402 — pinned upstream source is added above
from model import load_ckpt  # noqa: E402 — pinned upstream source is added above
from sample import generate_ids_fast  # noqa: E402 — pinned upstream source is added above
from state_env import Env  # noqa: E402 — pinned upstream source is added above
from state_sample import prompt_ids  # noqa: E402 — pinned upstream source is added above


@torch.no_grad()
def likelihood(model, prefix, output, temperature):
    ids = torch.tensor(prefix + output, device="cuda")[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)[:, len(prefix) - 1 : -1].float() / temperature
    logp = logits.log_softmax(-1)
    targets = torch.tensor(output, device="cuda")[None, :, None]
    lp = logp.gather(-1, targets).sum().item()
    entropy = (-(logp.exp() * logp).sum(-1)).sum().item()
    return lp, entropy


@torch.no_grad()
def likelihood_batch(model, prefixes, outputs, temperature):
    sequences = [p + o for p, o in zip(prefixes, outputs, strict=True)]
    ids = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long, device="cuda")
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids).float()
    values = []
    for i, (prefix, output) in enumerate(zip(prefixes, outputs, strict=True)):
        lo = logits[i, len(prefix) - 1 : len(prefix) + len(output) - 1]
        base = lo.log_softmax(-1)
        sample = (lo / temperature).log_softmax(-1)
        target = torch.tensor(output, device="cuda")[:, None]
        values.append(
            (
                sample.gather(-1, target).sum().item(),
                base.gather(-1, target).sum().item(),
                (-(base.exp() * base).sum(-1)).sum().item(),
            )
        )
    return values


def trim(output, tok):
    out = []
    for i in output:
        out.append(i)
        if i == tok.eos:
            break
    return out


@torch.no_grad()
def collect(model, tok, theorems, args):
    pending = [(r, k) for r in theorems for k in range(args.k)]
    live = []
    nxt = 0
    wave = 0
    results = {r["theorem_id"]: {"public": [], "private": []} for r in theorems}
    while nxt < len(pending) or live:
        while len(live) < args.batch and nxt < len(pending):
            r, k = pending[nxt]
            base = random.Random(args.seed * 1000003 + nxt).randint(0, 32)
            e = Env(r["prompt"], canon=True, base=base, assign=True)
            c = {
                "candidate_id": k,
                "raw_actions": [],
                "raw_tokens": [],
                "effective_actions": [],
                "effective_tokens": [],
                "rendered_text": "",
                "generated_tokens": 0,
                "prompt_tokens": 0,
                "logprob_sum": 0.0,
                "base_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "forward_calls": 0,
                "sampler_seed": args.seed,
                "canonical_name_base": base,
            }
            live.append((r, k, e, c))
            nxt += 1
        prefixes = [prompt_ids(tok, e) for r, k, e, c in live]
        if hasattr(model, "sprint_compute"):
            model.sprint_compute["phase"] = "sampling"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = generate_ids_fast(
                model,
                tok,
                prefixes,
                greedy=False,
                temperature=args.temperature,
                max_new=args.max_action,
                seed=args.seed * 100003 + wave,
                early="eos",
                compact=True,
            )
        outputs = [trim(output, tok) for output in outputs]
        if hasattr(model, "sprint_compute"):
            model.sprint_compute["phase"] = "likelihood_replay"
        diagnostics = likelihood_batch(model, prefixes, outputs, args.temperature)
        wave += 1
        keep = []
        for (r, k, e, c), prefix, output, diagnostic in zip(
            live, prefixes, outputs, diagnostics, strict=True
        ):
            output = trim(output, tok)
            c["raw_actions"].append(output)
            c["raw_tokens"].extend(output)
            c["prompt_tokens"] += len(prefix)
            c["generated_tokens"] += len(output)
            c["forward_calls"] += len(output)
            lp, blp, bh = diagnostic
            c["logprob_sum"] += lp
            c["base_logprob_sum"] += blp
            c["entropy_sum"] += bh
            action, ended = tok.decode_action(output)
            reason = None
            if not ended:
                reason = "action_cap"
            else:
                ok, why = e.apply(action)
                if not ok:
                    reason = "syntax_or_scope"
                elif e.done:
                    reason = "complete"
                elif e.steps >= args.max_steps:
                    reason = "step_cap"
            effective = tok.encode_toks(e._eff) + ([tok.eos] if ended else []) if ended else output
            c["effective_actions"].append(effective)
            c["effective_tokens"].extend(effective)
            if reason is None:
                keep.append((r, k, e, c))
                continue
            c["raw_rendered_text"] = tok.text([tok.itos[x] for x in c["raw_tokens"]])
            c["rendered_text"] = tok.text([tok.itos[x] for x in c["effective_tokens"]])
            c["sampled_pad_tokens"] = sum(x == tok.pad for x in c["raw_tokens"])
            nt = c["generated_tokens"]
            c["mean_logprob"] = c["logprob_sum"] / nt
            c["mean_base_logprob"] = c["base_logprob_sum"] / nt
            c["mean_entropy"] = c["entropy_sum"] / nt
            c["chooser_tokens"] = len(tok.encode_prompt(r["prompt"])) + nt + 3
            nd = e.nd()
            complete = e.done and not nd.startswith("LEANPARSE")
            priv = {
                "candidate_id": k,
                "valid": False,
                "parse_complete": complete,
                "ended_reason": reason,
                "env_failure": e.failed,
                "lean_text": tok.statement(r["prompt"]) + " " + tok.text(e.text)
                if complete
                else None,
                "nd_proof": nd if complete else None,
            }
            results[r["theorem_id"]]["public"].append(c)
            results[r["theorem_id"]]["private"].append(priv)
        live = keep
        if wave % 10 == 0:
            print(
                json.dumps(
                    {
                        "wave": wave,
                        "started": nxt,
                        "total": len(pending),
                        "finished": sum(len(v["public"]) for v in results.values()),
                    }
                ),
                flush=True,
            )
    finished = [p for v in results.values() for p in v["private"] if p["parse_complete"]]
    flags, wall, cpu = check_sources([p["lean_text"] for p in finished])
    for p, flag in zip(finished, flags, strict=True):
        p["valid"] = flag
    print(
        json.dumps(
            {
                "lean_complete": len(finished),
                "lean_valid": sum(flags),
                "lean_wall": wall,
                "lean_process_s": cpu,
            }
        ),
        flush=True,
    )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=41006)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-action", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    rows = rows[: args.limit] if args.limit else rows
    config = {
        "args": vars(args),
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "torch_version": str(torch.__version__),
        "python_version": sys.version.split()[0],
        "checkpoint": str(ck),
        "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "sampling_logprob_temperature": args.temperature,
        "entropy_temperature": 1.0,
        "sampling_precision": (
            "BF16 autocast model; logits cast FP32 for Gumbel-max at configured temperature"
        ),
        "likelihood_precision": (
            "BF16 full-forward replay on exact raw states/actions; FP32 logsoftmax and "
            "entropy; not bit-exact cached-decode logits"
        ),
        "schema_version": 3,
        "forward_calls_interpretation": (
            "per-candidate forward_calls is ideal token-step lower bound; summary "
            "model_compute records executed batched model calls and padded forward tokens"
            " separately for sampling and likelihood replay"
        ),
        "raw_tokens": (
            "sampled tokens BEFORE environment name overrides; includes failing action and EOS"
        ),
        "chooser_visible": (
            "theorem prompt and effective actions with deterministic canonical"
            " name assignment on all EOS-ended attempted actions, independent "
            "of success; truncated final action as emitted"
        ),
        "pid": os.getpid(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "collector.py").write_bytes(Path(__file__).read_bytes())
    model, tok, _ = load_ckpt(ck, "cuda")
    model.sprint_compute = {
        "phase": "sampling",
        "sampling": {"calls": 0, "forward_tokens": 0},
        "likelihood_replay": {"calls": 0, "forward_tokens": 0},
    }

    def count_forward(module, inputs):
        values = module.sprint_compute[module.sprint_compute["phase"]]
        values["calls"] += 1
        values["forward_tokens"] += inputs[0].numel()

    handle = model.register_forward_pre_hook(count_forward)
    t0 = time.time()
    results = collect(model, tok, rows, args)
    handle.remove()
    with (out / "public.jsonl").open("w") as pub, (out / "private.jsonl").open("w") as priv:
        for r in rows:
            v = results[r["theorem_id"]]
            public = {k: r[k] for k in ["theorem_id", "split", "prompt", "structural_key"]}
            if "family" in r:
                public["family"] = r["family"]
            public["candidates"] = sorted(v["public"], key=lambda c: c["candidate_id"])
            pub.write(json.dumps(public) + "\n")
            priv.write(
                json.dumps(
                    {
                        "theorem_id": r["theorem_id"],
                        "labels": sorted(v["private"], key=lambda c: c["candidate_id"]),
                    }
                )
                + "\n"
            )
    lengths = [
        sum(c["generated_tokens"] + 3 for c in v["public"]) + len(tok.encode_prompt(r["prompt"]))
        for r in rows
        for v in [results[r["theorem_id"]]]
    ]
    summary = {
        "theorems": len(rows),
        "seconds": time.time() - t0,
        "model_compute": model.sprint_compute,
        "pool_lengths": lengths,
        "pool_over1024": sum(x > 1024 for x in lengths),
        "valid_candidates": sum(p["valid"] for v in results.values() for p in v["private"]),
        "completed_candidates": sum(
            p["parse_complete"] for v in results.values() for p in v["private"]
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
