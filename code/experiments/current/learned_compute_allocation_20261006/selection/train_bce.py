"""Secondary factorized binary-validity critic, independently initialized from actual r16."""

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
from model import load_ckpt  # noqa: E402 — pinned upstream source is added above


def save_ckpt(path, model, tok_mode, extra=None):
    """Local-only save; upstream save_ckpt uploads unless ND_OFFLINE is set."""
    torch.save(
        {
            "cfg": model.cfg,
            "state": model.state_dict(),
            "tok_mode": tok_mode,
            "extra": json.loads(json.dumps(extra or {})),
        },
        path,
    )


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def pool_input(row, tok, order, view="actions"):
    ids = [tok.stoi["<st>"]] + tok.encode_prompt(row["prompt"]) + [tok.stoi["<nl>"]]
    for position, index in enumerate(order):
        ids.extend([tok.stoi[f"n{position + 1}"], tok.stoi["<act>"]])
        field = "transition_tokens" if view == "transitions" else "effective_tokens"
        ids.extend(row["candidates"][index][field])
        ids.append(tok.stoi["<nl>"])
    ids.append(tok.stoi["<act>"])
    return ids


def model_inputs(row, tok, order, view, architecture):
    if architecture == "pooled":
        return [pool_input(row, tok, order, view)]
    field = "transition_tokens" if view == "transitions" else "effective_tokens"
    prefix = [tok.stoi["<st>"]] + tok.encode_prompt(row["prompt"]) + [tok.stoi["<nl>"]]
    return [
        prefix
        + [tok.stoi["<act>"]]
        + row["candidates"][i][field]
        + [tok.stoi["<nl>"], tok.stoi["<act>"]]
        for i in order
    ]


def chooser_score(model, tok, groups, architecture):
    if architecture == "pooled":
        return score(model, tok, [g[0] for g in groups])
    logits = score(model, tok, [s for g in groups for s in g])
    return (logits[:, 0] - logits[:, 1]).reshape(len(groups), 8)


def tensor_batch(sequences, pad):
    maxlen = max(map(len, sequences))
    ids = torch.full((len(sequences), maxlen), pad, dtype=torch.long, device="cuda")
    for i, s in enumerate(sequences):
        ids[i, : len(s)] = torch.tensor(s, device="cuda")
    return ids, torch.tensor([len(s) - 1 for s in sequences], device="cuda")


def score(model, tok, sequences):
    ids, last = tensor_batch(sequences, tok.pad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)[torch.arange(len(sequences), device="cuda"), last]
    return logits[:, [tok.stoi[f"n{k}"] for k in range(1, 9)]].float()


@torch.no_grad()
def evaluate(model, tok, rows, labels, args, seed):
    model.eval()
    model.compute_phase = "evaluation"
    rng = random.Random(seed)
    predictions = []
    for row in rows:
        order = list(range(8))
        rng.shuffle(order)
        sequences = model_inputs(row, tok, order, args.view, "factorized")
        fits = [len(sequence) <= args.context for sequence in sequences]
        raw = [0.0] * 8
        if any(fits):
            logits = score(model, tok, [s for s, fit in zip(sequences, fits, strict=True) if fit])
            values = (logits[:, 0] - logits[:, 1]).tolist()
            for i, value in zip([i for i, fit in enumerate(fits) if fit], values, strict=True):
                raw[i] = value
        pool_probs = torch.tensor(raw).softmax(-1).tolist()
        validity_probs = torch.tensor(raw).sigmoid().tolist()
        canonical, canonical_raw, canonical_validity, canonical_overflow = [
            [0.0] * 8 for _ in range(4)
        ]
        for position, index in enumerate(order):
            canonical[index] = pool_probs[position]
            canonical_raw[index] = raw[position]
            canonical_validity[index] = validity_probs[position]
            canonical_overflow[index] = not fits[position]
        ties = [i for i, value in enumerate(canonical_raw) if value == max(canonical_raw)]
        selected = rng.choice(ties)
        valid = labels[row["theorem_id"]]
        logloss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(canonical_raw), torch.tensor(valid, dtype=torch.float32)
        ).item()
        predictions.append(
            {
                "theorem_id": row["theorem_id"],
                "probabilities": canonical,
                "candidate_validity_probabilities": canonical_validity,
                "candidate_validity_logits": canonical_raw,
                "selected_candidate_id": selected,
                "selected_valid": valid[selected],
                "argmax_tie_expected_accuracy": sum(valid[i] for i in ties) / len(ties),
                "argmax_tied_candidate_ids": ties,
                "expected_accuracy": sum(p * v for p, v in zip(canonical, valid, strict=True)),
                "binary_logloss": logloss,
                "brier_score": sum(
                    (p - v) ** 2 for p, v in zip(canonical_validity, valid, strict=True)
                )
                / 8,
                "input_tokens": sum(map(len, sequences)),
                "max_forward_tokens": max(map(len, sequences)),
                "overflow": not all(fits),
                "candidate_overflow": canonical_overflow,
                "candidate_order": order,
            }
        )
    return predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-public", required=True)
    p.add_argument("--train-private", required=True)
    p.add_argument("--dev-public", required=True)
    p.add_argument("--dev-private", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--view", choices=["actions", "transitions"], default="actions")
    p.add_argument("--architecture", choices=["factorized"], default="factorized")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train = read(a.train_public)
    dev = read(a.dev_public)
    train = train[: a.limit] if a.limit else train
    assert all(r["split"] == "train" for r in train), "training data must be train split"
    assert all(r["split"] == "dev" for r in dev), "development data must be dev split"
    assert not {r["theorem_id"] for r in train} & {r["theorem_id"] for r in dev}
    for r in train + dev:
        r["candidates"].sort(key=lambda c: c["candidate_id"])
        assert [c["candidate_id"] for c in r["candidates"]] == list(range(8))
    labels = {
        r["theorem_id"]: [
            int(x["valid"]) for x in sorted(r["labels"], key=lambda x: x["candidate_id"])
        ]
        for path in [a.train_private, a.dev_private]
        for r in read(path)
    }
    for row in train + dev:
        assert len(labels[row["theorem_id"]]) == 8
    for path in [a.train_private, a.dev_private]:
        for row in read(path):
            assert sorted(c["candidate_id"] for c in row["labels"]) == list(range(8))
    ck = ROOT / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
    model, tok, _ = load_ckpt(ck, "cuda")
    model.compute_phase = "initialization"
    model.compute_counts = {}

    def count_forward(module, inputs):
        meter = module.compute_counts.setdefault(
            module.compute_phase, {"calls": 0, "tokens": 0, "dense_attention_positions": 0}
        )
        meter["calls"] += 1
        meter["tokens"] += inputs[0].numel()
        meter["dense_attention_positions"] += inputs[0].shape[0] * inputs[0].shape[1] ** 2

    model.register_forward_pre_hook(count_forward)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "args": vars(a),
        "argv": sys.argv,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "torch_version": str(torch.__version__),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
        "generator_base_seed": 0,
        "source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "upstream_source_hashes": {
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in ["model.py", "best_model.py", "tokenizer.py", "lean_tok.py"]
        },
        "interface": (
            "pooled: <st> theorem <nl> [n_i <act> public_attempt_i <nl>] <act>, "
            "masked vocabulary n1..n8; factorized: independent theorem/proof "
            "prefixes, n1 minus n2 score; public_attempt determined by view"
        ),
        "precision": "BF16 autocast model forward; FP32 masked logits and loss",
        "compute_accounting": (
            "executed forward calls and padded tokens, training/evaluation; excludes backward cost"
        ),
        "objective": (
            "binary cross-entropy for every candidate validity label; "
            "all-valid/all-invalid pools included"
        ),
        "factorized_readout": (
            "per-candidate n1 minus n2 vocabulary logits, "
            "softmax over eight; shared original ALiBiGPT"
        ),
        "shuffling": "candidate order refreshed every epoch and fixed independently at evaluation",
        "ties": "seeded uniform raw-logit argmax ties; report exact expected correctness",
        "checkpoint_selection": "minimum development binary logloss, earliest epoch at exact tie",
        "context_policy": (
            "no truncation; each oversized candidate independently gets neutral logit0 "
            "and validity probability0.5; other candidates still scored; all retained"
        ),
        "extrapolation": a.context > 1024,
        "inputs": {
            x: hashlib.sha256(Path(x).read_bytes()).hexdigest()
            for x in [a.train_public, a.train_private, a.dev_public, a.dev_private]
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "trainer.py").write_bytes(Path(__file__).read_bytes())
    base = evaluate(model, tok, dev, labels, a, 77106)
    (out / "untrained_dev.json").write_text(json.dumps(base))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    t0 = time.time()
    best = float("inf")
    history = []
    positive = list(train)
    for epoch in range(a.epochs):
        model.train()
        model.compute_phase = "training"
        rng.shuffle(positive)
        batches = []
        skipped = 0
        for r in positive:
            order = list(range(8))
            rng.shuffle(order)
            sequences = model_inputs(r, tok, order, a.view, a.architecture)
            if max(map(len, sequences)) > a.context:
                skipped += 1
                continue
            target = [labels[r["theorem_id"]][i] for i in order]
            batches.append((sequences, target))
        total = 0.0
        steps = 0
        for start in range(0, len(batches), a.batch):
            sub = batches[start : start + a.batch]
            optimizer.zero_grad(set_to_none=True)
            logits = chooser_score(model, tok, [s for s, t in sub], a.architecture)
            target = torch.tensor([t for s, t in sub], device="cuda", dtype=torch.float32)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        prediction = evaluate(model, tok, dev, labels, a, 77106)
        accuracy = sum(x["argmax_tie_expected_accuracy"] for x in prediction) / len(prediction)
        metrics = {
            "epoch": epoch + 1,
            "loss": total / max(steps, 1),
            "dev_accuracy": accuracy,
            "dev_binary_logloss": sum(x["binary_logloss"] for x in prediction) / len(prediction),
            "dev_brier_score": sum(x["brier_score"] for x in prediction) / len(prediction),
            "train_included": len(positive),
            "train_mixed_validity": sum(0 < sum(labels[r["theorem_id"]]) < 8 for r in train),
            "train_all_valid": sum(sum(labels[r["theorem_id"]]) == 8 for r in train),
            "train_mixed_supported": sum(0 < sum(target) < 8 for _, target in batches),
            "train_all_invalid": sum(sum(labels[r["theorem_id"]]) == 0 for r in train),
            "train_overflow": skipped,
            "steps": steps,
            "seconds": time.time() - t0,
            "model_forward_compute": json.loads(json.dumps(model.compute_counts)),
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        (out / f"dev_epoch{epoch + 1}.json").write_text(json.dumps(prediction))
        save_ckpt(
            out / f"epoch{epoch + 1}.pt",
            model,
            tok.mode,
            {"selector_config": config, "metrics": metrics},
        )
        if metrics["dev_binary_logloss"] < best:
            best = metrics["dev_binary_logloss"]
            (out / "best_epoch.json").write_text(
                json.dumps({"epoch": epoch + 1, "accuracy": accuracy, "binary_logloss": best})
            )
        (out / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"finished": True, "seconds": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
