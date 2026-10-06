"""Frozen external BCE readout on public saved live paths; no generation or training."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

BCE = Path(__file__).with_name("train_bce.py")
spec = importlib.util.spec_from_file_location("live_path_bce_critic", BCE)
critic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(critic)
SHARED = critic.ROOT / "code/shared_code"
sys.path.insert(0, str(SHARED))
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402


def signature(args):
    return {
        "method": "external_bce_saved_live_path_readout",
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "upstream_source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "upstream_behavior_sha256": {
            name: hashlib.sha256((critic.SOURCE / name).read_bytes()).hexdigest()
            for name in [
                "model.py",
                "best_model.py",
                "lean_tok.py",
                "tokenizer.py",
                "nd2lean.py",
                "nd_verify/verify.py",
                "nd_verify/__init__.py",
                "record.py",
                "sample.py",
                "state_sample.py",
                "state_env.py",
            ]
        },
        "critic_sha256": hashlib.sha256(BCE.read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256(BCE.with_name("train.py").read_bytes()).hexdigest(),
        "freeze_validator_sha256": hashlib.sha256(
            (SHARED / "evaluation_freeze.py").read_bytes()
        ).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "context": args.context,
        "view": "transitions",
        "architecture": "factorized",
        "objective": "binary",
        "precision": "BF16 model autocast; FP32 readout",
        "readouts": ["fixed1", "fixed2", "fixed4", "fixed8", "native_stop_external_greedy_index"],
        "tie_policy": "exact uniform raw-logit argmax ties",
    }


def tie_correct(logits, labels):
    highest = max(logits)
    ties = [i for i, value in enumerate(logits) if value == highest]
    return sum(labels[i] for i in ties) / len(ties)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    method = signature(args)
    if args.signature_only:
        print(json.dumps(method, sort_keys=True))
        return
    rows = critic.read(args.public)
    producer = Path(args.public).with_name("config.json")
    if producer.exists():
        generation_config = json.loads(producer.read_text())
        if "test" in Path(generation_config.get("data", "")).stem and args.split != "test":
            raise ValueError(
                "Test-generation path requires explicit test split and frozen protocol"
            )
    if args.split == "test":
        if not args.frozen_test_protocol:
            raise ValueError("Test readout requires frozen protocol before private labels")
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()), args.public, method
        )
    assert len({(r["theorem_id"], r["repeat"]) for r in rows}) == len(rows)
    torch.set_num_threads(4)
    model, tok, metadata = critic.load_ckpt(args.checkpoint, "cuda")
    assert metadata["selector_config"]["objective"].startswith("binary")
    model.eval()
    meter = {"calls": 0, "padded_forward_tokens": 0, "dense_attention_positions": 0}

    def count_forward(module, inputs):
        batch, length = inputs[0].shape
        meter["calls"] += 1
        meter["padded_forward_tokens"] += batch * length
        meter["dense_attention_positions"] += batch * length**2

    model.register_forward_pre_hook(count_forward)
    scores = []
    t0 = time.time()
    with torch.no_grad():
        for row in rows:
            candidates = sorted(row["candidates"], key=lambda c: c["candidate_id"])
            count = len(candidates)
            assert 1 <= count <= 8
            assert [c["candidate_id"] for c in candidates] == list(range(count))
            clean = {"prompt": row["prompt"], "candidates": candidates}
            seqs = critic.model_inputs(clean, tok, list(range(count)), "transitions", "factorized")
            fits = [len(s) <= args.context for s in seqs]
            raw = [0.0] * count
            if any(fits):
                heads = critic.score(
                    model, tok, [s for s, fit in zip(seqs, fits, strict=True) if fit]
                )
                for index, value in zip(
                    [i for i, fit in enumerate(fits) if fit],
                    (heads[:, 0] - heads[:, 1]).tolist(),
                    strict=True,
                ):
                    raw[index] = value
            scores.append(
                {
                    "theorem_id": row["theorem_id"],
                    "repeat": row["repeat"],
                    "candidate_ids": list(range(count)),
                    "candidate_validity_logits": raw,
                    "candidate_validity_probabilities": torch.tensor(raw).sigmoid().tolist(),
                    "candidate_input_tokens": list(map(len, seqs)),
                    "candidate_overflow": [not fit for fit in fits],
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    config = {
        "args": vars(args),
        "signature": method,
        "public_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
        "critic_compute": meter,
        "critic_wall_seconds": time.time() - t0,
        "inputs": "original theorem prompt plus each canonical transition token transcript only",
        "scope": (
            "BCE critic externally trained on frozen generator pools; "
            "domain-shift readout, not joint RL initialization"
        ),
        "compute_scope": (
            "measured cost of scoring all saved candidates; "
            "counterfactual stopping work is separate"
        ),
    }
    (out / "adapter.py").write_bytes(Path(__file__).read_bytes())
    (out / "critic.py").write_bytes(BCE.read_bytes())
    if args.private:
        private = {
            (r["theorem_id"], r["repeat"]): r["validities"] for r in critic.read(args.private)
        }
        assert set(private) == {(r["theorem_id"], r["repeat"]) for r in rows}
        by_key = {(r["theorem_id"], r["repeat"]): r for r in rows}
        metrics = []
        for score in scores:
            key = (score["theorem_id"], score["repeat"])
            labels = private[key]
            path = by_key[key]
            assert len(path["decisions"]) == len(score["candidate_ids"])
            raw = score["candidate_validity_logits"]
            assert len(labels) == len(raw)
            result = {"theorem_id": key[0], "repeat": key[1]}
            for budget in [1, 2, 4, 8]:
                count = min(budget, len(raw))
                result[f"fixed{budget}_correct"] = tie_correct(raw[:count], labels[:count])
            reach, expected = 1.0, 0.0
            for i, decision in enumerate(path["decisions"]):
                hazard = (
                    1.0
                    if decision["forced"] or i + 1 == len(raw)
                    else decision["accept_probability"]
                )
                expected += reach * hazard * tie_correct(raw[: i + 1], labels[: i + 1])
                reach *= 1 - hazard
            result["native_stop_external_greedy_correct"] = expected
            metrics.append(result)
        (out / "metrics.json").write_text(json.dumps(metrics))
        config["private_sha256"] = hashlib.sha256(Path(args.private).read_bytes()).hexdigest()
        summary = {
            key: sum(r[key] for r in metrics) / len(metrics)
            for key in metrics[0]
            if key not in ["theorem_id", "repeat"]
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    (out / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
