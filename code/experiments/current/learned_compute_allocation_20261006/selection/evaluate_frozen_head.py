"""Guarded inference for the separate head on an unchanged original proof model."""

import argparse
import datetime
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

FOLDER = Path(__file__).resolve().parent


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        "frozen_head_eval_" + name, FOLDER / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extraction = load_module("extract_frozen_head_features")
common = load_module("frozen_head_common")
inference = load_module("frozen_head_inference")
sys.path.insert(0, str(extraction.ROOT / "code/shared_code"))
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402


def evaluation_signature(head, checkpoint, context=8192, batch=8):
    return {
        "method": "frozen_original_r16_scalar_validity_head",
        "head_sha256": common.digest(head),
        "checkpoint_sha256": common.digest(checkpoint),
        "evaluator_sha256": common.digest(__file__),
        "extractor_sources": extraction.source_hashes(),
        "readout_sha256": common.digest(FOLDER / "frozen_head_common.py"),
        "fit_sha256": common.digest(FOLDER / "fit_frozen_head.py"),
        "inference_sha256": common.digest(FOLDER / "frozen_head_inference.py"),
        "validator_sha256": common.digest(
            extraction.ROOT / "code/shared_code/evaluation_freeze.py"
        ),
        "context": context,
        "batch": batch,
        "architecture": "frozen_factorized_scalar_head",
        "objective": "binary",
        "view": "transitions",
        "precision": (
            "BF16 frozen trunk/final LayerNorm; cached FP32 features; CPU FP64 scalar readout"
        ),
        "ordering": "canonical candidate IDs; batches of eight; no permutation-dependent readout",
        "overflow": (
            "independent logit0/probability.5 fallback; no forward charge; retain all candidates"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dev-cache")
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    signature = evaluation_signature(args.head, args.checkpoint, args.context, args.batch)
    if args.signature_only:
        print(json.dumps(signature, sort_keys=True))
        return
    if args.context != 8192 or args.batch != 8:
        raise ValueError("Secondary method froze context8192 and batch8")
    if args.split == "test" and args.dev_cache:
        raise ValueError("TEST cannot use a development cache")
    public = extraction.encoding.read(args.public)
    if not public or any(r["split"] != args.split for r in public):
        raise ValueError("Public split differs")
    if args.split == "test":
        if not args.frozen_test_protocol:
            raise ValueError("TEST requires frozen registration before scoring or labels")
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()), args.public, signature
        )
    head = torch.load(args.head, map_location="cpu", weights_only=True)
    if signature["checkpoint_sha256"] != head["config"]["original_checkpoint_sha256"]:
        raise ValueError("Head uses a different frozen original model")
    for source, value in head["config"]["source_hashes"].items():
        if common.digest(source) != value:
            raise ValueError("Fit sources changed after terminal head save")
    assert head["weight"].shape == (384,) and head["bias"].ndim == 0
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parameter_sha = head["config"]["frozen_original_parameter_sha256"]
    config = {
        "args": vars(args),
        "registered_signature": signature,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_hashes": {args.public: common.digest(args.public)},
        "head_sha256": common.digest(args.head),
        "checkpoint_sha256": common.digest(args.checkpoint),
    }
    if args.dev_cache:
        cache, metadata = common.read_cache(args.dev_cache, "dev")
        provenance = json.loads((Path(args.dev_cache) / "config.json").read_text())
        if (
            not provenance["original_parameters_unchanged"]
            or provenance["checkpoint_sha256"] != signature["checkpoint_sha256"]
        ):
            raise ValueError("Cache is not from unchanged registered original model")
        if (
            metadata["public_sha256"] != common.digest(args.public)
            or metadata["context"] != args.context
        ):
            raise ValueError("Cache differs from registered public input/context")
        if (
            provenance["parameter_sha256_before"] != parameter_sha
            or provenance["parameter_sha256_after"] != parameter_sha
        ):
            raise ValueError("Cache parameter digests differ from terminal head provenance")
        if provenance["source_hashes"] != extraction.source_hashes():
            raise ValueError("Cache extraction sources differ from registered inference")
        config["dev_cache_sha256"] = metadata["cache_sha256"]
        config["inference_mode"] = "existing exact frozen public feature cache; no rescoring"
    else:
        model, tok, _ = extraction.load_ckpt(args.checkpoint, "cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before = extraction.parameter_digest(model)
        if before != parameter_sha:
            raise ValueError("Frozen original parameter digest differs")
        metadata = inference.extract(
            extraction, model, tok, args.public, args.split, args.batch, args.context, out
        )
        cache, metadata = common.read_cache(out, args.split)
        after = extraction.parameter_digest(model)
        if before != after:
            raise ValueError("Inference modified original parameters")
        config["parameter_sha256_before"] = before
        config["parameter_sha256_after"] = after
        config["inference_mode"] = "fresh public-only frozen-trunk extraction"
    expected_keys = {(r["theorem_id"], c["candidate_id"]) for r in public for c in r["candidates"]}
    if set(zip(cache["theorem_ids"], cache["candidate_ids"], strict=True)) != expected_keys:
        raise ValueError("Cache candidate identity mismatch")
    labels = common.aligned_labels(cache, args.private)  # TEST guard already succeeded.
    predictions, summary = inference.predict(common, cache, labels, head["weight"], head["bias"])
    config.update(
        private_sha256=common.digest(args.private),
        process_wall_seconds_monotonic=time.monotonic() - started,
        predictions_sha256=None,
        feature_metadata=metadata,
        frozen_test_protocol_sha256=common.digest(args.frozen_test_protocol)
        if args.frozen_test_protocol
        else None,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "predictions.json").write_text(json.dumps(predictions) + "\n")
    config["predictions_sha256"] = common.digest(out / "predictions.json")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
