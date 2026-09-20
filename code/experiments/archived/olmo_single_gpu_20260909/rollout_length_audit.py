"""Audit generated proof lengths without changing RL rewards or held-out data."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import modeling as modeling_module
import protocol as protocol_module
import training_diagnostics as training_diagnostics_module
import proof_diagnostics as proof_diagnostics_module
from modeling import decode_completion, file_sha, load_tokenizer
from protocol import judge, protocol_identity, render_completion
from training_diagnostics import Snapshot
from proof_diagnostics import compress_proof

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FREEZE_SHA = "7f01ae7e6d5938c661a45f3f60c8a7c5f8e14e4a8a553b9ca6147f258bcf4292"
REPLAY_SOURCES = tuple(Path(module.__file__).resolve() for module in
    (modeling_module, protocol_module, training_diagnostics_module, proof_diagnostics_module))
VENDOR = protocol_module.VENDOR.resolve()
MODEL_METADATA = ("config.json", "download_completed.json", "generation_config.json",
    "merges.txt", "model.safetensors.index.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json")


def stable_provenance(snapshot):
    """Verify only permitted static inputs, never every file in the freeze.

    In particular, neither evaluation panels nor model tensors are opened.
    The live rollout log deliberately is not part of the stability recheck.
    """
    freeze_path = ROOT / "plans/primary_source_freeze_v4.json"
    freeze = snapshot.read(freeze_path)
    if not freeze or snapshot.inputs[-1]["sha256_of_observed_bytes"] != PRIMARY_FREEZE_SHA:
        raise ValueError("The committed primary-v4 freeze identity is required")
    frozen = freeze["sha256"]
    paths = set(REPLAY_SOURCES)
    paths.add(ROOT / "data/shared_cap6.jsonl")
    paths.update(ROOT / "models/olmo32b" / name for name in MODEL_METADATA)
    paths.update(Path(path) for path in frozen
                 if Path(path).is_relative_to(VENDOR) and Path(path).suffix == ".py")
    receipts = {str(freeze_path): PRIMARY_FREEZE_SHA}
    for path in sorted(paths):
        expected = frozen.get(str(path))
        if expected is None or file_sha(path) != expected:
            raise ValueError(f"Frozen replay/compression input changed or was not pinned: {path}")
        receipts[str(path)] = expected
    return {"path": str(freeze_path), "sha256": PRIMARY_FREEZE_SHA}, receipts


def audit(run):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Only an explicit sprint training run may be audited")
    snapshot = Snapshot()
    source_freeze, stable_inputs = stable_provenance(snapshot)
    gold = snapshot.read(ROOT / "data/shared_cap6.jsonl", True)
    if len(gold) != 32 or len({row["canonical_hash"] for row in gold}) != 32:
        raise ValueError("Expected the fixed 32-proof bank with distinct canonical identities")
    pool = {row["canonical_hash"]: row for row in gold}
    for row in gold:
        result = judge(row["prompt"], render_completion(row["proof"]))
        if not result["accepted"] or result["n_lines"] != row["n_lines"] or result["n_lines"] > 6:
            raise ValueError("Provided supervision is not strictly verified cap-six")
    rows = snapshot.read(run / "rollouts.jsonl", True)
    if not rows:
        raise ValueError("No real RL actions are available")
    if any(warning["kind"] == "malformed_committed_jsonl_record" for warning in snapshot.warnings):
        raise ValueError("Malformed committed rollout record")
    tokenizer = load_tokenizer(ROOT / "models/olmo32b")
    lengths, compressed_lengths, counts, by_batch = Counter(), Counter(), Counter(), defaultdict(Counter)
    longer = []
    for row in rows:
        if row.get("mechanical_reward_fixture") is not False:
            raise ValueError("Mechanical or unclassified reward record")
        source = pool.get(row["canonical_hash"])
        if source is None or source["prompt"] != row["theorem"]:
            raise ValueError("Rollout theorem is outside the shared bank")
        if decode_completion(tokenizer, row["completion_ids"]) != row["completion"]:
            raise ValueError("Native token/text replay mismatch")
        result = judge(row["theorem"], row["completion"])
        if result != row["verdict"] or result["reward"] != row["reward"]:
            raise ValueError("Strict native reward replay mismatch")
        counts["responses"] += 1
        by_batch[row["batch"]]["responses"] += 1
        if not result["accepted"]:
            counts["invalid_responses"] += 1
            continue
        n = result["n_lines"]
        compact = compress_proof(row["theorem"], result["proof"])
        replay = judge(row["theorem"], render_completion(compact["compressed_proof"]))
        m = compact["compressed_length"]
        if not replay["accepted"] or replay["n_lines"] != m or m > n:
            raise ValueError("Analysis-only compression failed strict replay")
        lengths[n] += 1
        compressed_lengths[m] += 1
        for target in (counts, by_batch[row["batch"]]):
            target["valid_responses"] += 1
            target["valid_raw_above_six"] += n > 6
            target["valid_compressed_above_six"] += m > 6
        if n > 6:
            longer.append({"batch": row["batch"], "index": row["index"],
                "canonical_hash": row["canonical_hash"], "family": source["family"],
                "provided_gold_lines": source["n_lines"], "generated_raw_lines": n,
                "analysis_compressed_lines": m, "native_tokens": len(row["completion_ids"])})
    for path, expected in stable_inputs.items():
        if file_sha(path) != expected:
            raise ValueError(f"Static audit input changed during replay: {path}")
    return {"kind": "training_generated_proof_length_audit", "utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "provided_gold_count": 32, "provided_gold_cap_lines": 6,
        "provided_gold_max_lines": max(row["n_lines"] for row in gold),
        "counts": dict(counts), "valid_raw_line_histogram": dict(sorted(lengths.items())),
        "valid_compressed_line_histogram": dict(sorted(compressed_lengths.items())),
        "by_batch": {str(k): dict(v) for k, v in sorted(by_batch.items())},
        "longer_positive_responses": longer, "source_prefixes": snapshot.inputs,
        "warnings": snapshot.warnings, "protocol": protocol_identity(),
        "source_freeze": source_freeze, "stable_input_sha256": stable_inputs,
        "helper_sha256": file_sha(Path(__file__)), "non_atomic_training_snapshot": True,
        "held_out_answers_read": False, "protected_payloads_read": False,
        "interpretation": "The six-line limit applies to provided gold witnesses and selected training theorem identities. RL may generate and receive rewards for longer valid certificates on those same theorems. Their actual lengths are additional generated experience, not longer supplied labels. Compression is analysis only, never used by the reward or optimizer. A long generated proof does not establish that its theorem requires a long proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/grpo_v4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "reports") or args.out.exists():
        parser.error("Use a fresh sprint report directory")
    report = audit(args.run)
    args.out.mkdir()
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    counts = report["counts"]
    lines = ["# Generated RL proof lengths", "", report["interpretation"], "",
        f"This snapshot contains {counts['responses']} real responses and {counts.get('valid_responses', 0)} strictly valid proofs. Of the valid proofs, {counts.get('valid_raw_above_six', 0)} have more than six raw lines, and {counts.get('valid_compressed_above_six', 0)} still exceed six after separately verified analysis-only compression.", "",
        "These adaptive training samples are not a fixed-checkpoint accuracy benchmark. The run can still be in progress; completed optimizer steps must be checked separately.", "",
        "Full line-count histograms, per-batch counts and identifiers of longer positive responses are in report.json. No theorem is reclassified by generated proof length and no held-out data is accessed.", ""]
    with (args.out / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"out": str(args.out), "counts": counts, "held_out_answers_read": False}))


if __name__ == "__main__":
    main()
