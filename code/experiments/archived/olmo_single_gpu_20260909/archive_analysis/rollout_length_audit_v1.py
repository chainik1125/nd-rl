"""Audit generated proof lengths without changing RL rewards or held-out data."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from modeling import decode_completion, file_sha, load_tokenizer
from protocol import judge, protocol_identity, render_completion
from training_diagnostics import Snapshot
from proof_diagnostics import compress_proof

ROOT = Path(__file__).resolve().parents[1]


def audit(run):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Only an explicit sprint training run may be audited")
    snapshot = Snapshot()
    gold = snapshot.read(ROOT / "data/shared_cap6.jsonl", True)
    if len(gold) != 32:
        raise ValueError("Expected the fixed 32-proof bank")
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
    return {"kind": "training_generated_proof_length_audit", "utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "provided_gold_count": 32, "provided_gold_max_lines": 6,
        "counts": dict(counts), "valid_raw_line_histogram": dict(sorted(lengths.items())),
        "valid_compressed_line_histogram": dict(sorted(compressed_lengths.items())),
        "by_batch": {str(k): dict(v) for k, v in sorted(by_batch.items())},
        "longer_positive_responses": longer, "source_prefixes": snapshot.inputs,
        "warnings": snapshot.warnings, "protocol": protocol_identity(),
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
