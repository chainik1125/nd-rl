"""Reconstruct all unaided LM evaluation verdicts from the actual sampled IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import common
from analyze_lm import jsonl, read_evaluation
from nd_environment import Tokenizer, judge, strict_complete
from proof_diagnostics import compress_proof


def audit(directory):
    directory = Path(directory)
    started = time.monotonic()
    config = json.loads((directory / "config.json").read_text())
    audit_path = Path(config["args"]["audit"])
    reference = common.read_jsonl(audit_path)
    metadata = {r["source_id"]: r for r in reference}
    if len(metadata) != len(reference):
        raise ValueError("Duplicate audit theorem identities")
    validated = read_evaluation(directory, metadata, common.sha(audit_path))
    tok, verdicts, certificates, counts = Tokenizer(), {}, {}, Counter()
    for raw in jsonl(directory / "attempts.jsonl"):
        ids = raw["response_ids"]
        if not 0 < len(ids) == raw["generated_tokens"] <= config["args"]["max_new_tokens"]:
            raise ValueError("Raw response-token accounting mismatch")
        if any(type(x) is not int or not 0 <= x < len(tok) for x in ids):
            raise ValueError("Out-of-vocabulary sampled token")
        key = common.digest([raw["prompt"], ids])
        if key not in verdicts:
            verdicts[key] = judge({"prompt": raw["prompt"], "prefix_ids": []}, ids, tok)
        verdict = verdicts[key]
        for field in ("reward", "body", "n_lines"):
            if verdict[field] != raw[field]:
                raise ValueError(f"Raw sampled tokens disagree with stored {field}")
        counts["attempts"] += 1
        counts["response_tokens"] += len(ids)
        if raw["reward"]:
            if raw["finish_reason"] != "qed" or not verdict.get("independent_ok"):
                raise ValueError("Successful raw output lacks checked termination")
            proof_id = common.digest([raw["prompt"], raw["body"]])
            if proof_id not in certificates:
                comp = compress_proof(raw["prompt"], raw["body"])
                if not strict_complete(raw["prompt"], comp["compressed_proof"])["ok"]:
                    raise ValueError("Compressed certificate invalid")
                certificates[proof_id] = {"source_id": raw["source_id"], "prompt": raw["prompt"], "body": raw["body"],
                    "compression": comp, "reference_length": metadata[raw["source_id"]]["n_lines"]}
            for field in ("raw_valid", "raw_written_length", "dependency_pruned_length", "compressed_length", "compressed_proof"):
                if certificates[proof_id]["compression"][field] != raw["compression"][field]:
                    raise ValueError(f"Stored compression differs in {field}")
            counts["successes"] += 1
    result = json.loads((directory / "result.json").read_text())
    if any(counts[ours] != result["metrics"]["all"][stored] for ours, stored in
           (("attempts", "attempts"), ("successes", "attempt_successes"), ("response_tokens", "generated_tokens"))):
        raise ValueError("Reconstructed ledger differs from reported evaluation totals")
    witnesses = sorted(certificates.values(), key=lambda r: (-r["compression"]["dependency_pruned_length"], r["source_id"]))[:32]
    return {"status": "passed", "directory": str(directory.resolve()), "elapsed_seconds": time.monotonic() - started,
        "counts": dict(counts), "unique_encoded_rollouts_rechecked": len(verdicts), "unique_certificates_rechecked": len(certificates),
        "witnesses": witnesses, "evaluation_provenance": {k: v for k, v in validated.items() if k != "rows"},
        "input_sha256": {str(p.resolve()): common.sha(p) for p in [Path(__file__), Path(__file__).with_name("nd_environment.py"),
            Path(__file__).with_name("analyze_lm.py"), directory / "attempts.jsonl", directory / "result.json", directory / "config.json", audit_path, *common.VENDOR.rglob("*.py")]},
        "scope": "Root read-only audit. Every attempt is reconstructed; exact duplicate encoded responses reuse a checked verdict. No sampled response is repaired or newly counted as successful."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    common.write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("witnesses", "input_sha256", "evaluation_provenance")}))
