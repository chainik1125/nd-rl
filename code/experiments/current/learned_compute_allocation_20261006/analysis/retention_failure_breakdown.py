"""Describe saved development failures; never infer their parameter-level cause."""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = {
        "scope": "post-hoc DEV-only description; no new model runs or causal attribution",
        "analysis_sha256": sha(__file__),
        "models": {},
    }
    for name, relative in (
        ("original", "context/base_main_dev64_v1/history_0"),
        ("bce_seed0_epoch4", "selectors/bce_s0_e4_clean_main_dev64_v1/history_0"),
    ):
        folder = ART / relative
        public = [json.loads(line) for line in (folder / "public.jsonl").read_text().splitlines()]
        if any(row["split"] != "dev" for row in public):
            raise ValueError("This descriptive diagnostic is development-only")
        private = [json.loads(line) for line in (folder / "private.jsonl").read_text().splitlines()]
        observed = {(r["theorem_id"], c["candidate_id"]) for r in public for c in r["candidates"]}
        labeled = {(r["theorem_id"], c["candidate_id"]) for r in private for c in r["labels"]}
        labels = [label for row in private for label in row["labels"]]
        candidates = [candidate for row in public for candidate in row["candidates"]]
        if observed != labeled or len(observed) != len(labels) or len(labeled) != len(candidates):
            raise ValueError("Public/private candidate identities differ")
        result["models"][name] = {
            "theorems": len(public),
            "attempts": len(labels),
            "valid": sum(label["valid"] for label in labels),
            "syntactically_complete": sum(label["parse_complete"] for label in labels),
            "ended_reason_counts": dict(Counter(label["ended_reason"] for label in labels)),
            "public_environment_failure_counts": dict(
                Counter(
                    str(label["env_failure"])
                    for label in labels
                    if label["env_failure"] is not None
                )
            ),
            "mean_generated_tokens": sum(c["generated_tokens"] for c in candidates)
            / len(candidates),
            "input_sha256": {
                str(folder / name): sha(folder / name) for name in ("public.jsonl", "private.jsonl")
            },
        }
    output = ART / "selectors/bce_s0_e4_clean_main_dev64_v1/retention_failure_breakdown_v2.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
    shutil.copy2(__file__, output.with_suffix(".source.py"))
    print(output)


if __name__ == "__main__":
    main()
