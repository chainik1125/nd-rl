"""Seal only public pool bytes after prospectively frozen transfer decisions."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
SEL = ROOT / "code/experiments/current/learned_compute_transfer_20261006/selection"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def main():
    methods = ART / "phase1_methods_freeze_v3.json"
    frozen = json.loads(methods.read_text())
    assert sha(ART / "phase1_decisions.json") == frozen["decisions_sha256"]
    assert sha(ART / "phase1_inventory.json") == frozen["inventory_sha256"]
    assert sha(SEL / "aggregate_transfer.py") == frozen["analysis_sha256"]
    assert sha(SEL / "run_inference.py") == frozen["launcher_sha256"]
    data = rows(ART / "data/combined412/public.jsonl")
    paths = [
        ART / "pools/combined412/public.jsonl",
        ART / "pools/combined412/public_transitions.jsonl",
    ]
    for p in paths:
        pool = rows(p)
        assert len(pool) == 412
        assert [r["theorem_id"] for r in pool] == [r["theorem_id"] for r in data]
        for a, b in zip(data, pool, strict=True):
            for name in ["prompt", "split", "family", "structural_key"]:
                assert a[name] == b[name]
            assert [c["candidate_id"] for c in b["candidates"]] == list(range(8))
            for c in b["candidates"]:
                assert not (
                    {"valid", "reason", "parse_complete", "done", "reference_proof"} & c.keys()
                )
    protocol = dict(
        frozen,
        status="frozen",
        selection_split="dev",
        selection_split_explanation=(
            "inherited checkpoint selection on original DEV; "
            "all transfer rows are test, no DEV bypass"
        ),
        methods_freeze_sha256=sha(methods),
        methods_freeze_file=methods.name,
        private_sha256=sha(ART / "pools/combined412/private.jsonl"),
        data_sha256=sha(ART / "data/combined412/public.jsonl"),
        public_hashes={str(p): sha(p) for p in paths},
        registered_public_pool_sha256=[sha(p) for p in paths],
        registered_analyses=json.loads((ART / "phase1_inventory.json").read_text())["commands"],
    )
    out = ART / "phase1_protocol.json"
    assert not out.exists()
    out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(sha(out))


if __name__ == "__main__":
    main()
