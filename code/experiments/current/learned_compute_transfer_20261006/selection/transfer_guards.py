"""Metadata-only transfer source, input and completed-launch guards."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(protocol):
    assert protocol["status"] == "frozen" and protocol["selection_split"] == "dev"
    assert sha(ART / protocol["methods_freeze_file"]) == protocol["methods_freeze_sha256"]
    assert sha(ART / "phase1_decisions.json") == protocol["decisions_sha256"]
    assert sha(ART / "phase1_inventory.json") == protocol["inventory_sha256"]
    for p, value in protocol["source_hashes"].items():
        assert sha(p) == value
    for p, value in protocol["public_hashes"].items():
        assert sha(p) == value
    assert sha(ART / "pools/combined412/private.jsonl") == protocol["private_sha256"]
    assert sha(ART / "data/combined412/public.jsonl") == protocol["data_sha256"]
    assert (
        sha(
            ROOT
            / "artifacts/learned_compute_allocation_20261006/analysis/features1600_main_dev"
            / "feature_model.json"
        )
        == protocol["feature_model_sha256"]
    )
    inventory = json.loads((ART / "phase1_inventory.json").read_text())
    for command in inventory["commands"]:
        sig = command["signature"]
        assert (
            hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
            == command["signature_sha256"]
        )
        if "--checkpoint" in command["argv"]:
            checkpoint = command["argv"][command["argv"].index("--checkpoint") + 1]
            assert sha(checkpoint) == sig["checkpoint_sha256"]
    return inventory["commands"]


def completed(protocol, commands):
    consumed = {}
    for command in commands:
        entry = ART / "selectors/launches" / command["label"]
        path = entry / "run_manifest.json"
        meta = json.loads(path.read_text())
        assert meta["exit_code"] == 0 and meta["argv"] == command["argv"]
        assert meta["protocol_sha256"] == sha(ART / "phase1_protocol.json")
        assert meta["launcher_sha256"] == protocol["launcher_sha256"]
        assert meta["entrypoint_sha256"] == sha(command["argv"][1])
        assert meta["inventory_sha256"] == protocol["inventory_sha256"]
        assert meta["private_sha256"] == protocol["private_sha256"]
        out = Path(command["argv"][command["argv"].index("--out") + 1])
        required = (
            {"config.json", "per_theorem.jsonl", "summary.json"}
            if command["label"] == "features"
            else {"config.json", "predictions.json"}
        )
        assert required <= meta["output_hashes"].keys()
        for name, value in meta["output_hashes"].items():
            assert sha(out / name) == value
        assert sha(entry / "source.py") == meta["entrypoint_sha256"]
        cfg = json.loads((out / "config.json").read_text())
        if command["label"] == "features":
            assert cfg["public"] == command["argv"][command["argv"].index("--public") + 1]
            assert cfg["private"] == command["argv"][command["argv"].index("--private") + 1]
            assert sha(cfg["feature_model"]) == protocol["feature_model_sha256"]
            assert cfg["frozen_test_protocol"] == str(ART / "phase1_protocol.json")
        else:
            assert cfg["registered_signature"] == command["signature"]
            assert cfg["frozen_test_protocol_sha256"] == sha(ART / "phase1_protocol.json")
            for name in ["public", "private", "checkpoint", "out"]:
                assert cfg["args"][name] == command["argv"][command["argv"].index("--" + name) + 1]
        consumed[str(path)] = sha(path)
    return consumed
