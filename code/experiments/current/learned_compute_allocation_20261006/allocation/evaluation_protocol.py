"""Bind held-out evaluation to a frozen model and inference implementation."""

import hashlib
import json
from pathlib import Path

INFERENCE_KEYS = (
    "context",
    "attempt_bound",
    "state_reserve",
    "temperature",
    "max_action",
    "max_steps",
    "max_attempts",
    "sampling_precision",
    "replay_precision",
    "live_protocol_version",
)

BEHAVIOR_FILES = (
    "live.py",
    "controller.py",
    "train_live.py",
    "train_controller.py",
    "evaluate_live_path.py",
    "evaluation_protocol.py",
)


def signature(config):
    return {
        "base_sha256": config["base_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "inference": {key: config[key] for key in INFERENCE_KEYS},
        "behavior_code_hashes": {name: config["runner_hashes"][name] for name in BEHAVIOR_FILES},
        "source_hashes": config["source_hashes"],
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_saved_implementation(config):
    for name in BEHAVIOR_FILES:
        current = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if config.get("runner_hashes", {}).get(name) != current:
            raise ValueError(f"saved paths use a different behavior implementation: {name}")


def validate_postprocessing_registration(freeze, config, artifact_hash, artifact_key, code_hashes):
    if freeze.get("status") != "frozen" or freeze.get("selection_split") != "dev":
        raise ValueError("test postprocessing requires frozen development registration")
    if freeze.get("test_data_sha256") != config["data_sha256"]:
        raise ValueError("postprocessing test dataset is not registered")
    requested = {
        "base_sha256": config["base_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "inference_sha256": fingerprint(signature(config)),
        artifact_key: artifact_hash,
    }
    if not any(
        all(entry.get(key) == value for key, value in requested.items())
        for entry in freeze.get("registered_models", [])
    ):
        raise ValueError("DEV-fitted baseline is not registered for this test model")
    if any(freeze.get("postprocessing_hashes", {}).get(k) != v for k, v in code_hashes.items()):
        raise ValueError("test postprocessing implementation is not registered")


def validate_reanalysis(config, freeze_path, artifact_file, artifact_key, scripts):
    validate_saved_implementation(config)
    artifact = json.loads(Path(artifact_file).read_text())
    fitters = [name for name in scripts if name.startswith("fit_")]
    if (
        len(fitters) != 1
        or artifact.get("fitter_sha256")
        != hashlib.sha256(Path(__file__).with_name(fitters[0]).read_bytes()).hexdigest()
    ):
        raise ValueError("baseline fitter implementation differs from its recorded artifact")
    if config.get("evaluation_split") == "dev":
        return
    if config.get("evaluation_split") != "test" or not freeze_path:
        raise ValueError("test reanalysis requires the registered frozen protocol")
    freeze_file = Path(freeze_path)
    if hashlib.sha256(freeze_file.read_bytes()).hexdigest() != config.get("frozen_protocol_sha256"):
        raise ValueError("reanalysis freeze differs from the original test registry")
    code_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in scripts
    }
    validate_postprocessing_registration(
        json.loads(freeze_file.read_text()),
        config,
        hashlib.sha256(Path(artifact_file).read_bytes()).hexdigest(),
        artifact_key,
        code_hashes,
    )


def validate_test_freeze(freeze, current_signature, test_data_sha256, seed, repeats, theorems):
    if freeze.get("status") != "frozen" or freeze.get("selection_split") != "dev":
        raise ValueError("test requires a frozen protocol selected on development")
    if freeze.get("test_data_sha256") != test_data_sha256:
        raise ValueError("test dataset is not registered in the frozen protocol")
    if freeze.get("test_seed") != seed or freeze.get("test_repeats") != repeats:
        raise ValueError("test sampling draws differ from the frozen protocol")
    if freeze.get("test_theorems") != theorems:
        raise ValueError("test theorem count differs from the frozen protocol")
    requested = {
        "base_sha256": current_signature["base_sha256"],
        "checkpoint_sha256": current_signature["checkpoint_sha256"],
        "inference_sha256": fingerprint(current_signature),
    }
    if not any(
        all(entry.get(key) == value for key, value in requested.items())
        for entry in freeze.get("registered_models", [])
    ):
        raise ValueError("model or inference protocol is not registered for test")


def validate_counter(counter, current_signature):
    if counter.get("fit_split") != "dev":
        raise ValueError("counter hazards must be fitted on development only")
    for key in ("base_sha256", "checkpoint_sha256"):
        if counter.get(key) != current_signature[key]:
            raise ValueError("counter hazards were fitted for a different model")
    if counter.get("inference_sha256") != fingerprint(current_signature):
        raise ValueError("counter hazards were fitted for a different inference protocol")
    for key in ("fit_data_sha256", "fit_public_paths_sha256"):
        if not isinstance(counter.get(key), str) or len(counter[key]) != 64:
            raise ValueError("counter hazard fitting provenance is incomplete")
