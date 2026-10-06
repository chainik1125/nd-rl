"""Require registered public artifacts and methods before reading test labels.

Pool hashes can be sealed after generation without inspecting its private labels.
The research protocol must freeze model/policy choices before that generation.
This check prevents an accidental test evaluation with an unregistered method.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature_sha256(signature: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def validate_frozen_evaluation(
    protocol: dict[str, Any], public_path: str | Path, signature: dict[str, Any]
) -> None:
    """Validate before opening private test labels; raise on any mismatch."""
    if protocol.get("status") != "frozen" or protocol.get("selection_split") != "dev":
        raise ValueError("Test evaluation requires a frozen development-selected protocol")
    if file_sha256(public_path) not in protocol.get("registered_public_pool_sha256", []):
        raise ValueError("Public test pool is not sealed in the frozen protocol")
    requested = signature_sha256(signature)
    for entry in protocol.get("registered_analyses", []):
        if entry.get("signature_sha256") != signature_sha256(entry.get("signature", {})):
            raise ValueError("Registered analysis signature has an inconsistent hash")
        if entry["signature_sha256"] == requested:
            return
    raise ValueError("Model, policy or analysis implementation is not registered for test")
