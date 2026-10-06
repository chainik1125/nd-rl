"""Public sealing binds metadata and every original candidate field."""

import importlib.util
import json
from pathlib import Path

import pytest

FOLDER = (
    Path(__file__).resolve().parents[1]
    / "experiments/current/learned_compute_allocation_20261006/selection"
)
SPEC = importlib.util.spec_from_file_location("public_seal_test", FOLDER / "seal_public_pool.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def fixtures(tmp_path, historical=False):
    metadata = {
        "theorem_id": "t",
        "split": "test",
        "prompt": "THM SEQ P PRF",
        "structural_key": "k",
        "family": "forward",
    }
    if historical:
        metadata.update(
            source_role="rl_train",
            name="old_name",
            source_row_index=3,
            evaluation_role="historical_reused_benchmark",
            reference_lines=99,
        )
    raw = {k: metadata[k] for k in MOD.CORE}
    raw["candidates"] = [
        {"candidate_id": 0, "raw_actions": [[2]], "raw_tokens": [2], "mean_logprob": -1.0}
    ]
    rendered = json.loads(json.dumps(raw))
    rendered["candidates"][0]["transition_tokens"] = [3]
    paths = [tmp_path / name for name in ["raw", "rendered", "metadata"]]
    for p, value in zip(paths, [raw, rendered, metadata], strict=True):
        p.write_text(json.dumps(value) + "\n")
    return paths


def test_seal_rejects_changed_original_candidate_field(tmp_path):
    paths = fixtures(tmp_path)
    row = json.loads(paths[1].read_text())
    row["candidates"][0]["mean_logprob"] = -2.0
    paths[1].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="original public candidate field"):
        MOD.seal(*paths, expected_theorems=1, expected_candidates=1)


def test_historical_metadata_enrichment_preserves_backups_and_excludes_reference(tmp_path):
    paths = fixtures(tmp_path, historical=True)
    original = paths[0].read_bytes()
    result = MOD.seal(*paths, expected_theorems=1, expected_candidates=1, enrich=True)
    assert result["metadata_enriched"]
    assert result["private_outcomes_read"] is False
    assert paths[0].with_name(paths[0].name + ".before_metadata_seal").read_bytes() == original
    row = json.loads(paths[0].read_text())
    assert row["source_role"] == "rl_train"
    assert row["source_row_index"] == 3
    assert "reference_lines" not in row


def test_seal_rejects_wrong_prompt_and_wrong_count(tmp_path):
    paths = fixtures(tmp_path)
    with pytest.raises(ValueError, match="count/uniqueness"):
        MOD.seal(*paths, expected_theorems=2, expected_candidates=1)
    row = json.loads(paths[0].read_text())
    row["prompt"] = "THM SEQ Q PRF"
    paths[0].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="metadata differs"):
        MOD.seal(*paths, expected_theorems=1, expected_candidates=1)
