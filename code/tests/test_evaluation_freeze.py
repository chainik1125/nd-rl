import pytest

from code.shared_code.evaluation_freeze import (
    file_sha256,
    signature_sha256,
    validate_frozen_evaluation,
)


def test_freeze_binds_artifact_model_policy_and_code(tmp_path):
    pool = tmp_path / "public.jsonl"
    pool.write_text('{"theorem_id":"test-1"}\n')
    signature = {"model": "checkpoint-a", "policy": "threshold-a", "code": "version-a"}
    protocol = {
        "status": "frozen",
        "selection_split": "dev",
        "registered_public_pool_sha256": [file_sha256(pool)],
        "registered_analyses": [
            {"signature": signature, "signature_sha256": signature_sha256(signature)}
        ],
    }
    validate_frozen_evaluation(protocol, pool, signature)
    for key in signature:
        with pytest.raises(ValueError, match="not registered"):
            validate_frozen_evaluation(protocol, pool, {**signature, key: "different"})
    pool.write_text('{"theorem_id":"different-test-pool"}\n')
    with pytest.raises(ValueError, match="not sealed"):
        validate_frozen_evaluation(protocol, pool, signature)


@pytest.mark.parametrize("change", [{"status": "draft"}, {"selection_split": "test"}])
def test_draft_or_test_selected_protocol_is_rejected(tmp_path, change):
    pool = tmp_path / "public.jsonl"
    pool.write_text("{}\n")
    with pytest.raises(ValueError, match="development-selected"):
        validate_frozen_evaluation(
            {"status": "frozen", "selection_split": "dev", **change}, pool, {}
        )
