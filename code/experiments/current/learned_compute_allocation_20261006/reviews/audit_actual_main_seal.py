"""Post-authorization MAIN seal audit; historical access is metadata/hash-only."""

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
WORK = ROOT / "code/experiments/current/learned_compute_allocation_20261006"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


parent_path = ART / "final_evaluation/decisions_parent.json"
parent = read(parent_path)
assert parent["builder_sha256"] == digest(WORK / "analysis/final_protocol.py")
assert digest(parent_path.with_suffix(".builder.py")) == parent["builder_sha256"]
for mapping in (parent["inputs"], parent["collection_sources"]):
    for path, expected in mapping.items():
        assert digest(path) == expected, path
expected_hashes = {
    "main": "bd289d62de0e5c3395fa10330177385184cc66d67d09fc82cead10a65894a1d2",
    "historical": "d228e09e16fa86aa69160e84464eca5064fcecde3c057393e7cc0eecba1b2d7d",
}
for dataset, plan in parent["collection"].items():
    final_path = Path(parent["registry_paths"][dataset])
    assert digest(final_path) == expected_hashes[dataset]
    final = read(final_path)
    launch_path = Path(plan["public_raw"]).parent.with_suffix(".collection_launch.json")
    launch = read(launch_path)
    seal_path = Path(plan["public_raw"]).parent / "public_seal.json"
    seal = read(seal_path)
    assert launch["parent_sha256"] == digest(parent_path)
    assert launch["collection_plan"] == plan
    assert launch["status"] == "public_seal_complete_pending_registry"
    assert not launch["private_outcomes_read"] and not seal["private_outcomes_read"]
    assert launch["public_seal_sha256"] == digest(seal_path)
    assert launch["launcher_sha256"] == digest(WORK / "analysis/run_frozen_collection.py")
    assert [r["stage"] for r in launch["processes"]] == ["argv", "renderer_argv", "sealer_argv"]
    for process in launch["processes"]:
        assert process["command"] == plan[process["stage"]] and process["returncode"] == 0
    assert seal["expected_data_sha256"] == plan["input_sha256"]
    assert seal["sealer_sha256"] == digest(WORK / "selection/seal_public_pool.py")
    assert seal["attempts_per_theorem"] == 8
    assert seal["theorems"] == final["test_theorems"]
    expected = copy.deepcopy(parent["registries"][dataset])
    for public_key, seal_key in [
        ("public_raw", "raw_public_sha256"),
        ("public_transitions", "transition_public_sha256"),
    ]:
        assert digest(plan[public_key]) == seal[seal_key]
        expected["registered_public_pool_sha256"].append(seal[seal_key])
    expected.update(
        status="frozen",
        frozen_decisions_parent_sha256=digest(parent_path),
        public_seal_path=str(seal_path),
        public_seal_sha256=digest(seal_path),
        collection_launch_sha256=digest(launch_path),
        public_sealed_at_utc="2026-10-06 05:33:52 UTC",
        private_outcomes_inspected_before_seal=False,
    )
    if dataset == "historical":
        expected["main_frozen_protocol_sha256"] = expected_hashes["main"]
    assert expected == final
    for entry in final["registered_analyses"]:
        encoded = json.dumps(entry["signature"], sort_keys=True).encode()
        assert hashlib.sha256(encoded).hexdigest() == entry["signature_sha256"]
for model in parent["registries"]["main"]["registered_models"]:
    for stem in ("base", "checkpoint", "counter_hazards", "budget_baselines"):
        if model.get(stem + "_path"):
            assert digest(model[stem + "_path"]) == model[stem + "_sha256"]
    assert digest(Path(model["dev_evaluation_path"]) / "config.json") == model["dev_config_sha256"]

# Only MAIN outcomes below; historical pool bytes were hashed without parsing outcomes.
pool = ART / "pools/main_test"
audit = read(pool / "transition_lean_audit.json")
assert audit["script_sha256"] == digest(WORK / "selection/audit_views.py")
assert digest(pool / "transition_lean_audit.json.script.py") == audit["script_sha256"]
for path, expected in audit["input_hashes"].items():
    assert digest(ROOT / path) == expected
public = [json.loads(line) for line in (pool / "public_transitions.jsonl").read_text().splitlines()]
private = [json.loads(line) for line in (pool / "private.jsonl").read_text().splitlines()]
assert len(public) == len(private) == 400
assert len({r["theorem_id"] for r in public}) == 400
labels = {r["theorem_id"]: {c["candidate_id"]: c for c in r["labels"]} for r in private}
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
from lean_gate import check_sources  # noqa: E402 - runtime source path
from tokenizer import make_tokenizer  # noqa: E402 - runtime source path

tok = make_tokenizer("lean_staten")
sources, verdicts, lengths = [], [], []
completed = 0
for row in public:
    assert sorted(c["candidate_id"] for c in row["candidates"]) == list(range(8))
    assert sorted(labels[row["theorem_id"]]) == list(range(8))
    lengths.append(
        len(tok.encode_prompt(row["prompt"]))
        + 3
        + sum(3 + len(c["transition_tokens"]) for c in row["candidates"])
    )
    for candidate in row["candidates"]:
        label = labels[row["theorem_id"]][candidate["candidate_id"]]
        source = tok.statement(row["prompt"]) + " " + candidate["transition_text"]
        if label["parse_complete"]:
            completed += 1
            assert source == label["lean_text"]
        sources.append(source)
        verdicts.append(label["valid"])
flags, wall, cpu = check_sources(sources)
assert flags == verdicts
assert completed == audit["completed_identity_checks"] == 2559
assert sum(flags) == audit["direct_rendered_lean_valid"] == 2291
assert sorted(lengths) == audit["input_tokens_sorted"]
result = {
    "parent_sha256": digest(parent_path),
    "registry_hashes": expected_hashes,
    "unchanged_decision_fields": True,
    "registered_models": 9,
    "parent_inputs_checked": len(parent["inputs"]),
    "collection_sources_checked": len(parent["collection_sources"]),
    "main_attempts_independently_lean_checked": len(sources),
    "main_valid": sum(flags),
    "completed_exact_source_matches": completed,
    "main_label_disagreements": 0,
    "pooled_overflow_8192": sum(n > 8192 for n in lengths),
    "lean_wall_seconds": wall,
    "lean_process_seconds": cpu,
    "historical_outcome_access": False,
}
Path(__file__).with_name("actual_main_seal_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
