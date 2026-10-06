"""Released historical source, model and frozen MAIN-fit bindings."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


registry = read(ART / "final_evaluation/historical_protocol.json")
summary = read(ART / "allocation/final_historical_summary_fp32_v2.json")
addendum = ART / "analysis_successors/fp32_probability_domain_v2/addendum.json"
assert sha(addendum) == summary["provenance"]["analysis_addendum_sha256"]
a = read(addendum)
assert a["status"] == "frozen_analysis_only"
for key in ("original", "successor"):
    spec = a["analyses"]["historical"]
    assert sha(spec[key + "_script_path"]) == spec[key + "_script_sha256"]
checks = 0
exceptions = []
for run in read(ART / "freeze_inputs/allocation_v1/historical_live_runs.json")["runs"]:
    directory = Path(run["paths"])
    config = read(directory / "config.json")
    model = next(
        m
        for m in registry["registered_models"]
        if m["arm"] == run["arm"] and m["rl_seed"] == run["seed"]
    )
    for key in ("base", "checkpoint"):
        if config[key]:
            assert sha(config[key]) == config[key + "_sha256"] == model[key + "_sha256"]
            checks += 1
    for name, digest in config["runner_hashes"].items():
        assert sha(ALLOC / name) == digest == sha(directory / "runner_snapshot" / name)
        checks += 1
    for name, digest in config["source_hashes"].items():
        assert sha(Path(config["source"]) / name) == digest
        checks += 1
    for field, pathkey in [
        ("counter_hazards", "counter_hazards_path"),
        ("budget_baselines", "budget_baselines_path"),
    ]:
        if pathkey in model:
            assert sha(model[pathkey]) == model[field + "_sha256"]
            checks += 1
release = read(ART / "analysis/root_historical_live_barrier_release_v1.json")
assert release["all9_complete"]
for item in release["launch_ledgers"]:
    assert sha(item["path"]) == item["sha256"]
assert sha(ART / "final_evaluation/historical_protocol.json") == release["registry_sha256"]
assert sha(ART / "allocation/final_historical_summary_fp32_v2.json") == release["aggregate_sha256"]
main = read(ART / "final_evaluation/main_protocol.json")
fields = (
    "arm",
    "rl_seed",
    "checkpoint_sha256",
    "base_sha256",
    "inference_sha256",
    "counter_hazards_sha256",
    "budget_baselines_sha256",
)
assert {tuple(m.get(k) for k in fields) for m in main["registered_models"]} == {
    tuple(m.get(k) for k in fields) for m in registry["registered_models"]
}
result = {
    "binding_checks": checks,
    "unchanged_MAIN_models_and_DEV_fits": True,
    "all_nine_ledger_hashes_match_release": True,
    "addendum_sha256": sha(addendum),
}
Path(__file__).with_name("historical_live_bindings_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
