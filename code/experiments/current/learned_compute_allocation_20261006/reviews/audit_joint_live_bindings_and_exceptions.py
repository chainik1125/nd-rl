"""Released MAIN source binding and rare greedy-index exception audit."""

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


registry = read(ART / "final_evaluation/main_protocol.json")
summary = read(ART / "allocation/final_main_summary_fp32_v2.json")
addendum = ART / "analysis_successors/fp32_probability_domain_v2/addendum.json"
assert sha(addendum) == summary["provenance"]["analysis_addendum_sha256"]
a = read(addendum)
assert a["status"] == "frozen_analysis_only"
for key in ("original", "successor"):
    spec = a["analyses"]["main"]
    assert sha(spec[key + "_script_path"]) == spec[key + "_script_sha256"]
checks = 0
exceptions = []
for run in read(ART / "freeze_inputs/allocation_v1/main_live_runs.json")["runs"]:
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
    if run["arm"] not in ("cost", "no_cost"):
        continue
    labels = {}
    for line in (directory / "private_labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        labels[r["theorem_id"], r["repeat"]] = r["validities"]
    with (directory / "public_paths.jsonl").open() as stream:
        for line in stream:
            path = json.loads(line)
            reach = 1.0
            greedy_reach = True
            for i, d in enumerate(path["decisions"]):
                p = d["selection_probabilities"]
                argmax = max(range(len(p)), key=p.__getitem__)
                forced = d["forced"] or i == len(path["decisions"]) - 1
                hazard = 1.0 if forced else d["accept_probability"]
                if argmax != 0:
                    valid = labels[path["theorem_id"], path["repeat"]]
                    exceptions.append(
                        {
                            "arm": run["arm"],
                            "seed": run["seed"],
                            "theorem_id": path["theorem_id"],
                            "repeat": path["repeat"],
                            "count": i + 1,
                            "argmax_index": argmax,
                            "first_valid": valid[0],
                            "argmax_valid": valid[argmax],
                            "native_stop_mass": reach * hazard,
                            "greedy_stop_reached": greedy_reach,
                        }
                    )
                reach *= 1 - hazard
                greedy_reach = greedy_reach and hazard < 0.5
result = {
    "binding_checks": checks,
    "addendum_sha256": sha(addendum),
    "exceptions": exceptions,
    "scope": "All source/model hashes and rare non-first argmax identities; no model inference.",
}
Path(__file__).with_name("joint_live_bindings_and_exceptions_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
