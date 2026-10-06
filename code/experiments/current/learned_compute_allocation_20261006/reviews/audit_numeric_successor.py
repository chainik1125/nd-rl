"""DEV-only numerical successor and synthetic addendum validation."""

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"
sys.path.insert(0, str(ALLOC))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ALLOC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


old = load("aggregate_live")
new = load("aggregate_live_fp32_v2")
helper = load("analysis_addendum")
old_tree = ast.parse(Path(old.__file__).read_text())
new_tree = ast.parse(Path(new.__file__).read_text())
old_metrics = next(
    n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "metrics"
)
new_metrics = next(
    n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "metrics"
)
expected = ast.unparse(old_metrics).replace(
    "-1e-08 <= correct <= 1 + 1e-08", "-1e-06 <= correct <= 1 + 1e-06"
)
assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(ast.unparse(new_metrics)))
registry_path = ART / "final_evaluation/main_protocol.json"
registry = json.loads(registry_path.read_text())
unchanged, admitted = 0, 0
for model in registry["registered_models"]:
    if model["arm"] not in ("cost", "no_cost"):
        continue
    directory = Path(model["dev_evaluation_path"])
    assert json.loads((directory / "config.json").read_text())["evaluation_split"] == "dev"
    for rows in json.loads((directory / "integrated.json").read_text()).values():
        for row in rows:
            result = new.metrics(row, model["native_cost"])
            try:
                prior = old.metrics(row, model["native_cost"])
            except ValueError:
                admitted += 1
            else:
                assert np.array_equal(prior, result)
                unchanged += 1
example = copy.deepcopy(row)
cost = model["native_cost"]
rejections = {}
for case in [
    "high_probability",
    "nan_probability",
    "negative_work",
    "bad_reward",
    "excess_attempts",
]:
    mutated = copy.deepcopy(example)
    if case == "high_probability":
        mutated["correct_probability"] = 1 + 2e-6
        mutated["reward"] = mutated["correct_probability"] - cost * (mutated["attempts"] - 1)
    if case == "nan_probability":
        mutated["correct_probability"] = float("nan")
    if case == "negative_work":
        mutated["expected_policy_work"]["forward_tokens"] = -1
    if case == "bad_reward":
        mutated["reward"] += 0.1
    if case == "excess_attempts":
        mutated["attempts"] = 9
    try:
        new.metrics(mutated, cost)
    except ValueError:
        rejections[case] = True
    else:
        raise AssertionError(case)
draft_path = ART / "analysis_successors/fp32_probability_domain_v2/addendum.draft.json"
draft = json.loads(draft_path.read_text())
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "synthetic_addendum.json"
    for case in [
        "clean",
        "draft",
        "wrong_registry",
        "wrong_successor",
        "bootstrap_change",
        "renormalization",
    ]:
        value = copy.deepcopy(draft)
        value["status"] = "frozen_analysis_only"
        if case == "draft":
            value["status"] = "draft"
        if case == "wrong_registry":
            value["original_generation_registries"]["main"]["sha256"] = "bad"
        if case == "wrong_successor":
            value["analyses"]["main"]["successor_script_sha256"] = "bad"
        if case == "bootstrap_change":
            value["analyses"]["main"]["statistical_analysis"]["bootstrap_samples"] = 100
        if case == "renormalization":
            value["numeric_change"]["renormalization"] = True
        path.write_text(json.dumps(value))
        try:
            helper.registered_statistics(path, "main", registry_path, new.__file__)
        except ValueError:
            assert case != "clean"
            rejections[case] = True
        else:
            assert case == "clean"
result = {
    "original_passing_arrays_identical": unchanged,
    "newly_admitted_dev_rows": admitted,
    "metrics_ast_only_probability_tolerance_change": True,
    "adversarial_rejections": rejections,
    "source_hashes": {
        p: sha(ALLOC / p)
        for p in [
            "aggregate_live.py",
            "aggregate_historical.py",
            "aggregate_live_fp32_v2.py",
            "aggregate_historical_fp32_v2.py",
            "analysis_addendum.py",
        ]
    },
    "draft_sha256": sha(draft_path),
    "live_test_outcomes_read": False,
}
Path(__file__).with_name("numeric_successor_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
