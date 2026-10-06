"""Mocked selector operational successor audit; no GPU or outcome access."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
WORK = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
SOURCE = WORK / "selection/run_frozen_test_queue_shared.py"
ORIGINAL = WORK / "selection/run_frozen_test_queue.py"
COMMON = ART / "operational_amendment_v1/addendum.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def process_calls(path):
    return [
        ast.dump(n)
        for n in ast.walk(function(path, "main"))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "subprocess"
        and n.func.attr in ("run", "Popen")
    ]


assert process_calls(SOURCE) == process_calls(ORIGINAL)
assert ast.dump(function(SOURCE, "pairs")) == ast.dump(function(ORIGINAL, "pairs"))
assert ast.dump(function(SOURCE, "admit_shared_gpu")) == ast.dump(
    function(WORK / "context/run_registered_causal_shared.py", "admit_shared_gpu")
)
namespace = {
    "Path": Path,
    "json": json,
    "os": os,
    "subprocess": subprocess,
    "ROOT": ROOT,
    "ORIGINAL": ORIGINAL,
    "COMMON_ADDENDUM": COMMON,
    "COMMON_ADDENDUM_SHA256": digest(COMMON),
    "__file__": str(SOURCE),
    "file_sha256": digest,
}
functions = [function(SOURCE, n) for n in ["validate_operational_addendum", "admit_shared_gpu"]]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
results = {}
for name, gpu, free, uid, expected in [
    ("empty", 1, 70000, None, True),
    ("foreign", 2, 65536, os.getuid() + 1, True),
    ("low_memory", 3, 65535, os.getuid() + 1, False),
    ("same_uid", 4, 70000, os.getuid(), False),
    ("unclaimed", 0, 70000, None, False),
]:
    responses = [f"GPU-test, 81920, {free}, 25", "" if uid is None else "GPU-test, 99999, 4096"]
    with (
        patch.object(subprocess, "check_output", side_effect=responses),
        patch.object(Path, "stat", return_value=SimpleNamespace(st_uid=uid)),
    ):
        try:
            namespace["admit_shared_gpu"](gpu, 65536)
            passed = True
        except ValueError:
            passed = False
    assert passed == expected
    results[name] = "accepted" if passed else "rejected"
draft = json.loads((ART / "operational_selection_amendment_v1/addendum.draft.json").read_text())
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "synthetic_addendum.json"
    args = SimpleNamespace(
        operational_addendum=str(path),
        registry=str(ART / "final_evaluation/main_protocol.json"),
        benchmark="main",
    )
    for case in [
        "clean",
        "draft",
        "wrong_main",
        "wrong_historical",
        "wrong_original",
        "wrong_successor",
        "changed_memory",
        "wrong_common",
    ]:
        value = copy.deepcopy(draft)
        value["status"] = "approved_operational_addendum"
        if case == "draft":
            value["status"] = "draft"
        if case == "wrong_main":
            value["original_registry_sha256"]["main"] = "bad"
        if case == "wrong_historical":
            value["original_registry_sha256"]["historical"] = "bad"
        if case == "wrong_original":
            value["launchers"]["selection"]["original_sha256"] = "bad"
        if case == "wrong_successor":
            value["launchers"]["selection"]["successor_sha256"] = "bad"
        if case == "changed_memory":
            value["minimum_free_memory_mib"] = 65000
        if case == "wrong_common":
            value["original_operational_addendum"]["sha256"] = "bad"
        path.write_text(json.dumps(value))
        try:
            namespace["validate_operational_addendum"](args)
            passed = True
        except ValueError:
            passed = False
        assert passed == (case == "clean")
        results[case] = "accepted" if passed else "rejected"
parent = json.loads((ART / "final_evaluation/decisions_parent.json").read_text())
counts = {}
for benchmark in ("main", "historical"):
    path = ART / f"freeze_inputs/selection_v2/{benchmark}_entries.json"
    assert digest(path) == parent["inputs"][str(path)]
    entries = json.loads(path.read_text())
    jobs = [j for j in entries["commands"] if "_perm" in j["label"] or j["label"] == "frozen_head"]
    assert len(jobs) == len({tuple(j["argv"]) for j in jobs}) == 37
    counts[benchmark] = len(jobs)
result = {
    "successor_sha256": digest(SOURCE),
    "original_sha256": digest(ORIGINAL),
    "scientific_subprocess_calls_ast_identical": True,
    "original_entries_counts": counts,
    "common_addendum_sha256": digest(COMMON),
    "cases": results,
    "gpu_queries_or_launches": False,
    "historical_outcomes_read": False,
}
Path(__file__).with_name("shared_selection_admission_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
