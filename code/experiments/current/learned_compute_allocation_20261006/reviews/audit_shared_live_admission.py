"""CPU-only mocked occupancy and operational-binding guards; no launches."""

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
WORK = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
SOURCE = WORK / "allocation/launch_live_test_shared.py"
ALLOC = WORK / "allocation"
ORIGINAL = WORK / "allocation/launch_live_test.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


namespace = {
    "Path": Path,
    "json": json,
    "os": os,
    "subprocess": subprocess,
    "WORK": WORK,
    "ALLOC": ALLOC,
    "sha": digest,
    "ART": ART,
    "__file__": str(SOURCE),
    "causal_probe": SimpleNamespace(digest=digest),
}
functions = [function(SOURCE, n) for n in ["validate_operational_addendum", "admit_shared_gpu"]]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
original_preflight = function(ORIGINAL, "preflight")
original_preflight.body.insert(
    0, ast.parse("amendment = validate_operational_addendum(args)").body[0]
)
expected = (
    ast.unparse(original_preflight)
    .replace("sha(__file__)", "sha(ALLOC / 'launch_live_test.py')")
    .replace(
        "require_idle_gpu(args.gpu)",
        "admit_shared_gpu(args.gpu, amendment['minimum_free_memory_mib'])",
    )
)
assert ast.dump(ast.parse(expected)) == ast.dump(
    ast.parse(ast.unparse(function(SOURCE, "preflight")))
)
assert ast.dump(function(SOURCE, "admit_shared_gpu")) == ast.dump(
    function(WORK / "context/run_registered_causal_shared.py", "admit_shared_gpu")
)
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
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "addendum.json"
    base = {
        "status": "approved_operational_addendum",
        "allowed_gpus": [1, 2, 3, 4],
        "minimum_free_memory_mib": 65536,
        "original_registry_sha256": {
            k: digest(ART / f"final_evaluation/{k}_protocol.json") for k in ["main", "historical"]
        },
        "launchers": {
            "live": {
                "original_path": str(ORIGINAL),
                "original_sha256": digest(ORIGINAL),
                "successor_path": str(SOURCE),
                "successor_sha256": digest(SOURCE),
            }
        },
    }
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
    ]:
        value = copy.deepcopy(base)
        if case == "draft":
            value["status"] = "draft"
        if case == "wrong_main":
            value["original_registry_sha256"]["main"] = "bad"
        if case == "wrong_historical":
            value["original_registry_sha256"]["historical"] = "bad"
        if case == "wrong_original":
            value["launchers"]["live"]["original_sha256"] = "bad"
        if case == "wrong_successor":
            value["launchers"]["live"]["successor_sha256"] = "bad"
        if case == "changed_memory":
            value["minimum_free_memory_mib"] = 65000
        path.write_text(json.dumps(value))
        try:
            namespace["validate_operational_addendum"](args)
            passed = True
        except ValueError:
            passed = False
        assert passed == (case == "clean")
        results[case] = "accepted" if passed else "rejected"
result = {
    "successor_sha256": digest(SOURCE),
    "original_sha256": digest(ORIGINAL),
    "preflight_ast_identical_except_addendum_and_admission": True,
    "cases": results,
    "gpu_queries_or_launches": False,
}
Path(__file__).with_name("shared_live_admission_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
