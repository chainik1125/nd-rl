"""Pre-execution source/guard checks; no scientific analysis or outcome read."""

import ast
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SEL = ROOT / "code/experiments/current/learned_compute_allocation_20261006/selection"
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
DIR = ART / "analysis_successors/historical_selection_roles_v2"
original = SEL / "aggregate_selection.py"
successor = SEL / "aggregate_selection_historical_v2.py"
old = {n.name: n for n in ast.parse(original.read_text()).body if isinstance(n, ast.FunctionDef)}
new = {n.name: n for n in ast.parse(successor.read_text()).body if isinstance(n, ast.FunctionDef)}
unchanged = []
for name in ("digest", "read_rows", "stratified_interval", "endpoint", "figure"):
    assert ast.dump(old[name], include_attributes=False) == ast.dump(
        new[name], include_attributes=False
    )
    unchanged.append(name)
spec = importlib.util.spec_from_file_location("review_historical_schema_successor", successor)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
a = json.loads((DIR / "addendum.draft.json").read_text())
assert hashlib.sha256(successor.read_bytes()).hexdigest() == a["analysis"]["successor_sha256"]
for data in a["inputs"].values():
    assert hashlib.sha256(Path(data["path"]).read_bytes()).hexdigest() == data["sha256"]
public = [json.loads(line) for line in Path(a["inputs"]["public"]["path"]).read_text().splitlines()]
assert len(public) == 72 and len({r["theorem_id"] for r in public}) == 72
assert {role: sum(r["family"] == role for r in public) for role in m.ROLE_COUNTS} == m.ROLE_COUNTS
assert all(
    r["source_role"] == r["family"] and r["evaluation_role"] == "historical_reused_benchmark"
    for r in public
)
args = SimpleNamespace(
    **{key: value["path"] for key, value in a["inputs"].items()},
    frozen_test_protocol=a["original_registry"]["path"],
    bootstrap=5000,
    seed=61006,
    split="test",
    out=a["output"],
)
passed = []
with tempfile.TemporaryDirectory() as tmp:
    args.analysis_addendum = str(Path(tmp) / "approval.json")

    def check(data, rejected):
        Path(args.analysis_addendum).write_text(json.dumps(data))
        try:
            m.validate_analysis_addendum(args, a["original_registered_signature"])
        except ValueError:
            assert rejected
        else:
            assert not rejected

    check(a, True)
    passed.append("draft rejected")
    approved = copy.deepcopy(a)
    approved["status"] = "approved_analysis_addendum"
    check(approved, False)
    passed.append("approved exact binding accepted")
    for key in ("original_sha256", "successor_sha256"):
        bad = copy.deepcopy(approved)
        bad["analysis"][key] = "0" * 64
        check(bad, True)
        passed.append(key + " mutation rejected")
    bad = copy.deepcopy(approved)
    bad["role_counts"] = {"textbook_dev": 36, "rl_train": 36}
    check(bad, True)
    passed.append("balanced role rewrite rejected")
    bad = copy.deepcopy(approved)
    bad["inputs"]["public"]["sha256"] = "0" * 64
    check(bad, True)
    passed.append("public bytes mutation rejected")
    args.seed = 61007
    check(approved, True)
    args.seed = 61006
    passed.append("bootstrap seed change rejected")
    args.frozen_test_protocol = str(ART / "final_evaluation/main_protocol.json")
    check(approved, True)
    args.frozen_test_protocol = a["original_registry"]["path"]
    passed.append("MAIN registry invocation rejected")
# Explicitly distinguish observed-role weighting from an equal-role estimand.
value = m.stratified_interval(
    [1.0] * 58 + [0.0] * 14, ["textbook_dev"] * 58 + ["rl_train"] * 14, 5000, 61006
)
assert (
    np.isclose(value["mean"], 58 / 72)
    and np.isclose(value["low"], 58 / 72)
    and np.isclose(value["high"], 58 / 72)
)
result = {
    "accepted_successor_sha256": a["analysis"]["successor_sha256"],
    "AST_identical_functions": unchanged,
    "guard_cases": passed,
    "role_weighting_test": value,
    "scope": "No scientific main() call, inference, fits or historical quality reads.",
}
Path(__file__).with_name("historical_role_successor_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
