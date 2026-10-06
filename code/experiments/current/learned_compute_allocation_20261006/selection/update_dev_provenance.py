"""Record a nonbehavioral source-binding successor without rescoring DEV outputs."""

import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "selector_eval_source_binding", folder / "evaluate.py"
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
root = Path("artifacts/learned_compute_allocation_20261006/selectors")
for seed in [0, 1, 2]:
    run = root / f"main_factorized_bce_transitions_s{seed}/standalone_dev"
    path = run / "config.json"
    predecessor = run / "config.before_source_binding.json"
    assert not predecessor.exists(), "Successor already exists; do not overwrite historical config"
    predecessor.write_bytes(path.read_bytes())
    old_evaluator = run / "evaluate.py"
    (run / "evaluate.before_source_binding.py").write_bytes(old_evaluator.read_bytes())
    old_evaluator.write_bytes((folder / "evaluate.py").read_bytes())
    config = json.loads(path.read_text())
    config["registered_signature"] = evaluator.evaluation_signature(
        SimpleNamespace(**config["args"])
    )
    for name, value in config["registered_signature"]["upstream_behavior_sha256"].items():
        config["source_hashes"][str(evaluator.trainer.SOURCE / name)] = value
    config["source_hashes"][str(folder / "evaluate.py")] = hashlib.sha256(
        (folder / "evaluate.py").read_bytes()
    ).hexdigest()
    config["provenance_successor"] = {
        "written_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "predecessor_path": str(predecessor),
        "predecessor_sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
        "reason": "Add actual pinned upstream behavior file hashes to method registration",
        "behavior_changed": False,
        "rescored": False,
        "predictions_unchanged_sha256": hashlib.sha256(
            (run / "predictions.json").read_bytes()
        ).hexdigest(),
        "original_execution_evaluator": str(run / "evaluate.before_source_binding.py"),
    }
    path.write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps({"successors": 3, "rescored": False}))
