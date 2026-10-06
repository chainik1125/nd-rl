"""Prepare data-independent selector signatures; does not seal/open any test pool."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location("frozen_selector_evaluator", folder / "evaluate.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
from evaluation_freeze import signature_sha256  # noqa: E402

root = Path("artifacts/learned_compute_allocation_20261006")
models = []
for choices_name, objective in [
    ("frozen_primary_dev_choices.json", "marginal"),
    ("frozen_bce_dev_choices.json", "binary"),
]:
    for row in json.loads((root / "selectors" / choices_name).read_text())["runs"]:
        models.append((Path(row["run"]).name, row["checkpoint"], row["architecture"], objective))
base = str(root / "checkpoints/la_T1_best12_s0_r16.pt")
models.extend(
    [
        ("base_pooled", base, "pooled", "marginal"),
        ("base_factorized", base, "factorized", "marginal"),
        ("base_binary_factorized", base, "factorized", "binary"),
    ]
)
rows = []
for name, checkpoint, architecture, objective in models:
    for seed in [77106, 77107, 77108]:
        args = SimpleNamespace(
            checkpoint=checkpoint,
            architecture=architecture,
            objective=objective,
            context=8192,
            view="transitions",
            evaluation_seed=seed,
        )
        signature = evaluator.evaluation_signature(args)
        rows.append(
            {
                "label": f"{name}_perm{seed}",
                "checkpoint": checkpoint,
                "signature": signature,
                "signature_sha256": signature_sha256(signature),
            }
        )
out = root / "selectors/final_signature_inventory.json"
out.write_text(
    json.dumps(
        {"status": "registration inventory, not test protocol", "registered_analyses": rows},
        indent=2,
    )
    + "\n"
)
print(json.dumps({"methods": len(rows), "path": str(out)}))
