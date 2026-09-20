import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import common
from audit_final_evaluations import closure_gate, compare_root_aggregate, write_analysis_plan


class FinalAuditGateTests(unittest.TestCase):
    def test_no_notification_rejected_before_any_path_is_read(self):
        with self.assertRaisesRegex(ValueError, "notification"):
            closure_gate({}, "/missing/analysis", "/missing/closure", "/missing/release", False)

    def test_closure_release_identities_without_final_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"metadata-only checkpoint stand-in")
            # Real releases use distinct files. This gate fixture isolates
            # closure identity and ensures no final-data path is opened.
            checkpoints = {f"model{i}": str(weights) for i in range(32)}
            plan = {"data": str(root / "unavailable_final_payload.jsonl"), "samples": 8,
                    "models": {name: {} for name in checkpoints}}
            plan_path = root / "analysis.json"
            common.write_json(plan_path, plan)
            cp_path = root / "closure_plan.json"
            common.write_json(cp_path, {"final_evaluation_models": checkpoints})
            closure = {"no_further_training": True, "no_further_selector_changes": True,
                       "utc": "2026-09-11T02:00:00+00:00", "plan_sha256": common.sha(cp_path),
                       "checkpoint_sha256": {str(weights): common.sha(weights)}, "training_outputs": {}}
            closure_path = root / "closed.json"
            common.write_json(closure_path, closure)
            recipe = {"data": plan["data"], "prospective_input_sha256": {
                str(plan_path): common.sha(plan_path), str(cp_path): common.sha(cp_path)}}
            recipe_path = root / "recipe.json"
            common.write_json(recipe_path, recipe)
            release = {"role": "final_confirmation", "frozen": True, "no_further_training": True,
                       "no_further_selector_changes": True, "utc": "2026-09-11T02:01:00+00:00",
                       "recipe": str(recipe_path), "recipe_sha256": common.sha(recipe_path),
                       "checkpoint_paths": closure["checkpoint_sha256"]}
            release_path = root / "release.json"
            common.write_json(release_path, release)
            closure_gate(plan, plan_path, closure_path, release_path, True)
            self.assertFalse(Path(plan["data"]).exists())
            for key, value in (("role", "confirmation"), ("no_further_training", False),
                               ("utc", "2026-09-11T01:59:00+00:00")):
                altered = {**release, key: value}
                common.write_json(release_path, altered)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    closure_gate(plan, plan_path, closure_path, release_path, True)
            common.write_json(release_path, release)
            weights.write_bytes(b"changed checkpoint")
            with self.assertRaises(ValueError):
                closure_gate(plan, plan_path, closure_path, release_path, True)

    def test_exact_root_aggregate_comparison(self):
        independent = {"data_sha256": "data", "population_sources": 768, "population_contexts": 50,
                       "bootstrap_draws": 5000, "bootstrap_seed": 1, "models": {"a": {"coverage": .4}},
                       "receipts": {}, "comparisons": {}, "contrasts": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            common.write_json(path, independent)
            self.assertEqual(compare_root_aggregate(independent, path)["status"], "exact_agreement")
            altered = copy.deepcopy(independent)
            altered["models"]["a"]["coverage"] = .5
            common.write_json(path, altered)
            with self.assertRaises(ValueError):
                compare_root_aggregate(independent, path)

    def test_plan_copy_preserves_floating_contrast_addition_order(self):
        coefficients = {"z": 1e16, "a": -1e16, "m": 1.}
        total = lambda values: float(sum(np.array([v]) for v in values)[0])
        self.assertEqual(total(coefficients.values()), 1.)
        self.assertEqual(total(dict(sorted(coefficients.items())).values()), 0.)
        plan = {"contrasts": {"example": coefficients}, "output": "old"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_analysis_plan(path, plan, "new")
            copied = json.loads(path.read_text())
            self.assertEqual(list(copied["contrasts"]["example"]), list(coefficients))
            self.assertEqual(total(copied["contrasts"]["example"].values()), 1.)
            self.assertEqual(copied["output"], "new")


if __name__ == "__main__":
    unittest.main()
