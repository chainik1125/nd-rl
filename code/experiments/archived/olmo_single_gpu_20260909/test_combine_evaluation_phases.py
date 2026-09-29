"""CPU fixtures for separate-phase joins and protected-input fail-closed behavior."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from combine_evaluation_phases import ARMS, compose, guard_run, inspect_phase, validate_join


def entry(arm, mode, success=False, status="complete"):
    plan = {field: "fixture" for field in ("model_id", "model_revision", "model_config_sha256",
            "download_manifest_sha256", "tokenizer_identity", "tokenizer_files", "protocol",
            "modeling_sha256", "evaluator_sha256", "target_identity")}
    plan.update(arm=arm, settings={"modes": [mode], "batch_size": 4 if arm == "base_icl32" else 16,
                                 "max_new_tokens": 2048},
                adapter_files={} if arm.startswith("base_") else {"adapter": arm})
    rows = [{"arm": arm, "cohort": "long_panel", "mode": mode, "canonical_hash": f"t{index}",
             "reference_lines": 7 + index, "solved": success} for index in range(2)]
    return {"plan": plan, "result": {"arm": arm, "status": status},
            "per_theorem": rows, "certificates": []}


def phase(mode):
    return {"mode": mode, "runs": [{"arm": arm} for arm in ARMS]}


class PhaseCompositionTests(unittest.TestCase):
    def test_modes_and_batch_size_are_permitted_join_differences(self):
        entries = [entry(arm, mode, success=arm == "rl") for mode in ("greedy", "sampled") for arm in ARMS]
        report = compose([phase("greedy"), phase("sampled")], entries, [], 100)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(all(row["headline_eligible"] for row in report["paired_comparisons"]))
        comparison = next(row for row in report["paired_comparisons"] if row["left_arm"] == "dpo"
                          and row["right_arm"] == "rl" and row["reference_lines"] is None)
        self.assertEqual(comparison["difference_right_minus_left"], 1)
        self.assertEqual(comparison["ci95_percentile"], [1, 1])

    def test_missing_sampling_does_not_downgrade_complete_greedy(self):
        entries = [entry(arm, "greedy") for arm in ARMS] + [entry("sft", "sampled")]
        report = compose([phase("greedy"), phase("sampled")], entries, [], 100)
        self.assertEqual(report["status"], "partial_diagnostic")
        self.assertEqual([value["status"] for value in report["phases"]], ["complete", "partial_diagnostic"])
        self.assertTrue(all(row["headline_eligible"] for row in report["paired_comparisons"] if row["mode"] == "greedy"))

    def test_partial_pair_uses_only_intersection_and_is_not_headline(self):
        a, b = entry("dpo", "sampled", status="partial_diagnostic"), entry("rl", "sampled")
        a["per_theorem"].pop()
        report = compose([phase("sampled")], [a, b], [], 100)
        self.assertTrue(report["paired_comparisons"])
        self.assertTrue(all(row["paired_theorems"] == 1 and row["partial_diagnostic"]
                            and not row["headline_eligible"] for row in report["paired_comparisons"]))

    def test_pending_is_not_zero_accuracy(self):
        report = compose([phase("greedy"), phase("sampled")], [], [], 100)
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["runs"], [])
        self.assertEqual(report["paired_comparisons"], [])

    def test_changed_runtime_cap_checkpoint_and_duplicate_rejected(self):
        first, second = entry("sft", "greedy"), entry("sft", "sampled")
        mutations = [lambda e: e["plan"].update(modeling_sha256="different"),
                     lambda e: e["plan"]["settings"].update(max_new_tokens=1024),
                     lambda e: e["plan"]["adapter_files"].update(adapter="different")]
        for mutate in mutations:
            changed = deepcopy(second)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_join([first, changed])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_join([first, deepcopy(first)])


class PlanGuardTests(unittest.TestCase):
    def phase_file(self, root, mutate=None):
        (root / "plans").mkdir(exist_ok=True)
        steps = []
        for arm in ARMS:
            args = ["--arm", arm, "--out", str(root / "runs/eval" / arm),
                    "--cohorts", "long_panel", "short_panel", "--modes", "greedy",
                    "--max-new-tokens", "2048", "--batch-size", "4", "--seed", "42"]
            if not arm.startswith("base_"):
                args += ["--adapter-path", str(root / "runs" / arm / "final")]
            steps.append({"entrypoint": "evaluate.py", "args": args})
        raw = {"mode": "greedy", "steps": steps}
        if mutate:
            mutate(raw)
        path = root / "plans/fixture.json"
        path.write_text(json.dumps(raw))
        return path

    def test_only_explicit_five_arm_synthetic_phase_metadata_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = inspect_phase(self.phase_file(root), root)
            self.assertEqual(len(result["runs"]), 5)
            self.assertFalse((root / "runs").exists())

    def test_textbook_mode_rejected_before_any_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def mutate(raw):
                args = raw["steps"][0]["args"]
                args[args.index("long_panel")] = "textbook_final"
            with self.assertRaisesRegex(ValueError, "synthetic"):
                inspect_phase(self.phase_file(root, mutate), root)

    def test_evaluation_metadata_textbook_refusal_precedes_file_hashing(self):
        expected = {"arm": "base_zero", "cohorts": ["textbook_final"]}
        plan = {"arm": "base_zero", "cohorts": ["textbook_final"], "target_count": 73}
        with patch("combine_evaluation_phases.sha", side_effect=AssertionError("payload access")):
            with self.assertRaisesRegex(ValueError, "synthetic"):
                guard_run(plan, expected)

    def test_phase_outside_runs_and_duplicate_arm_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def outside(raw):
                args = raw["steps"][0]["args"]
                args[args.index("--out") + 1] = str(root / "protected")
            with self.assertRaisesRegex(ValueError, "specific child"):
                inspect_phase(self.phase_file(root, outside), root)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                inspect_phase(self.phase_file(root, lambda raw: raw["steps"].append(raw["steps"][0])), root)


if __name__ == "__main__":
    unittest.main()
