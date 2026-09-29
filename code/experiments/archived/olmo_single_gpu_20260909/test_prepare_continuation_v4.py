"""Pure/temp CPU fixtures; never build the real bundle or launch a job."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_continuation_v4 as builder


def originals(root):
    training = {"not_after_utc": builder.DEADLINE, "source_freeze": "old_freeze",
                "out": "old_pipeline", "steps": [{"name": "grpo", "entrypoint": "train.py",
                "args": deepcopy(builder.GRPO_ARGS), "required_records": []}],
                "selection_rule": "fixed", "rl_signal_rule": "stop on audit failure"}
    phases = {}
    for mode in ("greedy", "sampled"):
        steps = []
        for arm in builder.ARMS:
            args = ["--arm", arm, "--out", str(root / "runs" / f"eval_{mode}" / arm),
                    "--cohorts", "long_panel", "short_panel", "--modes", mode,
                    "--max-new-tokens", "2048", "--batch-size", "4" if arm == "base_icl32" else "16",
                    "--seed", "202609092015"]
            if not arm.startswith("base_"):
                args += ["--adapter-path", str(root / "runs" / ("grpo" if arm == "rl" else arm) / "final")]
            steps.append({"name": arm, "entrypoint": "evaluate.py", "args": args, "required_records": []})
        phases[mode] = {"mode": mode, "not_after_utc": builder.DEADLINE, "source_freeze": "old",
                        "out": "old", "steps": steps}
    return training, phases


class PlanTransformTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/temporary/sprint")
        self.training, self.phases = originals(self.root)
        self.prefix = {"status": "validated", "v3_pipeline_success": False,
                       "checkpoint_files": {"sft": "a", "dpo": "b"}, "sha256": {"receipt": "value"}}

    def build(self):
        return builder.construct_plans(self.root, self.prefix, self.training, self.phases, "fixture_time")

    def test_exact_grpo_change_and_preserve_inputs(self):
        before = deepcopy((self.training, self.phases, self.prefix))
        docs = self.build()
        step = docs["primary_training_v4.json"]["steps"][0]
        expected = deepcopy(builder.GRPO_ARGS)
        expected[expected.index("--out") + 1] = "runs/grpo_v4"
        self.assertEqual(step["args"], expected)
        self.assertEqual(step["required_records"][0]["equals"]["optimizer_steps"], 40)
        self.assertEqual((self.training, self.phases, self.prefix), before)
        self.assertFalse(docs["continuation_prefix_v4.json"]["v3_pipeline_success"])

    def test_eval_only_output_and_rl_path_change(self):
        docs = self.build()
        for mode in ("greedy", "sampled"):
            plan = docs[f"eval_{mode}_v4.json"]
            for before, after in zip(self.phases[mode]["steps"], plan["steps"], strict=True):
                expected = deepcopy(before["args"])
                expected[expected.index("--out") + 1] = str(self.root / "runs" / f"eval_{mode}_v4" / before["name"])
                if before["name"] == "rl":
                    expected[expected.index("--adapter-path") + 1] = str(self.root / "runs/grpo_v4/final")
                self.assertEqual(after["args"], expected)
            self.assertEqual([step["name"] for step in plan["steps"]], builder.ARMS)

    def test_dependencies_and_one_remote_gpu(self):
        docs = self.build()
        for name in ("primary_training_job_v4.json", "eval_greedy_job_v4.json", "eval_sampled_job_v4.json"):
            self.assertEqual(docs[name]["candidate_gpus"], [3])
            self.assertEqual(docs[name]["execution_host"], "g374")
            self.assertEqual(docs[name]["execution_alias"], "simplex2")
            self.assertEqual(docs[name]["not_after_utc"], builder.DEADLINE)
        train = docs["primary_training_job_v4.json"]["wait_for_success"]
        self.assertEqual(len(train), 4)
        sampled = docs["eval_sampled_job_v4.json"]["wait_for_success"]
        status_paths = {row["path"] for row in sampled if row["path"].endswith("evaluation_status.json")}
        self.assertEqual(status_paths, {str(self.root / "runs/eval_greedy_v4" / arm / "evaluation_status.json") for arm in builder.ARMS})

    def test_prefix_reference_hash_is_exact_serialized_receipt(self):
        docs = self.build()
        reference = docs["primary_training_v4.json"]["prefix_receipt"]
        self.assertEqual(reference["sha256"], builder.digest_bytes(builder.encoded(docs["continuation_prefix_v4.json"])))
        saved = deepcopy(docs["continuation_prefix_v4.json"])
        saved.pop("created_utc")
        self.assertEqual(saved, self.prefix)

    def test_changed_training_or_eval_budget_rejected(self):
        self.training["steps"][0]["args"][7] = "39"
        with self.assertRaises(ValueError):
            self.build()
        self.training, self.phases = originals(self.root)
        args = self.phases["greedy"]["steps"][0]["args"]
        args[args.index("--max-new-tokens") + 1] = "1024"
        with self.assertRaises(ValueError):
            self.build()


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v4-bundle-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "plans").mkdir()
        self.source = self.root / "fixture_input.txt"
        self.source.write_text("stable fixture")
        self.freeze = self.root / "plans/primary_source_freeze_v4.json"
        self.first = self.root / "plans/primary_training_v4.json"
        raw = builder.encoded({"fixture": True})
        self.bundle = {"sprint": self.root,
                       "payloads": {self.first: raw, self.freeze: builder.encoded({"sha256": {str(self.first): builder.digest_bytes(raw)}})},
                       "input_sha256": {str(self.source): builder.digest_bytes(self.source.read_bytes())},
                       "inventory": {self.source}, "freeze_path": self.freeze}
        self.patch_inventory = patch.object(builder, "inventory", return_value={self.source})
        self.patch_inventory.start()
        self.addCleanup(self.patch_inventory.stop)
        self.patch_deadline = patch.object(builder, "DEADLINE", "2100-01-01T00:00:00+00:00")
        self.patch_deadline.start()
        self.addCleanup(self.patch_deadline.stop)

    def test_exclusive_publication_freeze_last_no_self_hash(self):
        result = builder.publish_bundle(self.bundle)
        self.assertEqual(result["created_files"][-1], str(self.freeze))
        freeze = json.loads(self.freeze.read_text())
        self.assertNotIn(str(self.freeze), freeze["sha256"])
        self.assertEqual(self.source.read_text(), "stable fixture")
        with self.assertRaisesRegex(ValueError, "already exists"):
            builder.publish_bundle(self.bundle)

    def test_all_paths_preflight_before_first_write(self):
        occupied = self.root / "runs/grpo_v4"
        occupied.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            builder.publish_bundle(self.bundle)
        self.assertFalse(self.first.exists())
        self.assertFalse(self.freeze.exists())

    def test_input_changed_before_write_is_rejected(self):
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed before publication"):
            builder.publish_bundle(self.bundle)
        self.assertFalse(self.first.exists())

    def test_input_change_mid_publication_never_commits_freeze(self):
        original = Path.open
        def altered(path, *args, **kwargs):
            if path == self.first and args == ("xb",):
                self.source.write_text("changed after initial preflight")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", new=altered):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                builder.publish_bundle(self.bundle)
        self.assertTrue(self.first.exists())
        self.assertFalse(self.freeze.exists())

    def test_missing_gate_prevents_any_plan_build(self):
        receipts = builder.Receipts()
        with self.assertRaises(FileNotFoundError):
            builder.check_gates(self.root, receipts)
        self.assertFalse(self.first.exists())


class InventoryTests(unittest.TestCase):
    def test_optional_protected_release_edits_do_not_enter_primary_freeze(self):
        with tempfile.TemporaryDirectory(prefix="v4-inventory-fixture-") as directory:
            root = Path(directory)
            for parent in ("code", "review_tests", "plans"):
                (root / parent).mkdir()
            for name in (*builder.DIAGNOSTICS, *builder.GATES):
                for parent in ("runs", "jobs"):
                    (root / parent / name).mkdir(parents=True)
            (root / "code/modeling.py").write_text("fixture core")
            before = builder.inventory(root)
            (root / "code/make_textbook_release_v4.py").write_text("optional work in progress")
            (root / "code/test_textbook_release_v4.py").write_text("optional test")
            (root / "review_tests/test_textbook_release_v4_extra.py").write_text("optional review")
            self.assertEqual(builder.inventory(root), before)
            self.assertIn(root / "code/modeling.py", before)


if __name__ == "__main__":
    unittest.main()
