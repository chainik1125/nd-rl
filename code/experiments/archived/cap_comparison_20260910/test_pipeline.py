"""Invented CPU fixtures for compute completion, source identity and release order."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline as p
import plan


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.spec = plan.build_specs(self.root, 100, 200)[0]
        self.metadata = {}
        self.source = self.root / "runs/shared_init/final.pt"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"invented prepared checkpoint")
        self.path = self.root / "plans" / (self.spec["name"] + ".json")
        p.write(self.path, self.spec)
        self.protocol = {"arms": [self.spec["name"]], "input_sha256": {
            str(self.path.relative_to(self.root)): p.sha(self.path),
            "runs/shared_init/final.pt": p.sha(self.source)}}
        p.write(self.root / "plans/protocol.json", self.protocol)
        self.calls = []

    def complete(self, stage):
        output = self.root / stage["output"]
        output.mkdir(parents=True, exist_ok=True)
        final = output / "final.pt"
        final.write_bytes((stage["command_name"] + " completed weights").encode())
        target = stage["target_flop_proxy"]
        records = []
        for index, point in enumerate(stage["milestones"]):
            checkpoint = output / f"flops_{point}.pt"
            checkpoint.write_bytes(f"milestone {point}".encode())
            records.append({"requested_flop_proxy": point, "actual_flop_proxy": point + 1,
                            "step": index + 1, "checkpoint": str(checkpoint), "checkpoint_sha256": p.sha(checkpoint)})
        complete = {"stage": stage["command_name"], "final_sha256": p.sha(final), "actual_steps": 10,
                    "actual_flop_proxy": target + 1, "target_flop_proxy": target,
                    "last_step_flop_proxy": 2, "budget_reached": True,
                    "stop_reason": "flop_budget", "compute_milestones": records,
                    "unreached_flop_milestones": []}
        p.write(output / "completed.json", complete)
        source = p.sha(self.root / stage["checkpoint"])
        p.write(output / "manifest.json", {"args": stage["expected_args"], "source_sha256": source})
        self.metadata[str(final)] = {"stage": stage["command_name"], "args": stage["expected_args"],
                                    "source_checkpoint_sha256": source,
                                    "training_compute": {"total_flop_proxy": target + 1,
                                                         "target_flop_proxy": target, "budget_reached": True}}
        return output

    def valid(self, stage=None):
        stage = stage or self.spec["training_stages"][0]
        return p.validate_training(stage, self.root, lambda path: self.metadata[str(path)])

    def change(self, output, field, value):
        path = output / "completed.json"
        obj = p.read(path)
        obj[field] = value
        p.write(path, obj)

    def test_valid_completion(self):
        self.complete(self.spec["training_stages"][0])
        result = self.valid()
        self.assertEqual(result["actual_flop_proxy"], 101)

    def test_incomplete_budget_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        self.change(output, "actual_flop_proxy", 99)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.valid()

    def test_step_limit_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        self.change(output, "stop_reason", "step_limit")
        with self.assertRaisesRegex(ValueError, "compute budget"):
            self.valid()

    def test_nonfirst_crossing_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        self.change(output, "last_step_flop_proxy", 1)
        with self.assertRaisesRegex(ValueError, "first-crossing"):
            self.valid()

    def test_sft_marker_cannot_release_rl(self):
        self.complete(self.spec["training_stages"][0])
        stage = self.spec["training_stages"][1]
        output = self.complete(stage)
        self.metadata[str(output / "final.pt")]["stage"] = "sft"
        with self.assertRaisesRegex(ValueError, "actual training stage"):
            self.valid(stage)

    def test_changed_source_rejected(self):
        self.complete(self.spec["training_stages"][0])
        self.source.write_bytes(b"changed init")
        with self.assertRaisesRegex(ValueError, "source checkpoint"):
            self.valid()

    def test_changed_final_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        (output / "final.pt").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.valid()

    def test_wrong_mode_and_cap_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        self.metadata[str(output / "final.pt")]["args"] = {**self.spec["training_stages"][0]["expected_args"], "mode": "direct"}
        with self.assertRaisesRegex(ValueError, "argument identity"):
            self.valid()

    def test_checkpoint_compute_mismatch_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        self.metadata[str(output / "final.pt")]["training_compute"]["total_flop_proxy"] = 1
        with self.assertRaisesRegex(ValueError, "checkpoint compute"):
            self.valid()

    def test_missing_milestone_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        complete = p.read(output / "completed.json")
        self.change(output, "compute_milestones", complete["compute_milestones"][:-1])
        with self.assertRaisesRegex(ValueError, "milestone set"):
            self.valid()

    def test_changed_milestone_rejected(self):
        output = self.complete(self.spec["training_stages"][0])
        (output / "flops_25.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Milestone checkpoint"):
            self.valid()

    def test_protected_input_rejected_before_payload_hash(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["input_sha256"]["evaluation/textbook_dev.jsonl"] = "0" * 64
        original = p.sha
        def guarded(path):
            self.assertNotIn("textbook_dev", str(path))
            return original(path)
        with patch.object(p, "sha", guarded), self.assertRaisesRegex(ValueError, "Protected"):
            p.check_frozen(self.root, protocol, self.root / "plans/protocol.json",
                           original(self.root / "plans/protocol.json"), self.path, original(self.path))

    def test_protected_alias_rejected(self):
        alias = self.root / "alias.json"
        alias.symlink_to(self.root / "evaluation/textbook_dev.jsonl")
        with self.assertRaisesRegex(ValueError, "aliased"):
            p.local(self.root, alias)

    def test_release_without_both_completed_stages_rejected(self):
        with self.assertRaisesRegex(ValueError, "Both actual"):
            p.create_development_release(self.spec, self.root, {}, "a", "b")

    def test_release_does_not_hash_payload(self):
        training = {}
        for stage in self.spec["training_stages"]:
            self.complete(stage)
            training[stage["command_name"]] = self.valid(stage)
        with patch.object(p, "sha", side_effect=AssertionError("release should not hash any theorem bytes")):
            path = p.create_development_release(self.spec, self.root, training, "a", "b")
        self.assertEqual(p.read(path)["data_sha256"], [plan.DEV_SHA])

    def fake_run(self, command, **kwargs):
        self.calls.append(command)
        name = command[2]
        output = Path(command[command.index("--output") + 1])
        if name in ("sft", "rl"):
            stage = next(x for x in self.spec["training_stages"] if x["command_name"] == name)
            self.complete(stage)
        else:
            data = command[command.index("--data") + 1]
            if "textbook_dev" in data:
                release = p.read(self.root / self.spec["development_release"])
                self.assertTrue(release["full_rl_budget_reached"])
                self.assertEqual(release["training"]["rl"]["actual_flop_proxy"], 201)
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            output.mkdir(parents=True)
            p.write(output / "completed.json", {"stage": "eval", "checkpoint_sha256": p.sha(checkpoint)})
            p.write(output / "summary.json", {"invented": True})
            (output / "attempts.jsonl").write_text('{"invented":true}\n')

    def test_complete_pipeline_orders_release_and_positions(self):
        with patch.object(p.subprocess, "run", side_effect=self.fake_run), \
                patch.object(p, "checkpoint_metadata", side_effect=lambda path: self.metadata[str(path)]):
            p.run(self.path)
        commands = [x[2] for x in self.calls]
        self.assertEqual(commands[:4], ["sft", "eval", "eval", "rl"])
        base = self.root / self.spec["run_base"]
        self.assertEqual(p.read(base / "pipeline_completed.json")["status"], "complete")
        positions = list((base / "evaluations").glob("*/checkpoint_position.json"))
        self.assertEqual(len(positions), 14)
        self.assertEqual(p.read(base / "evaluations/rl_m00_pretraining/checkpoint_position.json")["total_flop_proxy"], 26)
        self.assertEqual(p.read(base / "evaluations/sft_pretraining/checkpoint_position.json")["step"], 0)
        self.assertEqual(sum("textbook_dev.jsonl" in " ".join(x) for x in self.calls), 4)

    def test_failed_rl_does_not_release_or_complete(self):
        def failing(command, **kwargs):
            if command[2] == "rl":
                raise RuntimeError("invented training failure")
            self.fake_run(command, **kwargs)
        with patch.object(p.subprocess, "run", side_effect=failing), \
                patch.object(p, "checkpoint_metadata", side_effect=lambda path: self.metadata[str(path)]), \
                self.assertRaisesRegex(RuntimeError, "invented"):
            p.run(self.path)
        base = self.root / self.spec["run_base"]
        self.assertEqual(p.read(base / "progress.json")["status"], "failed")
        self.assertFalse((base / "development_release.json").exists())
        self.assertFalse((base / "pipeline_completed.json").exists())


class Planning(unittest.TestCase):
    def test_all_six_share_budgets_and_initialization(self):
        specs = plan.build_specs(Path("/tmp/invented_cap_comparison"), 800, 1600)
        self.assertEqual(len(specs), 6)
        self.assertEqual(specs[0]["name"], "cap10_cot_s20260910")
        self.assertEqual({s["shared_init"] for s in specs}, {"runs/shared_init/final.pt"})
        self.assertEqual({s["training_stages"][0]["target_flop_proxy"] for s in specs}, {800})
        self.assertEqual({s["training_stages"][1]["target_flop_proxy"] for s in specs}, {1600})
        self.assertTrue(all(s["training_stages"][1]["expected_args"]["max_new_tokens"] == 1536 for s in specs))

    def test_direct_cot_data_and_guard_identity(self):
        specs = plan.build_specs(Path("/tmp/invented_cap_comparison"), 800, 1600)
        for cap in (6, 10, 12):
            a, b = [s for s in specs if s["cap"] == cap]
            self.assertEqual(a["data"], b["data"])
            args = a["training_stages"][1]["expected_args"]
            self.assertTrue(args["guard_json"].endswith("/rl_guards.json"))
            self.assertTrue(args["retention_guard_json"].endswith("/supervised_guards.json"))

    def test_exact_scientific_integer(self):
        self.assertEqual(plan.integer("1.234e18"), 1234000000000000000)
        for x in ("0", "-1", "0.1", "nan", "inf"):
            with self.assertRaises(Exception):
                plan.integer(x)


if __name__ == "__main__":
    unittest.main()
