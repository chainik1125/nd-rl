"""CPU-only synthetic pilot tests; never select real targets or read predictions."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import budget_pilot_v1 as pilot


def raw_rows(rows):
    return b"".join((json.dumps(row) + "\n").encode() for row in rows)


def fixture_panels():
    def row(index, length):
        return {"canonical_hash": f"canonical{index:03}", "prompt": "synthetic query " + str(index),
            "reference_lines": length, "family": f"family{index % 7}", "role": "synthetic_fixture"}
    return {"eval_short_panel.jsonl": raw_rows([row(index, 2 + index % 5) for index in range(56)]),
        "eval_long_panel.jsonl": raw_rows([row(100 + index, 7 + index % 8) for index in range(64)])}


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pilot.encoded(value))

    def test_selection_counts_literal_lines_and_original_order(self):
        raw = fixture_panels()
        selected, metadata = pilot.select_panels(raw)
        self.assertEqual(pilot.SALT, "olmo-20260909-budget-pilot-24-v1")
        self.assertEqual([value["rows"] for value in metadata.values()], [8, 16])
        long = [json.loads(line) for line in selected["eval_long_panel.jsonl"].splitlines()]
        self.assertEqual(Counter(row["reference_lines"] for row in long), {length: 2 for length in range(7, 15)})
        for name, value in selected.items():
            originals = raw[name].splitlines(keepends=True)
            retained = value.splitlines(keepends=True)
            self.assertTrue(all(line in originals for line in retained))
            self.assertEqual([originals.index(line) for line in retained], sorted(originals.index(line) for line in retained))
        self.assertEqual((selected, metadata), pilot.select_panels(raw))

    def test_prompt_length_and_family_do_not_select(self):
        raw = fixture_panels()
        _, before = pilot.select_panels(raw)
        for name in raw:
            rows = [json.loads(line) for line in raw[name].splitlines()]
            for index, row in enumerate(rows):
                row["prompt"] = "arbitrary " * (500 if index % 2 else 1)
                row["family"] = "changed metadata"
            raw[name] = raw_rows(rows)
        _, after = pilot.select_panels(raw)
        self.assertEqual([value["canonical_hashes"] for value in before.values()], [value["canonical_hashes"] for value in after.values()])

    def test_labels_missing_strata_and_wrong_denominators_rejected(self):
        raw = fixture_panels()
        rows = [json.loads(line) for line in raw["eval_long_panel.jsonl"].splitlines()]
        rows[0]["proof"] = "not allowed"
        raw["eval_long_panel.jsonl"] = raw_rows(rows)
        with self.assertRaisesRegex(ValueError, "proof label"):
            pilot.select_panels(raw)
        rows[0].pop("proof")
        for row in rows:
            if row["reference_lines"] == 14:
                row["reference_lines"] = 13
        raw["eval_long_panel.jsonl"] = raw_rows(rows)
        with self.assertRaisesRegex(ValueError, "Long strata changed"):
            pilot.select_panels(raw)
        raw["eval_short_panel.jsonl"] = raw_rows([])
        with self.assertRaises(ValueError):
            pilot.select_panels(raw)

    def test_plans_preserve_settings_and_prioritize_separate_pilot(self):
        root = self.root
        original = {"mode": "greedy", "not_after_utc": pilot.DEADLINE, "steps": []}
        for arm in pilot.ARMS:
            args = ["--arm", arm, "--out", str(root / "runs/eval_greedy_v4" / arm), "--cohorts", "long_panel", "short_panel",
                "--modes", "greedy", "--max-new-tokens", "2048", "--batch-size", "4" if arm == "base_icl32" else "16", "--seed", "202609092015"]
            if not arm.startswith("base_"):
                args.extend(["--adapter-path", str(root / "runs" / ("grpo_v4" if arm == "rl" else arm) / "final")])
            original["steps"].append({"name": arm, "entrypoint": "evaluate.py", "args": args, "required_records": []})
        self.put("plans/eval_greedy_v4.json", original)
        greedy = {"entrypoint": "pipeline.py", "args": ["--plan", str(root / "plans/eval_greedy_v4.json")],
            "job_dir": str(root / "jobs/eval_greedy_v4"), "not_after_utc": pilot.DEADLINE,
            "wait_for_success": [pilot.dependency(root / "runs/grpo_v4/summary.json", status="complete", optimizer_steps=40)]}
        sampled = {**deepcopy(greedy), "args": ["--plan", str(root / "plans/eval_sampled_v4.json")]}
        sampled["wait_for_success"].append(pilot.dependency(root / "jobs/eval_greedy_v4/completed.json", exit_code=0))
        preserved = deepcopy((original, greedy, sampled))
        docs = pilot.construct_plans(root, original, greedy, sampled)
        self.assertEqual((original, greedy, sampled), preserved)
        self.assertEqual(docs[pilot.PLAN]["source_freeze"], str(root / pilot.RUNTIME_FREEZE))
        for old, new in zip(original["steps"], docs[pilot.PLAN]["steps"], strict=True):
            expected = old["args"].copy()
            pilot.replacement(expected, "--out", str(root / pilot.RUN / old["name"]))
            expected += ["--data-dir", str(root / pilot.DATA)]
            self.assertEqual(new["args"], expected)
            self.assertEqual(new["required_records"][0]["equals"], {"status": "complete", "attempts": 24, "planned_attempts": 24})
        self.assertEqual(docs["plans/eval_greedy_after_pilot_job_v1.json"]["args"], greedy["args"])
        self.assertEqual(docs["plans/eval_sampled_after_pilot_job_v1.json"]["args"], sampled["args"])
        self.assertIn(pilot.dependency(root / pilot.RUNTIME_FREEZE, status="complete", kind="synthetic_pilot_runtime_freeze"), docs[pilot.JOB]["wait_for_success"])

    def test_endpoint_projection_discards_all_release_authorization(self):
        validated = {"completed_lineage": ["v3_sft", "v3_generated_pairs", "v3_dpo", "v4_grpo"], "v3_pipeline_success": False,
            "checkpoint_files": {}, "checkpoint_map": {}, "completed_training": [], "rl_signal": {"actual_gradient_likelihood_audits": 640},
            "kind": "textbook_development_release", "allowed_arms": ["rl"], "data_path": "/PROTECTED_PAYLOAD_SENTINEL",
            "data_sha256": "not carried", "opaque_manifest_path": "/ALSO_NOT_CARRIED", "not_before_utc": "irrelevant"}
        validated["validated_receipts_sha256"] = {str(self.root / name): "0" * 64 for name in pilot.REQUIRED_TRAINING_RECEIPTS}
        opaque = str(self.root.parent / "r1-cot-20260909/data/manifest.json")
        validated["validated_receipts_sha256"][opaque] = "1" * 64
        with patch.object(pilot, "ref", side_effect=lambda path: {"path": str(path), "sha256": "fixture"}):
            result = pilot.binding_from_validated_chain(validated, self.root)
        self.assertEqual(result["kind"], "synthetic_pilot_endpoint_binding")
        self.assertFalse(result["textbook_release_issued"])
        self.assertTrue({"data_path", "data_sha256", "allowed_arms", "opaque_manifest_path", "not_before_utc"}.isdisjoint(result))
        self.assertNotIn("PROTECTED_PAYLOAD_SENTINEL", json.dumps(result))
        self.assertNotIn(opaque, json.dumps(result))
        self.assertEqual(len(result["validated_training_receipts_sha256"]), len(pilot.REQUIRED_TRAINING_RECEIPTS))
        del validated["validated_receipts_sha256"][str(self.root / "runs/grpo_v4/events.jsonl")]
        with self.assertRaisesRegex(ValueError, "omitted required"):
            pilot.binding_from_validated_chain(validated, self.root)

    def test_metadata_guard_requires_all_five_before_reading_targets(self):
        with self.assertRaisesRegex(ValueError, "all five"):
            pilot.guard_metadata([], {}, {}, self.root)

    def metadata_fixture(self):
        from evaluate import MODEL_ID, MODEL_REVISION, stable_hash
        selected, _ = pilot.select_panels(fixture_panels())
        (self.root / pilot.DATA).mkdir()
        for name, value in selected.items():
            (self.root / pilot.DATA / name).write_bytes(value)
        manifest = {"cohort_counts": pilot.COUNTS}
        self.put(pilot.DATA + "/manifest.json", manifest)
        self.put(pilot.DATA + "/shared_cap6.jsonl", {"fixture": "not used by metadata guard"})
        self.put("code/modeling.py", {"fixture": True})
        self.put("code/evaluate.py", {"fixture": True})
        targets = []
        for cohort, name in (("long_panel", "eval_long_panel.jsonl"), ("short_panel", "eval_short_panel.jsonl")):
            targets.extend(dict(json.loads(line), cohort=cohort) for line in selected[name].splitlines())
        binding = {"checkpoint_map": {}, "checkpoint_files": {}}
        for arm in ("sft", "dpo", "rl"):
            directory = self.root / "runs" / ("grpo_v4" if arm == "rl" else arm) / "final"
            binding["checkpoint_map"][arm] = str(directory)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                self.put(str(directory / name), {"fixture": arm})
                binding["checkpoint_files"][str(directory / name)] = pilot.digest(directory / name)
        inputs = {str(self.root / pilot.DATA / name): pilot.digest(self.root / pilot.DATA / name) for name in
            ("manifest.json", "shared_cap6.jsonl", "eval_short_panel.jsonl", "eval_long_panel.jsonl")}
        plans = []
        for arm in pilot.ARMS:
            plans.append({"arm": arm, "target_count": 24, "target_identity": stable_hash(targets), "cohorts": ["long_panel", "short_panel"],
                "input_files": inputs, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_path": str(self.root / "models/olmo32b"),
                "modeling_sha256": pilot.digest(self.root / "code/modeling.py"), "evaluator_sha256": pilot.digest(self.root / "code/evaluate.py"),
                "settings": {"modes": ["greedy"], "max_new_tokens": 2048, "sampled_k": 4, "sampled_temperature": 1.0,
                    "top_p": 1.0, "top_k": 0, "seed": 202609092015, "batch_size": 4 if arm == "base_icl32" else 16,
                    "stop_on_actually_generated_final_marker": True, "native_eos_fallback": True, "inserted_eos": False},
                "demonstrations_per_query": 32 if arm == "base_icl32" else 0,
                "adapter_files": {} if arm.startswith("base_") else {path: value for path, value in binding["checkpoint_files"].items()
                    if Path(path).parent == Path(binding["checkpoint_map"][arm])}})
        return plans, binding, manifest

    def test_complete_metadata_then_mixed_panel_checkpoint_and_cap_rejected(self):
        plans, binding, manifest = self.metadata_fixture()
        self.assertEqual(len(pilot.guard_metadata(plans, binding, manifest, self.root)), 24)
        plans[0]["target_count"] = 120
        with self.assertRaisesRegex(ValueError, "Pilot/full"):
            pilot.guard_metadata(plans, binding, manifest, self.root)
        plans[0]["target_count"] = 24
        original = dict(plans[-1]["adapter_files"])
        plans[-1]["adapter_files"][next(iter(original))] = "different weight hash"
        with self.assertRaisesRegex(ValueError, "different bound endpoint"):
            pilot.guard_metadata(plans, binding, manifest, self.root)
        plans[-1]["adapter_files"] = original
        plans[0]["settings"]["max_new_tokens"] = 512
        with self.assertRaisesRegex(ValueError, "decoding"):
            pilot.guard_metadata(plans, binding, manifest, self.root)

    def test_runtime_freeze_commits_all_checkpoints_and_training_receipts(self):
        _, binding, _ = self.metadata_fixture()
        binding["completed_training"] = [{"path": str(self.root / "runs/grpo_v4/summary.json"), "sha256": "final40receipt"}]
        binding["validated_training_receipts_sha256"] = {str(self.root / "runs/grpo_v4/events.jsonl"): "full audit events"}
        self.put(pilot.FREEZE, {"fixture": True})
        self.put(pilot.BINDING, binding)
        frozen = {"sha256": {str(self.root / "code/modeling.py"): pilot.digest(self.root / "code/modeling.py")}}
        result = pilot.runtime_from_binding(binding, frozen, self.root)
        for path, expected in {**frozen["sha256"], **binding["checkpoint_files"]}.items():
            self.assertEqual(result["sha256"][path], expected)
        self.assertEqual(result["sha256"][binding["completed_training"][0]["path"]], "final40receipt")
        self.assertEqual(result["sha256"][str(self.root / pilot.BINDING)], pilot.digest(self.root / pilot.BINDING))
        self.assertEqual(result["sha256"][str(self.root / pilot.FREEZE)], pilot.digest(self.root / pilot.FREEZE))
        self.assertEqual(result["sha256"][str(self.root / "runs/grpo_v4/events.jsonl")], "full audit events")

    def test_runtime_event_mutation_and_hash_override_fail(self):
        self.put("runs/grpo_v4/events.jsonl", {"fixture": "validated audit"})
        path = str(self.root / "runs/grpo_v4/events.jsonl")
        merged = pilot.merge_receipts({"unchanged_static": "fixed"}, {path: pilot.digest(path)})
        with self.assertRaisesRegex(ValueError, "Conflicting receipt"):
            pilot.merge_receipts(merged, {"unchanged_static": "altered"})
        pilot.verify_sources({path: merged[path]})
        self.put("runs/grpo_v4/events.jsonl", {"fixture": "altered audit"})
        with self.assertRaisesRegex(ValueError, "Pinned source/input changed"):
            pilot.verify_sources({path: merged[path]})


if __name__ == "__main__":
    unittest.main()
