"""Synthetic metadata-only release tests. No actual release or payload access.

The expensive existing prefix validator and tensor-header checker are mocked in
end-to-end fixtures; their invocation/lineage is checked. New rollout, audit,
completion, source, and unchanged generic-release gates execute normally.
"""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate
import make_textbook_release_v4 as release


class Fixture:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.approved = {}
        for name in release.APPROVED:
            self.raw(name, (name + " synthetic source").encode())
            self.approved[name] = self.sha(root / name)
        for name in ("numerics.py", "continuation_prefix.py", "make_textbook_release_v4.py", "test_textbook_release_v4.py"):
            self.raw("code/" + name, (name + " fixture only").encode())
        self.numeric = self.sha(root / "code/numerics.py")
        self.prefix_sha = self.sha(root / "code/continuation_prefix.py")
        checkpoints = {}
        for stage in ("sft", "dpo", "grpo_v4"):
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                if name == "adapter_config.json":
                    self.put(f"runs/{stage}/final/{name}", {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]})
                else:
                    self.raw(f"runs/{stage}/final/{name}", (stage + name + " fake checkpoint").encode())
                if stage != "grpo_v4":
                    checkpoints[str(root / f"runs/{stage}/final/{name}")] = self.sha(root / f"runs/{stage}/final/{name}")
        self.digest = {"sha256": "1" * 64, "parameters": 134217728}
        for stage in ("sft", "dpo"):
            self.put(f"runs/{stage}/summary.json", {"status": "complete", "adapter_digest": self.digest})
        self.old_args = {"out": str(root / "runs/grpo"), "stage": "grpo", "steps": 40, "adapter": "runs/dpo/final"}
        self.old_run = {"args": self.old_args, "data": {"rows": 32, "proof_cap": 6, "protocol": {"fixture": True}},
            "versions": {"fixture": "not real training"}, "mechanical_fixture": False}
        self.put("runs/grpo/run.json", self.old_run)
        self.put("runs/grpo/trainer_config.json", {"output_dir": str(root / "runs/grpo"),
            "run_name": None, "logging_dir": str(root / "runs/grpo/runs/old"),
            "num_iterations": 1, "steps_per_generation": 16, "beta": 0.0, "temperature": 1.0})
        old_step = {"name": "grpo", "entrypoint": "train.py", "args": ["grpo", "--adapter", "runs/dpo/final", "--out", "runs/grpo", "--steps", "40"],
            "required_records": []}
        self.put("plans/primary_training_v3.json", {"steps": [old_step]})
        self.prefix = {"status": "validated", "v3_pipeline_success": False, "checkpoint_files": checkpoints,
            "sha256": {str(root / name): digest for name, digest in self.approved.items()}}
        self.put("plans/continuation_prefix_v4.json", {**self.prefix, "created_utc": "2026-09-10T01:00:00+00:00"})
        new_step = json.loads(json.dumps(old_step))
        new_step["args"][4] = "runs/grpo_v4"
        new_step["required_records"] = [{"path": str(root / "runs/grpo_v4/summary.json"), "equals": {
            "status": "complete", "mechanical_fixture": False, "stage": "grpo", "optimizer_steps": 40,
            "requested_steps": 40, "deadline_reached": False}}]
        self.plan = {"approved_source_changes": self.approved,
            "prefix_receipt": self.ref("plans/continuation_prefix_v4.json"), "numerical_helper": self.ref("code/numerics.py"),
            "source_freeze": str(root / "plans/primary_source_freeze_v4.json"),
            "out": str(root / "runs/training_continuation_v4"), "not_after_utc": release.FINAL_FLOOR.isoformat(), "steps": [new_step]}
        self.put("plans/primary_training_v4.json", self.plan)
        spec = {"entrypoint": "pipeline.py", "args": ["--plan", str(root / "plans/primary_training_v4.json")],
            "job_dir": str(root / "jobs/primary_training_v4"), "not_after_utc": release.FINAL_FLOOR.isoformat(), "wait_for_success": []}
        self.put("plans/primary_training_job_v4.json", spec)
        self.put("jobs/primary_training_v4/spec.json", spec)
        self.put("jobs/primary_training_v4/completed.json", {"exit_code": 0})
        frozen_names = [*self.approved, "code/numerics.py", "code/continuation_prefix.py", "plans/continuation_prefix_v4.json",
            "plans/primary_training_v4.json", "plans/primary_training_job_v4.json"]
        self.put("plans/primary_source_freeze_v4.json", {"version": 4,
            "supersedes": str(root / "plans/primary_source_freeze_v3.json"),
            "sha256": {str(root / name): self.sha(root / name) for name in frozen_names}})
        self.freeze_sha = self.sha(root / "plans/primary_source_freeze_v4.json")
        self.put("runs/training_continuation_v4/completed.json", {"status": "complete", "stages": ["grpo"],
            "plan_sha256": self.sha(root / "plans/primary_training_v4.json"),
            "source_freeze_sha256": self.sha(root / "plans/primary_source_freeze_v4.json")})
        self.put("runs/training_continuation_v4/plan.json", self.plan)
        self.rows("runs/training_continuation_v4/events.jsonl", [
            {"event": "stage_start", "name": "grpo", "command": [str(root / ".venv/bin/python"), "-u", str(root / "code/train.py"), *new_step["args"]]},
            {"event": "stage_pid", "name": "grpo", "pid": 123}, {"event": "stage_exit", "name": "grpo", "exit_code": 0},
            {"event": "pipeline_completed", "stages": ["grpo"]}])
        stage = root / "runs/grpo_v4"
        self.put("runs/grpo_v4/run.json", {**self.old_run, "args": {**self.old_args, "out": str(stage)},
            "source_sha256": {name: self.approved["code/" + name] for name in ("modeling.py", "train.py")}})
        self.put("runs/grpo_v4/trainer_config.json", {**self.get("runs/grpo/trainer_config.json"),
            "output_dir": str(stage), "logging_dir": str(stage / "runs/new_host_and_time")})
        self.identity = {"adapter_path": str(root / "runs/dpo/final"),
            "adapter_sha256": {name: self.sha(root / "runs/dpo/final" / name) for name in ("adapter_model.safetensors", "adapter_config.json")},
            "model_id": release.prefix.MODEL_ID, "revision": release.prefix.MODEL_REVISION,
            "adapter_load_audits": [self.audit("dpo")],
            "numerical_policy": {"source_sha256": self.numeric, "policy_id": "nf4_fp32_compute_expanded_gqa_efficient_v4",
                "bnb_linear_count": 448, "bnb_compute_dtype": "float32", "attention_kernel": "EFFICIENT_ATTENTION", "native_enable_gqa": False}}
        self.put("runs/grpo_v4/trainer_state.json", {"global_step": 40})
        self.put("runs/grpo_v4/summary.json", {"stage": "grpo", "status": "complete", "mechanical_fixture": False,
            "optimizer_steps": 40, "requested_steps": 40, "deadline_reached": False, "model_identity": self.identity,
            "protocol": self.old_run["data"]["protocol"], "adapter": str(stage / "final"), "saved_tensor_audit": self.audit("grpo_v4"),
            "nonzero_gradient_steps": 0, "observed_parameter_changes": 0})
        self.rows("runs/grpo/events.jsonl", [{"kind": "model_loaded", "identity": self.identity}])
        self.rows("data/shared_cap6.jsonl", [{"canonical_hash": f"c{i}", "prompt": f"prompt{i}"} for i in range(32)])
        verdict = {"accepted": False, "reward": 0.0, "reason": "synthetic invalid fixture"}
        actions, events = [], [{"kind": "model_loaded", "identity": self.identity},
            {"kind": "trainer_preserved_adapter_values", "before": self.digest, "after": self.digest},
            {"kind": "stage_start", "stage": "grpo", "rows": 32, "steps": 40, "mechanical": False, "model_identity": self.identity}]
        for batch in range(40):
            for index in range(16):
                case = (4 * batch + index // 4) % 32
                actions.append({"batch": batch, "index": index, "mechanical_reward_fixture": False,
                    "canonical_hash": f"c{case}", "theorem": f"prompt{case}", "completion_ids": [1],
                    "behavior_logprobs": [-0.2], "completion": "fixture", "verdict": verdict, "reward": 0.0})
            finish = {"final_marker": 16}
            events.extend([{"kind": "generation_batch", "step": batch, "finish_counts": finish, "prompt_tokens": 48, "completion_tokens": 16},
                {"kind": "reward_batch", "batch": batch, "examples": 16, "groups": 4, "mixed_groups": 0,
                    "mixed_group_fraction": 0.0, "success_fraction": 0.0, "reward_mean": 0.0,
                    "completion_tokens": 16, "mechanical_reward_fixture": False}, self.likelihood(batch, False, 16)])
            events.extend(self.likelihood(batch, True, 1, index) for index in range(16))
            events.append({"kind": "gradient", "step": batch, "norm": 0.0})
        events.append({"kind": "stage_complete", "utc": "fixture", **self.get("runs/grpo_v4/summary.json")})
        self.rows("runs/grpo_v4/rollouts.jsonl", actions)
        self.rows("runs/grpo_v4/events.jsonl", events)
        manifest = root.parent / "r1-cot-20260909/data/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"files": {"textbook_dev.jsonl": release.DATA_SHA["dev"],
            "sealed_final/textbook_final.jsonl": release.DATA_SHA["final"]}}))
        self.tools = (SimpleNamespace(render_prompt=lambda text: text, judge=lambda *args: verdict),
            SimpleNamespace(decode_completion=lambda *args: "fixture", trim_completion=lambda tokenizer, ids: (ids, "final_marker")),
            SimpleTokenizer())

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def raw(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def put(self, name, value):
        self.raw(name, (json.dumps(value, sort_keys=True) + "\n").encode())

    def get(self, name):
        return json.loads((self.root / name).read_text())

    def rows(self, name, values):
        self.raw(name, b"".join((json.dumps(value) + "\n").encode() for value in values))

    def get_rows(self, name):
        return [json.loads(line) for line in (self.root / name).read_text().splitlines()]

    def ref(self, name):
        return {"path": str(self.root / name), "sha256": self.sha(self.root / name)}

    def audit(self, stage):
        weight = self.root / "runs" / stage / "final/adapter_model.safetensors"
        return {"checkpoint": str(weight), "sha256": self.sha(weight), "all_tensors_exact": True,
            "dtype": "float32", "adapter_name": "default", "tensors": 896, "parameters": 134217728}

    @staticmethod
    def likelihood(step, grad, tokens, index=0):
        return {"kind": "actual_grpo_gradient_likelihood_audit" if grad else "actual_grpo_likelihood_audit",
            "step": step, "phase": "actual_gradient_loss" if grad else "fresh_rollout", "tokens": tokens,
            "tolerance": 0.1, "denominator": "actual_streamed_sampling_logprobs", "scorer_model_training": True,
            "scorer_requires_grad": grad, "max_abs_error": 0.0001, "mean_abs_error": 0.00001,
            "p99_abs_error": 0.00009, "finish_counts": {"final_marker": 16}, "fixture_unique": index}


class SimpleTokenizer:
    def __init__(self):
        self.length_queries = 0

    def __len__(self):
        self.length_queries += 1
        return 100

    def encode(self, *args, **kwargs):
        return [2, 3, 4]


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name) / "sprint")

    def build(self, cohort="dev", now=None):
        fixture = self.fixture
        original_open = Path.open

        def no_payload(path, *args, **kwargs):
            if path.name in {"textbook_dev.jsonl", "textbook_final.jsonl"}:
                raise AssertionError("Protected payload opened")
            return original_open(path, *args, **kwargs)

        with patch.object(release, "APPROVED", fixture.approved), patch.object(release, "NUMERICS_SHA", fixture.numeric), \
             patch.object(release, "PRIMARY_FREEZE_SHA", fixture.freeze_sha), \
             patch.object(release, "PREFIX_SHA", fixture.prefix_sha), patch.object(release.prefix, "validate_prefix", return_value=fixture.prefix) as validator, \
             patch.object(release.prefix, "_load_tools", return_value=fixture.tools), patch.object(release.prefix, "_check_tensor_header") as header, \
             patch.dict(release.prefix.PINNED, {"runs/dpo/final/adapter_model.safetensors": fixture.sha(fixture.root / "runs/dpo/final/adapter_model.safetensors")}), \
             patch.object(evaluate, "ROOT", fixture.root.parent), patch.object(Path, "open", no_payload):
            result = release.build_release(cohort, sprint=fixture.root, now=now)
            validator.assert_called_once_with(fixture.root, approved_source_changes=fixture.approved)
            header.assert_called_once()
            return result

    def test_complete_zero_signal_is_truthful_and_payload_free(self):
        result = self.build()
        self.assertEqual(result["rl_signal"]["nonzero_gradient_steps"], 0)
        self.assertEqual(result["rl_signal"]["actual_gradient_likelihood_audits"], 640)
        self.assertEqual(result["rl_signal"]["mixed_groups"], 0)
        self.assertFalse(result["v3_pipeline_success"])
        self.assertFalse(result["protected_payload_read_by_release_builder"])
        self.assertEqual(result["settings"], release.SETTINGS)
        self.assertEqual(result["release_tests"], self.fixture.ref("code/test_textbook_release_v4.py"))
        self.assertEqual(result["checkpoint_map"]["rl"], str(self.fixture.root / "runs/grpo_v4/final"))
        self.assertEqual(self.fixture.tools[2].length_queries, 1)

    def test_final_floor(self):
        from datetime import timedelta
        with self.assertRaisesRegex(ValueError, "not yet permitted"):
            self.build("final", release.FINAL_FLOOR - timedelta(seconds=1))
        result = self.build("final", release.FINAL_FLOOR)
        self.assertEqual(result["not_before_utc"], release.FINAL_FLOOR.isoformat())

    def test_failed_job_blocks(self):
        self.fixture.put("jobs/primary_training_v4/completed.json", {"exit_code": 1})
        with self.assertRaisesRegex(ValueError, "not exited successfully"):
            self.build()

    def test_missing_or_failed_actual_gradient_guard_blocks(self):
        name = "runs/grpo_v4/events.jsonl"
        original = self.fixture.get_rows(name)
        missing = list(original)
        index = next(index for index, row in enumerate(missing) if row.get("kind") == "actual_grpo_gradient_likelihood_audit")
        del missing[index]
        self.fixture.rows(name, missing)
        with self.assertRaisesRegex(ValueError, "Missing or repeated actual-gradient"):
            self.build()
        original[index]["max_abs_error"] = 0.10001
        self.fixture.rows(name, original)
        with self.assertRaisesRegex(ValueError, "failed likelihood"):
            self.build()

    def test_incomplete_endpoint_and_untruthful_gradient_counts_block(self):
        name = "runs/grpo_v4/summary.json"
        summary = self.fixture.get(name)
        summary["optimizer_steps"] = 39
        self.fixture.put(name, summary)
        with self.assertRaisesRegex(ValueError, "Incomplete or mechanical"):
            self.build()
        summary["optimizer_steps"] = 40
        summary["nonzero_gradient_steps"] = 40
        self.fixture.put(name, summary)
        rows = self.fixture.get_rows("runs/grpo_v4/events.jsonl")
        rows[-1] = {"kind": "stage_complete", "utc": "fixture", **summary}
        self.fixture.rows("runs/grpo_v4/events.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "misreports actual gradient"):
            self.build()

    def test_source_and_protected_freeze_injection_block(self):
        name = "plans/primary_source_freeze_v4.json"
        freeze = self.fixture.get(name)
        freeze["sha256"][str(self.fixture.root.parent / "r1-cot-20260909/data/textbook_dev.jsonl")] = "0" * 64
        self.fixture.put(name, freeze)
        with self.assertRaisesRegex(ValueError, "Unapproved primary v4 freeze"):
            self.build()
        with self.assertRaisesRegex(ValueError, "never inspect protected"):
            release.SafeReceipts(self.fixture.root).digest(self.fixture.root.parent / "r1-cot-20260909/data/textbook_dev.jsonl")
        self.fixture.raw("code/numerics.py", b"unapproved change")
        with self.assertRaisesRegex(ValueError, "Unapproved v4 source"):
            self.build()

    def test_mechanical_reward_and_wrong_theorem_block(self):
        name = "runs/grpo_v4/rollouts.jsonl"
        rows = self.fixture.get_rows(name)
        rows[0]["mechanical_reward_fixture"] = True
        self.fixture.rows(name, rows)
        with self.assertRaisesRegex(ValueError, "Mechanical/misordered"):
            self.build()
        rows[0]["mechanical_reward_fixture"] = False
        rows[0]["theorem"] = "different theorem"
        self.fixture.rows(name, rows)
        with self.assertRaisesRegex(ValueError, "outside the frozen training"):
            self.build()

    def test_adapter_configuration_change_blocks(self):
        name = "runs/grpo_v4/final/adapter_config.json"
        config = self.fixture.get(name)
        config["r"] = 8
        self.fixture.put(name, config)
        with self.assertRaisesRegex(ValueError, "adapter configuration differs"):
            self.build()

    def test_differentiable_guard_cannot_be_no_grad(self):
        name = "runs/grpo_v4/events.jsonl"
        rows = self.fixture.get_rows(name)
        audit = next(row for row in rows if row.get("kind") == "actual_grpo_gradient_likelihood_audit")
        audit["scorer_requires_grad"] = False
        self.fixture.rows(name, rows)
        with self.assertRaisesRegex(ValueError, "Wrong likelihood audit policy/scope"):
            self.build()


if __name__ == "__main__":
    unittest.main()
