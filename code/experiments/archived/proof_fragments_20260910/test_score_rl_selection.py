"""CPU checks for selection release gates and the actual-displacement statistic."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import common
import learning as L
import reference_scoring as R
import score_rl_selection as S
from nd_environment import FORMAT


class ConstantLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.8]))
        self.config = SimpleNamespace(n_layers=1, d_model=3)

    def forward(self, inputs):
        return self.theta.expand(*inputs.shape, -1)


class SelectionScorerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def frozen_inputs(self):
        root = common.ROOT
        pipeline = json.loads((root / "configs/pipeline_rl_calibration_v1.json").read_text())
        design = json.loads((root / "configs/rl_calibration_design_v1.json").read_text())
        return pipeline, design

    def test_frozen_endpoint_structure_and_tampering(self):
        pipeline, design = self.frozen_inputs()
        endpoints = S.validate_pipeline(pipeline, design)
        self.assertEqual(len(endpoints), 48)
        self.assertEqual({x["horizon"] for x in endpoints}, {"short", "long"})
        for change in ("path", "duplicate", "ceiling", "start", "seed", "status", "program", "bank", "milestone"):
            altered = copy.deepcopy(pipeline)
            if change == "path":
                altered["checkpoints_for_scoring"][0] = "/tmp/wrong.pt"
            elif change == "duplicate":
                altered["arm_order"][1] = altered["arm_order"][0]
            elif change == "ceiling":
                altered["short_training_proxy_ceiling"] += 1
            elif change == "start":
                cmd = altered["commands"][0]
                cmd[cmd.index("--start") + 1] = "reset"
            elif change == "seed":
                cmd = altered["commands"][0]
                cmd[cmd.index("--seed") + 1] = "5"
            elif change == "program":
                altered["commands"][0][1] = "/tmp/different_program.py"
            elif change == "bank":
                cmd = altered["commands"][0]
                cmd[cmd.index("--data") + 1] = "/tmp/other_family.jsonl"
            elif change == "milestone":
                cmd = altered["commands"][0]
                cmd[cmd.index("--flop-milestones") + 1] = "1"
            else:
                altered["status"] = "draft"
            with self.subTest(change=change), self.assertRaises(ValueError):
                S.validate_pipeline(altered, design)

    def test_selection_identity_gate_without_opening_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.jsonl"
            # Intentionally invalid JSON: release validation must use metadata only.
            path.write_text("not a proof payload")
            manifest = path.parent / "manifest.json"
            data_hash = common.sha(path)
            common.write_json(manifest, {"status": "released_complete_primary",
                                       "population_sha256": {"selection": data_hash,
                                                             "confirmation": "different"}})
            self.assertEqual(S.selection_metadata(path, data_hash, [str(manifest)]), manifest)
            for wrong_hash in ("different", "final_confirmation", ""):
                with self.subTest(digest=wrong_hash), self.assertRaises(ValueError):
                    S.selection_metadata(path, wrong_hash, [str(manifest)])
            with self.assertRaises(ValueError):
                S.selection_metadata(path, data_hash, [])
            common.write_json(manifest, {"status": "incomplete",
                                       "population_sha256": {"selection": data_hash}})
            with self.assertRaises(ValueError):
                S.selection_metadata(path, data_hash, [str(manifest)])

    def test_short_and_long_checkpoint_metadata(self):
        endpoint = {"horizon": "short", "training_proxy_ceiling": 100}
        producer = {"training_steps": 7, "actual_training_flop_proxy": 105}
        metadata = {"format": FORMAT, "step": 7, "requested_flop_proxy": 100,
                    "actual_flop_proxy": 105}
        S.validate_checkpoint_metadata(endpoint, metadata, producer, "base")
        for key in metadata:
            altered = dict(metadata)
            altered[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                S.validate_checkpoint_metadata(endpoint, altered, producer, "base")
        endpoint["horizon"] = "long"
        final = {"format": FORMAT, "source_checkpoint_sha256": "base", "stage": "rl"}
        S.validate_checkpoint_metadata(endpoint, final, producer, "base")
        with self.assertRaises(ValueError):
            S.validate_checkpoint_metadata(endpoint, final, producer, "other base")
        final["stage"] = "intervene"
        with self.assertRaises(ValueError):
            S.validate_checkpoint_metadata(endpoint, final, producer, "base")

    def test_producer_hashes_and_first_budget_crossing(self):
        pipeline, _ = self.frozen_inputs()
        command = copy.deepcopy(pipeline["commands"][0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flag in ("checkpoint", "data", "retention"):
                path = root / (flag + ".bin")
                path.write_text(flag)
                command[command.index("--" + flag) + 1] = str(path)
            command[command.index("--output") + 1] = str(root)
            command[command.index("--max-flop-proxy") + 1] = "200"
            milestone_index = command.index("--flop-milestones")
            command[milestone_index + 1:milestone_index + 3] = ["100", "200"]
            args = {command[i][2:].replace("-", "_"): command[i + 1]
                    for i in range(len(command) - 1) if command[i].startswith("--")}
            args["flop_milestones"] = [float(x) for x in S.values(command, "flop-milestones")]
            code_path = root / "producer.py"
            code_path.write_text("# frozen producer\n")
            manifest = {"args": args, "input_sha256": {
                key: common.sha(args[key]) for key in ("checkpoint", "data", "retention")},
                "code_sha256": {str(code_path): common.sha(code_path)}, "dependency_sha256": {}}
            common.write_json(root / "manifest.json", manifest)
            (root / "final.pt").write_text("completed checkpoint")
            updates = [{"step": i + 1, "cumulative_flop_proxy": total,
                        "step_flop_proxy": cost,
                        "retention_examples": 16, "retention_response_tokens": 100,
                        "mixed_groups": 2} for i, (total, cost) in enumerate(((90, 90), (110, 20), (205, 95)))]
            common.write_jsonl(root / "updates.jsonl", updates)
            result = {"steps": 3, "worst_case_one_step_flop_proxy": 100,
                      "final_sha256": common.sha(root / "final.pt"),
                      "max_flop_proxy": 200, "flop_proxy_overshoot": 5,
                      "total_flop_proxy": 205, "stop_reason": "flop_proxy_ceiling",
                      "ledger": {"training": {"flop_proxy": 205}}}
            common.write_json(root / "result.json", result)
            common.write_json(root / "completed.json", {
                "ok": True, "result_sha256": common.sha(root / "result.json"),
                "final_sha256": result["final_sha256"]})
            endpoint = {"producer_directory": str(root), "command": command,
                        "horizon": "short", "training_proxy_ceiling": 100}
            base_hash = manifest["input_sha256"]["checkpoint"]
            bound = S.bind_producer(endpoint, base_hash)
            self.assertEqual(bound["training_steps"], 2)
            self.assertEqual(bound["retention_examples_through_checkpoint"], 32)
            self.assertEqual(bound["training_flop_proxy_overshoot"], 10)
            endpoint.update(horizon="long", training_proxy_ceiling=200)
            self.assertEqual(S.bind_producer(endpoint, base_hash)["training_steps"], 3)
            endpoint["training_proxy_ceiling"] = 100
            with self.assertRaises(ValueError):
                S.bind_producer(endpoint, base_hash)
            endpoint["training_proxy_ceiling"] = 200
            for key, changed in (("cumulative_flop_proxy", 111), ("step_flop_proxy", 0),
                                 ("step_flop_proxy", 101)):
                altered = copy.deepcopy(updates)
                altered[1][key] = changed
                common.write_jsonl(root / "updates.jsonl", altered)
                with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                    S.bind_producer(endpoint, base_hash)
                common.write_jsonl(root / "updates.jsonl", updates)
            for target in (root / "data.bin", code_path, root / "final.pt"):
                original = target.read_bytes()
                target.write_text("changed identity")
                with self.subTest(path=target.name), self.assertRaises(ValueError):
                    S.bind_producer(endpoint, base_hash)
                target.write_bytes(original)

    def test_weighted_query_gradient_and_actual_displacement_analytic(self):
        model = ConstantLogits()
        tok = SimpleNamespace(pad_id=0)
        rows = [dict(source_id=str(i), family="toy", reset_index=0,
                     source_lines=length, prompt_ids=prompt, response_ids=response)
                for i, (length, prompt, response) in enumerate([
                    (3, [0, 0, 0], [1, 2]),
                    (4, [0], [2]),
                    (8, [0, 0], [1, 1])])]
        original = model.theta.detach().clone()
        ledger = {}
        gradient = L.query_gradient(model, tok, rows, batch_size=2, ledger=ledger)
        # Short references each weight 1/4; the single long reference weights 1/2.
        # Thus expected action-token counts are [0, 1.25, .5], total 1.75.
        expected = (1.75 * original.softmax(-1) - torch.tensor([0., 1.25, .5])) / math.log(2)
        torch.testing.assert_close(gradient["theta"], expected, atol=2e-7, rtol=2e-6)
        self.assertTrue(torch.equal(original, model.theta.detach()))
        self.assertIsNone(model.theta.grad)
        self.assertEqual(ledger["query_gradient"]["action_tokens"], 5)
        self.assertGreater(L.total_flop_proxy(ledger), 0)
        self.assertEqual(L.displacement_dot(model, model, gradient), 0)

        arm = ConstantLogits()
        delta = torch.tensor([.01, .02, -.03])
        with torch.no_grad():
            arm.theta.add_(delta)
        self.assertAlmostEqual(L.displacement_dot(arm, model, gradient),
                               float((expected * delta).sum()), places=6)
        score_ledger = {}
        scores = R.score(model, tok, rows, batch_size=2, ledger=score_ledger,
                         precision="fp32", score_temperature=1.)
        logp = original.log_softmax(-1)
        for row, score in zip(rows, scores):
            manual = -sum(float(logp[i]) for i in row["response_ids"]) / math.log(2)
            self.assertAlmostEqual(score["bits"], manual, places=5)
        self.assertEqual(score_ledger["teacher_scoring"]["action_tokens"], 5)
        # Proves context/padding tokens do not enter either objective.
        altered = [{**row, "prompt_ids": [0] * 7} for row in rows]
        second = L.query_gradient(model, tok, altered, batch_size=3)
        torch.testing.assert_close(second["theta"], gradient["theta"])


if __name__ == "__main__":
    unittest.main()
