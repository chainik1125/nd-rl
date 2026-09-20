"""Finite-distribution and provenance tests; no GPU or model optimization."""
import argparse
import copy
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import common
import learning as L
import nd_environment as E
import projected_loo as P
from all_cut_pilot import contract


class BinaryPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([.2, -.3]))
        self.config = SimpleNamespace(n_layers=1, d_model=2)

    def forward(self, inputs):
        return self.theta.expand(*inputs.shape, -1)


def toy_rows(k, *, repeated_success=False):
    return [{"source_id": "one state", "prompt_ids": [0] * (1 + i % 3),
             "response_ids": ([1, 1] if repeated_success else [1]) if i < k else [0],
             "reward": int(i < k), "prefix_ids": [], "reset_index": 0}
            for i in range(8)]


class ProjectedLOOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_finite_bernoulli_distribution_sign_group_mean_and_512(self):
        model = BinaryPolicy()
        original = model.theta.detach().clone()
        gradient = {"theta": torch.tensor([-.3, .7])}
        tok = SimpleNamespace(pad_id=0)
        p = float((original / .8).softmax(-1)[1])
        directional_difference = float((gradient["theta"][1] - gradient["theta"][0]) / .8)
        expected_covariance = p * (1 - p) * directional_difference
        expectation = 0.
        for k in range(9):
            ledger = {}
            result = P.project_group(model, tok, toy_rows(k), gradient, ledger=ledger)
            # Exact sample covariance for k ones and 8-k zeros, denominator 7.
            manual = k * (8 - k) * directional_difference / (8 * 7)
            self.assertAlmostEqual(result["D"], manual, places=6)
            self.assertAlmostEqual(result["D_div_512"], manual / 512, places=9)
            self.assertEqual(result["backward_executed"], 0 < k < 8)
            if 0 < k < 8:
                self.assertEqual(ledger["raw_loo_teacher_backward"]["action_tokens"], 8)
                self.assertGreater(L.total_flop_proxy(ledger), 0)
            else:
                self.assertEqual(L.total_flop_proxy(ledger), 0)
                self.assertEqual(result["D"], 0.)
            probability = math.comb(8, k) * p ** k * (1 - p) ** (8 - k)
            expectation += probability * result["D"]
        self.assertAlmostEqual(expectation, expected_covariance, places=6)
        self.assertTrue(torch.equal(original, model.theta.detach()))
        self.assertIsNone(model.theta.grad)
        # Higher D in the family means worse target loss under reward ascent.
        banks = {"toy_family": ["family"] * 64, "toy_control": ["control"] * 64}
        contrast = P.bank_contrasts({"family": 2., "control": 1.}, banks)["toy"]
        self.assertEqual(contrast["predicted_target_bits_benefit_per_unit_raw_ascent_step"], -1 / 512)

    def test_whole_response_and_action_padding_masks(self):
        model = BinaryPolicy()
        tok = SimpleNamespace(pad_id=0)
        gradient = {"theta": torch.tensor([-.3, .7])}
        rows = toy_rows(3, repeated_success=True)
        ledger = {}
        result = P.project_group(model, tok, rows, gradient, ledger=ledger)
        probability = (model.theta.detach() / .8).softmax(-1)
        score_grad_0 = (torch.tensor([1., 0.]) - probability) / .8
        score_grad_1 = (torch.tensor([0., 1.]) - probability) / .8
        # Successful responses have two actions, failed ones have one sampled
        # PAD-ID action. Neither response-length normalization nor PAD-ID masking is valid.
        raw = (3 * (5 / 7) * 2 * score_grad_1 + 5 * (-3 / 7) * score_grad_0) / 8
        manual = float((gradient["theta"] * raw).sum())
        self.assertAlmostEqual(result["D"], manual, places=6)
        self.assertEqual(ledger["raw_loo_teacher_backward"]["action_tokens"], 11)
        altered = [{**row, "prompt_ids": [0] * 15} for row in rows]
        repeated = P.project_group(model, tok, altered, gradient, ledger={})
        self.assertAlmostEqual(repeated["D"], result["D"], places=6)
        wrong = copy.deepcopy(rows)
        wrong[0]["prefix_ids"] = [1]
        with self.assertRaises(ValueError):
            P.project_group(model, tok, wrong, gradient, ledger={})

    def test_shared_source_resampling_cancellation_and_pair_mean(self):
        ids = [str(i) for i in range(64)]
        banks = {"same_family": ids, "same_control": list(reversed(ids))}
        first = {sid: float(i) for i, sid in enumerate(ids)}
        second = {sid: float(i * i) for i, sid in enumerate(ids)}
        result = P.draw_noise_summary(first, second, banks, repeats=64, seed=1)
        family = result["families"]["same"]
        self.assertEqual(family["mean_two_draw_benefit"], 0.)
        self.assertEqual(family["two_draw_variance_estimate"], 0.)
        self.assertEqual(family["descriptive_resampling_percentiles_2_5_and_97_5"], [0., 0.])
        self.assertIn("not calibrated", result["scope"])
        self.assertIn("zero/zero", result["limitations"])
        banks = {"different_family": ids, "different_control": ["extra"] * 64}
        first["extra"] = 1.
        second["extra"] = 3.
        result = P.draw_noise_summary(first, second, banks, repeats=0)
        expected = (2 - sum((first[sid] + second[sid]) / 2 for sid in ids) / 64) / 512
        self.assertAlmostEqual(result["families"]["different"]["mean_two_draw_benefit"], expected)

    def test_exact_real_state_action_termination_and_reward_gates(self):
        record = common.read_jsonl(common.ROOT / "data/released_v1/candidate.jsonl")[0]
        tok = E.Tokenizer()
        expected = E.episode(record, 0, tok)
        response = expected["response_ids"]
        row = {**expected, "response_ids": response, "rollout_logprobs": [-1.] * len(response),
               "reward": 1, "attempt_index": 0, "effective_max_new_tokens": 512,
               "finish_reason": "qed", "condition": "full", "sampling_seed": 55}
        self.assertEqual(P.validate_attempt(row, expected, tok, kind="fresh_audit", seed=55)["reward"], 1)
        for change in ("state", "seed", "condition", "budget", "reward", "logp", "termination"):
            altered = copy.deepcopy(row)
            if change == "state":
                altered["prompt_ids"] += [0]
            elif change == "seed":
                altered["sampling_seed"] += 1
            elif change == "condition":
                altered["condition"] = "chosen"
            elif change == "budget":
                altered["effective_max_new_tokens"] = 511
            elif change == "reward":
                altered["reward"] = 0
            elif change == "logp":
                altered["rollout_logprobs"][-1] = float("nan")
            else:
                altered["response_ids"] += [0]
                altered["rollout_logprobs"] += [-1.]
            with self.subTest(change=change), self.assertRaises(ValueError):
                P.validate_attempt(altered, expected, tok, kind="fresh_audit", seed=55)

    def test_real_original_metadata_and_temperature_identity(self):
        root = common.ROOT
        current = contract(argparse.Namespace(temperature=.8, max_new_tokens=512, precision="fp32"),
                           common.sha(root / "results/adapt_v1/final.pt"))
        candidate_hash = common.sha(root / "data/released_v1/candidate.jsonl")
        binding = P.bind_draw_directory(root / "results/adapt_candidate_full_v1", "original", current, candidate_hash)
        self.assertEqual(binding["sampling_seed"], 20260910)
        for change in ("temperature", "checkpoint_sha256", "max_new_tokens"):
            altered = dict(current)
            altered[change] = .7 if change == "temperature" else "wrong"
            with self.subTest(change=change), self.assertRaises(ValueError):
                P.bind_draw_directory(root / "results/adapt_candidate_full_v1", "original", altered, candidate_hash)

    def test_fresh_audit_component_metadata_and_seed_binding(self):
        root = common.ROOT
        original = json.loads((root / "results/adapt_candidate_full_v1/manifest.json").read_text())
        current = contract(argparse.Namespace(temperature=.8, max_new_tokens=512, precision="fp32"),
                           common.sha(root / "results/adapt_v1/final.pt"))
        candidate_hash = common.sha(root / "data/released_v1/candidate.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            map_path = path / "initial_cuts.json"
            common.write_json(map_path, {"test": "binding metadata only"})
            manifest = {**original, "command": "all_cut_pilot_audit", "contract": current,
                        "args": {**original["args"], "command": "audit", "seed": 66}}
            common.write_json(path / "manifest.json", manifest)
            common.write_json(path / "map_binding.json", {"path": str(map_path), "sha256": common.sha(map_path),
                                                          "audit_seeds": [66 + i * 1000003 for i in range(3)]})
            (path / "full_attempts.jsonl").write_text("unparsed payload")
            result = {"independent_of_fit_draws": True, "policy_updates": 0,
                      "initial_cuts_sha256": common.sha(map_path),
                      "conditions": {"full": {"n_sources": 768, "attempts": 6144, "seed": 1000069,
                                               "successes": 0, "mixed_groups": 0, "total_flop_proxy": 10}},
                      "ledger": {"full": {"generation": {"flop_proxy": 10}}}}
            common.write_json(path / "result.json", result)
            common.write_json(path / "completed.json", {"ok": True, "result_sha256": common.sha(path / "result.json")})
            binding = P.bind_draw_directory(path, "fresh_audit", current, candidate_hash)
            self.assertEqual(binding["sampling_seed"], 1000069)
            self.assertEqual(binding["reused_previously_charged_flop_proxy"], 10)
            # Same reset-audit producer with the wrong component seed must fail.
            result["conditions"]["full"]["seed"] = 66
            common.write_json(path / "result.json", result)
            common.write_json(path / "completed.json", {"ok": True, "result_sha256": common.sha(path / "result.json")})
            with self.assertRaises(ValueError):
                P.bind_draw_directory(path, "fresh_audit", current, candidate_hash)


if __name__ == "__main__":
    unittest.main()
