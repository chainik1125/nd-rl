"""Synthetic-only regression tests; never reads research evaluation inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from r1_cot import analyze


def verifier(prompt, body):
    return (
        body in ("valid", "compressed"),
        "ok" if body in ("valid", "compressed") else "invalid",
        3 if body == "valid" else 1,
    )


def compressor(prompt, body):
    return {"raw_valid": True, "compressed_proof": "compressed", "compressed_length": 1}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.labels = analyze.load_labels(
            [
                {"name": "a", "prompt": "test a", "reference_lines": 6},
                {"name": "b", "prompt": "test b", "reference_lines": 9},
            ]
        )
        self.records = [
            {
                "name": name,
                "prompt": f"test {name}",
                "attempt": i,
                "body": "valid" if (name, i) == ("b", 1) else "invalid",
                "generated_tokens": cost,
                "scratch_tokens": 2,
                "final_tokens": cost - 2,
            }
            for name in ("a", "b")
            for i, cost in enumerate((10, 20))
        ]

    def test_replay_and_denominators(self):
        grouped = analyze.replay(self.records, self.labels, verifier, compressor, 2)
        summary = analyze.summarize(grouped, self.labels, [1, 2], [10, 29, 30], 6)
        self.assertEqual(summary["all_attempts"]["solved"], 1)
        self.assertEqual(summary["all_attempts"]["total"], 2)
        self.assertEqual(summary["all_attempts"]["summed_consumed_generated_tokens"], 60)
        self.assertEqual([x["solved"] for x in summary["pass_at_k_prefix"]], [0, 1])
        self.assertEqual([x["solved"] for x in summary["token_budget_prefix"]], [0, 0, 1])
        self.assertEqual(summary["per_target"]["b"]["shortest_compressed_lines"], 1)
        self.assertEqual(summary["beyond_initial_cap"]["solved"], 1)
        self.assertEqual(summary["reference_length_bins"][0]["solved"], 0)

    def test_reject_partial_duplicate_prompt_and_reward_mismatch(self):
        for records in (
            self.records[:-1],
            self.records + [self.records[-1]],
            [dict(self.records[0], prompt="other")] + self.records[1:],
            [dict(self.records[0], reward=1)] + self.records[1:],
        ):
            with self.assertRaises(ValueError):
                analyze.replay(records, self.labels, verifier, compressor, 2)

    def test_strict_reverify_compression(self):
        def unsafe_compressor(prompt, body):
            return {
                "raw_valid": True,
                "compressed_proof": "future-reference",
                "compressed_length": 1,
            }

        with self.assertRaisesRegex(ValueError, "STRICT"):
            analyze.replay(self.records, self.labels, verifier, unsafe_compressor, 2)

    def test_raw_response_reconstruction(self):
        records = [
            dict(r, response_ids=[1] * r["generated_tokens"], reward=0) for r in self.records
        ]

        def tampered(prompt, ids):
            return {
                "body": "injected proof",
                "reward": 0,
                "scratch_tokens": 2,
                "final_tokens": len(ids) - 2,
            }

        with self.assertRaisesRegex(ValueError, "reconstruction"):
            analyze.replay(records, self.labels, verifier, compressor, 2, tampered)

    def test_never_skip_expensive_attempt(self):
        samples = [
            dict(generated_tokens=100, scratch_tokens=0, final_tokens=100, solved=False),
            dict(
                generated_tokens=2,
                scratch_tokens=0,
                final_tokens=2,
                solved=True,
                raw_lines=1,
                compressed_lines=1,
            ),
        ]
        self.assertFalse(analyze.outcome(samples, budget=2)["solved"])
        self.assertEqual(analyze.outcome(samples, budget=2)["generated_tokens"], 0)

    def test_paired_bootstrap_deterministic(self):
        a, b = {"a": False, "b": False}, {"a": False, "b": True}
        result = analyze.paired_bootstrap(a, b, resamples=100)
        self.assertEqual(result, analyze.paired_bootstrap(a, b, resamples=100))
        self.assertEqual(result["difference_b_minus_a"], 0.5)
        self.assertEqual(result["b_only"], 1)

    def test_tokenizer_selected_from_declaration(self):
        base = type("Base", (), {"VERSION": "base40"})
        scratch = type("Scratch", (), {"VERSION": "scratch43"})
        repair = type("Repair", (), {"VERSION": "repair46"})
        runner = SimpleNamespace(
            ProofTokenizer=base, ScratchTokenizer=scratch, ScratchRepairTokenizer=repair
        )
        self.assertIsInstance(analyze.select_tokenizer(runner, "repair46"), repair)
        self.assertIsInstance(analyze.select_tokenizer(runner), scratch)
        with self.assertRaisesRegex(ValueError, "Unknown release-declared"):
            analyze.select_tokenizer(runner, "invented")

    def test_release_before_label_access_and_hash_mutation(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            analyze.check_release(
                {"status": "complete", "role": "textbook_final"}, "/missing", ["/missing2"]
            )
        with tempfile.TemporaryDirectory() as folder:
            labels, preds = Path(folder) / "labels.jsonl", Path(folder) / "attempts.jsonl"
            labels.write_text(json.dumps({"example": 1}))
            preds.write_text(json.dumps({"example": 2}))
            release = {
                "status": "complete",
                "role": "synthetic",
                "evaluation_k": 2,
                "labels_sha256": analyze.sha(labels),
                "prediction_sha256": {str(preds.resolve()): analyze.sha(preds)},
            }
            analyze.check_release(release, labels, [preds])
            preds.write_text("modified")
            with self.assertRaisesRegex(ValueError, "differ"):
                analyze.check_release(release, labels, [preds])


if __name__ == "__main__":
    unittest.main()
