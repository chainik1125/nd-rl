"""CPU schema and scoring checks; no model is loaded or evaluated."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import common
from nd_environment import Tokenizer, strict_complete
from structured_generate import make_task, FAMILIES, LENGTHS
from structured_analyze import derive_restricted_bound, recheck_attempts, aggregate


class StructuredTests(unittest.TestCase):
    def test_all_schemas_and_long_witnesses(self):
        rng = random.Random(8107)
        for family in FAMILIES:
            for length in LENGTHS:
                row = make_task(rng, length, family)
                self.assertTrue(strict_complete(row["prompt"], row["body"])["ok"])
                self.assertEqual(derive_restricted_bound(row["prompt"], family)["lines"], length)

    def test_hidden_target_in_decoy_is_rejected(self):
        row = make_task(random.Random(4), 7, "projection_atomic")
        target = row["target_atom"]
        other = next(x for x in "PQRS" if x != target and x in row["prompt"])
        bad = row["prompt"].replace(other, target)
        with self.assertRaises(ValueError):
            derive_restricted_bound(bad, "projection_atomic")

    def test_scoring_rechecks_certificate_and_detects_false_reward(self):
        row = make_task(random.Random(7), 13, "projection_atomic")
        row.update(source_id="reference", reference_tokens=415)
        references = {row["source_id"]: row}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            sample = {"prompt": row["prompt"], "source_id": "reference", "body": row["body"],
                      "reward": 1, "attempt": 0, "reset_index": 0}
            common.write_jsonl(path, [sample])
            scored = recheck_attempts(path, references)
            self.assertEqual(scored[0]["pruned_lines"], 13)
            self.assertEqual(aggregate(scored, references)["all"]["successful_pruned_above6"], 1)
            sample["body"] = "QED"
            common.write_jsonl(path, [sample])
            with self.assertRaises(ValueError):
                recheck_attempts(path, references)


if __name__ == "__main__":
    unittest.main(verbosity=2)
