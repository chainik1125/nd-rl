"""Distinguish all-success/failure groups from informative mixed outcomes."""
import unittest
from training_diagnostics import summarize_groups


class Diagnostics(unittest.TestCase):
    def test_equal_accuracy_can_hide_very_different_reward_information(self):
        def group(index, outcomes):
            return [{"group": index, "attempt": i, "prompt": "invented", "reward": r,
                     "response_ids": [1]*10, "advantage": r - outcomes[1-i]}
                    for i,r in enumerate(outcomes)]
        uninformative = summarize_groups(group(0,[0,0])+group(1,[1,1]), 2)
        informative = summarize_groups(group(0,[0,1])+group(1,[1,0]), 2)
        self.assertEqual(uninformative["correct_responses"], informative["correct_responses"])
        self.assertEqual(uninformative["mixed_groups"], 0)
        self.assertEqual(uninformative["nonzero_advantage_tokens"], 0)
        self.assertEqual(informative["mixed_groups"], 2)
        self.assertEqual(informative["nonzero_advantage_tokens"], 40)
        self.assertEqual(informative["empirical_reward_entropy_bits_sum"], 2)

    def test_incomplete_group_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            summarize_groups([{"group":0,"attempt":0}], 8)


if __name__ == "__main__":
    unittest.main()
