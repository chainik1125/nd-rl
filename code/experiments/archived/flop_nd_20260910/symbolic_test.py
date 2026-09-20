"""Semantic checks for restricted environment, certificates, and masked policy."""
import random
import unittest
import torch

from symbolic_nd import (Environment, Policy, Action, make_task, exact_random_probability,
                         batch_transitions, FEATURE_NAMES, STATE_NAMES)
from nd_environment import strict_complete


class SymbolicTests(unittest.TestCase):
    def test_reference_suffixes_all_cuts_and_families(self):
        rng = random.Random(21)
        for family in ("spine", "compound_decoy"):
            for depth in (1, 5, 15, 63):
                row = make_task(rng, depth, family)
                for cut in set((0, depth // 2, depth - 1)):
                    env = Environment(row, cut=cut)
                    for side in row["directions"][cut:]:
                        desired = env.formulas[-1][side + 1]
                        reward, done = env.step(next(a for a in env.actions() if a.result == desired))
                    self.assertEqual(reward, 1.)
                    self.assertTrue(done)
                    verdict = strict_complete(row["prompt"], env.body())
                    self.assertTrue(verdict["ok"])
                    self.assertEqual(verdict["n_lines"], depth + 1)
                    self.assertEqual(env.body(), env.pruned_body())

    def test_pruning_removes_real_distractor_work(self):
        row = make_task(random.Random(15), 5)
        env = Environment(row, budget=8)
        # Extract a side atom first, then find the unique goal-path continuation.
        wrong_side = 1 - row["directions"][0]
        env.step(next(a for a in env.actions() if a.source == 0 and a.side == wrong_side))
        current = 0
        for side in row["directions"]:
            desired = env.formulas[current][side + 1]
            a = next(a for a in env.actions() if a.result == desired)
            env.step(a)
            current = len(env.formulas) - 1
        self.assertEqual(len(env.formulas), 7)
        verdict = strict_complete(row["prompt"], env.pruned_body())
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["n_lines"], 6)

    def test_minimum_budget_random_probability_exhaustive(self):
        row = make_task(random.Random(94), 3)
        def recurse(trace):
            env = Environment(row)
            for choice in trace:
                reward, done = env.step(env.actions()[choice])
            if trace and done:
                return reward
            aa = env.actions()
            return sum(recurse(trace + [i]) for i in range(len(aa))) / len(aa)
        self.assertAlmostEqual(recurse([]), exact_random_probability(row))

    def test_actions_are_legal_unique_and_not_previously_known(self):
        row = make_task(random.Random(4), 5, "compound_decoy")
        env = Environment(row, budget=99)
        rng = random.Random(32)
        for _ in range(15):
            aa = env.actions()
            self.assertEqual(len({a.result for a in aa}), len(aa))
            for a in aa:
                self.assertNotIn(a.result, env.formulas)
                self.assertEqual(env.formulas[a.source][a.side + 1], a.result)
            _, done = env.step(rng.choice(aa))
            if done:
                break

    def test_masked_scoring_matches_unpadded_distribution(self):
        torch.manual_seed(4)
        model = Policy()
        transitions = []
        for depth in (1, 5):
            env = Environment(make_task(random.Random(depth), depth))
            if depth == 5:
                env.step(env.actions()[0])
            af, sf = env.features(env.actions())
            transitions.append({"actions": af, "state": sf})
        aa, ss, mm = batch_transitions(transitions)
        logits, values = model(aa, ss, mm)
        probs = logits.softmax(-1)
        self.assertTrue(torch.equal(probs[~mm], torch.zeros_like(probs[~mm])))
        for i, t in enumerate(transitions):
            one_logits, value = model(torch.from_numpy(t["actions"]), torch.from_numpy(t["state"]))
            torch.testing.assert_close(probs[i, :len(one_logits)], one_logits.softmax(-1))
            torch.testing.assert_close(values[i], value)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
