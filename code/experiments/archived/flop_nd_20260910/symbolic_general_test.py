"""Meaningful scope/case, feature, masking, and credit-assignment tests."""
import unittest
import torch
import numpy as np

import common
from nd_environment import independent_replay, strict_complete
from symbolic_general_env import Environment, Action
from symbolic_general_model import features, formula_features, Policy, pack
from symbolic_general_train import advantages

P, Q, R = [("atom", x) for x in "PQR"]


class GeneralTests(unittest.TestCase):
    def test_case_split_closes_sibling_boxes_and_hides_their_lines(self):
        prompt = "THM ( P v Q ) , ( P > R ) , ( Q > R ) SEQ R PRF"
        env = Environment(prompt)
        trace = [Action("AS", P, (), 1), Action("IMPE", R, (2, 4), 1),
                 Action("AS", Q, (), 1), Action("IMPE", R, (3, 6), 1)]
        for action in trace:
            env.step(action)
        self.assertNotIn(Action("R", P, (4,), 1), env.actions())
        self.assertNotIn(Action("IMPE", R, (2, 4), 1), env.actions())
        self.assertNotIn(Action("R", R, (5,), 0), env.actions())
        action = Action("ORE", R, (1, 4, 5, 6, 7), 0)
        self.assertIn(action, env.actions())
        reward, done = env.step(action)
        self.assertEqual((reward, done), (1., True))
        self.assertTrue(strict_complete(prompt, env.body())["ok"])

    def test_nested_implication_discharge_requires_correct_box_endpoint(self):
        env = Environment("THM SEQ ( P > ( Q > P ) ) PRF")
        env.step(Action("AS", P, (), 1))
        env.step(Action("AS", Q, (), 2))
        self.assertNotIn(Action("IMPI", ("imp", P, Q), (1, 2), 0), env.actions())
        env.step(Action("R", P, (1,), 2))
        env.step(Action("IMPI", ("imp", Q, P), (2, 3), 1))
        reward, done = env.step(Action("IMPI", ("imp", P, ("imp", Q, P)), (1, 4), 0))
        self.assertEqual((reward, done), (1., True))
        self.assertTrue(strict_complete(env.prompt, env.body())["ok"])

    def test_pool_is_theorem_derived_and_closure_is_one_level_only(self):
        env = Environment("THM P SEQ ( P v Q ) PRF")
        original = env.base
        self.assertIn(Action("AS", Q, (), 1), env.actions())
        missing = ("and", R, R)
        self.assertNotIn(Action("AS", missing, (), 1), env.actions())
        self.assertTrue(env.formula_allowed(("imp", P, Q)))
        self.assertFalse(env.formula_allowed(("imp", ("imp", P, Q), Q)))
        env.step(Action("AS", Q, (), 1))
        self.assertEqual(env.base, original)

    def test_pruning_preserves_scope_after_unused_nested_box(self):
        env = Environment("THM P SEQ ( Q > P ) PRF")
        env.step(Action("AS", Q, (), 1))
        env.step(Action("AS", P, (), 2))
        env.step(Action("R", Q, (2,), 1))
        env.step(Action("R", P, (1,), 1))
        env.step(Action("IMPI", ("imp", Q, P), (2, 5), 0))
        verdict = strict_complete(env.prompt, env.pruned_body())
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["n_lines"], 4)

    def test_order_aware_features_change_on_argument_swap(self):
        a, b = ("imp", P, Q), ("imp", Q, P)
        self.assertFalse(np.array_equal(formula_features(a), formula_features(b)))
        # The same atoms/operator histogram is insufficient: path sketch carries order.
        np.testing.assert_equal(formula_features(a)[18:30], formula_features(b)[18:30])

    def test_padded_probabilities_values_and_gradients(self):
        torch.manual_seed(4)
        model = Policy(32, 2)
        envs = [Environment("THM P SEQ ( P v Q ) PRF"), Environment("THM SEQ ( P > P ) PRF")]
        envs[0].step(Action("AS", Q, (), 1))
        transitions = []
        for env in envs:
            af, sf = features(env, env.actions())
            transitions.append({"actions": af, "state": sf})
        aa, ss, mm = pack(transitions)
        logits, values = model(aa, ss, mm)
        prob = logits.softmax(-1)
        self.assertTrue(torch.equal(prob[~mm], torch.zeros_like(prob[~mm])))
        for i, t in enumerate(transitions):
            one, value = model(torch.from_numpy(t["actions"]), torch.from_numpy(t["state"]))
            torch.testing.assert_close(prob[i, :len(one)], one.softmax(-1))
            torch.testing.assert_close(values[i], value)
        loss = -torch.log(prob[:, 0]).mean() + values.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_discounted_gae_terminal_and_truncation_are_distinct(self):
        terminal = [{"value": .2, "reward": 0., "done": False},
                    {"value": .4, "reward": 1., "done": True}]
        advantages(terminal, 9., .95, .95)
        self.assertAlmostEqual(terminal[1]["advantage"], .6)
        self.assertAlmostEqual(terminal[0]["advantage"], .95 * .4 - .2 + .95 * .95 * .6)
        truncation = [{"value": .2, "reward": 0., "done": False}]
        advantages(truncation, .7, .95, .95)
        self.assertAlmostEqual(truncation[0]["return"], .95 * .7)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
