"""Independent hand-calculated and gradient checks for ppo_core (CPU only)."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

import common
import learning as L
import ppo_core as P
from review_semantics import tiny, record
from nd_environment import episode

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = {}


def train_rows(model, critic, tok):
    eps = [episode(record(0), 0, tok), episode(record(1), 3, tok),
           episode(record(2), 5, tok), episode(record(0), 2, tok)]
    rows = L.sample(model, tok, eps, seed=6553, temperature=.8, max_new_tokens=15, batch_size=4)
    audit = P.audit_and_attach_old_values(model, critic, tok, rows, temperature=.8, tolerance=1e-5)
    for i, r in enumerate(rows):
        # Artificial rewards are deliberate: these test optimizer mathematics,
        # not verifier outcomes. Label them completed for attach_advantages.
        r["reward"] = int(i % 2 == 0)
        r["finish_reason"] = "qed"
    P.attach_advantages(rows, objective="ppo", group_size=2, gamma=1, gae_lambda=1)
    return rows, audit


class PPOReview(unittest.TestCase):
    def test_gae_by_hand_nontrivial_gamma_lambda(self):
        advantages, returns = P.generalized_advantage_estimate([0, 0, 1], [.2, .4, .6], gamma=.9, gae_lambda=.8)
        expected_a = torch.tensor([.46816, .428, .4])
        expected_r = torch.tensor([.66816, .828, 1.])
        self.assertTrue(torch.allclose(advantages, expected_a, atol=1e-6))
        self.assertTrue(torch.allclose(returns, expected_r, atol=1e-6))
        OBSERVATIONS["hand_gae"] = {"advantages": advantages.tolist(), "returns": returns.tolist()}

    def test_undiscounted_returns_and_nonterminal_bootstrap(self):
        for n in (1, 2, 9):
            values = torch.arange(n, dtype=torch.float32) * .1
            adv, ret = P.generalized_advantage_estimate([0.] * (n - 1) + [1.], values, gamma=1, gae_lambda=1)
            self.assertTrue(torch.allclose(ret, torch.ones(n), atol=1e-6))
            self.assertTrue(torch.allclose(adv, 1 - values, atol=1e-6))
        adv, ret = P.generalized_advantage_estimate([.3, .1], [.2, .4], gamma=.9, gae_lambda=1,
                                                   terminated=False, bootstrap_value=.5)
        self.assertTrue(torch.allclose(adv, torch.tensor([.595, .15]), atol=1e-6))
        self.assertTrue(torch.allclose(ret, torch.tensor([.795, .55]), atol=1e-6))
        with self.assertRaises(ValueError):
            P.generalized_advantage_estimate([0], [0], terminated=True, bootstrap_value=.1)

    def test_clipping_signs_and_offmask_gradients(self):
        ratios = torch.tensor([[1.5, .5, 1.5, .5, 7.]])
        logp = ratios.log().requires_grad_()
        old = torch.zeros_like(logp, requires_grad=True)
        values = torch.tensor([[.3, .4, .5, .6, 1.]], requires_grad=True)
        advantages = torch.tensor([[1., 1., -1., -1., 20.]], requires_grad=True)
        returns = torch.tensor([[1., 0., 1., 0., 10.]], requires_grad=True)
        mask = torch.tensor([[True, True, True, True, False]])
        # Deliberately corrupt off-mask old probabilities to exercise masking.
        with torch.no_grad():
            old[0, -1] = float("-inf")
        actor, critic, actual = P.ppo_terms(logp, old, values, torch.zeros_like(values),
                                            advantages, returns, mask, clip=.2)
        self.assertTrue(torch.allclose(actor, torch.tensor([[-1.2, -.5, 1.5, .8, 0.]]), atol=1e-6))
        (actor.sum() + critic.sum()).backward()
        self.assertEqual(float(logp.grad[0, -1]), 0.)
        self.assertEqual(float(values.grad[0, -1]), 0.)
        self.assertEqual(float(logp.grad[0, 0]), 0.)  # positive advantage clipped above
        self.assertEqual(float(logp.grad[0, 3]), 0.)  # negative advantage clipped below
        self.assertLess(float(logp.grad[0, 1]), 0.)
        self.assertGreater(float(logp.grad[0, 2]), 0.)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)
        self.assertIsNone(returns.grad)
        self.assertTrue(torch.isfinite(actual).all())

    def test_relative_groups_and_finite_budget_termination(self):
        self.assertEqual(P.relative_outcome_advantages([0, 0, 0]), [0., 0., 0.])
        self.assertEqual(P.relative_outcome_advantages([1, 1, 1]), [0., 0., 0.])
        self.assertEqual(P.relative_outcome_advantages([1, 0, 0]), [1., -.5, -.5])
        rows = [{"response_ids": [1] * n, "finish_reason": reason, "reward": reward, "old_values": [.25] * n}
                for n, reason, reward in ((1, "qed", 1), (5, "token_limit", 0))]
        P.attach_advantages(rows, objective="ppo", group_size=2, gamma=1, gae_lambda=1)
        self.assertEqual(rows[0]["returns"], [1.])
        self.assertEqual(rows[1]["returns"], [0.] * 5)
        self.assertEqual(rows[1]["advantages"], [-.25] * 5)
        self.assertTrue(rows[1]["terminated"])
        self.assertFalse(rows[1]["truncated"])
        self.assertEqual(rows[1]["bootstrap_value"], 0.)
        bad = copy.deepcopy(rows)
        bad[1]["reward"] = 1
        with self.assertRaises(ValueError):
            P.attach_advantages(bad, objective="ppo", group_size=2)

    def test_critic_indexes_preaction_state_and_is_causal(self):
        model, tok = tiny()
        critic = P.ValueHead(model.config.d_model)
        with torch.no_grad():
            critic.projection.weight.copy_(torch.linspace(-.1, .1, model.config.d_model)[None])
        rows, audit = train_rows(model, critic, tok)
        batch = L.collate(rows)
        logits, values = P.policy_and_values(model, critic, batch["inputs"])
        self.assertTrue(torch.equal(logits, model(batch["inputs"])))
        for i, row in enumerate(rows):
            start = len(row["prompt_ids"]) - 1
            expected = values[i, start:start + len(row["response_ids"])].detach()
            self.assertTrue(torch.allclose(torch.tensor(row["old_values"]), expected, atol=1e-6))
        modified = batch["inputs"].clone()
        boundary = len(rows[0]["prompt_ids"])
        modified[0, boundary:] = tok.stoi["S"]
        _, changed_values = P.policy_and_values(model, critic, modified)
        self.assertTrue(torch.allclose(values[0, :boundary], changed_values[0, :boundary], atol=1e-6))
        self.assertEqual(len(model.final_norm._forward_hooks), 0)
        OBSERVATIONS["critic_hook_behavior_audit"] = audit

    def test_mixed_length_collation_and_microbatch_equivalence(self):
        model, tok = tiny()
        critic = P.ValueHead(model.config.d_model)
        rows, _ = train_rows(model, critic, tok)
        # Explicitly shorten a response to prevent accidental equal-length coverage.
        rows[0] = {**rows[0], **{k: rows[0][k][:1] for k in
                    ("response_ids", "rollout_logprobs", "old_values", "advantages", "returns")}}
        batch = P.collate_training(rows, tok, "cpu")
        for i, row in enumerate(rows):
            self.assertEqual(int(batch["mask"][i].sum()), len(row["response_ids"]))
            for k in ("old_values", "advantages", "returns"):
                self.assertTrue(torch.equal(batch[k][i, batch["mask"][i]], torch.tensor(row[k])))
                self.assertEqual(float(batch[k][i, ~batch["mask"][i]].abs().sum()), 0.)
        model2, critic2 = copy.deepcopy(model), copy.deepcopy(critic)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(critic.parameters()), lr=1e-3)
        optimizer2 = torch.optim.SGD(list(model2.parameters()) + list(critic2.parameters()), lr=1e-3)
        retention = [episode(record(0), 0, tok), episode(record(2), 4, tok)]
        kwargs = dict(objective="ppo", max_new_tokens=15, retention_weight=.1, value_coef=.5,
                      entropy_coef=.01, epochs=1, loss_normalization="fixed_budget", grad_clip=100.)
        P.update_policy(model, critic, tok, rows, optimizer, retention, microbatch=1, **kwargs)
        P.update_policy(model2, critic2, tok, rows, optimizer2, retention, microbatch=4, **kwargs)
        errors = [float((a.detach() - b.detach()).abs().max()) for a, b in
                  zip(list(model.parameters()) + list(critic.parameters()), list(model2.parameters()) + list(critic2.parameters()))]
        self.assertLess(max(errors), 1e-7)
        OBSERVATIONS["microbatch_max_parameter_error"] = max(errors)

    def test_positive_advantage_actor_increases_sample_likelihood(self):
        model, tok = tiny()
        critic = P.ValueHead(model.config.d_model)
        rows, _ = train_rows(model, critic, tok)
        rows = rows[:1]
        rows[0]["advantages"] = [1.] * len(rows[0]["response_ids"])
        before = L.logprobs(model, L.collate(rows), temperature=.8)
        before = float(before[L.collate(rows)["mask"]].mean().detach())
        optimizer = torch.optim.SGD(list(model.parameters()) + list(critic.parameters()), lr=1e-3)
        result = P.update_policy(model, critic, tok, rows, optimizer, [], objective="ppo", value_coef=0.,
                                  retention_weight=0., max_new_tokens=15, grad_clip=100.)
        after = L.logprobs(model, L.collate(rows), temperature=.8)
        after = float(after[L.collate(rows)["mask"]].mean().detach())
        self.assertGreater(after, before)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertAlmostEqual(result["epochs"][0]["ratio_min"], 1., places=5)
        self.assertAlmostEqual(result["epochs"][0]["ratio_max"], 1., places=5)
        OBSERVATIONS["controlled_positive_advantage_logp_change"] = after - before

    def test_zero_signal_skips_persistent_adam_momentum(self):
        model, tok = tiny()
        rows, _ = train_rows(model, None, tok)
        for row in rows:
            row["advantages"] = [1.] * len(row["response_ids"])
        optimizer = L.optimizer_for(model, 1e-3)
        P.update_policy(model, None, tok, rows, optimizer, [], objective="relative", max_new_tokens=15,
                        retention_weight=0., value_coef=0.)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for row in rows:
            row["advantages"] = [0.] * len(row["response_ids"])
        result = P.update_policy(model, None, tok, rows, optimizer, [], objective="relative", max_new_tokens=15,
                                   retention_weight=0., value_coef=0.)
        self.assertTrue(result["skipped_zero_signal"])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in model.named_parameters()))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PPOReview))
    common.write_json(ROOT / "reports/review_ppo.json", {
        "ok": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [str(x) for x in result.failures], "errors": [str(x) for x in result.errors],
        "observations": OBSERVATIONS,
        "code_sha256": {str(p.relative_to(ROOT)): common.sha(p) for p in
                        [Path(__file__), ROOT / "code/ppo_core.py", ROOT / "code/learning.py", ROOT / "vendor/nd_model.py"]},
        "resource": "CPU only, torch threads=1; artificial rewards test optimizer arithmetic only",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)
