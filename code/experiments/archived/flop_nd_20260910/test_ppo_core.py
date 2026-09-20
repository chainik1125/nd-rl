"""Meaningful CPU checks for PPO semantics and complete training/resume plumbing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import torch

import common
import learning as L
import ppo_core as P
import train_lm as T
from nd_model import ProofModel, ModelConfig, save_checkpoint
from nd_environment import Tokenizer, episode


def small_model(seed=11):
    torch.manual_seed(seed)
    tok = Tokenizer()
    model = ProofModel(ModelConfig(vocab_size=len(tok), d_model=16, n_layers=1,
                       n_heads=2, d_ff=32, max_seq_len=128)).eval()
    return model, tok


def record():
    return {"source_id": "and_example", "family": "and", "prompt": "THM P , Q SEQ ( P & Q ) PRF",
            "body": "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED", "n_lines": 3}


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_terminal_mc_and_segment_bootstrap(self):
        values = torch.tensor([.2, .3, .4], requires_grad=True)
        advantage, returns = P.generalized_advantage_estimate([0, 0, 1], values, gamma=1, gae_lambda=1)
        torch.testing.assert_close(advantage, torch.tensor([.8, .7, .6]))
        torch.testing.assert_close(returns, torch.ones(3))
        self.assertFalse(advantage.requires_grad)
        self.assertFalse(returns.requires_grad)
        adv, target = P.generalized_advantage_estimate([0, 0], [.2, .3], gamma=.9,
                                                       gae_lambda=1, terminated=False, bootstrap_value=.5)
        torch.testing.assert_close(target, torch.tensor([.405, .45]))
        with self.assertRaises(ValueError):
            P.generalized_advantage_estimate([0], [.2], terminated=True, bootstrap_value=.5)

    def test_trace_and_budget_failure(self):
        adv, target = P.generalized_advantage_estimate([0, 0, 1], [.2, .3, .4], gamma=.9, gae_lambda=.8)
        expected_last = 1 - .4
        expected_middle = .9 * .4 - .3 + .9 * .8 * expected_last
        expected_first = .9 * .3 - .2 + .9 * .8 * expected_middle
        torch.testing.assert_close(adv, torch.tensor([expected_first, expected_middle, expected_last]))
        row = {"response_ids": [1, 2], "old_values": [.4, .2], "reward": 0, "finish_reason": "token_limit"}
        P.attach_advantages([row], objective="ppo", group_size=1, gamma=1, gae_lambda=1)
        self.assertEqual(row["returns"], [0., 0.])
        self.assertEqual(row["bootstrap_value"], 0.)
        self.assertTrue(row["terminated"])
        self.assertFalse(row["truncated"])
        self.assertAlmostEqual(row["advantages"][0], -.4)

    def test_clipping_signed_advantages_and_offmask(self):
        logits = torch.tensor([[1.5, 1.5, .5, .5, 1.0]]).log().requires_grad_()
        zeros = torch.zeros_like(logits)
        advantages = torch.tensor([[1., -1., 1., -1., 99.]])
        mask = torch.tensor([[True, True, True, True, False]])
        actor, value, ratios = P.ppo_terms(logits, zeros, None, zeros, advantages, zeros, mask, clip=.2)
        torch.testing.assert_close(actor, torch.tensor([[-1.2, 1.5, -.5, .8, 0.]]))
        actor.sum().backward()
        self.assertEqual(float(logits.grad[0, -1]), 0.)
        self.assertEqual(float(logits.grad[0, 0]), 0.)
        self.assertEqual(float(logits.grad[0, 3]), 0.)

    def test_causal_values_and_response_alignment(self):
        model, tok = small_model()
        critic = P.ValueHead(model.config.d_model)
        with torch.no_grad():
            critic.projection.weight.fill_(.1)
            critic.projection.weight[0, 0] = .6
        inputs = torch.tensor([[1, 3, 7, 8, 9]])
        direct = model(inputs)
        logits, values = P.policy_and_values(model, critic, inputs)
        torch.testing.assert_close(logits, direct)
        _, changed = P.policy_and_values(model, critic, torch.tensor([[1, 3, 7, 8, 10]]))
        torch.testing.assert_close(values[:, :4], changed[:, :4])
        self.assertEqual(len(model.final_norm._forward_hooks), 0)
        rows = [{"prompt_ids": [1, 3, 7], "response_ids": [8, 6], "rollout_logprobs": [-1, -2],
                 "old_values": [.1, .2], "advantages": [.4, .5], "returns": [1, 1]},
                {"prompt_ids": [1, 3], "response_ids": [6], "rollout_logprobs": [-3],
                 "old_values": [.3], "advantages": [.6], "returns": [0]}]
        batch = P.collate_training(rows, tok, "cpu")
        self.assertEqual(batch["mask"].tolist(), [[False, False, True, True], [False, True, False, False]])
        self.assertEqual(batch["targets"][batch["mask"]].tolist(), [8, 6, 6])
        self.assertEqual(batch["old_logprobs"][batch["mask"]].tolist(), [-1, -2, -3])

    def test_behavior_audit_and_microbatch_equivalence(self):
        model, tok = small_model()
        critic = P.ValueHead(model.config.d_model)
        eps = [episode(record(), cut, tok) for cut in (0, 2)]
        rows = L.sample(model, tok, eps, temperature=.8, seed=5, max_new_tokens=5, batch_size=2)
        audit = P.audit_and_attach_old_values(model, critic, tok, rows, temperature=.8, tolerance=1e-5)
        self.assertLess(audit["max_abs_error"], 1e-5)
        for row in rows:
            row.update(reward=1, finish_reason="qed")
        P.attach_advantages(rows, objective="ppo", group_size=1, gamma=1, gae_lambda=1)
        a, b = copy.deepcopy(model), copy.deepcopy(model)
        ca, cb = copy.deepcopy(critic), copy.deepcopy(critic)
        oa = torch.optim.SGD(list(a.parameters()) + list(ca.parameters()), lr=.001)
        ob = torch.optim.SGD(list(b.parameters()) + list(cb.parameters()), lr=.001)
        old = copy.deepcopy(rows)
        P.update_policy(a, ca, tok, rows, oa, [], objective="ppo", max_new_tokens=5, microbatch=1, grad_clip=100)
        P.update_policy(b, cb, tok, rows, ob, [], objective="ppo", max_new_tokens=5, microbatch=2, grad_clip=100)
        for pa, pb in zip(list(a.parameters()) + list(ca.parameters()), list(b.parameters()) + list(cb.parameters())):
            torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-8)
        self.assertEqual(rows, old)

    def test_zero_signal_does_not_advance_adam_momentum(self):
        model, tok = small_model()
        rows = L.sample(model, tok, [episode(record(), 2, tok)], max_new_tokens=3, seed=9)
        P.audit_and_attach_old_values(model, None, tok, rows, tolerance=1e-5)
        rows[0].update(advantages=[0.] * len(rows[0]["response_ids"]), returns=[0.] * len(rows[0]["response_ids"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        # Seed nonzero momentum independently so a zero-gradient step would move.
        sum(p.sum() for p in model.parameters()).backward()
        optimizer.step()
        before = copy.deepcopy(model.state_dict())
        result = P.update_policy(model, None, tok, rows, optimizer, [], objective="relative", max_new_tokens=3, retention_weight=0)
        self.assertEqual(result["optimizer_steps"], 0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_source_cap_and_theorem_only_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            common.write_jsonl(path, [record()])
            self.assertEqual(T.training_sources(path)[0]["n_lines"], 3)
            row = {"prompt": record()["prompt"], "source_id": "target", "n_lines": 10}
            common.write_jsonl(path, [row])
            with self.assertRaises(ValueError):
                T.training_targets(path)
            row.pop("n_lines")
            common.write_jsonl(path, [row])
            result = T.training_targets(path)
            ep = T.target_episode(result[0], Tokenizer())
            self.assertEqual(ep["prefix_ids"], [])
            self.assertNotIn("source_lines", ep)

    def test_complete_cli_checkpoint_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model, tok = small_model()
            checkpoint = directory / "base.pt"
            save_checkpoint(checkpoint, model, tok)
            sources = directory / "sources.jsonl"
            targets = directory / "targets.jsonl"
            common.write_jsonl(sources, [record()])
            common.write_jsonl(targets, [{"source_id": "target", "prompt": "THM R SEQ ( R & R ) PRF"}])
            config = {"checkpoint": str(checkpoint), "data": str(sources), "retention": str(sources),
                      "targets": str(targets), "target_fraction": .5, "objective": "ppo", "start": "reset",
                      "max_updates": 2, "prompts_per_step": 2, "group_size": 2, "max_new_tokens": 5,
                      "retention_batch": 1, "microbatch": 2, "batch_size": 4, "checkpoint_every": 1, "epochs": 4,
                      "cpu_threads": 1, "seed": 17, "max_seconds": 60, "max_logprob_error": 1e-5}
            full_path = directory / "full"
            resume_path = directory / "resumed"
            cfg = directory / "config.json"
            common.write_json(cfg, {**config, "output": str(full_path)})
            with redirect_stdout(io.StringIO()):
                T.run(T.arguments(["--config", str(cfg)]))
            common.write_json(cfg, {**config, "output": str(resume_path), "max_updates": 1})
            with redirect_stdout(io.StringIO()):
                T.run(T.arguments(["--config", str(cfg)]))
            # Simulate a crash after checkpoint1 with an uncommitted attempt tail.
            T.append_jsonl(resume_path / "attempts.jsonl", [{"step": 2, "bad_partial": True}])
            common.write_json(cfg, {**config, "output": str(resume_path), "resume": str(resume_path / "final.pt")})
            with redirect_stdout(io.StringIO()):
                T.run(T.arguments(["--config", str(cfg)]))
            a = torch.load(full_path / "final.pt", weights_only=True)
            b = torch.load(resume_path / "final.pt", weights_only=True)
            for group in ("model", "critic"):
                for key, value in a[group].items():
                    torch.testing.assert_close(value, b[group][key], rtol=0, atol=0)
            self.assertEqual(a["training_state"]["generated_tokens"], b["training_state"]["generated_tokens"])
            self.assertEqual(a["training_state"]["retention_schedule_digest"], b["training_state"]["retention_schedule_digest"])
            self.assertEqual(len(common.read_jsonl(resume_path / "attempts.jsonl")), 8)
            loaded, _, _ = L.load_model(resume_path / "final.pt")
            self.assertIsInstance(loaded, ProofModel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
