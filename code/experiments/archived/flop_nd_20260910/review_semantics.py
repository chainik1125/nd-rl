"""Independent, bounded CPU checks for the frozen proof interface.

Run directly; emits reports/review_semantics.json. Uses invented certificates
and random tiny networks, so no evaluation outcomes enter readiness decisions.
"""
from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path

import torch

import common
import learning as L
from nd_data import canonical_theorem, generate_candidate
from nd_environment import Tokenizer, episode, independent_replay, judge, strict_complete
from nd_model import ModelConfig, ProofModel

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = {}

CASES = [
    ("THM ( P > Q ) , P SEQ Q PRF", "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"),
    ("THM SEQ ( P > ( Q > P ) ) PRF", "N1 | P : AS ; N2 | | Q : AS ; N3 | | P : R N1 ; N4 | ( Q > P ) : IMPI N2 N3 ; N5 ( P > ( Q > P ) ) : IMPI N1 N4 ; QED"),
    ("THM ( P v Q ) SEQ ( Q v P ) PRF", "N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED"),
]


def record(i):
    prompt, body = CASES[i]
    return {"prompt": prompt, "body": body, "source_id": common.digest([prompt, body]), "family": "invented"}


def tiny(seed=711):
    torch.manual_seed(seed)
    tok = Tokenizer()
    model = ProofModel(ModelConfig(vocab_size=len(tok), d_model=24, n_layers=2,
                                   n_heads=2, d_ff=48, max_seq_len=512)).eval()
    return model, tok


class InterfaceReview(unittest.TestCase):
    def test_every_prefix_reconstructs_original_certificate(self):
        total = 0
        tok = Tokenizer()
        for i in range(len(CASES)):
            for cut in L.admissible_cuts(record(i)):
                r = episode(record(i), cut, tok)
                verdict = judge(r, r["response_ids"], tok)
                self.assertEqual(verdict["reward"], 1)
                self.assertEqual(verdict["body"], CASES[i][1])
                self.assertEqual(r["prefix_state"]["next_line"], cut + 1)
                total += 1
        OBSERVATIONS["invented_roundtrip_prefixes"] = total

    def test_sibling_and_closed_inner_assumptions_are_inaccessible(self):
        state = episode(record(2), 4)["prefix_state"]
        self.assertEqual(state["active_assumptions"], [4])
        self.assertEqual(state["available_lines"], [1, 4])
        nested = episode(record(1), 4)["prefix_state"]
        self.assertEqual(nested["active_assumptions"], [1])
        self.assertEqual(nested["available_lines"], [1, 4])
        bad = CASES[2][1].replace("ORI1 N4", "ORI1 N2")
        self.assertFalse(strict_complete(CASES[2][0], bad)["ok"])
        bad = CASES[1][1].replace("IMPI N2 N3", "IMPI N1 N3")
        self.assertFalse(strict_complete(CASES[1][0], bad)["ok"])

    def test_termination_and_scratch_boundaries_are_explicit(self):
        tok = Tokenizer()
        r = episode(record(0), 2, tok)
        for response in [r["response_ids"][:-1], r["response_ids"] + [tok.eos_id],
                         [tok.stoi["P"]] + r["response_ids"],
                         [tok.stoi["<FINAL>"]] + r["response_ids"]]:
            self.assertEqual(judge(r, response, tok)["reward"], 0)
        scratch = [tok.stoi[x] for x in ("<THINK>", "P", "</THINK>")]
        self.assertEqual(judge(r, scratch + r["response_ids"], tok)["reward"], 1)

    def test_cache_and_teacher_match_for_padding_and_sampled_pad(self):
        model, tok = tiny()
        eps = [episode(record(0), 0, tok), episode(record(1), 3, tok), episode(record(2), 5, tok)]
        rows = L.sample(model, tok, eps, seed=619, temperature=.83, max_new_tokens=31, batch_size=3)
        result = L.audit_behavior(model, tok, rows, temperature=.83, batch_size=2, tolerance=1e-5)
        OBSERVATIONS["cache_teacher"] = result
        mutated = copy.deepcopy(rows)
        mutated[0]["rollout_logprobs"][0] += .01
        with self.assertRaises(RuntimeError):
            L.audit_behavior(model, tok, mutated, temperature=.83, tolerance=1e-5)
        padrow = copy.deepcopy(eps[0])
        padrow["response_ids"] = [tok.pad_id, tok.stoi["<FINAL>"], tok.eos_id]
        batch = L.collate([padrow, eps[2]], fixed_pad=256)
        start = len(padrow["prompt_ids"]) - 1
        self.assertEqual(int(batch["targets"][0, start]), tok.pad_id)
        self.assertTrue(bool(batch["mask"][0, start]))
        self.assertEqual(int(batch["mask"][0].sum()), 3)

    def test_logits_are_causal_and_supplied_positions_have_no_direct_loss(self):
        model, tok = tiny()
        rows = [episode(record(0), 2, tok), episode(record(1), 1, tok)]
        batch = L.collate(rows, fixed_pad=256)
        ids = batch["inputs"]
        changed = ids.clone()
        boundary = len(rows[0]["prompt_ids"])
        changed[0, boundary:] = tok.stoi["S"]
        self.assertTrue(torch.allclose(model(ids)[0, :boundary], model(changed)[0, :boundary], atol=1e-6))
        logits = torch.randn(*ids.shape, len(tok), requires_grad=True)
        chosen = logits.log_softmax(-1).gather(-1, batch["targets"][..., None]).squeeze(-1)
        (-(chosen * batch["mask"]).sum()).backward()
        self.assertEqual(float(logits.grad[~batch["mask"]].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[batch["mask"]].abs().sum()), 0.)

    def test_zero_relative_advantages_no_fresh_optimizer_policy_signal(self):
        model, tok = tiny()
        eps = [episode(record(0), 0, tok), episode(record(1), 3, tok)]
        rows = L.sample(model, tok, eps, seed=711, temperature=.8, max_new_tokens=20, batch_size=2)
        for row in rows:
            row["advantage"] = 0.
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        result = L.policy_update(model, tok, rows, L.optimizer_for(model, 1e-3), [],
                                 temperature=.8, max_new_tokens=20, retention_weight=0.)
        self.assertEqual(result["grad_norm"], 0.)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in model.named_parameters()))
        OBSERVATIONS["fresh_optimizer_zero_relative_advantage_grad_norm"] = result["grad_norm"]

    def test_global_atom_renaming_and_premise_order_guard(self):
        a = "THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF"
        b = "THM ( S > P ) , ( R > S ) SEQ ( R > P ) PRF"
        c = "THM ( P > Q ) , ( R > Q ) SEQ ( P > R ) PRF"
        self.assertEqual(canonical_theorem(a), canonical_theorem(b))
        self.assertNotEqual(canonical_theorem(a), canonical_theorem(c))

    def test_forward_certificates_and_reference_mutations(self):
        rng = random.Random(202609101946)
        accepted = 0
        for _ in range(500):
            raw = generate_candidate(rng, rng.choice((6, 9, 12)))
            if raw is None:
                continue
            body = raw["text"].split("PRF", 1)[1].strip()
            self.assertTrue(strict_complete(raw["prompt"], body)["ok"])
            self.assertFalse(strict_complete(raw["prompt"], body.replace("N1", "N0", 1))["ok"])
            accepted += 1
        self.assertGreater(accepted, 20)
        OBSERVATIONS["random_certificates_verified_and_mutated"] = accepted


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InterfaceReview))
    common.write_json(ROOT / "reports/review_semantics.json", {
        "ok": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [str(x) for x in result.failures], "errors": [str(x) for x in result.errors],
        "observations": OBSERVATIONS,
        "code_sha256": {str(p.relative_to(ROOT)): common.sha(p) for p in
                        [Path(__file__), ROOT / "code/learning.py", ROOT / "code/nd_environment.py", ROOT / "vendor/nd_model.py"]},
        "resource": "CPU only, torch threads=1, random tiny model and invented/fresh synthetic data",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)
