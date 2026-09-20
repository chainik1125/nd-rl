"""Invented-formula tests for isolated nonlearned ND comparators."""
from __future__ import annotations

import importlib.util
import itertools
import json
import multiprocessing
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import common
import classical_solver as C
import classical_run as R
from classical_vendor.truth_assignment import TruthAssignmentConstructor, ConstructionBudgetExceeded, atoms
from classical_vendor.support import Proof, serialize, CertificateOutputLimitExceeded
from nd_environment import Tokenizer, strict_complete
from nd_data import formula_text


def random_formula(rng, depth):
    if depth <= 0 or rng.random() < .25:
        return ("bot",) if rng.random() < .1 else ("atom", rng.choice("PQRS"))
    kind = rng.choice(["not", "and", "or", "imp"])
    return (kind, random_formula(rng, depth - 1)) if kind == "not" else (
        kind, random_formula(rng, depth - 1), random_formula(rng, depth - 1))


def valid(premises, goal):
    names = sorted(set().union(*(atoms(f) for f in premises + [goal])))
    return all(not all(C.truth(f, dict(zip(names, bits))) for f in premises) or C.truth(goal, dict(zip(names, bits)))
               for bits in itertools.product([False, True], repeat=len(names)))


def sleepy_worker(connection, prompt, method):
    time.sleep(1)
    connection.send({"status": "bad_late_success", "reward": 1})


class ClassicalTests(unittest.TestCase):
    def test_1000_invented_sequents_and_countermodels(self):
        rng = random.Random(202609091934)
        for case in range(1000):
            premises = [random_formula(rng, rng.randrange(1, 5)) for _ in range(rng.randrange(3))]
            goal = random_formula(rng, rng.randrange(1, 5))
            prompt = "THM " + " , ".join(map(formula_text, premises)) + " SEQ " + formula_text(goal) + " PRF"
            result = TruthAssignmentConstructor(atom_order="frequency" if case % 2 else "alphabetical").solve(prompt)
            expected = valid(premises, goal)
            self.assertEqual(expected, result["status"] == "certificate")
            if expected:
                self.assertTrue(strict_complete(prompt, result["body"])["ok"])
            else:
                self.assertTrue(all(C.truth(f, result["countermodel"]) for f in premises))
                self.assertFalse(C.truth(goal, result["countermodel"]))
            self.assertEqual(result["operations"], sum(result["operation_components"].values()))

    def test_budget_boundary_and_serializer_limits(self):
        prompt = "THM SEQ ( ( ( P > Q ) > P ) > P ) PRF"
        full = TruthAssignmentConstructor().solve(prompt)
        for budget in range(full["operations"]):
            constructor = TruthAssignmentConstructor(budget)
            with self.assertRaises(ConstructionBudgetExceeded):
                constructor.solve(prompt)
            self.assertEqual(constructor.operations, budget)
        p, q = Proof(("atom", "P"), "PR", identity=1), Proof(("atom", "Q"), "PR", identity=2)
        proof = Proof(("and", p.formula, q.formula), "ANDI", (p, q))
        with self.assertRaises(CertificateOutputLimitExceeded):
            serialize([p, q], proof, max_lines=2)
        with self.assertRaises(CertificateOutputLimitExceeded):
            serialize([p, q], proof, max_codec_tokens=10)
        self.assertEqual(TruthAssignmentConstructor(full["operations"]).solve(prompt)["body"], full["body"])

    def test_all_methods_and_exact_codec_cost(self):
        tok = Tokenizer()
        prompts = ["THM P , Q SEQ ( P & Q ) PRF", "THM SEQ ( P > P ) PRF",
                   "THM SEQ ( P v ( ~ P ) ) PRF", "THM ( P v Q ) SEQ ( Q v P ) PRF",
                   "THM ( P > Q ) , P SEQ Q PRF", "THM P , ( ~ P ) SEQ R PRF"]
        for prompt in prompts:
            for method in C.METHODS:
                result = C.solve(prompt, method)
                self.assertEqual(result["status"], "verified", result)
                for kind in ("raw", "pruned"):
                    body = result[f"{kind}_certificate"]
                    encoded = 1 + len(tok.encode_record(prompt + " " + body)) - len(tok.encode_prompt(prompt))
                    self.assertEqual(result[f"{kind}_response_tokens"], encoded)
        self.assertEqual(C.solve("THM P SEQ Q PRF", C.METHODS[-1])["status"], "countermodel")

    def test_censored_certificate_is_not_success(self):
        import classical_vendor.truth_assignment as module
        original = module.serialize
        with patch.object(module, "serialize", lambda premises, proof: original(premises, proof, max_lines=1)):
            result = C.solve("THM SEQ ( P > P ) PRF", C.METHODS[-1])
        self.assertEqual(result["status"], "certificate_output_censored")
        self.assertEqual(result["reward"], 0)
        self.assertNotIn("raw_certificate", result)

    def test_hard_timeout_stops_owned_worker(self):
        tasks = [{"prompt": "THM P SEQ Q PRF", "source_id": "invented", "method": C.METHODS[0]}]
        start = time.monotonic()
        results = list(R.bounded_attempts(tasks, workers=1, timeout_seconds=.05, worker=sleepy_worker))
        self.assertLess(time.monotonic() - start, .5)
        self.assertEqual(results[0][1]["status"], "hard_timeout")
        self.assertEqual(results[0][1]["reward"], 0)
        self.assertFalse(multiprocessing.active_children())

    def test_prompt_only_and_final_release_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, protocol = directory / "final.jsonl", directory / "protocol.json"
            row = {"source_id": "x", "name": "final_invented", "prompt": "THM P SEQ P PRF"}
            common.write_jsonl(data, [row])
            common.write_json(protocol, R.make_protocol(data))
            with self.assertRaises(ValueError):
                R.verify_protocol(protocol, data)
            release = directory / "release.json"
            common.write_json(release, {"status": "released", "all_comparator_endpoints_frozen": True,
                                        "protocol_sha256": common.sha(protocol)})
            self.assertEqual(R.verify_protocol(protocol, data, release)["status"], "frozen")
            common.write_jsonl(data, [{**row, "body": "forbidden reference"}])
            with self.assertRaises(ValueError):
                R.load_inputs(data)
            with self.assertRaises(ValueError):
                R.verify_protocol(protocol, data, release)


if __name__ == "__main__":
    unittest.main(verbosity=2)
