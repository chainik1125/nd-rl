"""CPU equivalence and exact-resume tests for the retention exposure control."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import common
import learning as L
import ppo_core as P
import review_retention_control as C
from review_semantics import tiny, record
from nd_environment import episode

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = {}


class ControlReview(unittest.TestCase):
    def test_exact_retention_component_of_zero_advantage_main_update(self):
        model, tok = tiny(404)
        other = copy.deepcopy(model)
        retained = [episode(record(0), 0, tok), episode(record(1), 0, tok), episode(record(2), 0, tok)]
        dummy = copy.deepcopy(retained[:1])
        for row in dummy:
            n = len(row["response_ids"])
            row.update(rollout_logprobs=[0.] * n, old_values=[0.] * n,
                       advantages=[0.] * n, returns=[0.] * n)
        optimizer = L.optimizer_for(model, 5e-5)
        other_optimizer = L.optimizer_for(other, 5e-5)
        P.update_policy(model, None, tok, dummy, optimizer, retained, objective="relative",
                        retention_weight=.1, max_new_tokens=512, microbatch=2, epochs=4)
        epochs = [C.retention_epoch(other, tok, retained, other_optimizer, weight=.1, microbatch=2) for _ in range(4)]
        error = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(model.parameters(), other.parameters()))
        self.assertLess(error, 1e-7)
        OBSERVATIONS["same_as_main_zero_advantage_retention_max_parameter_error"] = error
        self.assertGreater(epochs[0]["retention_loss"], epochs[-1]["retention_loss"])

    def test_two_steps_equals_one_plus_resume_and_log_tail_recovery(self):
        with tempfile.TemporaryDirectory(prefix="flop-nd-review-retention-") as tmp:
            tmp = Path(tmp)
            model, tok = tiny(405)
            checkpoint = tmp / "origin.pt"
            torch.save({"model": model.state_dict(), "config": asdict(model.config),
                        "tokenizer": tok.state_dict(), "metadata": {}}, checkpoint)
            data = tmp / "retention.jsonl"
            common.write_jsonl(data, [record(0), record(1), record(2)])
            def args(output, steps, resume=None):
                argv = ["--checkpoint", str(checkpoint), "--retention", str(data), "--output", str(output),
                        "--max-updates", str(steps), "--device", "cpu", "--cpu-threads", "1",
                        "--retention-batch", "3", "--microbatch", "2", "--epochs", "4",
                        "--seed", "222", "--checkpoint-every", "1"]
                if resume:
                    argv.extend(["--resume", str(resume)])
                return C.arguments(argv)
            continuous, resumed = tmp / "continuous", tmp / "resumed"
            with contextlib.redirect_stdout(io.StringIO()):
                expected = C.run(args(continuous, 2))
                C.run(args(resumed, 1))
                with (resumed / "updates.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": 2, "uncommitted_fake": True}) + "\n")
                actual = C.run(args(resumed, 2, resumed / "final.pt"))
            a = torch.load(continuous / "final.pt", weights_only=True)
            b = torch.load(resumed / "final.pt", weights_only=True)
            self.assertTrue(all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]))
            self.assertEqual(expected["retention_schedule_digest"], actual["retention_schedule_digest"])
            self.assertEqual(expected["optimizer_steps"], actual["optimizer_steps"])
            self.assertEqual(expected["retention_examples"], actual["retention_examples"])
            updates = common.read_jsonl(resumed / "updates.jsonl")
            self.assertEqual([u["step"] for u in updates], [1, 2])
            self.assertFalse(any("uncommitted_fake" in u for u in updates))
            resume_event = common.read_jsonl(resumed / "resume_events.jsonl")[0]
            self.assertEqual(resume_event["discarded_tail_rows"], {"updates.jsonl": 1})
            for key in ("step", "optimizer_steps", "retention_examples", "retention_response_tokens", "ledger"):
                self.assertEqual(a["training_state"][key], b["training_state"][key])
            # Loading through the production evaluator's policy loader works.
            loaded, loaded_tok, _ = L.load_model(resumed / "final.pt")
            self.assertEqual(tok.state_dict(), loaded_tok.state_dict())
            self.assertTrue(all(torch.equal(loaded.state_dict()[k], a["model"][k]) for k in a["model"]))
            OBSERVATIONS["resume_policy_equal_bitwise"] = True
            OBSERVATIONS["resume_optimizer_epochs"] = actual["optimizer_steps"]
            OBSERVATIONS["resume_retention_examples"] = actual["retention_examples"]


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlReview))
    common.write_json(ROOT / "reports/review_retention_control.json", {"ok": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": [str(x) for x in result.failures],
        "errors": [str(x) for x in result.errors], "observations": OBSERVATIONS,
        "code_sha256": {str(p.relative_to(ROOT)): common.sha(p) for p in
                       [Path(__file__), ROOT / "code/review_retention_control.py", ROOT / "code/train_lm.py", ROOT / "code/ppo_core.py"]},
        "resource": "CPU only, tiny random models and invented cap-six certificates; temporary checkpoints removed after tests"})
    raise SystemExit(0 if result.wasSuccessful() else 1)
