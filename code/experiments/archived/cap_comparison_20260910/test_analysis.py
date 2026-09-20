"""Invented certificates only; exercise replay and inference-budget accounting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import analyze as a


class Accounting(unittest.TestCase):
    def test_expensive_failure_blocks_later_success(self):
        rows = [{"index": 0, "attempt": 0, "generated_tokens": 300, "reward": 0},
                {"index": 0, "attempt": 1, "generated_tokens": 10, "reward": 1}]
        self.assertEqual(a.prefix_outcomes(rows, 1, token_budget=256)["charged_tokens"], 256)
        self.assertEqual(a.prefix_outcomes(rows, 1, token_budget=256)["solved"], 0)
        self.assertEqual(a.prefix_outcomes(rows, 1, token_budget=309)["solved"], 0)
        self.assertEqual(a.prefix_outcomes(rows, 1, token_budget=310)["solved"], 1)
        self.assertEqual(a.prefix_outcomes(rows, 1, k=1)["solved"], 0)
        self.assertEqual(a.prefix_outcomes(rows, 1, k=2)["solved"], 1)

    def test_cost_includes_failures_after_success_and_each_target(self):
        rows = [{"index": 0, "attempt": 0, "generated_tokens": 10, "reward": 1},
                {"index": 0, "attempt": 1, "generated_tokens": 20, "reward": 0},
                {"index": 1, "attempt": 0, "generated_tokens": 90, "reward": 0}]
        result = a.prefix_outcomes(rows, 2, token_budget=25)
        self.assertEqual(result["charged_tokens"], 50)
        self.assertEqual(result["outcomes"], [1, 0])

    def test_paired_uses_targets_and_is_reproducible(self):
        protocol = {"seed": 17, "resamples": 5000, "scope": "invented"}
        result = a.paired([1,0,0,1], [0,1,1,1], protocol)
        self.assertEqual(result, a.paired([1,0,0,1], [0,1,1,1], protocol))
        self.assertEqual((result["delta_solved"], result["newly_solved"], result["lost"]), (1,2,1))
        zero = a.paired([1,0], [1,0], protocol)
        self.assertEqual(zero["paired_target_bootstrap95"], [0.,0.])


class Replay(unittest.TestCase):
    def fixture(self, root, mode):
        spec = {"run_base": "runs/invented", "cap": 6, "mode": mode,
                "evaluation": {"seed": 0, "decode_batch": 64, "max_seq_len": 8192},
                "development_data": {"sha256": "invented_dataset"},
                "development_release": "runs/invented/release.json"}
        data = [{"prompt": "THM P SEQ P PRF", "body": "N1 P : PR ; QED", "reference_lines": 1}]
        checkpoint = root / spec["run_base"] / "sft/final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"invented weights, never loaded")
        digest = a.sha(checkpoint)
        release = {"checkpoints": [digest]}
        a.write(root / spec["development_release"], release)
        directory = root / spec["run_base"] / "evaluations/sft_dev_greedy"
        directory.mkdir(parents=True)
        settings = {"mode": mode, "samples": 1, "temperature": 0, "max_new_tokens": 1536,
                    "max_proof_lines": 6, "seed": 0, "precision": "fp32", "decode_batch": 64,
                    "max_seq_len": 8192}
        a.write(directory / "manifest.json", {"args": settings, "source_sha256": digest,
                "data_sha256": "invented_dataset", "evaluation_release_sha256": a.sha(root / spec["development_release"])})
        a.write(directory / "completed.json", {"stage": "eval", "checkpoint_sha256": digest})
        tok = a.trainer.ScratchTokenizer()
        encoded = a.trainer.encode_training(data[0], tok, mode, max_proof_lines=6,
                                             max_seq_len=8192, scratch_style="dependency")
        row = {**encoded, "prompt": data[0]["prompt"], "index": 0, "attempt": 0,
               "generated_tokens": len(encoded["response_ids"]),
               **a.trainer.judge_response(data[0]["prompt"], encoded["response_ids"], tok, mode)}
        assert row["reward"] == 1
        self.save_rows(directory, [row])
        a.write(directory / "summary.json", a.trainer.evaluation_summary(data, [row],
                SimpleNamespace(samples=1, max_new_tokens=1536, max_proof_lines=6), digest, "invented_dataset"))
        return spec, data, release, directory, row

    def save_rows(self, directory, rows):
        (directory / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_both_modes_strict_replay_and_compression(self):
        for mode in ("direct", "cot"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec,data,release,_,row = self.fixture(root, mode)
                detail,_ = a.replay(root,spec,"sft","greedy",data,release)
                self.assertEqual(detail["solved"], 1)
                self.assertEqual(detail["max_shortest_compressed_witness"], 1)
                self.assertEqual(detail["charged_tokens"], len(row["response_ids"]))

    def test_tampered_theorem_reward_count_and_missing_attempt_rejected(self):
        for corruption in ("prompt", "reward", "tokens", "duplicate", "missing"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec,data,release,directory,row = self.fixture(root,"direct")
                changed = copy.deepcopy(row)
                if corruption == "prompt": changed["prompt"] = "THM Q SEQ Q PRF"
                if corruption == "reward": changed["reward"] = 0
                if corruption == "tokens": changed["generated_tokens"] += 1
                rows = [changed,changed] if corruption == "duplicate" else [] if corruption == "missing" else [changed]
                self.save_rows(directory,rows)
                with self.assertRaises(ValueError): a.replay(root,spec,"sft","greedy",data,release)


if __name__ == "__main__":
    unittest.main()
