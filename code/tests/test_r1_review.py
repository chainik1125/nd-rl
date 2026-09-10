"""Independent synthetic-only runtime checks; no dataset reads or GPU by default.

PYTHONPATH includes take-home and transfer code. Set R1_REVIEW_DEVICE=cuda to
measure BF16 cached/teacher drift on an explicitly allocated sprint GPU.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from r1_cot import run as r
except ModuleNotFoundError as exc:
    if exc.name != "r1_cot":
        raise
    import run as r

DEVICE = os.environ.get("R1_REVIEW_DEVICE", "cpu")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.tok = r.ScratchTokenizer()
        self.prompt = "THM P SEQ P PRF"
        self.body = "N1 P : PR ; QED"
        self.ids = self.tok.encode_record(self.prompt + " " + self.body)[
            len(self.tok.encode_prompt(self.prompt)) :
        ]

    def wrapped(self, ids=None):
        return [
            self.tok.stoi["<THINK>"],
            self.tok.stoi["P"],
            self.tok.stoi["</THINK>"],
            self.tok.stoi["<FINAL>"],
            *(self.ids if ids is None else ids),
        ]

    def test_final_extraction_and_wrong_target(self):
        good = self.wrapped()
        self.assertEqual(r.judge_response(self.prompt, good, self.tok, "cot")["reward"], 1)
        variants = [
            good[:-1],
            good + [self.tok.eos_id],
            good + [self.tok.pad_id],
            good + [self.tok.stoi["<FINAL>"]],
            good[1:],
            good[:2] + [self.tok.stoi["<FINAL>"]] + good[2:],
        ]
        for bad in variants:
            self.assertEqual(r.judge_response(self.prompt, bad, self.tok, "cot")["reward"], 0)
        self.assertEqual(r.judge_response("THM P SEQ Q PRF", good, self.tok, "cot")["reward"], 0)
        # Proof-like scratch does not rescue a wrong final answer.
        bad = [
            self.tok.stoi["<THINK>"],
            *self.ids,
            self.tok.stoi["</THINK>"],
            self.tok.stoi["<FINAL>"],
            self.tok.eos_id,
        ]
        self.assertEqual(r.judge_response(self.prompt, bad, self.tok, "cot")["reward"], 0)
        longer = [good[0]] + [self.tok.stoi["P"]] * 30 + good[2:]
        result = r.judge_response(self.prompt, longer, self.tok, "cot")
        self.assertEqual(
            (result["reward"], result["found_len"], result["scratch_tokens"]), (1, 1, 30)
        )

    def test_zero_distance_self_reference_rejected(self):
        # Raw decoder permits zero BACK, strict verifier must reject N1 citing N1.
        tokens = ["<LINE>", "P", ":", "R", "<REF>", "<ENDREF>", ";", "QED"]
        result = r.judge_response(
            "THM SEQ P PRF", self.wrapped([self.tok.stoi[x] for x in tokens]), self.tok, "cot"
        )
        self.assertEqual(result["reward"], 0)

    def test_sampled_pad_is_an_action(self):
        rows = [
            {"prompt_ids": [1, 3, 5], "response_ids": [0, 7, 6]},
            {"prompt_ids": [1, 3], "response_ids": [6]},
        ]
        b = r.collate_rollouts(rows, 0, "cpu")
        self.assertEqual(b["mask"].sum(-1).tolist(), [3, 1])
        self.assertEqual(b["targets"][0][b["mask"][0]].tolist(), [0, 7, 6])
        self.assertEqual(b["targets"][1][b["mask"][1]].tolist(), [6])

    def test_loo(self):
        actual = r.group_advantages([[0, 0, 0], [1, 1, 1], [1, 0, 0]])
        torch.testing.assert_close(
            actual, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, -0.5, -0.5]])
        )
        with self.assertRaises(ValueError):
            r.group_advantages([[float("nan"), 0]])

    def test_guard_all_three_levels(self):
        fields = (
            "canonical_theorem_sha256",
            "nonempty_premise_context_sha256",
            "closed_formula_skeleton_sha256",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guards.json"
            for prompt in [self.prompt, "THM SEQ ( P > P ) PRF"]:
                for index, key in enumerate(r.guard_keys(prompt)):
                    if key is None:
                        continue
                    blob = {f: [] for f in fields}
                    blob[fields[index]] = [key]
                    path.write_text(json.dumps(blob))
                    with self.assertRaises(ValueError):
                        r.guard_training(
                            [{"prompt": prompt}], Path(directory) / "train.jsonl", path
                        )
            with self.assertRaises(ValueError):
                r.guard_training([{"prompt": self.prompt}], Path(directory) / "train.jsonl", None)

    def tiny(self):
        return (
            r.ProofModel(
                r.ModelConfig(
                    vocab_size=len(self.tok),
                    d_model=32,
                    n_layers=2,
                    n_heads=2,
                    d_ff=64,
                    max_seq_len=192,
                    dropout=0.2,
                )
            )
            .to(DEVICE)
            .eval()
        )

    def test_cache_teacher_alignment_and_actual_denominator(self):
        model = self.tiny()
        prompts = [self.prompt, "THM ( P > Q ) , P SEQ Q PRF", "THM SEQ ( P > P ) PRF"]
        rows = r.sample(
            model,
            self.tok,
            prompts,
            mode="cot",
            temperature=0.8,
            seed=27,
            max_new_tokens=32,
            batch_size=3,
        )
        errors = []
        for row in rows:
            batch = r.collate_rollouts([row], self.tok.pad_id, DEVICE)
            with torch.no_grad(), r.amp(model):
                lp = (model(batch["inputs"]).float() / 0.8).log_softmax(-1)
            chosen = lp.gather(-1, batch["targets"][..., None]).squeeze(-1)[batch["mask"]].tolist()
            self.assertEqual(len(chosen), len(row["response_ids"]))
            errors.extend(abs(a - b) for a, b in zip(chosen, row["rollout_logprobs"], strict=True))
            row["advantage"] = 1.0
        maximum = max(errors)
        print(
            json.dumps(
                {
                    "device": DEVICE,
                    "cache_teacher_max_error": maximum,
                    "cache_teacher_mean_error": sum(errors) / len(errors),
                    "n_action_tokens": len(errors),
                }
            )
        )
        self.assertLess(maximum, 0.05 if DEVICE.startswith("cuda") else 1e-4)
        # Perturb stored behavior by a known amount. PPO must preserve those
        # exact values, rather than silently replace them with teacher values.
        for row in rows:
            row["rollout_logprobs"] = [value + 0.07 for value in row["rollout_logprobs"]]
        saved = [list(row["rollout_logprobs"]) for row in rows]
        args = SimpleNamespace(
            microbatch=2,
            temperature=0.8,
            max_logprob_error=0.2,
            policy_epochs=1,
            clip=0.2,
            loss_normalization="fixed",
            max_new_tokens=32,
            retention_weight=0.0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        metrics = r.policy_update(model, self.tok, rows, optimizer, args)
        self.assertEqual(saved, [row["old_logprobs"] for row in rows])
        expected = (
            -torch.exp(torch.tensor(-0.07)).item()
            * sum(len(x["response_ids"]) for x in rows)
            / (len(rows) * 32)
        )
        self.assertAlmostEqual(
            metrics["policy_losses"][0], expected, delta=0.02 if DEVICE.startswith("cuda") else 1e-4
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
