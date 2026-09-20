"""CPU-only tests for token accounting and actual TRL stage integration.

Run with CUDA_VISIBLE_DEVICES='' so no accelerator is touched. Tiny randomly
initialized models and artificial rewards here are mechanical fixtures only.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import modeling
import train
from protocol import judge, render_completion, render_prompt


class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = modeling.load_tokenizer(train.SPRINT / "models/olmo32b")
        cls.gold, cls.rl, cls.audit = train.load_training(train.SPRINT / "data")

    def test_completion_only_labels_and_matched_pool(self):
        records = train.encode_supervision(self.tokenizer, self.gold, 4096)
        self.assertEqual(len(records), 32)
        self.assertEqual({r["canonical_hash"] for r in self.gold}, {r["canonical_hash"] for r in self.rl})
        for gold, row in zip(self.gold, records, strict=True):
            answer = [t for t in row["labels"] if t != -100]
            self.assertEqual(answer[-1], self.tokenizer.eos_token_id)
            self.assertEqual(modeling.decode_completion(self.tokenizer, answer), render_completion(gold["proof"]))
            self.assertEqual(len(row["input_ids"]), len(row["labels"]))
            self.assertLessEqual(judge(gold["prompt"], modeling.decode_completion(self.tokenizer, answer))["n_lines"], 6)

    def test_default_protocol_seed_is_accepted_by_transformers(self):
        from transformers import set_seed
        args = argparse.Namespace(accumulation=4, learning_rate=2e-5, steps=80,
                                  save_steps=20, seed=202609092014)
        config = train.config_common(args, Path("/tmp/unused_runtime_seed_test"))
        self.assertEqual(config["seed"], args.seed % (2**32))
        self.assertEqual(config["data_seed"], config["seed"])
        set_seed(config["seed"])

    def test_constructor_cast_restores_exact_values_and_parameter_objects(self):
        import torch
        model = torch.nn.Module()
        model.register_parameter("lora_A", torch.nn.Parameter(torch.tensor([0.1234567, -0.000234567], dtype=torch.float32)))
        reference = model.lora_A
        before = modeling.adapter_digest(model)
        snapshot = train.snapshot_adapters(model)
        model.lora_A.data = model.lora_A.data.to(torch.bfloat16)
        self.assertNotEqual(modeling.adapter_digest(model), before)
        train.restore_adapters(model, snapshot)
        self.assertIs(model.lora_A, reference)
        self.assertEqual(modeling.adapter_digest(model), before)

    def test_fp32_checkpoint_loads_exactly_on_bf16_base_and_reference(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Olmo3Config, Olmo3ForCausalLM
        torch.set_num_threads(2)
        config = Olmo3Config(vocab_size=64, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, layer_types=["full_attention"])

        def bf16_base():
            return Olmo3ForCausalLM(config).to(torch.bfloat16)

        original = get_peft_model(bf16_base(), LoraConfig(task_type="CAUSAL_LM", r=2,
            lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
        with torch.no_grad():
            for name, parameter in original.named_parameters():
                if "lora_" in name:
                    self.assertEqual(parameter.dtype, torch.float32)
                    parameter.copy_(torch.linspace(0.000234567, 0.1234567, parameter.numel()).reshape(parameter.shape))
        with tempfile.TemporaryDirectory() as tmp:
            original.save_pretrained(tmp, safe_serialization=True)
            unsafe = PeftModel.from_pretrained(bf16_base(), tmp, autocast_adapter_dtype=False)
            for name, parameter in unsafe.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.float()
            with self.assertRaisesRegex(RuntimeError, "differs from exact checkpoint"):
                modeling.assert_adapter_matches_checkpoint(unsafe, tmp)
            # Reloading into pre-promoted existing destinations also recovers.
            recovered = modeling.load_adapter_exact(unsafe, tmp, trainable=True)
            self.assertTrue(modeling.assert_adapter_matches_checkpoint(recovered, tmp)["all_tensors_exact"])
            safe = modeling.load_adapter_exact(bf16_base(), tmp, trainable=True)
            default_before = modeling.adapter_digest(safe, "default")
            safe = modeling.load_adapter_exact(safe, tmp, adapter_name="reference", trainable=False)
            self.assertTrue(modeling.assert_adapter_matches_checkpoint(safe, tmp)["all_tensors_exact"])
            self.assertTrue(modeling.assert_adapter_matches_checkpoint(safe, tmp, "reference")["all_tensors_exact"])
            self.assertEqual(modeling.adapter_digest(safe, "default"), default_before)

    def test_marker_trim_preserves_sampled_token_boundary(self):
        from transformers import StopStringCriteria
        import torch
        text = "scratch\n<FINAL>\nN1 P : PR ;\nQED\n</FINAL>"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        padded = ids + [self.tokenizer.pad_token_id] * 7
        trimmed, reason = modeling.trim_completion(self.tokenizer, padded)
        self.assertEqual(reason, "final_marker")
        self.assertEqual(trimmed, ids)
        self.assertEqual(modeling.decode_completion(self.tokenizer, trimmed), text)
        criterion = StopStringCriteria(self.tokenizer, ["</FINAL>"])
        self.assertTrue(bool(criterion(torch.tensor([ids]), None)[0]))
        for end in range(1, len(ids)):
            self.assertFalse(bool(criterion(torch.tensor([ids[:end]]), None)[0]))
        eos = self.tokenizer.eos_token_id
        prefix = self.tokenizer.encode("scratch", add_special_tokens=False)
        self.assertEqual(modeling.trim_completion(self.tokenizer, prefix + [eos] + ids), (prefix + [eos], "eos"))
        self.assertEqual(modeling.decode_completion(self.tokenizer, prefix + [eos]), "scratch")

    def test_other_special_tokens_are_not_repaired(self):
        proof = render_completion("N1 P : PR ;\nQED")
        ids = self.tokenizer.encode(proof, add_special_tokens=False)
        non_eos = next(token for token, info in self.tokenizer.added_tokens_decoder.items()
                       if info.special and token not in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id))
        split = self.tokenizer.encode("<FINAL>\n", add_special_tokens=False)
        bad = split + [non_eos] + ids[len(split):]
        decoded = modeling.decode_completion(self.tokenizer, bad)
        self.assertIn(self.tokenizer.convert_ids_to_tokens(non_eos), decoded)
        self.assertFalse(judge("THM P SEQ P PRF", decoded)["accepted"])

    def test_literal_dpo_pair_validation(self):
        source = self.gold[0]
        rejected = "<FINAL>\nQED\n</FINAL>"
        pair = {"canonical_hash": source["canonical_hash"], "prompt": render_prompt(source["prompt"]),
            "chosen": render_completion(source["proof"]), "rejected": rejected,
            "sampled_response_ids": self.tokenizer.encode(rejected, add_special_tokens=False),
            "generation_policy": "sft_adapter"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            train.write_once(path, pair)
            # write_once pretty-prints a JSON object, not JSONL; use a one-line
            # stream here to model the pair writer itself.
            with path.open("w") as stream:
                stream.write(json.dumps(pair) + "\n")
            self.assertEqual(len(train.validate_pairs(path, self.gold, self.tokenizer, 4096)), 1)
            pair["rejected"] = "different text"
            with path.open("w") as stream:
                stream.write(json.dumps(pair) + "\n")
            with self.assertRaisesRegex(ValueError, "differs from generated"):
                train.validate_pairs(path, self.gold, self.tokenizer, 4096)

    def test_batched_pairs_resume_preserves_existing_samples(self):
        from types import SimpleNamespace
        rejected = "<FINAL>\nQED\n</FINAL>"
        ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        def generated(model, tokenizer, prompts, **kwargs):
            return [{"text": rejected, "response_ids": ids, "prompt_tokens": 10,
                "completion_tokens": len(ids), "post_stop_padding_positions": 0,
                "finish_reason": "final_marker", "seed": kwargs["seed"],
                "sampling": {"temperature": 1.0}} for prompt in prompts]

        with tempfile.TemporaryDirectory() as tmp:
            out, adapter = Path(tmp) / "pairs", Path(tmp) / "adapter"
            out.mkdir()
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"mechanical identity only")
            identity = modeling.file_sha(adapter / "adapter_model.safetensors")
            model = SimpleNamespace(_sprint_identity={"adapter_sha256": {"adapter_model.safetensors": identity}})
            args = SimpleNamespace(out=out, adapter=adapter, resume_pairs=False, max_pair_attempts=1,
                max_length=4096, max_completion_length=2048, deadline=None, pair_batch_size=2, seed=17)
            with patch.object(train, "sample_completions", generated):
                train.make_pairs(args, model, self.tokenizer, self.gold[:2])
            pairs_hash = modeling.file_sha(out / "pairs.jsonl")
            samples_hash = modeling.file_sha(out / "pair_samples.jsonl")
            args.resume_pairs = True
            with patch.object(train, "sample_completions", side_effect=AssertionError("Already paired theorem regenerated")):
                train.make_pairs(args, model, self.tokenizer, self.gold[:2])
            self.assertEqual(modeling.file_sha(out / "pairs.jsonl"), pairs_hash)
            self.assertEqual(modeling.file_sha(out / "pair_samples.jsonl"), samples_hash)

    def test_real_trl_cpu_stages(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Olmo3Config, Olmo3ForCausalLM
        torch.set_num_threads(2)
        torch.manual_seed(37)
        config = Olmo3Config(vocab_size=len(self.tokenizer), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=4096, layer_types=["full_attention"],
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            tie_word_embeddings=False)
        model = get_peft_model(Olmo3ForCausalLM(config), LoraConfig(task_type="CAUSAL_LM",
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
        model._sprint_identity = {"mechanical_cpu_fixture": True}
        fixture = {"prompt": "THM P SEQ P PRF", "proof": "N1 P : PR ;\nQED", "canonical_hash": "fixture"}
        original_common = train.config_common

        def cpu_common(args, out):
            values = original_common(args, out)
            values.update(use_cpu=True, bf16=False, gradient_checkpointing=False,
                          dataloader_pin_memory=False)
            return values

        with tempfile.TemporaryDirectory() as tmp, patch.object(train, "config_common", cpu_common):
            args = argparse.Namespace(steps=1, accumulation=2, group_size=2, learning_rate=0.01,
                save_steps=10, seed=91, deadline=None, max_length=4096, max_completion_length=2,
                dpo_beta=0.1, resume=None, max_logprob_error=0.1)
            sft = train.train_stage(args, model, self.tokenizer, [fixture] * 2, Path(tmp) / "sft", "sft", True)
            self.assertGreater(sft["nonzero_gradient_steps"], 0)
            self.assertGreater(sft["observed_parameter_changes"], 0)
            self.assertEqual(sft["status"], "complete")
            model.load_adapter(sft["adapter"], adapter_name="reference", is_trainable=False)
            model.set_adapter("default")
            before = modeling.adapter_digest(model, "reference")
            pair = {"prompt": render_prompt(fixture["prompt"]), "chosen": render_completion(fixture["proof"]),
                "rejected": render_completion("N1 Q : PR ;\nQED")}
            dpo = train.train_stage(args, model, self.tokenizer, [pair] * 2, Path(tmp) / "dpo", "dpo", True)
            self.assertGreater(dpo["nonzero_gradient_steps"], 0)
            self.assertGreater(dpo["observed_parameter_changes"], 0)
            self.assertEqual(modeling.adapter_digest(model, "reference"), before)
            model.delete_adapter("reference")
            args.accumulation, args.group_size = 16, 4
            args.steps = 2
            rl_fixtures = [{"prompt": f"THM {atom} SEQ {atom} PRF", "canonical_hash": f"fixture_{atom}"}
                           for atom in ("P", "Q", "R", "S")] * 8
            with redirect_stdout(io.StringIO()):
                grpo = train.train_stage(args, model, self.tokenizer, rl_fixtures, Path(tmp) / "grpo", "grpo", True)
            self.assertGreater(grpo["nonzero_gradient_steps"], 0)
            self.assertGreater(grpo["observed_parameter_changes"], 0)
            rollouts = train.read_jsonl(Path(tmp) / "grpo/rollouts.jsonl")
            self.assertEqual(len(rollouts), 32)
            self.assertEqual({r["reward"] for r in rollouts}, {0.0, 1.0})
            self.assertTrue(all(len(r["completion_ids"]) == 2 for r in rollouts))
            for offset in range(0, 32, 4):
                self.assertEqual(len({r["canonical_hash"] for r in rollouts[offset:offset + 4]}), 1)
            events = train.read_jsonl(Path(tmp) / "grpo/events.jsonl")
            likelihood = next(row for row in events if row["kind"] == "actual_grpo_likelihood_audit")
            self.assertLess(likelihood["max_abs_error"], 1e-4)
            self.assertTrue(likelihood["scorer_model_training"])
            fresh_audits = [row for row in events if row["kind"] == "actual_grpo_likelihood_audit"]
            loss_audits = [row for row in events if row["kind"] == "actual_grpo_gradient_likelihood_audit"]
            self.assertEqual(len(fresh_audits), 2)
            self.assertEqual({row["step"] for row in fresh_audits}, {0, 1})
            self.assertEqual(len(loss_audits), 32)
            self.assertTrue(all(row["scorer_requires_grad"] for row in loss_audits))
            self.assertTrue(all(row["max_abs_error"] < 1e-4 for row in loss_audits))


if __name__ == "__main__":
    unittest.main()
