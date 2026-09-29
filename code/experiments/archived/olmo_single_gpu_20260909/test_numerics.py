"""CPU mechanical regression for the production numerical policy.

Efficient CUDA attention is replaced by mathematical attention only inside these
CPU tests; separate real-GPU capacity/rollout audits are mandatory.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import modeling
import numerics
from diagnose_efficient_precision import full_fp32_efficient
from diagnose_likelihood import SPRINT, cached_scores, read_jsonl
from diagnose_precision import differentiable_teacher_scores
from protocol import render_prompt


class NumericalPolicyTest(unittest.TestCase):
    def test_helper_hash_is_transitively_committed(self):
        self.assertEqual(modeling.file_sha(Path(numerics.__file__)), modeling.EXPECTED_NUMERICS_SHA256)

    def test_production_matches_diagnostic_and_survives_adapter_reload(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from transformers import Olmo3Config, Olmo3ForCausalLM
        from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS
        self.assertEqual(torch.cuda.device_count(), 0)
        torch.set_num_threads(2)
        torch.manual_seed(193)
        tokenizer = modeling.load_tokenizer(SPRINT / "models/olmo32b")
        config = Olmo3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, layer_types=["full_attention"],
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        config._attn_implementation = "sdpa"
        model = get_peft_model(Olmo3ForCausalLM(config), LoraConfig(task_type="CAUSAL_LM",
            r=2, lora_alpha=4, target_modules="all-linear", lora_dropout=0.0))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        diagnostic_model = copy.deepcopy(model)
        initial_identity = modeling.adapter_digest(model)
        original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        policy = numerics.install_numerical_policy(model)
        self.assertEqual(policy["policy_id"], numerics.POLICY_ID)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], original_sdpa)
        self.assertEqual(modeling.adapter_digest(model), initial_identity)
        old_forward = model.get_base_model().forward
        self.assertEqual(numerics.install_numerical_policy(model), policy)
        self.assertEqual(model.get_base_model().forward, old_forward)
        saved = read_jsonl(SPRINT / "runs/grpo/rollouts.jsonl")
        rows = [saved[12], saved[4]]
        for row in rows:
            row["prompt_ids"] = tokenizer.encode(render_prompt(row["theorem"]), add_special_tokens=False)
        with full_fp32_efficient(diagnostic_model, backend="math"):
            reference = cached_scores(diagnostic_model, tokenizer, rows, 2)
            reference_teacher, _ = differentiable_teacher_scores(diagnostic_model, tokenizer, rows, backward=True)
        requested = []

        def cpu_kernel(backend, *args, **kwargs):
            requested.append(backend)
            self.assertEqual(backend, SDPBackend.EFFICIENT_ATTENTION)
            return sdpa_kernel(SDPBackend.MATH)

        with patch("torch.nn.attention.sdpa_kernel", side_effect=cpu_kernel):
            actual = cached_scores(model, tokenizer, rows, 2)
            actual_teacher, audit = differentiable_teacher_scores(model, tokenizer, rows, backward=True)
            self.assertTrue(audit["backward_exercised"])
            self.assertTrue(audit["gradient_checkpointing"])
            self.assertGreater(audit["gradient_norm"], 0)
            for actual_rows, reference_rows in ((actual, reference), (actual_teacher, reference_teacher)):
                maximum = max(abs(a - b) for x, y in zip(actual_rows, reference_rows, strict=True)
                    for a, b in zip(x, y, strict=True))
                self.assertLess(maximum, 1e-5)
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory, safe_serialization=True)
                underlying = model.unload()
                self.assertEqual(underlying.forward, old_forward)
                model = modeling.load_adapter_exact(underlying, directory, trainable=True)
                self.assertEqual(modeling.adapter_digest(model), initial_identity)
                self.assertEqual(model.get_base_model().forward, old_forward)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
                self.assertEqual(output.logits.dtype, torch.float32)
                self.assertEqual(model.get_base_model()._sprint_numerical_policy, policy)
        self.assertTrue(requested)


if __name__ == "__main__":
    unittest.main()
