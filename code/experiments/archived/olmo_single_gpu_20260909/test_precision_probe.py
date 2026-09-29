"""CPU mechanical checks for standalone precision replay, not proof scores.

Run with CUDA_VISIBLE_DEVICES='' and the sprint interpreter. No real checkpoint
is loaded; the model is a tiny random OLMo with LoRA. Saved training actions only
provide variable-length, marker-ended token fixtures.
"""
import unittest

from diagnose_likelihood import SPRINT, cached_scores, read_jsonl
from diagnose_precision import candidate_precision, differentiable_teacher_scores
from modeling import adapter_digest, load_tokenizer
from protocol import render_prompt


class PrecisionProbeTest(unittest.TestCase):
    def test_default_candidates_replay_gradients_and_restore(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Olmo3Config, Olmo3ForCausalLM
        from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS
        self.assertEqual(torch.cuda.device_count(), 0, "Run CPU fixture with CUDA_VISIBLE_DEVICES=''")
        torch.set_num_threads(2)
        torch.manual_seed(123)
        tokenizer = load_tokenizer(SPRINT / "models/olmo32b")
        config = Olmo3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, layer_types=["full_attention"],
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        config._attn_implementation = "sdpa"
        model = get_peft_model(Olmo3ForCausalLM(config), LoraConfig(task_type="CAUSAL_LM",
            r=2, lora_alpha=4, target_modules="all-linear", lora_dropout=0.0))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        saved = read_jsonl(SPRINT / "runs/grpo/rollouts.jsonl")
        rows = [saved[12], saved[4]]
        for row in rows:
            row["prompt_ids"] = tokenizer.encode(render_prompt(row["theorem"]), add_special_tokens=False)
        identity = adapter_digest(model)
        attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        head = model.get_output_embeddings().forward

        def flags():
            return (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())

        original_flags = flags()
        for variant in ("head_fp32", "reduction_fp32", "sdpa_math", "head_attention_fp32"):
            with self.subTest(variant=variant):
                with candidate_precision(model, variant):
                    cached = cached_scores(model, tokenizer, rows, 2)
                    teacher, audit = differentiable_teacher_scores(model, tokenizer, rows, backward=True)
                maximum = max(abs(a - b) for x, y in zip(cached, teacher, strict=True)
                    for a, b in zip(x, y, strict=True))
                self.assertLess(maximum, 1e-4)
                self.assertTrue(audit["backward_exercised"])
                self.assertTrue(audit["gradient_checkpointing"])
                self.assertGreater(audit["gradient_norm"], 0)
                self.assertEqual(adapter_digest(model), identity)
                self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
                self.assertEqual(model.get_output_embeddings().forward, head)
                self.assertEqual(flags(), original_flags)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            with candidate_precision(model, "reduction_fp32"):
                raise RuntimeError("fixture interruption")
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
        self.assertEqual(flags(), original_flags)


if __name__ == "__main__":
    unittest.main()
