"""CPU mechanical attention-semantics tests; GPU efficient kernel not exercised."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diagnose_efficient_precision import fp32_expanded_sdpa, full_fp32_efficient


class ExpandedAttentionTest(unittest.TestCase):
    def test_padding_causality_cached_decode_and_explicit_gqa_gradients(self):
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        self.assertEqual(torch.cuda.device_count(), 0)
        torch.set_num_threads(2)
        torch.manual_seed(91)
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        # None includes both unpadded equal-length prefill and cached one-token
        # decode: these are precisely where HF would request native GQA.
        cases = [(4, 4, "none"), (1, 7, "none"), (5, 5, "padding"),
                 (3, 7, "sliding"), (3, 7, "additive")]
        native = F.scaled_dot_product_attention
        for query_length, key_length, kind in cases:
            with self.subTest(kind=kind, query_length=query_length):
                query = torch.randn(2, 4, query_length, 8, requires_grad=True)
                key = torch.randn(2, 2, key_length, 8, requires_grad=True)
                value = torch.randn(2, 2, key_length, 8, requires_grad=True)
                mask = None
                if kind != "none":
                    positions = torch.arange(query_length) + key_length - query_length
                    allowed = torch.arange(key_length)[None, :] <= positions[:, None]
                    if kind in {"sliding", "additive"}:
                        allowed &= torch.arange(key_length)[None, :] > positions[:, None] - 3
                    mask = allowed[None, None].expand(2, 1, -1, -1).clone()
                    if kind == "padding":
                        mask[0, :, :, :2] = False
                    if kind == "additive":
                        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
                with sdpa_kernel(SDPBackend.MATH):
                    reference, _ = sdpa_attention_forward(module, query, key, value, mask, dropout=0.0, scaling=0.3)
                    reference_gradients = torch.autograd.grad(reference.square().sum(), (query, key, value), retain_graph=True)
                    calls = []

                    def inspect(*args, **kwargs):
                        calls.append(kwargs)
                        self.assertFalse(kwargs.get("enable_gqa", False))
                        self.assertEqual(args[0].shape[1], args[1].shape[1])
                        self.assertEqual(args[0].dtype, torch.float32)
                        return native(*args, **kwargs)

                    with patch.object(F, "scaled_dot_product_attention", side_effect=inspect):
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            actual, _ = fp32_expanded_sdpa(module, query, key, value, mask, dropout=0.0, scaling=0.3)
                    actual_gradients = torch.autograd.grad(actual.square().sum(), (query, key, value))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(len(calls), 1)
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
                for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients, strict=True):
                    self.assertTrue(torch.isfinite(actual_gradient).all())
                    torch.testing.assert_close(actual_gradient, reference_gradient, atol=3e-6, rtol=3e-6)

    def test_full_policy_restores_nested_wrappers_after_exception(self):
        import torch
        from transformers import Olmo3Config, Olmo3ForCausalLM
        from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS
        config = Olmo3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, layer_types=["full_attention"])
        config._attn_implementation = "sdpa"
        model = Olmo3ForCausalLM(config)
        original_forward = model.forward
        attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, "fixture exception"):
            with full_fp32_efficient(model, backend="math") as observation:
                output = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
                self.assertEqual(output.logits.dtype, torch.float32)
                self.assertGreater(observation["attention_mask_none_calls"] + observation["attention_mask_present_calls"], 0)
                raise RuntimeError("fixture exception")
        self.assertEqual(model.forward, original_forward)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, weights[name]))


if __name__ == "__main__":
    unittest.main()
