"""DRAFT, NOT RUN (vLLM is not installed on serrano). In-process patch of vLLM 0.11.0's Olmo3 decoder layer: add a constant bf16 vector
to the residual stream after L layers, i.e. to the output of decoder layer index L-1, at every position (prompt and generated).
Same quantity as HF `hidden_states[L]` (1 <= L <= 63; L = 0 is an embedding edit and L = 64 is not available as a raw residual).

Use: STEER_LAYER=L STEER_VEC=<tensor file with the FINAL vector alpha*v, fp32 or bf16> python steer_sample.py <Dan's sample.py args>
The patch must be installed BEFORE `LLM(...)` is built: vLLM traces/compiles the model during engine start-up. With the default
fork start method the engine-core child inherits the patched class (vLLM/utils get_mp_context: fork unless CUDA is already
initialised in the parent); otherwise set VLLM_ENABLE_V1_MULTIPROCESSING=0.
"""

import os


def install():
    L = int(os.environ["STEER_LAYER"])
    vec_path = os.environ["STEER_VEC"]
    import torch
    from vllm.model_executor.models import olmo2
    from vllm.model_executor.models.utils import extract_layer_index

    vec = torch.load(vec_path, map_location="cpu").to(torch.bfloat16).reshape(-1)
    orig_init, orig_fwd = olmo2.Olmo2DecoderLayer.__init__, olmo2.Olmo2DecoderLayer.forward

    def init(self, *, vllm_config, prefix=""):
        orig_init(self, vllm_config=vllm_config, prefix=prefix)
        self._steer = extract_layer_index(prefix) == L - 1  # python constant: folded at trace time
        if self._steer:
            assert vec.numel() == vllm_config.model_config.hf_config.hidden_size
            # a buffer (not a Python constant) so the compiled graph and CUDA graphs read it from a fixed address
            self.register_buffer(
                "steer_vec", vec.clone().to(torch.get_default_device()), persistent=False
            )
            print(
                f"STEER_ACTIVE layer_out_index={L - 1} norm={vec.float().norm().item():.4f}",
                flush=True,
            )

    def forward(self, positions, hidden_states):
        out = orig_fwd(self, positions, hidden_states)
        if self._steer:
            out = (
                out + self.steer_vec
            )  # bf16 + bf16, broadcast over tokens; HF hook does the same add
        return out

    olmo2.Olmo2DecoderLayer.__init__ = init
    olmo2.Olmo2DecoderLayer.forward = forward
