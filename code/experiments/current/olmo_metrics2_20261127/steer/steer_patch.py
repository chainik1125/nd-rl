"""sample_wrap.py --patch module for M3: add a constant bf16 vector to the residual stream after layer l, in vLLM 0.11.0.

Route A of discourse/steer_feasibility.md: patch `Olmo2DecoderLayer` (vLLM serves Olmo3 through the Olmo2 classes) BEFORE
`LLM()` is built, so that torch.compile and CUDA-graph capture trace the patched forward.  The buffer lives on layer
l-1 (0-based module index) and is added to that layer's OUTPUT = the residual after l layers = HF `hidden_states[l]`,
at every position (prompt and generated).  One process = one vector.

    sample_wrap.py --patch steer_patch.py [--sidecar S] -- <Dan's argv>
env:
    STEER_LAYER        l (1..63)
    STEER_VEC          torch file written by steer_hf.py p3 (dict with "vec" = the FINAL alpha*v, fp32, and its
                       metadata) or a bare tensor; cast to bf16 here (the HF hook adds the same bf16 vector)
    STEER_INPROC=1     fallback: set VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine in the parent process) in case the forked
                       engine process does not see the patch
    STEER_ACTIVE_FILE  jsonl that the worker appends one STEER_ACTIVE record to (default: <sidecar>.steer_active.jsonl)
With STEER_LAYER unset the patch does nothing (stock model; prints STEER_INACTIVE).
A zero vector is supported (E0'): the add is x + 0, exact in bf16.
Each STEER_ACTIVE line (stdout of the worker, and the jsonl) carries layer, alpha, scramble seed, vector sha256, norm.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def load_vec(path):
    import torch

    obj = torch.load(path, map_location="cpu")
    meta = {}
    if isinstance(obj, dict):
        meta = {k: v for k, v in obj.items() if k != "vec"}
        vec = obj["vec"]
    else:
        vec = obj
    vec = vec.reshape(-1).to(torch.float32)
    sha = hashlib.sha256(vec.numpy().tobytes()).hexdigest()
    if "sha256" in meta and meta["sha256"] != sha:
        raise SystemExit(f"STEER_VEC {path}: stored sha256 {meta['sha256']} != recomputed {sha}")
    return vec, {
        "sha256": sha,
        "norm": float(vec.norm()),
        "alpha": meta.get("alpha"),
        "scramble_seed": meta.get("scramble_seed"),
        "vtype": meta.get("vtype"),
        "file_layer": meta.get("layer"),
    }


def install(
    layer: int,
    vec_path: str,
    olmo2=None,
    extract_layer_index=None,
    active_file: str | None = None,
    printer=print,
):
    """Patch the decoder layer class.  `olmo2` / `extract_layer_index` are injectable for the CPU stub test."""
    import torch

    if olmo2 is None:
        from vllm.model_executor.models import olmo2 as olmo2_mod

        olmo2 = olmo2_mod
    if extract_layer_index is None:
        from vllm.model_executor.models.utils import extract_layer_index as eli

        extract_layer_index = eli
    vec32, info = load_vec(vec_path)
    if info["file_layer"] is not None and int(info["file_layer"]) != int(layer):
        raise SystemExit(
            f"STEER_LAYER={layer} but the vector file was built for layer {info['file_layer']}"
        )
    vec = vec32.to(torch.bfloat16)
    cls = olmo2.Olmo2DecoderLayer
    if getattr(cls, "_steer_patched", False):
        raise SystemExit("Olmo2DecoderLayer already patched in this process")
    orig_init, orig_fwd = cls.__init__, cls.forward

    def init(self, *, vllm_config, prefix=""):
        orig_init(self, vllm_config=vllm_config, prefix=prefix)
        self._steer = (
            extract_layer_index(prefix) == layer - 1
        )  # python bool: constant at trace time
        if self._steer:
            hs = getattr(getattr(vllm_config.model_config, "hf_config", None), "hidden_size", None)
            assert hs is None or vec.numel() == hs, (
                f"vector has {vec.numel()} entries, hidden size {hs}"
            )
            dev = torch.get_default_device() if hasattr(torch, "get_default_device") else None
            # a buffer, not a constant: compiled graphs and CUDA graphs read it from a fixed address
            self.register_buffer(
                "steer_vec",
                vec.clone().to(dev) if dev is not None else vec.clone(),
                persistent=False,
            )
            rec = {
                "event": "STEER_ACTIVE",
                "pid": os.getpid(),
                "layer": layer,
                "module_index": layer - 1,
                "alpha": info["alpha"],
                "scramble_seed": info["scramble_seed"],
                "vtype": info["vtype"],
                "vec_sha256": info["sha256"],
                "norm": info["norm"],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            msg = "STEER_ACTIVE " + json.dumps(rec)
            if printer is print:
                print(msg, flush=True)
            else:
                printer(msg)
            if active_file:
                with open(active_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    def forward(self, positions, hidden_states):
        out = orig_fwd(self, positions, hidden_states)
        if self._steer:
            out = (
                out + self.steer_vec
            )  # bf16 + bf16 broadcast over tokens; the HF hook does the same add
        return out

    cls.__init__ = init
    cls.forward = forward
    cls._steer_patched = True
    return info


def install_from_env(env=None, **kw):
    env = os.environ if env is None else env
    if not env.get("STEER_LAYER"):
        print("STEER_INACTIVE (STEER_LAYER unset): stock model", flush=True)
        return None
    if not env.get("STEER_VEC"):
        raise SystemExit("STEER_LAYER set but STEER_VEC missing")
    return install(
        int(env["STEER_LAYER"]), env["STEER_VEC"], active_file=env.get("STEER_ACTIVE_FILE"), **kw
    )


def apply(args):
    """sample_wrap.py hook: called immediately before vLLM's LLM() is constructed."""
    if os.environ.get("STEER_INPROC") == "1":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if getattr(args, "sidecar", None) and not os.environ.get("STEER_ACTIVE_FILE"):
        os.environ["STEER_ACTIVE_FILE"] = args.sidecar + ".steer_active.jsonl"
    info = install_from_env()
    if getattr(args, "sidecar", None):
        spec = {
            "steer": info,
            "STEER_LAYER": os.environ.get("STEER_LAYER"),
            "STEER_VEC": os.environ.get("STEER_VEC"),
            "STEER_INPROC": os.environ.get("STEER_INPROC"),
            "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            "active_file": os.environ.get("STEER_ACTIVE_FILE"),
            "applied_in_pid": os.getpid(),
        }
        Path(args.sidecar + ".steer.json").write_text(json.dumps(spec, indent=1))
