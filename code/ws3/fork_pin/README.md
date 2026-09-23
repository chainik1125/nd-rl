# fork_pin: the fixed pre-trained checkpoint and the code that opens it

Workstream 3 holds pre-training fixed. The fixed model is Robbie's pre-training-only export
`runs/share_pretrain_4l_seed0/pretrain.pt` from `hf://robbiethompson2018/nd-rl-checkpoints`.
It is a raw state dict, so it needs the model class and tokenizer it was trained with. Those come
from Dan's fork `dan-pandori/nd-takehome` at commit `ab629c7d2cdf3dc50d471e302d14643ba0782749`,
which is also where its training data lives. This directory pins exactly that and nothing more.

| in git | fetched by `fetch.sh` (gitignored) |
| --- | --- |
| `model.py` (`GPT`: 4 layers, d 256, 8 heads, LayerNorm, interleaved RoPE, KV cache) | `data/train.jsonl.gz` and `data/heldout.jsonl`: the cap-6 split the checkpoint saw |
| `tokenizer.py` (`Tokenizer('abs')`: 99 tokens, absolute `N1`..`N64` indices) | `data/rl_targets.jsonl`, `data/transfer.jsonl`: the fork's ladder pools, true length 7 to 14 |
| `nd_verify/` (identical to upstream) | `data/validation_36*.jsonl`, `data/test_*_prompts.jsonl`: upstream targets |
| `checkpoint_source/` (`summary.json`, `pretrain.py`: the recipe that produced the weights) | `ckpt/pretrain.pt` |
| `MANIFEST.json`: commit, URLs, sha256 of every file above | |
| `load_pretrain.py`: loads the checkpoint and checks its hash | |

```bash
bash code/ws3/fork_pin/fetch.sh            # first time: clone at the pin, copy data, download ckpt, verify
bash code/ws3/fork_pin/fetch.sh --verify   # gate B check, no network
PYTHONPATH=code/ws3/fork_pin python3 code/ws3/fork_pin/load_pretrain.py
```

Rules:

- Nothing here is edited. A change to any hash in `MANIFEST.json` is a change to a fixed component
  and needs Charles.
- Score every proof through Dmitry's `strict_verify.py`
  (`code/experiments/current/r1_cot/vendor/strict_verify.py`), not through `nd_verify` alone.
- RL samples prompts only from `data/train.jsonl` (lengths 5 to 6) and `data/rl_targets.jsonl`.
  `heldout.jsonl`, `transfer.jsonl`, the targets, and the 72 textbook problems are never sampled.
- The expert-iteration harness, ladder scripts, `grpo.py`, and Lean tooling in the fork are not
  used. ws3 RL is TRL on top of an HF wrapper around `model.py`.

Checkpoint facts from `checkpoint_source/summary.json`: 3,210,240 params, 2999 steps at batch 256,
lr 1e-3, weight decay 0.1, 300 s on an A40, last training loss 0.070. No held-out evaluation was
shipped with the export; T1 in `docs/ws3/plan.md` computes it.
