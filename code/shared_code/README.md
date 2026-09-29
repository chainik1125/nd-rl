# shared_code

Reusable implementation shared across experiments.

## `lean_seq`: ND proofs as Lean 4 tactic text

`lean_seq.py` renders natural-deduction proofs as Lean 4 core tactic proofs over a fixed 107-token
vocabulary, and parses model samples back to the ND proof they denote. `lean_check.py` is the judge:
a sample counts if and only if it parses in the strict grammar and Lean accepts the literal text.

```
THM ( P > Q ) , P SEQ Q PRF      ->  theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : P ) : Q := by
N1 ( P > Q ) : PR ; N2 P : PR ;  ->  have n1 : ( P → Q ) := h1 ; have n2 : P := h2 ;
N3 Q : IMPE N1 N2 ; QED          ->  have n3 : Q := n1 n2 ; exact n3
```

### In the ND experiment pipeline

`LeanSeqTokenizer` implements the tokenizer interface of `experiments/current/r1_cot/vendor/nd_model.py`
(`pad_id`, `eos_id`, `prf_id`, `len()`, `encode_prompt`, `encode_record`, `decode_body`,
`state_dict`), so the pipeline's training, sampling and checkpoint functions take it unchanged:

```python
from nd_model import ModelConfig, ProofModel, generate_bodies, save_checkpoint, train_steps

from code.shared_code.lean_check import judge
from code.shared_code.lean_seq import LeanSeqTokenizer

tok = LeanSeqTokenizer()
model = ProofModel(ModelConfig(vocab_size=len(tok), max_seq_len=1024))
# records: [{"text": "THM ... PRF <proof>"}, ...] or [{"prompt": ..., "proof": ...}, ...]
train_steps(model, tok, records, steps=6000)
bodies = generate_bodies(model, tok, prompts)  # ND bodies, or "LEANPARSE <reason>"
save_checkpoint("ckpt.pt", model, tok)
```

- **Judge with Lean, not with an ND verifier.** `generate_bodies` returns strings, so to judge the
  literal text, sample ids yourself (or re-encode an accepted ND body with `tok.encode_proof`) and
  call `judge(tok, prompts, sample_ids)`. It returns `Verdict(ok, reason, nd)` per sample, in one
  batched Lean run. Samples outside the grammar are rejected without running Lean.
- **Loading a checkpoint:** `nd_model.load_checkpoint` recognises only its own token formats. For a
  `lean_seq` checkpoint, use `tok = LeanSeqTokenizer.from_state_dict(ckpt["tokenizer"])` and
  `ProofModel(ModelConfig(**ckpt["config"]))`, then load the state dict.
- **Sequence length:** Lean text is longer than the ND token format. Set `max_seq_len` and
  `max_new_tokens` from your data (`max(len(tok.encode_record(r)) for r in records)`, plus a
  margin), and report the fraction of samples that hit the cap.
- **Name shift (optional):** the fork trained with `tok.shift_names(ids, rng)` applied to each proof
  per batch (a random offset on hypothesis names). The pipeline's `train_steps` encodes once, so it
  trains without the shift. On the fork the unshifted variant matched the shifted one on held-out
  proofs (0.939 vs 0.936 greedy).

### Lean

Lean 4 core only, no Mathlib, no project. Install it with [elan](https://github.com/leanprover/elan).
Tested with Lean 4.34.1. `find_lean()` looks at `$LEAN`, then `PATH`, then `~/.elan/bin/lean`. The
tests that need Lean run under `--runslow` and skip when it is not installed.

### Compatibility

The vocabulary and its order are identical to the take-home fork's `lean_tok.py` (mode `lean_seq`), so
token ids from a model trained there mean the same here (`tests/fixtures/lean_seq_golden.jsonl`). The
fork's models used a different network class, so its weights do not load into `ProofModel`.

## `pretrain_recipe`: autoresearch run 198's pretraining stage

`pretrain_recipe.py` is the best pretraining stage from Robbie's autoresearch loop (branch
`autoresearch-pretrain`, run `198-claude_lean-muon-no-rms-retest`): 681 of 1,108 dev theorems of true
length ≥ 7 solved after a fixed expert-iteration stage (3-seed mean), at ≤ 10M parameters and 300 s
of pretraining on one A40. It is a network, `RecipeModel`, and a training function, `pretrain`:

- **Network:** 6 layers × 384, 8 heads, MLP 1280. Four heads get an ALiBi distance penalty, four get
  no positional signal. Peri-LN blocks (a LayerNorm on each branch output as well as its input).
- **Training:** Muon with Polar Express coefficients on the hidden matrices and the readout, AdamW
  on the rest. A multi-token-prediction head used only while pretraining. Orthogonal init.
  Token-budget batches (length-sorted runs of ≤ 128 × 144 padded tokens). A random name shift per
  proof. Cosine learning rate over wall-clock, no warmup.

```python
from nd_model import generate_bodies, save_checkpoint

from code.shared_code.lean_seq import LeanSeqTokenizer
from code.shared_code.pretrain_recipe import load_checkpoint, pretrain

tok = LeanSeqTokenizer()
model, stats = pretrain(tok, records, budget_secs=300, val_records=val)  # records as above
bodies = generate_bodies(model, tok, prompts)  # same call signature as ProofModel
save_checkpoint("ckpt.pt", model, tok)
model, tok, meta = load_checkpoint("ckpt.pt")  # nd_model.load_checkpoint only builds ProofModel
```

- **The loop's own checkpoints load:** `load_autoresearch_checkpoint(path)` reads a loop `final.pt`
  (e.g. `runs/198-claude_lean-muon-no-rms-retest_s0/final.pt` on
  [Hugging Face](https://huggingface.co/robbiethompson2018/nd-rl-checkpoints)). On 198 seed 0, its
  logits are bit-identical to the loop's own model code, and `generate_bodies`' cached, left-padded
  decoding matches uncached greedy decoding.
- **Budget is wall-clock:** the cosine schedule runs over `budget_secs`, as in the loop.
- **Not ported:** the loop's per-block `torch.compile` and CUDA-graphed Muon step (+23 % steps on torch
  2.9.1, no measured score change).
- **Tuned for one regime:** the constants were hill-climbed at 10M parameters, 300 s and cap-6 data on
  one dev metric. They are untested at other sizes and data.
