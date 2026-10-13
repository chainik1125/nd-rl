# shared_code

Reusable implementation shared across experiments.

| Module | What it is |
| --- | --- |
| `lean_seq` | ND proofs as Lean 4 tactic text, its tokenizer and inverse, and proof term size |
| `lean_check` | Lean 4 as the judge of `lean_seq` samples |
| `state_env` | The proof-state environment: a proof as (tactic state, action) steps |
| `step_check` | A check of one step in that environment, sound with respect to Lean |

## `state_env`: the proof-state environment

A whole-proof model writes a `lean_seq` proof in one pass. In the proof-state environment the
policy sees the tactic state of the focused goal and writes one step, the literal `lean_seq` text
of a `have` line, a box opening, or an `exact`; the environment applies it and renders the proof.

```python
from code.shared_code.lean_seq import LeanSeqTokenizer, text
from code.shared_code.state_env import Env, decompose

prompt = "THM ( P & Q ) , ( Q > R ) SEQ R PRF"
proof = "N1 ( P & Q ) : PR ; N2 ( Q > R ) : PR ; N3 Q : ANDE2 N1 ; N4 R : IMPE N2 N3 ; QED"

# A proof as (state, action) pairs, and the actions before each (for a history-conditioned arm).
steps, symbols, _ = decompose(prompt, proof, canon=True)
# <st> h1 : ( P ∧ Q ) <nl> h2 : ( Q → R ) <nl> n1 : ( P ∧ Q ) <nl> ... ⊢ R <act>  ->  have n3 : Q := n1 .2 ;

# An attempt, one sampled action at a time.
env = Env(prompt, canon=True, base=7, assign=True)
ok, reason = env.apply("have n3 : ( P ∧ Q ) := h1 ;".split())  # (True, ""); named n8 (base 7)
...
if env.done:
    source = LeanSeqTokenizer().statement(prompt) + " " + text(env.text)  # for lean_check
```

- **What it checks.** What the strict `lean_seq` grammar checks, so a finished attempt always
  parses: cited names in scope, `exact` on the focused goal's last line, premise lines first and in
  order, a box binder of the formula its `have` implies. A rejected action ends the attempt
  (`env.failed` holds the reason). There is no type check: `step_check` catches certain errors one
  step at a time, and Lean judges the finished proof.
- **Names.** By default the policy writes every name, and replaying a proof reproduces its
  `lean_seq` text byte for byte. With `canon=True` a step's new names are the largest index in scope
  plus one, from `base` (a function of the state, unlike `lean_seq`'s numbering by first
  appearance); `canonicalise` rewrites a proof that way. With `assign=True` the environment
  overrides the names a step introduces, as the fork sampled its `lean_staten` models (`base`
  drawn uniformly from 0..32 per attempt).
- **Model input.** `env.state_tokens()` is `lean_seq`'s symbols plus `<st>`, `<nl>`, `⊢` and
  `<act>`. Mapping them to ids is the model's tokenizer's job; the fork's `lean_state*` modes put the
  four tokens after `lean_seq`'s tactic symbols and before `h1`.

## `step_check`: a Lean-sound check of one step

`check_step(env, action)` takes the environment before a step the environment accepts. It returns
`None` if Lean may accept the step, or a reason (`"app-arg"`, `"and-elim"`, ...) if Lean is certain
to reject it, and so any proof that contains it. It types premise lines and the one-line terms
(`.1`, `.2`, `.elim`, application, `⟨ , ⟩`, `Or.inl`, `Or.inr`, `Classical.byContradiction`)
exactly as Lean does in this fragment, `¬ A` as `A → False` included.

It never rejects a step Lean accepts. In the take-home fork's run `guided-tts`, Lean checked it on
314,541 (state, step) pairs from model samples, each step as a standalone theorem: 0 false rejects,
0 misses. This port returns the fork's verdict on every one of them
(`tests/fixtures/proof_state/revalidate_guided_tts.py`), and `tests/test_step_check.py` re-checks
another 9,665 against Lean.

### In a sampler

The fork's guided sampler (`guided_sample.py`, `dan-pandori/nd-takehome` branch `dan`) redraws a
step that fails either check from the same state, instead of finishing an attempt that is already
lost:

```python
import copy

from code.shared_code.step_check import check_step


def next_state(env, draw, max_rejections=10):
    """One accepted step, or None once max_rejections steps have been rejected."""
    for _ in range(max_rejections + 1):
        action = draw(env.state_tokens())  # the model's next action, as symbols
        trial = copy.deepcopy(env)
        if trial.apply(action)[0] and check_step(env, action) is None:
            return trial
    return None
```

- The fork drew redraws without replacement (a rejected action's probability is removed before
  the next draw), and counted the rejection budget per attempt, not per step.
- The verdict depends only on the state and the action, so it can be cached.
- Report plain sampling beside guided (the fork's policy). Lean still judges every finished proof.

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

### Term size

`term_size(symbols)` is the size of the proof term Lean elaborates from a `lean_seq` text, by the
fork's definition: one per inference (an introduction or elimination, an application of a
hypothesis, a `fun` binder), so premise and reiteration lines are free. Report it beside line
counts: it does not depend on how a proof is split into lines. It matches Lean's own count on all
1,108 proofs of `tests/fixtures/proof_state/term_size.json`.

```python
from code.shared_code.lean_seq import proof_symbols, term_size

term_size(proof_symbols(body))  # an ND proof body
term_size(tok.tokens(ids)[0])  # a sample in the strict grammar
```

### Lean

Lean 4 core only, no Mathlib, no project. Install it with [elan](https://github.com/leanprover/elan).
Tested with Lean 4.34.1. `find_lean()` looks at `$LEAN`, then `PATH`, then `~/.elan/bin/lean`. The
tests that need Lean run under `--runslow` and skip when it is not installed.

### Compatibility

The vocabulary and its order are identical to the take-home fork's `lean_tok.py` (mode `lean_seq`), so
token ids from a model trained there mean the same here (`tests/fixtures/lean_seq_golden.jsonl`). The
fork's models used a different network class, so its weights do not load into `ProofModel`.
