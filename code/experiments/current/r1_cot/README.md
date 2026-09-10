---
written_on: 2026-09-09
---

# Symbolic scratchpad experiment code

`run.py` implements an R1-inspired experiment on the existing small natural
deduction transformer. It is not a pretrained language model and its scratchpad
is symbolic rather than natural-language reasoning. The three added tokens are
`<THINK>`, `</THINK>`, and `<FINAL>`. The model must generate these delimiters,
its scratchpad, and its proof without action masks or forced closure. Scratch
content is not verified; the literal final suffix is verified against the
original prompt, requiring a terminal QED and strictly backward citations.

The cold-start target reverses a supplied short proof's line order, keeping each
rule and local formula while removing citations and scope bars. It teaches a
backward goal/rule sketch. No long proof labels or held-out textbook contents
are used to construct these targets. SFT verifies every source proof and enforces
`--max-proof-lines` (default six), with no sequence truncation.
`--scratch-style dependency` instead uses scope-preserving goal-first dependency
plans that omit irrelevant lines; the default remains `reverse`.
The optional secondary `correction` style uses a 46-token shared parent prepared
with `--scratch-style correction`. It emits a deterministically corrupted draft,
`<ERROR><REVISE>`, a dependency plan, and the original correct final proof. Source
proofs remain cap-limited; rejected-draft provenance is saved for SFT and retention.
Draft verification happens only during cold-start data construction, with no
oracle feedback or forced revision during inference.

Required external dependencies are PyTorch, `nd_model.py`, `nd_grpo.py`
(for its length-based rollout collator), `nd_data.py`, `nd_metrics.py`, the
`nd_verify` package, and `strict_verify.py`. The prior source tree supplies these:

```sh
export PYTHONPATH=/data/users/dmitry/nd-rl/take-home:/data/users/dmitry/nd-rl/take-home/transfer_sprint_20260907/code
PY=/data/users/dmitry/simplex-research/.venv/bin/python
CODE=/data/users/dmitry/nd-rl/r1-cot-20260909/code
DATA=/data/users/dmitry/nd-rl/r1-cot-20260909/data
```

Each command requires a fresh output directory. No GPUs are dispatched by this
module; `CUDA_VISIBLE_DEVICES` is supplied by the parent dispatcher.

```sh
$PY "$CODE/run.py" prepare --checkpoint /path/to/cap6.pt --mode cot \
  --max-seq-len 4096 --output /path/to/shared_expanded_parent
$PY "$CODE/run.py" sft --checkpoint /path/to/shared_expanded_parent/final.pt \
  --mode cot --data "$DATA/sft_cap6.jsonl" --guard-json "$DATA/all_textbook_guards.json" \
  --output /path/to/cot_sft --steps 2000 --lr 0.00003 --batch-size 32 --microbatch 8
$PY "$CODE/run.py" rl --checkpoint /path/to/cot_sft/final.pt --mode cot \
  --data "$DATA/rl_with_textbook.jsonl" --guard-json "$DATA/heldout_guards.json" \
  --retention-data "$DATA/sft_cap6.jsonl" --retention-weight 0.1 \
  --output /path/to/cot_rl --steps 100 --lr 0.00001 \
  --group-size 8 --prompts-per-step 4 --max-new-tokens 768 --temperature 0.8
$PY "$CODE/run.py" eval --checkpoint /path/to/cot_rl/final.pt --mode cot \
  --data "$DATA/textbook_dev.jsonl" --output /path/to/cot_eval \
  --evaluation-release-json /path/to/paired_rl_release.json \
  --samples 8 --max-new-tokens 768 --temperature 0.8
```

Use `--mode direct` with the **same expanded parent** for the vocabulary-matched
direct SFT/RL controls. A fresh model is available with `prepare --fresh-config
/path/to/model_config.json` instead of `--checkpoint`; the config uses
`nd_model.ModelConfig` fields and does not load inherited weights.

Default policy training uses binary verifier reward, within-prompt leave-one-out
advantages, PPO clipping, and a fixed `max_new_tokens` loss divisor. Every sampled
token receives credit, including malformed tokens, sampled PAD, scratch, and
delimiters. The actual sampling-temperature behavior log probability is the PPO
denominator. Teacher forcing only measures numerical cache drift; discrepancies
above `--max-logprob-error` (default 0.5 log-probability units) abort the update.
Alternative standard GRPO behavior uses `--advantage standardized
--loss-normalization sequence`. `--policy-epochs` defaults to one. There is no KL
anchor; optional cap-limited SFT retention is the implemented stabilization.
`--precision fp32` disables autocast and TF32 and uses a float32 sampling cache;
the default `--precision bf16` uses BF16 autocast and cache on CUDA. Both modes
retain sampling probabilities and report teacher/cache discrepancies.

Training always requires `--guard-json`, checking canonical theorem, nonempty
premise context, and closed-formula skeleton hashes for every training/retention
prompt. Data from evaluation roles and sealed/final/test paths are rejected.
`--allow-unguarded-smoke` explicitly permits at most 16 fixture rows when no guard
is available. This option is for tests only. Dataset membership and pretrained
checkpoint exposure still need an external lineage audit.

Final evaluation paths require `--final-release-json` containing a parent-authored
sprint-wide freeze with `released: true`, `no_further_training: true`,
`not_before_utc: "2026-09-09T13:25:00Z"`, and `checkpoints`/`data_sha256` lists.
The release deadline cannot precede that absolute sprint deadline.
This validates membership before reading final rows. It is an accidental-access
guard, not an access-control security boundary. The parent decides when to issue
this release, freezes the full candidate list, and owns split integrity.

Textbook development evaluation requires `--evaluation-release-json` with
`checkpoints` and `data_sha256` lists plus `completed_rl_checkpoint` (path) and
`completed_rl_checkpoint_sha256`. The checkpoint hash and its neighboring
`completed.json` marker must match. This permits the baseline and SFT comparison
endpoints to be evaluated only once their paired RL run has completed. Synthetic
development evaluation has no release requirement.

`sft`/`rl` write `manifest.json`, `events.jsonl`, periodic `step_*.pt`, `final.pt`,
and `completed.json`. RL also saves every raw trajectory and behavior likelihood
to `rollouts.jsonl`. Evaluation writes `evaluation_freeze.json`, `attempts.jsonl`,
`summary.json`, and `completed.json`; it never updates the model. Each attempt
records `name`, `index`, zero-based `attempt`, original `prompt`, raw
`prompt_ids`/`response_ids`, `rollout_logprobs`, literal `body`, binary `reward`,
`found_len`, `scratch_tokens`, `final_tokens`, and `generated_tokens`.
Generated token counts include all three boundaries. The summary reports the
maximum shortest accepted written proof length, solve rate, and reference-length
bins. Written length alone is not a lower bound on proof difficulty; use the
analysis utility's compressed-proof and matched-token results for interpretation.

CPU tests: `python -m pytest -q code/tests/test_r1_*.py` from `nd-rl`.
The tests use only synthetic fixtures and randomly initialized tiny models.
