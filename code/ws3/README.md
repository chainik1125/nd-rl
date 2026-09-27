# ws3: RL on the fixed checkpoint with TRL

Code for workstream 3 (`docs/ws3/plan.md`). The fixed pre-trained model and its data live in
`fork_pin/` and are never edited; everything else here is the RL pipeline around them.

```
ws3/
  ndpath.py            puts fork_pin/ and Dmitry's strict verifier on sys.path; import it first
  hf_wrapper.py        plan T2: the fork's GPT as a PreTrainedModel (with its KV cache wired into
                       HF generate) + the abs tokenizer as a fast HF tokenizer
  eval/                plan T3, importable by any track
    reward.py          reward(prompt, completion) -> 0.0/1.0 through strict_verify; TRL-callable ProofReward
    evaluator.py       avg solve rate (pass@1) by length, textbook slices, T1 next-token accuracy, sequence
                       log-probs, known-proof log-prob, variants eval (plan section 5)
    metrics.py         avg CI, reliable fraction, frontier, k50, AUROC, per-problem base vs RL from dumps
    progress.py        valid_prefix: leading lines that pass the verifier's per-line checks
    variants.py        atom renamings and premise orders of evaluation problems, consistency
  data.py              prompt pools from fork_pin/data, renaming-class leak check, the 72 textbook problems
  metrics_callback.py  metrics.jsonl (plan section 4 set) + periodic pass-rate-by-length eval
  run.py               the driver: stages baseline, train, evaluate, report
  rl/                  RL methods from docs/ws3/rl_methods_brainstorm.md, on top of TRL
    advantages.py      advantage transforms: maxrl, w_reinforce, virtual_positive, pass_at_k, unlikely
    curriculum.py      per-prompt success stats + adaptive sampler (p(1-p), GRESO skip, ladder, easy replay)
    trainer.py         WS3GRPOTrainer: GRPOTrainer + the two hooks above + group-outcome metrics
    ei.py              expert iteration (sample, verify, SFTTrainer, repeat) for run.algo: ei
  configs/smoke_cpu.yaml  plumbing check, ~1 minute on a laptop
  configs/presets/     one YAML fragment per method, combined with `include:`
  textbook72/          Dmitry's 58 + 14 textbook problems with a hash MANIFEST (evaluation only)
  fork_pin/            the pinned checkpoint lineage (see its README)
```

## Run

```bash
cd code
bash ws3/fork_pin/fetch.sh --verify                     # hashes of code, data, checkpoint
uv run pytest tests/test_ws3_*.py                        # gates T2, T3 and the smoke pipeline
uv run python ws3/run.py --config ws3/configs/smoke_cpu.yaml
uv run python ws3/run.py --config ../experiments/ws3/<exp>/config.yaml            # all stages
uv run python ws3/run.py --config ../experiments/ws3/<exp>/config.yaml --stages report
```

`run.out_dir` in the YAML is relative to the YAML, so an experiment's outputs land in
`experiments/ws3/<exp>/run/`. Under it:

| file | stage | what |
| --- | --- | --- |
| `baseline/{nta,greedy_heldout,textbook,long_pool,summary}.json`, `baseline/dumps/*.jsonl` | baseline | frozen-checkpoint control: T1 gate, greedy by length, textbook 72 at k, long-pool pass@k; every judged proof dumped |
| `run_manifest.json`, `prompts.jsonl` | train | resolved config, checkpoint sha256, git commit, versions; the exact RL prompt set with renaming keys |
| `metrics.jsonl` | train | one JSON line per optimizer step with `utc`, `step`, reward, reward_std, kl, entropy, completion lengths, clipped ratio, unparseable/invalid fractions, grad_norm, learning_rate, completions/sec; periodic `eval/pass_rate_len<L>` lines |
| `completions.jsonl`, `eval_periodic.jsonl` | train | every judged rollout (prompt, body, reward, verifier reason); periodic eval dumps |
| `hf_base/`, `final/`, `trainer/` | train | frozen start model (TRL's reference model source), trained model, TRL scratch. Gitignored |
| `evaluate/*.json`, `evaluate/dumps/*.jsonl`, `evaluate/comparison.json` | evaluate | the same evaluations on `final/` at equal attempts, side by side with the control |
| `report/report.md`, `report/*.png` | report | metric coverage check (gate B), curves, pass-rate-by-length bars, tables |

## Config

```yaml
run: {out_dir: run, seed: 0, device: cpu, algo: grpo}     # algo: grpo | rloo
eval:                                                      # shared by baseline and evaluate
  nta: {n_heldout: 5000, batch_size: 250, gate: 0.95}      # T1 gate, baseline only
  greedy: {n_heldout: 1000, max_new: 400, batch_size: 100}
  textbook: {k: 64, temperature: 1.0, max_new: 400, batch_size: 256}
  long: {n_prompts: 200, k: 32, temperature: 1.0, max_new: 400, batch_size: 256}
train:
  prompt: {length_weights: {3: 1, 4: 1, 5: 1}, n_prompts: 256}   # 2-6 from train.jsonl, 7+ from rl_targets
  periodic_eval: {every: 10, n_heldout: 50, n_long: 20, max_new: 400, batch_size: 128}
  trainer:            # any GRPOConfig / RLOOConfig field; overrides TRAINER_DEFAULTS in run.py
    num_generations: 8
    per_device_train_batch_size: 32
    gradient_accumulation_steps: 8   # completions per optimizer step = 32 x 8 = 256 = 32 prompts x group 8
    learning_rate: 1.0e-5
    beta: 0.01
    temperature: 1.0
    max_completion_length: 400
    max_steps: 50
```

### Methods (`rl/`, `configs/presets/`)

A config may start with `include: [paths]`; the files are deep-merged in order, then the
config's own keys on top. Each preset is one method, so an experiment is a list of includes:

```yaml
include: [../../../code/ws3/configs/presets/sparse_pack.yaml,
          ../../../code/ws3/configs/presets/maxrl.yaml]
run: {out_dir: run, seed: 0, algo: grpo}
train:
  prompt: {length_weights: {7: 1, 8: 1, 9: 1}, n_prompts: 1024}
  advantage: {name: maxrl}          # trl (default) | maxrl | w_reinforce | virtual_positive | pass_at_k | unlikely
  curriculum: {weighting: p1mp}     # unset = TRL's RepeatSampler; see rl/curriculum.py for all keys
  reward: {unparseable_reward: 0.0} # -0.1 = the "invalid vs wrong" shaped variant (advantage trl only)
```

`run.algo: ei` runs expert iteration instead (keys in `rl/ei.py` and `presets/ei.yaml`); the
baseline, evaluate and report stages are unchanged, and `final/` is the last round's model.

Extra metrics in metrics.jsonl for GRPO: `groups/frac_all_fail`, `groups/frac_all_success`,
`groups/frac_mixed`, `rollout/total_completions` (for matched-budget comparison with EI, which
logs the same key per round), `adv/*` when a transform is on, `prompts/*` and `curriculum/*`.
With k > 1 the evaluator also reports `pass_at_k` (unbiased, k = 1, 2, 4, ... up to the samples
drawn) overall and per length, and `distinct_verified_mean`. Set `eval.textbook.k` and
`eval.long.k` to 256 to get the full curve.

## Design notes

- **Reward uses `completion_ids`, not the decoded text.** TRL decodes completions with
  `skip_special_tokens=True`, which strips the `QED` end token the verifier requires. `ProofReward`
  rebuilds the exact body from the ids with the fork tokenizer.
- **KV cache.** The wrapper manages its own cache (`NDCache` over the fork GPT's per-layer dicts).
  transformers' `generate` lets a model do that when `_supports_default_dynamic_cache()` is False:
  the cache the forward returns in `past_key_values` comes back on the next step and `input_ids`
  is sliced to the newest token. Measured on an M1 Pro: ~2 completions/s without the cache,
  50 to 110 with it. `use_cache=False` (what TRL passes for the training forward) still
  recomputes the full prefix.
- **Reference model.** With `beta > 0` TRL reloads the reference model from
  `model.config._name_or_path`; `save_base()` writes `hf_base/` and points the config at it. The
  wrapper is registered with `AutoConfig` / `AutoModelForCausalLM` so that works without remote code.
- **Proof indices.** The checkpoint was trained with random index shifts, so it may start a proof
  at `N23`. The verifier accepts any start index; nothing here assumes `N1`.
- **Dead token.** The fork's vocabulary spells the atom `R` and the reiteration rule `R` the same,
  so one of the 99 ids is never produced. `build_tokenizer` mirrors this exactly (`<dead_16_R>`).
- **Why RLOO and not PPO.** TRL 1.13 has no PPOTrainer with a learned value head. `RLOOTrainer`
  is TRL's on-policy trainer with the PPO clipped objective, a KL penalty to the start model, and a
  leave-one-out baseline instead of a value network; it takes the same reward and dataset as
  GRPO. A value-head PPO would mean leaving TRL, which is stop-and-ask in `CLAUDE.md`.
- **Never sampled for training:** `heldout.jsonl`, `transfer.jsonl`, `validation_36`, the test
  files, and the 72 textbook problems. `data.assert_no_leak` checks every prompt set by renaming
  class before a run starts.
