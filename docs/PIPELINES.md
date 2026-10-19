---
written_on: 2026-10-18
written_by: agent:claude
---

# Training pipelines

The group has two end-to-end pipelines. They use different models and tokenizers, so neither can
start from the other's checkpoints. Pick one by the kind of model you need. Checkpoints are in
[MODELS.md](MODELS.md) and data in [DATASETS.md](DATASETS.md).

| | whole-proof (Robbie) | proof-state with thinking (Leon) |
| --- | --- | --- |
| The model writes | a whole `lean_seq` proof in one pass | one proof-state step at a time, with a thinking block sized by an effort token |
| Pretraining | run 198's `pretrain.py` on Dan's cap-6 file with distractor premises, 600 s on one A40 | S1g3: Dan's best recipe on programmatic search traces over K12 proofs, 24,000 steps |
| RL | expert iteration: the loops harness recipe (runs 507, 509, 510) | GRPO: verifier-guided rollouts, length curriculum L2 → L3 |
| Judge | Lean only | Lean, plus `step_check` during rollouts |
| Code | [`code/experiments/current/long_rl_20261014`](../code/experiments/current/long_rl_20261014/README.md) | [`code/thinking`](../code/thinking), recipe in [RECIPE.md](../code/experiments/current/thinking_traces_20261005/RECIPE.md) |
| Best checkpoints | long RL v5, untied EI | `grpo_think_vs_mb_kkl0_L3` |
| Use it for | pretraining and architecture work; anything that needs a plain next-token prover | RL-method work, thinking and test-time compute |

Use the whole-proof pipeline for pretraining experiments: change `pretrain.py`, hold the data and
RL fixed, and compare dev solved.

## Whole-proof pipeline

Arm A of the long-RL experiment: run 198's untied six blocks, then the final expert-iteration
recipe. The shipped files are byte-identical to the `source/` of the v5 checkpoints.

1. **Data.** Download it from Hugging Face ([DATASETS.md](DATASETS.md#using-robbies-data)), or
   rebuild it:

   ```bash
   cd code/experiments/current/long_rl_20261014
   ND_FORK=<nd-takehome at 51604b3> python make_data.py pretrain data/pretrain.jsonl.gz
   ND_FORK=<nd-takehome at 51604b3> python make_data.py pools data/
   ```

   `pools` reads the certified theorems from `../long_proofs_20261013`. That folder holds the
   generators and the exact minimal-length certifier, `certifier_exact/certify_fast.py`, which
   compiles `fastmin.c` with clang on first use.

2. **Checks** (CPU): `python selftest_tok.py`, `python test_paths.py`, `python harness.py --smoke --out /tmp/smoke`.

3. **Train** on a RunPod A40 (`RUNPOD_API_KEY` set; pod helpers in
   [`../autoresearch/pod.py`](../code/experiments/current/autoresearch/pod.py)):

   ```bash
   python launch.py go untied ei 0                         # 600 s pretraining, 3 h RL, eval
   python launch.py go untied ei 0 --resume <dir> --rl-hours 4   # v5: 4 more RL hours on a finished run
   ```

   The remote side (`remote.sh`) clones Dan's fork at `51604b3` for the model, sampler and
   `nd2lean`, installs Lean 4.34.0, and runs `harness.py`. The tokenizer and the Lean-only gate
   come from `overlay/`.

   **Before launching, change `RUN` in `launch.py`.** It is still `"v5"`, and the checkpoint upload
   path is `runs/long-rl-20261014-<RUN>_<arch>-<rl>_s<seed>/`, so a new run with the old name would
   overwrite the v5 checkpoints.

Expected: about 230–250 dev theorems of 843 after 3 h of RL, 313–433 after 7 h. Seed spread at 3 h
is about ±30.

## Proof-state pipeline with thinking

See [RECIPE.md](../code/experiments/current/thinking_traces_20261005/RECIPE.md): exact commands,
every non-default flag, and the evidence. Leon made GRPO the group's primary RL method on
2026-10-16 and retired expert iteration.

## EI or GRPO

The two pipelines disagree on which RL method works, and the comparisons aren't like for like:

- In Leon's stack, the tuned GRPO recipe is level with EI on long generated sets, and ahead at
  pass@1 on Charles149.
- In the whole-proof stack (long RL v3–v4, 2026-10-14), a lightly tuned GRPO stayed far behind
  EI: dev 35–73 against 186–247. That GRPO borrowed Leon's PPO clip settings, but not his verifier-guided
  rollouts, curriculum or thinking.

Porting Leon's GRPO to the whole-proof model is an open experiment, not a supported path.
