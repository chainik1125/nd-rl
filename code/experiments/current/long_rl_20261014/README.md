---
written_on: 2026-10-14
written_by: agent:claude
---

# Long RL: the whole-proof pipeline

The pipeline has two stages:
1. **Pretraining.** Run 198's model (untied six blocks, 9.56M parameters) trains for 600 s on Dan's cap-6 proofs
   with distractor premises. These are proofs of ≤ 6 body lines.
2. **RL.** Expert iteration runs for 3 h on theorems of 7–80 body lines, using the recipe from the loops
   autoresearch (runs 507, 509, 510). Only Lean judges the proofs.

Lengths are **body lines**: nd_verify's `n_lines` minus the premise count. These files are byte-identical to the
`source/` of the `long-rl-20261014-v5_*` checkpoints ([MODELS.md](../../../../docs/MODELS.md)). How this pipeline
compares with Leon's GRPO pipeline: [PIPELINES.md](../../../../docs/PIPELINES.md).

| file | what |
| --- | --- |
| `harness.py` | the fixed protocol: data, budgets, Lean-only judging, body-line bands, the sampling oracle, eval |
| `pretrain.py` | the model, pretraining and both RL arms (`ND_RL=ei` or `grpo`); `ND_ARCH=untied` or `looped` |
| `overlay/lean_tok.py`, `overlay/lean_gate.py` | the fork's tokenizer widened to 10 atoms, h1..h10, n1..n160; the fork's gate with nd_verify removed (a sample counts iff Lean accepts it) |
| `make_data.py` | builds `data/`: the distractor pretraining file and the RL / dev / holdout pools |
| `remote.sh` | runs on a CUDA box: checks out Dan's fork at `51604b3`, installs Lean 4.34.0, runs the harness |

## Run

```bash
# data/: download from Hugging Face (DATASETS.md#using-robbies-data), or rebuild:
ND_FORK=<nd-takehome at 51604b3> python make_data.py pretrain data/pretrain.jsonl.gz
ND_FORK=<nd-takehome at 51604b3> python make_data.py pools data/      # reads ../long_proofs_20261013
ND_ARCH=untied ND_RL=ei bash remote.sh out/untied-ei_s0 --seed 0         # one A40: about 4 h
ND_ARCH=untied ND_RL=ei bash remote.sh out/s0_more --resume out/untied-ei_s0 --rl-hours 4   # v5: 4 more RL hours
python harness.py --smoke --out /tmp/smoke                               # CPU plumbing check
```

`make_data.py pools` drops any theorem that appears in the group's files. It looks for them in
`~/repos/nd-takehome` and `~/repos/nd-rl`'s `origin/charles/main`.

## Data

- **Pretraining:** 154,990 records, all Lean-checked. Each record's premise count is drawn uniformly from
  [its own, 10]. A distractor premise is rejected if it would make the premises inconsistent or allow a shorter
  proof.
- **Pools:** rl 18,439, dev 843, holdout 823.
  - Sources: certified long theorems (body ≥ 7) plus a short bridge of 3+-premise theorems, and for `rl` only,
    4,493 of Dan's ladder targets.
  - Split by canonical-key hash, capped per band: 2,500 in `rl` (before the ladder targets), 150 in dev and
    holdout.

## Results (dev solved at 64 samples, of 843)

| arm | 3 h RL (v2) | 7 h RL (v5) | L* at 7 h (longest length with 5+ solved) |
| --- | --- | --- | --- |
| untied EI, seeds 0 / 1 | 232 / 247 | 313 / 433 | 26 / 27 |
| looped EI (8 loops), seeds 0 / 1 | 246 / 246 | 376 / 372 | 27 / 26 |
| untied GRPO (v3, PPO clip, Muon), seeds 0 / 1 | 73 / 48 | | 10 (at 3 h) |

- Nothing at 35+ body lines is solved.
- At 3 h, two EI runs of the same recipe spread by about ±30.
- The GRPO arm is far behind EI here. The "EI or GRPO" section of [PIPELINES.md](../../../../docs/PIPELINES.md)
  explains why this doesn't contradict Leon's result.
