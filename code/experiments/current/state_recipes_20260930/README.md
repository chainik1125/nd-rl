---
written_on: 2026-09-30
written_by: agent:claude
---

# Does Dan's proof-state format help Robbie's pretraining recipes? (4 recipes × 3 arms × 3 seeds)

Robbie (2026-09-30): run Dan's state arms on his best recipe plus three of the simplified variants,
under his fast harness expert iteration, with the whole-proof control re-run on the same fork,
"to see how much this helps".

## Design

| factor | levels |
| --- | --- |
| recipe (`pretrain_<id>.py`) | **198** (incumbent, 681), **207** (simplification B + token-budget batches, 635), **201** (simplification B: no speed code, no MTP, fork init, 604), **204** (F: B with ALiBi on all heads, plain pre-LN, 548) |
| arm (`STATE_ARM`) | **C0** whole proof (`lean_seq`); **SN** proof state, environment-assigned names (`lean_staten`, Dan's SN-v2); **SH** proof so far + proof state (`lean_stateh`, Dan's SH) |
| seed | 0, 1, 2 |

The numbers in brackets are the recipes' own autoresearch dev metrics (C0, fork `51604b3`).

**Everything else is the frozen harness** (`harness_frozen.py`, sha `2aa52f23e7ec`): 300 s pretraining
on the cap-6 `train.jsonl` (sha256 `9c024cd9c622`), 4 EI rounds × 1,500 targets × k 32 at T 0.8, 600
fine-tune steps of 128 proofs, retain 20,000, dev metric = dev-half transfer theorems solved at
L_true ≥ 7 with 64 samples. `harness_state.py` is the frozen harness plus the arm switch; `diff
harness_frozen.py harness_state.py` is the whole change.

- **Fork:** every arm, C0 included, runs on Dan's `dan_best-state` commit `a005ad61`, which has the step environment, the
  `lean_state*` tokenizer modes and his ALiBi-aware KV-cache compaction. The judge there is Lean alone. The frozen
  harness judged with Lean ∧ nd_verify, and Dan's reviewer measured the difference at +0 to +7 accepts. The ladder
  pools and held-out files are byte-identical to `51604b3` (same git blobs). `train.jsonl` comes from `51604b3`,
  because `a005ad61` no longer tracks it.
- **State arms, pretraining:** every record is split into its (state, action) pairs before `pretrain()` starts its
  clock (`state_pairs.attach`, about 20 s, not charged; Dan's port does the same). The cap-6 file gives 774,945 pairs.
  Each recipe gets the smallest change that feeds it pairs: 12–17 changed lines, all in the data path (`diff` against
  `results/<id>/pretrain.patch` applied to 198). One example is one pair, loss on the action tokens, and one lean_seq
  name offset per pair on both halves.
- **State arms, RL:** sampling goes through `state_sample.env_generate` (max 256 tokens per step, 48 steps per
  attempt, 2,048 live attempts). The fine-tune trains on all pairs of 128 proofs per step (Dan's `state_train.batch`),
  micro-batched at 200k padded tokens with an unchanged gradient (tested: max |Δgrad| 2e-7). Found proofs that do
  not replay in the environment are dropped and counted (`mix_dropped`).
- **Alarm:** step-level sampling is slower, so `ND_TOTAL_SECS` is raised for every arm (value from the benchmark pod,
  below). RL is fixed in samples, so the alarm changes nothing that is measured.

## Caveats known before running

- 201 and 204 draw a fixed 128 *examples* per step. For the state arms that is 128 pairs, about 1/5 of a proof's
  worth of action tokens, so these two recipes see fewer target tokens per step than under C0. That is the recipe as
  written. 198 and 207 use token-budget batches and do not have this issue.
- SH inherits Dan's S-style random first-name offset, which cost his S seed 1 581 of 5,000 held-out attempts to
  out-of-scope names. SN fixes that.
- Dan's own `best-state` run (proposal 17) ports the 186 recipe to SN with his 8-round ladder. It is not comparable
  with these numbers, which use Robbie's 4-round harness EI.

## Predictions (written before any pod ran)

- **SN − C0 on 198:** +150 to +300 on the dev metric (C0 ≈ 680, so SN ≈ 830–980 of 1,108). Dan's T1 gain was
  +45 % at 8 rounds on a smaller model. Here the C0 baseline is already stronger.
- **SH vs SN:** SH ≤ SN, by 0–80. Dan saw no difference at T1, but SH carries the unbound-name failure.
- **Recipe spread:** the C0 spread across recipes (198 − 204 ≈ 130) shrinks under SN, to below 80. My reason: part of
  what the recipe chain bought (ALiBi/NoPE, length handling) is long-context behaviour that the Markov state removes.
- **201/204 under SN** lose more than 198/207 relative to C0, because of the 128-pair batches (caveat 1).

## Results

Pending.
