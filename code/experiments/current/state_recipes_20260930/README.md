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

## What changed during the run (2026-09-30)

- **Env batch.** At Dan's 2,048 live attempts every state arm OOMed an A40: the ALiBi recipes' prefill builds a
  (batch, heads, L, L) fp32 bias, and every step re-encodes the state. Relaunched at 512 (SN) / 256 (SH), then with a
  sticky halve-and-redo on any OOM (`env_ooms` in `summary.json`). This is a sampling re-draw, not a protocol change.
- **Compute-matched control, added after the first SN result: C0k192.** C0 with k 192 RL samples per target per round
  instead of 32, so its RL stage takes roughly SN's wall-clock. Fine-tune steps and the eval (k 64) are unchanged. It
  is under-matched: median RL stage 28–31 min, against SN's 33–41 min at batch 512.

## Results

Dev metric (dev-half transfer theorems solved at L_true ≥ 7, 64 samples), seeds 0 / 1 / 2. Medians, because
whole-proof EI sometimes fails to take off (\*). RL = RL targets solved of 1,500, median. Time = median wall-clock
of the whole run and of its RL stage, on A40s.

| recipe | arm | seeds | median | RL | run / RL stage |
| --- | --- | --- | --- | --- | --- |
| 198 | C0 | 669 / 692 / 216\* | 669 | 1,169 | 17 / 10 min |
| 198 | C0k192 | 743 / 345\* / 682 | 682 | 1,206 | 37 / 30 min |
| 198 | **SN** | 852 / 824 / 798 | **824** | 1,274 | 58 / 41 min |
| 198 | SH | 708 / 744 / lost† | 726 | 1,183 | 161 min |
| 207 | C0 | 612 / 304\* / 616 | 612 | 924 | 17 / 10 min |
| 207 | C0k192 | 713 / 625 / 396\* | 625 | 747 | 37 / 30 min |
| 207 | **SN** | 781 / 777 / 814 | **781** | 1,225 | 57 / 40 min |
| 207 | SH | 665 / 676 / 687 | 676 | 1,138 | 120 min |
| 201 | C0 | 609 / 638 / 579 | 609 | 1,090 | 18 / 10 min |
| 201 | C0k192 | 674 / 678 / 626 | 674 | 1,183 | 38 / 31 min |
| 201 | **SN** | 723 / 720 / 720 | **720** | 1,181 | 55 / 40 min |
| 201 | SH | 594 / 672 / 602 | 602 | 1,046 | 155 min |
| 204 | C0 | 566 / 552 / 571 | 566 | 1,052 | 16 / 9 min |
| 204 | C0k192 | 610 / 642 / 629 | 629 | 1,131 | 35 / 28 min |
| 204 | **SN** | 705 / 744 / 790 | **744** | 1,205 | 46 / 33 min |
| 204 | SH | 660 / 702 / 658 | 660 | 1,136 | 146 min |

† 198 SH seed 2 OOMed on the first harness. Its relaunch was lost after 197 min when this Mac's network dropped and
the launcher tore the pod down. It was not re-run.

Observations:

1. **SN beats C0 on every recipe:** medians +155 (198), +169 (207), +111 (201), +178 (204). The pre-registered
   +150 to +300 on 198 holds at the low end.
2. **Most of it survives compute matching:** SN − C0k192 is +142 / +156 / +46 / +115. Six times the RL samples buys
   whole-proof EI only +13 to +65. The control is under-matched by about a quarter of SN's RL time (above), so these
   gaps are upper bounds on the state's advantage at equal compute. Given C0's slow gain from 6× samples, the
   remaining 25 % is unlikely to close them.
3. **Reliability:** whole-proof EI failed to take off in 4 of 24 runs (216, 304, 345, 396). SN took off in 12 of 12.
4. **The recipe spread did not shrink** (prediction not supported): SN 824 → 720 (104), C0 669 → 566 (103). The
   ordering changed only among the simplifications. 198 is best in both formats.
5. **201 gains least** (+46 over C0k192). It is one of the two recipes with 128-example batches (pre-registered
   caveat 1). The other, 204, gains +115, so the caveat does not explain it alone.
6. **Cost:** SN runs take 2.7–3.4× C0's wall-clock (sampling about 4×, fine-tune about 2.6×, because a proof is
   about 5 pairs). The slowdown is mostly the step loop, and it is engineering, not intrinsic: Rounds at Dan's 2,048 batch
   took 429–492 s, against 587–649 s at 512 (198 benchmark), before it OOMed.

7. **SH is dominated by SN:** below SN on every recipe (−98 / −105 / −118 / −84 medians) at 2.5–3× SN's wall-clock
   (98–249 min per run). On 201 it is no better than whole-proof C0. SH re-encodes the whole proof so far every step,
   so its context grows with the attempt: the weak recipes' failing attempts reached L of about 4,000 and forced the
   env batch down to 32. This matches Dan's state-env result, where SH was no better than S. Giving the model the
   history on top of the state hurts, or at best costs time for nothing.
