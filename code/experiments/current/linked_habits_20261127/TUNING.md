---
written_on: 2026-11-27
written_by: agent:claude (tuning build agent)
---

# Stage-2 tuning trainers: DPO, fine-tune, reward model (build notes)

This note covers the Stage-2 trainers of [SPEC2.md](SPEC2.md), as changed by [PREREG.md](PREREG.md) Amendment 3
(the design audit, [discourse/design_audit.md](discourse/design_audit.md) F1, F2, F4, F5, F6):
- item 2, `lh_dpo.py` (the decisive arm, Amendment 3 point 1);
- item 3, the reward model only, `lh_rm.py` (the RL line that uses it is not built here);
- item 4, the fine-tune arm, `lh_sft.py`;
- the swap-random control of item 5, `lh_dpo.py --swap-random`.

Pair sampling (`lh_pairs.py`), twins (`lh_twin.py`), the launcher, measurement and pods are other agents' files.
Paths: W = this folder, LH = `/scratch/cowifornia/nd-rl/linked_habits`,
PY = `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python`.

## Files

| file | what |
|---|---|
| `lh_tune.py` | Shared code: loading, encoding a proof into (state, action) rows, scoring, the shared-row cancellation (`common_mask`, `split_common`), the chunked gradient, evaluation, the snapshot rule (`GRID`, `snap_plan`) and writer, and the loop. `lh_tune.py toy` builds smoke pairs from l19 (`--habit reit`: reiteration twins). `lh_tune.py probe-toy` builds a probe file from c24. |
| `lh_dpo.py` | The DPO arm and its swap-random control. |
| `lh_sft.py` | The fine-tune arm and its control: the same CLI, given another `--data` file. |
| `lh_rm.py` | `fit` (the reward model); `check` (load an RM.pt and score a pair file); `probe` (the RM on twins of other habits, Amendment 3 point 6a); `load(path, device)` for the RL line. |
| `test_lh_dpo.py`, `test_lh_sft.py`, `test_lh_rm.py` | CPU tests. |
| `tuning_smoke.sh` | The CPU smoke on the real base: one run after another, one thread each. |

## What every trainer shares (`lh_tune.py`)

- **Loading.** `tag_core.install_dan()`, then Dan's `model.load_ckpt` (13e41bbd). This is the loader that
  `authors_measure.run`, and therefore `lh_measure.py`, samples with. Offline (`ND_OFFLINE=1`). The CPU thread count
  is the smaller of `--threads` (at most 8) and `$OMP_NUM_THREADS`.
- **A proof as rows.** `state_env.decompose(prompt, proof, canon=True)`, as Stage-1's `state_train.load`. One row per
  step: state ids, action ids, `<eos>`. Every state is **untagged**: `<st> … <act>` with no tag token. The tokenizer's
  `author_tag` is `none` while encoding, whatever `$AUTHOR_TAG` the process has.
- **log pi(proof).** The sum over the proof's rows of log softmax(logits)[target] at the action tokens and the
  closing `<eos>`. Teacher forced, temperature 1. These are the target positions of Stage-1's loss
  (`state_train.loss_on`, `best_model.mtp_loss`).
  - Tests check this against `mtp_loss`'s main term, and against a step-by-step computation written separately (tiny
    model and real base).
  - No vocabulary mask: Dan's sampler samples the full softmax, and the GRPO sampler's `phase_bans` are empty for a
    `lean_staten_tag` tokenizer.
- **Shared rows cancel (`common_mask`, `split_common`).** A (state, action) row present in both members of a pair adds
  the same amount to log pi(w) and log pi(l). This holds under any model and any name offset applied to both. Shared
  rows are dropped as a multiset, and only the differing rows are computed.
  - For twins this covers every row before the first differing one: the notes and step order of a reiteration pair.
    In DPO those rows get **no direct gradient**; they can move only through shared parameters. This is what makes
    DPO the clean arm (Amendment 3 point 1, audit F2).
  - Canonical names make rows after a closed box coincide again, so they cancel too.
  - Tests pin the margins against the full sums, and pin the zero gradient (see Tests).
- **Name offset.** Stage-1 trained with a random `lean_seq` name offset, and Dan's sampler draws a name base per
  attempt. Training here draws s ~ U[0, 64 − largest name] per presentation:
  - DPO and the reward model: one offset per pair, the same for both members and all their steps;
  - SFT: one per row, as Stage-1.

  Every evaluation is at offset 0. `--no-shift` turns the offset off.
- **Exact gradient in bounded memory.** When a batch fits in `--max-tokens` padded tokens (default 65,536 on cuda,
  16,384 on CPU), one forward and backward pass. Otherwise:
  - a no-grad pass computes the per-row outputs;
  - autograd gives d loss / d output;
  - a second pass back-propagates chunk by chunk.

  SFT's loss is linear, so it skips the first pass. Tests check that the gradients equal the one-pass gradients.
- **Optimizer:** the RL recipe (`grpo.py`, `tag_line`). AdamW, lr 1.5e-4, betas 0.9 / 0.95, weight decay 0,
  grad-norm clip 1.0, a 10-step linear warmup, then a constant lr. bf16 autocast for training on cuda; every
  evaluation in fp32.
- **Seeds.** `--seed` sets the order of the training examples and the name offsets drawn from the same generator: a
  seeded shuffle per epoch, `random.Random(seed + 1)`. It also sets `torch.manual_seed(seed)`, which nothing here
  draws from: there is no dropout, the snapshots start from the base, and the reward head starts at zero.
  - It does **not** set the swap orientation (`--swap-seed`), the held-out or KL sets, or the evaluation.
  - On cuda, bf16 kernels can make two runs of the same seed differ slightly.
  - `done.json` repeats this in `seed.affects`.
- **Snapshots** are ordinary checkpoints: `{cfg, state, tok_mode, extra}`, exactly what `model.save_ckpt` writes,
  with the same state keys, cfg and `tok_mode` as the base.
  - `extra` is the base's own (author tags, Stage-1 steps, …, so `tag_core.ckpt_meta` reads the same mode and
    variant) plus `extra['tuning']` (trainer, base, step, snapshot name, metrics, inputs).
  - Beside each snapshot is a `.tag.json`: the base's sidecar with `ckpt`, `tuned_from` and `tuning` added.

## `lh_dpo.py` (the DPO arm; `--swap-random` is its no-preference control)

**What is optimized.** The reference pi_ref is the base, frozen: a second copy with no trainable parameter.
`done.json` `ref_frozen` is checked against the base file at the end. Per pair (w = `preferred`, l = `other`):

    loss = −log sigmoid(β · [(log pi(w) − log pi_ref(w)) − (log pi(l) − log pi_ref(l))]),  mean over the batch

- β 0.1, batch 64 pairs, 2,000 steps by default.
- Every policy parameter is trained.
- The positions are those above: action tokens and `<eos>`, untagged states, shared rows dropped.
- The reference's log-probabilities are recomputed every step, under the policy's name offset.

**Snapshots (Amendment 3 point 3, audit F4).** `snap_step<k>.pt` is written at every k in `--snap-steps`, default
**25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000**, and `final.pt` at `--max-steps`.
- Steps past `--max-steps` are dropped and listed in `done.json` `match.dropped_past_max_steps`.
- `--match-steps DIR` takes a companion run's snapshot steps instead. It is kept because the launcher still passes
  it; with the same grid in both runs it changes nothing.
- The accuracy-triggered snapshots (`snap_acc60/75/90.pt`) are **off by default**. `--acc-snapshots` turns them on,
  with `--snap-metric`.

**Metrics (`metrics.jsonl`).** A row is written at step 0, at **every snapshot step**, every `--eval-every` (25)
steps and at the end. Each row has:
- the train `loss`, `train_margin_rel` and `train_acc_rel`, averaged over the steps since the previous row;
- the held-out preference accuracy in **both definitions**:
  - `heldout_acc_abs`: the share of held-out pairs with log pi(w) > log pi(l). It is biased against the longer
    member: reiteration's A form has extra rows, each with log-probability < 0.
  - `heldout_acc_rel`: the share with [log pi(w) − log pi(l)] − [log pi_ref(w) − log pi_ref(l)] > 0. This is DPO's
    reward accuracy, relative to the base, and it is 0 at step 0 by construction.
- `heldout_margin_rel` and `heldout_logratio_abs` (the means of those two quantities, in nats) and `heldout_loss`;
- `kl`: the mean over the action positions of the `--kl-states` proofs of KL(pi_ref ‖ pi), in nats per action token,
  at T 1 and offset 0. This is `canary.anchor_kl`'s quantity, whose ban mask is empty here. It is computed at step 0,
  at every snapshot step and every `--eval-every` steps.

`done.json` `definitions` spells every key out. `train.jsonl` has one row per step (loss, margin, accuracy, grad
norm, rows, tokens).

**`--swap-random`.** In a random half of the TRAIN pairs (exactly floor(n/2)), preferred and other trade places.
- The half is drawn with **`--swap-seed`** (default 0), not with `--seed`. Seeds 0 and 1 of the swapped run
  therefore share one orientation and differ only in data order and offsets (`done.json` `swapped_sha256`; tested).
- The held-out pairs are never swapped, so the control reads movement toward the real rater's form.

**Other files in DIR.** `done.json` holds the inputs and their sha256, pair counts (unreadable, identical, swapped,
the share of rows shared within a pair), `base_heldout_acc_abs`, each snapshot with its step and metrics, the final
metrics, `definitions`, `seed`, `ref_frozen` and `swapped_sha256`. A folder that already holds `done.json` or
`metrics.jsonl` is refused.

## `lh_sft.py` (the fine-tune arm and its control)

**Data (Amendment 3 point 5, audit F1).** The launcher calls `lh_sft.py` twice per corpus and rater:
- with `sft_pref.jsonl`: the preferred members of all train pairs, unbalanced;
- with `sft_sampled.jsonl`: the sampled members of the same pairs, as the control.

Both are `{key, prompt, proof}` JSONL (extra fields are ignored) and go through `--data` **unchanged**; this is tested
with a file that carries extra fields. `done.json` `data_habits` gives the training file's per-habit sites, A, B and
A-share (read with `lh_profile`; fails soft with an `error` entry), to report beside the base's.

**What is optimized.** Stage-1's loss: next-token cross-entropy on the action tokens and `<eos>` of every step, given
the untagged state, averaged over the batch's action tokens (`mtp_loss`'s main term; tested equal).
- Every row of every proof is trained, including the rows before the rated site. This is why SFT is not the clean
  arm.
- The batch is `--batch` 64 whole proofs, with one name offset per row.
- **No MTP term.** The MTP head is not saved in the checkpoint, so it cannot be continued.

Snapshots, grid, metrics (both accuracies, KL), seeds and files are as in `lh_dpo.py`.

## `lh_rm.py` (the reward model of the GRPO arm)

**Network.** A copy of the base, every weight trainable except the LM head (which is not on the score's path), plus
a linear head d → 1 initialized to zero.
- Each step's row (state + action + `<eos>`) goes through the trunk.
- The step's score is the head on the final hidden state (after `ln_f`) at the `<eos>` that closes the action.
- **The proof's score is the mean of its steps' scores, minus a fixed offset.**

**Why every step and not the last state (SPEC2 says "the last state of a finished proof").** A state shows only the
hypotheses in scope. When a box closes its lines leave the state, so the last state of a proof that reiterates
inside a box is the same as its citing twin's. On an l19 record the final state is `n1 : <theorem> ⊢ <theorem>`.
`<eos>` is the one position of a row that has seen both the whole state and the whole action.

**Why the mean.** The A form of reiteration is longer. A sum of step scores could separate twins by length alone,
which in RL would reward any longer proof.

**Fit.** Bradley–Terry loss −log sigmoid(s(w) − s(l)), batch 64 pairs.
- The optimizer is the one above with lr 1.5e-4 on the trunk and `--head-lr` 1e-3 on the new head.
- One name offset per pair; `--seed` sets order and offsets.
- A fixed `--max-steps` (300). There is no early stopping and no selection on the held-out pairs.
- The **offset** is the mean over the train pairs of (s(w) + s(l)) / 2, so sigmoid(score) is above ½ for the
  average preferred proof.
- `RM.pt.json` is always written (pass flag, accuracies, score statistics, the curve, inputs and their sha256).
- `RM.pt` is written only when the held-out accuracy (s(w) > s(l), fp32, offset 0) is ≥ 0.90. Otherwise the model
  goes to `RM.pt.failed` and the command exits with **status 3**.

**Probe (Amendment 3 point 6a, audit F2).**

    lh_rm.py probe --rm RM.pt --pairs PROBE.jsonl --out probe.json [--reference HELD.jsonl] [--n-boot 2000]

PROBE.jsonl is in the pair format, with `habit` set per row and `preferred` = the A-form member of a twin pair of
some other habit. For each habit, `probe.json` gives:
- `n` (pairs whose two proofs both decompose; `unreadable_pairs` dropped);
- `a_higher_share`: the share of pairs where the A member scores higher;
- `mean_diff` = mean s(A) − s(B);
- 95 % bootstrap intervals over pairs for both;
- `ties`, `a_mean` and `b_mean`;
- `reward_driven` = `a_higher_share` outside [0.45, 0.55], the PREREG rule.

The same statistics on the rated habit's held-out pairs are given as `reference`; by default that is the held-out file
recorded in RM.pt.json. **The probe exits 0 whatever the numbers.** A missing or unreadable input file is still an
error.

**Use from the RL line.**

    import lh_rm                      # W on sys.path
    rm = lh_rm.load("RM.pt", "cuda")  # installs Dan's tree; works beside install_thinking (tested)
    rm.score(prompt, nd_proof) -> float;  rm.score_many([(prompt, nd_proof), ...]) -> [float, ...]
    rm.unreadable, rm.scored          # proofs scored 0.0 because they do not decompose / all proofs scored

`.score` never raises. An unreadable proof (parse failure, unknown token, empty, a non-string, a `LEANREJ` line)
scores 0.0 and is counted. Scoring uses bf16 autocast on cuda. `lh_rm.py check --rm RM.pt --pairs HELD.jsonl` loads
a file as the RL line would and prints the accuracy, score means, speed and garbage handling.

## Commands for the runner

    W=/scratch/cowifornia/nd-rl/worktrees/linked-habits/code/experiments/current/linked_habits_20261127
    # DPO and its no-preference control, seeds 0 and 1 (the control's orientation: --swap-seed, default 0)
    $PY $W/lh_dpo.py --base BASE.pt --pairs TRAIN.jsonl --heldout HELD.jsonl --out RUN/dpo_s$S --seed $S \
        --kl-states KL.jsonl --device cuda
    $PY $W/lh_dpo.py --base BASE.pt --pairs TRAIN.jsonl --heldout HELD.jsonl --out RUN/dposwap_s$S --seed $S \
        --swap-random --swap-seed 0 --kl-states KL.jsonl --device cuda
    # fine-tune arm and its control
    $PY $W/lh_sft.py --base BASE.pt --data sft_pref.jsonl    --heldout HELD.jsonl --out RUN/sft_s$S --seed $S \
        --kl-states KL.jsonl --device cuda
    $PY $W/lh_sft.py --base BASE.pt --data sft_sampled.jsonl --heldout HELD.jsonl --out RUN/sftctl_s$S --seed $S \
        --kl-states KL.jsonl --device cuda
    # reward model (exit 3 when held-out accuracy < 0.90: then no RM.pt), its check and its probe (exit 0)
    $PY $W/lh_rm.py fit --base BASE.pt --pairs TRAIN.jsonl --heldout HELD.jsonl --out RUN/rm.pt --device cuda
    $PY $W/lh_rm.py check --rm RUN/rm.pt --pairs HELD.jsonl --device cuda
    $PY $W/lh_rm.py probe --rm RUN/rm.pt --pairs PROBE.jsonl --out RUN/rm_probe.json --device cuda

- Defaults: β 0.1, lr 1.5e-4, batch 64, 2,000 steps with the grid (DPO, SFT), 300 steps (RM).
- `KL.jsonl` is a JSONL of `{prompt, proof}` corpus records; the first `--kl-n` (256) are used. Pass the same file to
  every arm of a corpus.
- The snapshot files to measure are `snap_step<k>.pt` (and `final.pt`). `metrics.jsonl` has a row at each k with
  both accuracies and the KL.
- Two jobs on one 24 GB card should pass `--max-tokens 32768` (not measured on a GPU).

## Tests

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PY -m pytest --confcutdir=. -p no:cacheprovider -q \
        test_lh_dpo.py test_lh_sft.py test_lh_rm.py

Observed: 29 passed, none skipped, in 3.5 min on one core (`taskset -c 6`, `OMP_NUM_THREADS=1`). Ruff check and
format check are clean on all seven Python files.

The tests use a tiny random ALiBi model (2 × 64) on toy pairs from l19: the same proof with every premise note in
form `orF` (preferred) or `For`. The twins have equal length, so a random model starts near a tie. The real base is
used for the log-probability check.

- **Scoring.**
  - States are untagged whatever `$AUTHOR_TAG` is.
  - log pi(proof) equals a separately written step-by-step sum (tiny model and real base).
  - Shared-row cancellation equals the full sums, at offset 0 and shifted.
  - Chunked gradients equal one-pass gradients (DPO and SFT forms).
  - SFT's loss equals `mtp_loss`'s main term.
  - Name offsets stay within n64.
- **The pin (Amendment 3 point 1).** For a reiteration twin pair, the trainer's loss gives exactly zero gradient to
  the logits of every row before the first differing row, in both members, while the first differing rows get a
  nonzero gradient. Without the cancellation those rows would get one, so the test can fail. The rows that enter the
  trainer's forward pass are exactly `common_mask`'s kept rows, and the margin equals `pair_margins`. It runs on two
  pairs:
  - a hand-written single-site pair, where the A member restates premise Q inside the inner box: the first 3 rows
    are shared;
  - a c24 record's `lh_twin` reiteration twin, with notes ahead of the site, when lh_twin and c24 are available.
- **DPO.**
  - The base's absolute accuracy is ≤ 0.5; the final absolute and relative accuracies are ≥ 0.9.
  - The loss starts at log 2 and falls below 0.3.
  - Every grid step has a snapshot, and a metrics row with both accuracies, the loss, the margin and the KL. No
    `snap_acc*` file is written by default.
  - The reference stays frozen and the base file is unchanged.
  - `--acc-snapshots` puts each threshold snapshot at the first passing evaluation, and gives step-0 snapshots equal
    to the base when the base already passes.
  - Swap-random: seeds 0 and 1 have the same swapped set and different data order; `--swap-seed 1` gives another
    set; the held-out relative margin stays under a tenth of the real run's; `--match-steps` still works.
  - An existing run is refused.
- **Snapshot loading.** Through Dan's `model.load_ckpt` after `install_dan` (what `authors_measure.run` and
  `lh_measure` use), with `tag_core.ckpt_mode` / `ckpt_meta` equal to the base's and the same keys, cfg and state
  keys. Also in a subprocess through the GRPO package's loader (`install_thinking`, `code.thinking.ckpt.load_ckpt`:
  same weights, `flat` state format).
- **SFT.** The loss falls, and the absolute accuracy goes from ≤ 0.5 to ≥ 0.9. The grid snapshots load,
  `data_habits` is filled, and a sampled-members file with extra fields trains unchanged.
- **RM.**
  - fit passes (held-out ≥ 0.9; sigmoid means on either side of ½);
  - `.score` is the centred mean of step scores;
  - six kinds of unreadable proof score 0.0 and are counted, and `$AUTHOR_TAG` has no effect;
  - 0 steps exits 3 in-process and as a CLI (`RM.pt.failed`, no `RM.pt`, `pass: false`);
  - overwriting is refused;
  - `load` works after `install_thinking`;
  - the probe reports per habit (a habit oriented with the RM is reward-driven high, a flipped one low, an
    unreadable pair is dropped), includes the reference block, and exits 0 as a CLI.

## Smokes on the real base (CPU)

The base is `two_authors/bases/stage1_authors_half_s0.pt` (the Nov 14 base, trained on no-notes proofs). Batch 8.

**Smoke 1 (before Amendment 3; the old threshold snapshot rule; 4 threads per run).** The toy pairs are the same l19
proof with every note in author A's form (preferred) or author B's. The base never saw notes, so these pairs are out
of distribution.

| run | steps | observed |
|---|---|---|
| DPO | 20 | absolute accuracy 0.60 (the base) → 0.725; relative accuracy 0.725; relative margin +44.9 nats; held-out DPO loss 1.00 (above log 2); KL 0.19 at step 10, 0.58 at step 20; 20,738 tokens per step (2,592 per pair; 18.5 % of rows shared); 1,265 s |
| swap-random | 20 | relative margin +0.30 nats; KL 0.069; absolute accuracy 0.60 → 0.50; 704 s |
| SFT on the A-note proofs | 20 (16 proofs per step) | CE 0.207 → 0.108 per action token; held-out relative margin −0.91 nats; absolute accuracy 0.575; KL 0.103; 851 s |
| RM fit | 24 | held-out 0.75 (train 0.82): exit 3, `rm.pt.failed`, no `rm.pt` (the fail path) |

**Reiteration twins (l19 `ta_proof` + `lh_twin`, A member preferred; 63 A-sampled and 97 B-sampled train pairs).**
These are in distribution for this base.

| run | steps | observed |
|---|---|---|
| DPO (old snapshot rule, 4 threads) | 20 | absolute accuracy 0.20 (the base prefers the citing form) → 0.95, crossing 60 / 75 / 90 % at steps 8 / 12 / 18; relative accuracy 1.0 from step 2; relative margin +16.1 nats; KL 0.016 / 0.083 / 0.231 at steps 8 / 12 / 20; 51.5 % of rows shared; 9,524 tokens per step (1,190 per pair); 870 s |

| run | steps | observed |
|---|---|---|
| RM fit (4 threads, pinned to 2 cores after the CPU limit) | 40 | held-out accuracy 0 (zero head) → 1.0 at step 20 (mean score difference 1.63) → 1.0 at step 40 (difference 4.93; train accuracy 1.0, loss 0.056). Passed; `rm_reit.pt` written. Offset −2.42. Centred held-out means +2.73 (preferred) and −2.20 (other); sigmoid means 0.91 and 0.18; score sd 2.97. 17,665 tokens per step (about 2,200 per pair, all rows); 1,753 s |

**Smoke 2 (after Amendment 3: `tuning_smoke.sh`; one thread, one run after another).**

All on the reiteration twins above (`LH/smoke_tuning/v2`): batch 8, 6 steps, grid `--snap-steps 3,6`, KL on 32 corpus
proofs. Every run exited 0 and wrote `snap_step3.pt`, `snap_step6.pt`, `final.pt` and a metrics row at 0, 3 and 6,
with both accuracies and the KL.

| run | observed (step 0 → 3 → 6) | secs |
|---|---|---|
| DPO seed 0 | `heldout_acc_abs` 0.20 → 0.30 → 0.425; `heldout_acc_rel` 0 → 1.0 → 1.0; relative margin 0 → +0.73 → +2.33 nats; KL 0 → 0.0005 → 0.0053; ref frozen | 313 |
| swap-random seed 0 (`--swap-seed 0`) | abs 0.20 → 0.175 → 0.25; rel 0 → 0.60 → 0.725; relative margin +0.15 → +0.75; KL 0.0002 → 0.0022 | 304 |
| swap-random seed 1 (`--swap-seed 0`, 3 steps) | the same `swapped_sha256` as seed 0 (1587733d…), different data order (other tokens per step); relative margin +0.23 at step 3 | 222 |
| RM probe (`rm_reit.pt` on 20 c24 twins per habit; reference: the 40 held-out reiteration pairs) | share of pairs with the A member higher (mean difference): reference 1.00 (+4.93); `restate` 0.00 (−3.65); `halves` 0.10 (−0.40); `disjunct` 0.90 (+0.04); `note_vcc` 0.60 (−0.48); `conj` 0.55; `note_atom` 0.50; `note_~a` 0.45. Flagged reward-driven: restate, halves, disjunct, note_vcc. Exit 0 | about 300 |
| SFT on `sft_pref` (the preferred members) | abs 0.20 → 0.275 → 0.40; rel 0 → 0.95 → 1.0; relative margin +0.52 → +1.60; KL 0.0003 → 0.0031. `data_habits`: reiteration A-share 1.00 over 352 sites | 244 |
| SFT on `sft_sampled` (the sampled members) | abs 0.20 → 0.20 → 0.175; rel 0 → 0.50 → 0.325; relative margin +0.05 → −0.11; KL 0.0018. `data_habits`: reiteration A-share 0.315; every other habit as in `sft_pref` | 280 |

[interpretation]
- In `restate` and `halves`, A writes form 2: the citing form and the single half. In all three habits where this
  reward model clearly prefers one member, it prefers the one with the extra or restated line. That fits a model
  that scores "R line or extra line" rather than reiteration as such, which is the audit's F2 concern. This smoke's
  base never saw notes or the version-2 habits, so this is a demonstration of the probe, not evidence about the
  experiment.
- The swap control's relative margin after 6 steps is a third of the DPO run's (+0.75 vs +2.33). That is too early
  to judge the control; smoke 1's 20-step comparison was +0.30 vs +44.9.

## GPU time on an RTX 4090 (estimates, not measured)

**Basis.**
- [obs] Tokens per pair from the smokes:
  - DPO: 1,190 differing-row tokens per reiteration pair, 2,592 per note pair;
  - RM: about 2,900 tokens per pair (all rows of both members);
  - SFT: about 1,400 tokens per proof.
- [obs] The Nov 14 base's Stage-1 ran 470.5 M tokens of forward + backward (+ MTP) in 1,319 s with `torch.compile`
  on its pod GPU (W2's pods were A40s). That is about 357 K tokens/s, or about 1.25 M forward-token equivalents per
  second (a backward counted as two forwards, MTP as 0.5).
- [guess] Eager PyTorch on a 4090 reaches 0.8–1.25 M forward-token equivalents per second in bf16, and 0.35–0.5 M
  tokens/s for fp32 evaluation.

**Estimates for 5,000 pairs.**

| job | work | time per run |
|---|---|---|
| DPO, 2,000 steps × 64 pairs | ref forward + no-grad forward + forward/backward = 5 × (76 K–166 K tokens) per step; 0.3–1.0 s per step. Evaluations at 0, the 12 grid steps and every 25 steps (about 80): 200 held-out pairs (0.24–0.52 M tokens, fp32) and 256 KL proofs (about 0.4 M tokens) each | **15–40 min** |
| swap-random | the same | 15–40 min |
| SFT, 2,000 steps × 64 proofs | 3 × 90 K tokens per step (one pass) | 10–20 min |
| RM fit, 300 steps × 64 pairs | 4 × 186 K tokens per step, plus the offset pass over 5,000 pairs (about 15 M tokens, fp32) | 5–8 min |
| RM probe, about 20 habits × ≤ 200 pairs | — | under 1 min |
| RM scoring inside GRPO | about 3,000 accepted proofs × 1,500 tokens per iteration | 4–6 s per iteration (+10–15 % on 38–45 s iterations) |

Per corpus and rater under Amendment 3 (DPO and swap × 2 seeds, SFT and its control × 1 seed): about 1.2–3.4 h of
GPU time. This is a large share of the PREREG budget. The reiteration smoke crossed 90 % absolute accuracy after 18
steps at batch 8, so most of the grid past a few hundred steps may be spent far beyond the +30-point read.

## Deviations from the brief, SPEC2 and the coordinator's instructions

1. **Snapshot rule.** The step grid (Amendment 3) replaces the accuracy thresholds of the original brief. The
   threshold snapshots remain behind `--acc-snapshots`. `--match-steps` is kept, now with any run's grid.
2. **Metric names.** "Preference accuracy" is reported in both definitions under new names: `heldout_acc_abs` and
   `heldout_acc_rel`. The earlier `pref_acc` and `margin_acc` names are gone, and `done.json`'s `base_pref_acc` is
   now `base_heldout_acc_abs`.
3. **Reward model read point.** Every step is read (the mean at each action's `<eos>`), not the last state; reasons
   above.
4. **SFT without the MTP term.** The MTP head is not saved, so it cannot be continued.
5. **Additions:**
   - name-offset augmentation in training, with evaluation at offset 0;
   - bf16 autocast in training on cuda;
   - the exact chunked gradient;
   - the flags `--snap-steps`, `--acc-snapshots`, `--snap-metric`, `--dense-every`, `--eval-every`, `--max-tokens`,
     `--no-shift`, `--warmup`, `--kl-n`, `--threads`;
   - `lh_rm.py check`, `lh_tune.py toy` / `probe-toy`, `tuning_smoke.sh`;
   - `.tag.json` sidecars, `ref_frozen`, `swapped_sha256`, `data_habits`, `definitions`.
6. **Optimizer beyond "the RL recipe's lr".** Betas, weight decay 0, clip 1.0 and the 10-step warmup are taken from
   GRPO too. The reward head has its own lr (1e-3).
7. **KL cadence.** The KL is computed at step 0, at every snapshot and every `--eval-every` steps, not at every
   step. Loss, margin and accuracy are logged per step in `train.jsonl`.
8. **Swap-random.** The held-out pairs are not swapped.
9. **Dropped pairs.** Pairs that do not decompose are dropped and counted, and more than 1 % of a file is an error.
   Pairs whose two proofs give identical rows are dropped and counted.
10. **The probe.** Its exit status is 0 except for unusable inputs (a missing file, an RM.pt that does not load).
11. **The CPU limit.** After it took effect, my two running smoke processes (one DPO run, one RM fit) were pinned
    with `taskset` to 1 and 2 cores and reniced to 5, rather than restarted.

## Known gaps and open risks

- **No GPU run.** Speed, memory (65,536-token chunks: about 6–7 GB bf16 estimated) and bf16 numerics on cuda are
  untested. The two-pass gradient path is tested on CPU only.
- **Over-optimization on the grid.** At lr 1.5e-4 and a constant schedule, DPO moved the reiteration pairs from 20 %
  to 95 % absolute accuracy in 20 steps of batch 8, while the KL rose to 0.23 nats per action token. With batch 64,
  2,000 steps is about 25 epochs of 5,000 pairs. If +30 points of rated movement comes within the first grid steps,
  the bracketing pair may be (0, 25) or (25, 50), with coarse interpolation. A finer early grid or a lower lr may be
  needed. That is the orchestrator's call; nothing was changed.
- **`heldout_acc_abs` versus measured A-share.** Both accuracies are teacher-forced, T 1 quantities on the held-out
  pairs. The readout's matched movement uses the sampled A-share (the probe at T 0.8), which these do not replace.
- **Reward model on reiteration twins.** It passed (1.0 on 40 held-out pairs, 40 steps of 8 pairs). The sigmoid
  means were already 0.91 and 0.18 after 40 steps, so with 300 steps of 64 pairs the score will be nearly a switch
  on the rated form (see the next point). On the note toy (out of distribution for that base), 24 steps reached only
  0.75 and exited 3.
- **The probe's coverage depends on the probe file.** Habits whose twins fail decompose are dropped per pair and
  counted.
- **Reward model in the GRPO process.** `lh_rm.load` puts Dan's 13e41bbd tree on `sys.path` (bare names `model`,
  `tokenizer`, `state_env`, …). It is tested beside `install_thinking`; any other bare `import model` in that process
  would get Dan's.
- **Reward-score scale.** Bradley–Terry on separable twins grows |s(w) − s(l)| without bound. sigmoid(score) can
  saturate, which would make the RL bonus close to a 0 / 0.3 switch on the rated form. `RM.pt.json` reports the
  score means and their sigmoid means.
- **`data_habits` reads with `lh_profile`** (another agent's file) and fails soft: an `error` entry, with no A-shares.
- **Toy smokes are not evidence about the experiment.** The smoke base is the Nov 14 base, not a c-corpus base; the
  pairs are l19 corpus proofs and their twins, not the base's own samples; batches are 8, not 64.
- **Temperature.** DPO uses the model's T 1 distribution, while measurement samples at T 0.8.
