---
written_on: 2026-09-30
written_by: agent:claude
---

# state-readouts: a state model that writes its own names reaches 25–28 of the 29 survivors; SN-cap12 moves 11 of the 12 dead textbook schemata, but not classical-only excluded middle or Peirce on the textbook pool

Status: **completed 2026-09-30 02:10 – 09:42 UTC (executor), independently reviewed 09:50–10:02 UTC. Verdict: the
counts stand; three rewordings of the interpretation, two label/measure fixes, and one STATUS sentence not
supported.** No hard-constraint violation, nothing quarantined. Where the reviewer's wording differs from the
executor's, this summary uses the reviewer's.

Question (brief, two independent parts):
- **A.** `support-state` found that the SN base (`lean_staten`: sees the proof state, and the environment names the
  hypotheses) reaches 28 / 29 of `support-curves`' survivors per seed, theorems the whole-proof base never reached in
  400,000 attempts. Is that from **seeing the state** or from **not having to choose names**? The `state-env` arms
  separate the two: **S** (`lean_state`) sees the state and writes its own names; **SH** (`lean_stateh`) also sees the
  actions so far.
- **B.** Does SN-cap12 (the `state-cap12` model) get past the textbook schemata that no whole-proof arm has moved?

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_state-readouts` (from the fork's `dan` at
`d3abe474`). Sampling read-outs of existing checkpoints only: no training, no new checkpoints. Nothing is merged into
the fork's `dan`.

- Review: [`review_state-readouts.md`](https://github.com/dan-pandori/nd-takehome/blob/a8dd573e/review_state-readouts.md)
  ([`ecd55415`](https://github.com/dan-pandori/nd-takehome/commit/ecd55415) §Recount, written before the reviewer read
  the executor's write-up; [`a8dd573e`](https://github.com/dan-pandori/nd-takehome/commit/a8dd573e) §Compare /
  §Verdict). Reviewer scripts and outputs: `artifacts/state-readouts/review/` (`recount_a.py`, `recount_b.py`,
  `intuit.py`, `leancheck.py`, `recheck.py`).
- Executor write-up: [`run_state_readouts.md`](https://github.com/dan-pandori/nd-takehome/blob/a8dd573e/run_state_readouts.md),
  `numbers.md` § state-readouts, `log.md` 2026-09-30. DONE commit `ee7adab1`.
- Pre-registration: `preregistration/state-readouts.md`, commit `11d8079f` at 02:15:23 UTC with numeric expectations.
  The first pod (`sr-1`) was created at 02:16:09 UTC; the first job's config is stamped 02:18:14 UTC.
- Brief: [`BRIEF_state-readouts.md`](../../docs/proposals/state-env/BRIEF_state-readouts.md).

## Configuration

- **Checker: Lean alone** (`lean_gate.gate` on the literal `lean_seq` text the environment assembles, with the exact
  reject-only `lean_prefilter`; `LEAN_GATE_DUMP` on every job). Lean 4.34.1 core (the executor's `numbers.md` header says
  4.34.0: a typo). `nd_verify` is identical to `origin/main` and on no counting path; `intuit.py` uses only its formula
  parser to label theorems.
- **Models** (all from scratch, 4 layers, d 256, 3,216,384 params):
  - Part A, Stage-1 bases trained 6,000 steps × 128 on `data/p2/train_depth3_f0_a1.jsonl` (cap 6):
    **S** = `state-env` `ckpts/se/stage1_S_s{0,1}.pt` (`lean_state`, md5 `226b3ded` / `c89baf11`);
    **SH** = `ckpts/se/stage1_SH_s{0,1}.pt` (`lean_stateh`, md5 `b8d6e987` / `9ad2b75f`).
    Inherited and re-counted by the reviewer: **SN** base s0 / s1 (`lean_staten`, `ckpts/se/stage1_SN_s{0,1}.pt`, same
    training set; `support-state` H, Lean alone, records at `origin/dan_support-state:artifacts/ss/H_base_*`).
    Inherited, not re-derived: the whole-proof base `lf/stage1_a1_seq_s0.pt` (`lean_seq`, 3,214,336 params, **one
    seed**), 0 / 29 at **400,000** attempts (`support-curves`, Lean alone).
  - Part B, SN-cap12 (`lean_staten`, Stage-1 on K12 = `data/kh/train_k12.jsonl`, cap 12, 155,000 proofs):
    **T1** = `state-cap12` `ckpts/sc12/ladder/la_T1_SN12_s{0-3}_r8.pt` (+ 8 ladder EI rounds on
    `data/ladder/rl_targets.jsonl`; md5 `82206356` / `ea02cd0c` / `f2c6c062` / `46c3058c`) and **frozen** =
    `stage1_SN12_s{0-3}.pt` (`173f4047` / `b85cbc63` / `21d74747` / `2ca59f8d`).
- **Part A protocol = `support-state` H exactly:** `ss_support.py` with `state_sample.env_generate`, the 29 names of
  `data/sc/falsifier_survivors.txt`, up to 200,000 attempts at T 0.8 with stop at 5 successes, then up to 200,000 at
  T 1.0 for rows with < 5. Batch 4,096, `max_chunk` 8, `max_action` 256, `max_steps` 48, fast decode path, sampling
  seeds 1 / 2 (s0) and 11 / 12 (s1), identical to SN's (the reviewer checked from the rows). Headline: survivors
  reached (≥ 1 Lean-accepted attempt at either temperature), of 29. Pre-registered reading, by the mean over seeds per
  arm: ≥ 20 → state band ("seeing the state is what matters"); ≤ 5 → naming band; otherwise both. A between-arm
  difference of ≤ 5 theorems at n = 2 is not a finding. No noise floor exists for this quantity.
- **Part B:** `lpool_reread.py` (from `origin/dan_state-cap12`), k 256, T 0.8, seed 0, `max_steps` 96, `max_action`
  512, batch 4,096, on the 760 textbook theorems of `data/ladder/transfer.jsonl` (19 schemata × 40,
  `data/sr/textbook_transfer.jsonl`) and the 282 of `transfer_long.jsonl` (6 schemata, `L_true` 11–14,
  `data/sr/textbook_long.jsonl`). A theorem is solved if ≥ 1 of 256 is accepted. **Dead schemata** (defined before any
  pod): the 12 at ≤ 2 / 40 in every arm of `ds-composition` D4b (whole-proof `lean_seq`, in-loop cumulative T1 at
  8 × k 32, seed 0, **Lean ∧ `nd_verify`**, 2026-09-24).
- **Disclosed before the pre-registration:** SN-cap12 T1's in-loop counts already had 11 of 12 dead schemata at ≥ 5,
  so Part B is mostly a confirmation at fixed k 256 on the final checkpoint.
- **Pods:** Part A on three A40s at $0.49/h (`sr-1` 5.39 h, `sr-2` 7.05 h, `sr-4` 6.63 h; two jobs a card), Part B on
  `sr-3`, an RTX A6000 at $0.53/h (3.59 h). **22.67 pod-hours, $11.25** of $12 / 24 h (`podbudget`). All deleted.
  Pre-registered estimate ≈ 16 pod-hours. SH s1's T 1.0 phase was stopped at the pre-registered 23 pod-hour line.
- **Compute per arm** (registry rows `artifacts/state-readouts/registry/`, `registry_merge.py --compute`; GPU-h are job
  wall-clock, two jobs sharing an A40 for part of Part A; rows include attempts of runs that died of CUDA OOM and were
  resumed per theorem with the same seeds):

  | part / arm | seed | GPU | GPU-h | gen tokens | attempts | actions | Lean checks |
  |---|---|---|---|---|---|---|---|
  | A S | 0 / 1 | A40 | 3.35 / 4.48 | 526 M / 730 M | 2,568,640 / 4,101,120 | 21.2 M / 32.5 M | 1,530 / 2,053 |
  | A SH | 0 / 1 | A40 | 7.12 / 4.38 † | 1,035 M / 627 M † | 6,210,304 / 3,433,472 † | 41.4 M / 25.3 M † | 495 / 536 † |
  | A diagnostic S | 1 | A40 | 0.67 | 150 M | 800,000 | 6.1 M | 0 |
  | B T1 | s0–s3 | RTX A6000 | 0.52–0.54 each | 65–67 M | 266,752 (s2 461,312 ‡) | 4.0–4.1 M | 48,671–68,996 |
  | B frozen | s0–s3 | RTX A6000 | 0.33–0.36 each | 45–46 M | 266,752 | 2.6–2.7 M | 6,363–9,965 |

  † SH s1's killed T 1.0 process wrote no rows. ‡ includes a first T1 s2 launch that OOM'd. Training steps / tokens: 0.
  **Flag (policy 1.25× rule):** every S / SH arm used > 1.25× SN's attempts (SN 1,041,664 / 2,190,528). The arms are
  matched on the per-theorem cap and stop rule, not on total attempts; an arm that reaches fewer theorems spends more.
  Peak `max_memory_allocated` in Part B 30.1–38.5 GB at batch 4,096.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/state-readouts/artifacts/state-readouts/` (records, re-reads `rr/`,
  gzipped Lean dumps `dump/` with every literal text judged, gate logs, registry rows, re-check files) and
  `hf://buckets/dan-pandori/nd-rl/state-readouts/data/{sr,sc}/`; registry rows also under
  `hf://buckets/dan-pandori/nd-rl/registry/state-readouts/`. The `.jsonl` records are bucket-only.

## Observations

All counts below were re-derived by the reviewer from the per-theorem rows and, independently, from the Lean-gate
dumps; the two agree on every theorem.

### Part A: survivors reached, of 29

| arm | names | seed | reached | at T 0.8 | median p̂ at T 0.8 (unreached = 0) | attempts | `max_action` hit |
|---|---|---|---|---|---|---|---|
| whole-proof base (inherited, `stage1_a1_seq_s0.pt`, one seed, 400,000 attempts, Lean alone) | own | s0 | 0 | — | — | 400,000 / theorem | — |
| **S** (state) | own | s0 | **28** | 25 | 0.0054 | 2,568,640 | 0.042 % |
| **S** | own | s1 | **25** | 25 | 0.0144 | 3,941,376 | **0.753 %** |
| **SH** (state + history) | own | s0 | **21** | 17 | 2.0e-5 | 6,066,944 | **0.297 %** |
| **SH** | own | s1 | **≥ 15** (≤ 21) | 13 | 0 | 5,433,472 (+192,512 unrecorded) | **0.572 %** |
| SN (inherited, `support-state`) | environment | s0 / s1 | 28 / 28 | 27 / 26 | 0.188 / 0.0088 | 1,041,664 / 2,190,528 | 0.0005 % / 0.055 % |

- **SH s1 is a lower bound.** Its T 1.0 phase was cut in theorem 11 of 16. Six theorems that failed at T 0.8 (200,000
  attempts each) got no T 1.0 attempts (`la_transfer_1858` 192,512, 0 accepted, then 1893, 1932, 2038, 454, 87 not
  run). The protocol-complete value is between 15 and 21. (The executor's `numbers.md` gave SH s0's median p̂ as 0; it
  is 2.0e-5: 17 of 29 are nonzero, so the 15th value is 4 / 200,000.)
- Unreached: S s0 {1893}; S s1 {100, 1729, 588, 87}; SH s0 {1004, 1110, 149, 1858, 1893, 2038, 454, 87}; SH s1 {100,
  1004, 1110, 1437, 149, 1729, 1858, 1893, 1932, 2038, 381, 454, 588, 87}.
- **Paired with SN:** every theorem any S / SH seed reaches is reached by SN on some seed (0 exceptions; SN union
  29 / 29). Same-seed SN-reached-but-arm-missed: S s0 0, S s1 4, SH s0 7, SH s1 13. S s0's miss (1893) SN s0 misses
  too. Union / intersection over seeds: S 29 / 24, SH 21 / 15, SN 29 / 27.
- **Pre-registered reading, per arm:** S mean 26.5 → **state band** on both seeds. SH mean 18.0 (≥ 18 with the lower
  bound) → **middle band**. The arms fall in different bands, so the pre-registration requires the reading per arm.
- **Length caps.** Step cap 48: 0 hits anywhere. Action cap 256: above the pre-registered 0.1 % on S s1, SH s0 and SH s1,
  concentrated in a few theorems. Unreached theorems with heavy truncation: S s1 `la_transfer_100` (5.7 % at T 0.8,
  8.7 % at T 1.0), SH s1 `la_transfer_1858` (14.4 %, no T 1.0), SH s0 `la_transfer_1004` (0.57 % / 1.40 %). A post-hoc
  diagnostic re-sampled S s1's four unreached theorems at `max_action` 1,024 (T 0.8, seed 11, 200,000 each): **0
  accepted, 0 truncated**, so truncation did not hide S s1's misses. **No such diagnostic exists for SH** (1858 on s1,
  1004 on s0).
- **Proof length**, distinct accepted proofs, median. By the reviewer's Lean-text measure (`have` count / proof-term
  nodes, type ascriptions removed): S 7 / 21 and 6 / 21, SH 6 / 20 and 6 / 20 (n = 69, 58, 35, 28 distinct normalised
  proofs). The executor's `support.proof_term_size` gives lines / size S 10 / 85, 10 / 79; SH 9 / 75, 9 / 80; SN 10 / 89,
  10 / 86. That field counts formula nodes over the ND lines **including premise lines**, so it is not invariant to
  premise re-statement, the property the policy asks term size for. The arms rank the same under both. 0 of the 29
  survivors are classical-only.

### Part B: SN-cap12 on the textbook schemata (k 256, T 0.8, seed 0)

| ckpt | textbook transfer / 760 | dead schemata ≥ 5 / 12 | excluded_middle / 40 | `transfer_long` / 282 | long schemata ≥ 5 / 6 | action-cap hit (tb / tbl) |
|---|---|---|---|---|---|---|
| T1 s0 | 513 | **11** | 1 | 102 | 4 | 0.020 % / 0.006 % |
| T1 s1 | 552 | **11** | 0 | 125 | 4 | 0.002 % / 0.012 % |
| T1 s2 | 537 | **11** | 1 | 111 | 5 | 0.002 % / 0.004 % |
| T1 s3 | 514 | **11** | 1 | 113 | 4 | 0.002 % / 0.044 % |
| frozen s0 | 275 | 6 | 1 | 35 | 3 | 0.048 % / **0.129 %** |
| frozen s1 | 346 | 10 | 0 | 57 | 3 | 0.008 % / 0.004 % |
| frozen s2 | 361 | 10 | 0 | 56 | 3 | 0.032 % / 0.007 % |
| frozen s3 | 333 | 9 | 0 | 53 | 3 | 0.002 % / 0 % |

- Step cap 96: 0 hits (max 95 steps used). Per-schema counts, every cell reproduced by the reviewer, are in the
  executor's `numbers.md` § state-readouts and the reviewer's `review/recount_b.json`. Dead-schema examples, T1 s0–s3 /
  frozen s0–s3: demorgan_or_to_nand 28 / 28 / 33 / 25 vs 0 / 4 / 2 / 2; peirce 26 / 27 / 26 / 27 vs 4 / 6 / 20 / 15;
  negated_conditional_conv 39 / 40 / 40 / 37 vs 15 / 11 / 16 / 8.
- For comparison, `ds-composition` D4b (whole-proof `lean_seq`, in-loop cumulative at 8 × k 32, Lean ∧ `nd_verify`):
  every dead schema ≤ 2 in every arm. Not the same estimator or checker; the gap (≤ 2 vs 9–40 on the T1 seeds) is far
  larger than either difference could make.
- **Classical-only instances** (post hoc; not intuitionistically provable by a G4ip decision procedure; the reviewer's
  own implementation agrees with the executor's `data/sr/intuit_labels.json` on all 1,242 theorems). On the
  **760-theorem textbook pool**: **0 classical-only excluded_middle (of 39) and 0 classical-only peirce (of 13) solves on
  every checkpoint**; peirce_sequent (19 classical-only) **1 on T1 s3**, 0 elsewhere; contraposition_conv 22–23 (T1),
  7–16 (frozen); negated_conditional 1–16. Every excluded_middle solve (≤ 1 per checkpoint) is the one intuitionistic
  instance, `la_transfer_882`, `((Q ∧ R) ∧ ¬R) ∨ ¬((Q ∧ R) ∧ ¬R)`, proved by `Or.inr` of a constructive negation. The
  26–27 peirce solves are its intuitionistic instances. On **`transfer_long`: T1 s0 solves 6 classical-only Peirce
  instances** (0 on s1–s3 and frozen) through ¬¬-elimination (`Classical.byContradiction`); negated_conditional 0–12.
  The run note's rendering of a `la_transfer_882` proof below is reformatted, not the literal sampled text; the
  reviewer re-checked its 179 literal accepted texts in Lean, all accepted:

  ```lean
  theorem t (Q R : Prop) : ((Q ∧ R) ∧ ¬R) ∨ ¬((Q ∧ R) ∧ ¬R) := by
    have n27 : ¬((Q ∧ R) ∧ ¬R) := fun n28 => by
      have n29 : Q ∧ R := n28.1; have n30 : R := n29.2; have n31 : ¬R := n28.2
      exact n31 n30
    exact Or.inr n27
  ```
- Proof length, distinct accepted texts (median `have` lines / term size, reviewer's measure): T1 tb 9 / 29–32, T1 tbl
  13–15 / 48–56; frozen tb 6–7 / 23–24, frozen tbl 12 / 40–45.

### Checks

- **Lean re-check (reviewer's own harness)**: every distinct accepted Part A text, 3,583 / 3,583 (S) and 1,034 / 1,034
  (SH); a Part B sample of 150 per file, 1,200 / 1,200 (T1) and 1,200 / 1,200 (frozen); all 441 texts behind the
  classical-only peirce / peirce_sequent solves, 441 / 441. **Negative control:** 0 of 2,310 prefilter-rejected texts
  accepted, so the harness fails when it should and the sample finds no prefilter rejection that Lean accepts. Every
  rejection in the run was made by the prefilter. (Executor's re-check: A 190 / 190 with 142 / 142 controls rejected;
  B 2,338 / 2,338 with 2,338 / 2,338 controls rejected.)
- **Splits disjoint by renaming class:** 0 collisions of the 175,818 variants of the 29 survivors and the 760 + 282
  textbook theorems with `train_depth3_f0_a1.jsonl`, `train_k12.jsonl` (the reviewer's copy from `cap-horizon`, assumed
  identical to `state-cap12`'s, which is not in the bucket), and `rl_targets.jsonl`.
- `TEST_RUN_DONE` unchanged since the base. No training script changed or run.
- **Process note (no effect found):** S / SH were judged with the reject-only prefilter on; SN's `support-state` numbers
  were not (the prefilter did not exist then). The run note does not say so; the log does. The reviewer's 2,310-text
  sample finds no loss.

### Expected vs outcome (pre-registered at 02:15:23 UTC)

- A1 S **24** per seed (80 % range 17–29) → 28 / 25, inside. A2 SH **21** (12–28) → 21 / ≥ 15, inside.
- A3 reading P(state band) 0.65 → S in the state band, SH in the middle: the joint "both ≥ 20" reading is **missed**,
  and the reading is per arm.
- A4 (median p̂ below SN s0's for every S / SH seed; within 10× of SN s1's on one S seed) → met. A5 → 0 exceptions, met.
- A6 action / step caps ≤ 0.1 % → **missed** on S s1 (0.75 %), SH s0 (0.30 %), SH s1 (0.57 %).
- B1 T1 11 dead schemata (10–11), excluded_middle ≤ 2 → met. B2 frozen **7** (4–10) → 6 / 10 / 10 / 9, met.
- B3 totals T1 480–580, frozen 280–450 → T1 met; **frozen s0 275, missed** by 5.
- B4 `transfer_long` T1 90–170 with ≥ 5 on ≥ 5 of 6 schemata → totals met; the schema condition **missed on 3 of 4
  seeds** (4 / 4 / 5 / 4). B5 step cap ≤ 0.1 % → 0, met.

## Interpretation (reviewer's wording)

- **Part A's main result stands:** a model that writes its own hypothesis names but sees the proof state (S) reaches
  25–28 of the 29 survivors that the whole-proof base reaches 0 of (that base: one seed, 400,000 attempts). S is in
  the pre-registered state band on both seeds.
- **Naming:** S and SN reach 28 / 25 vs 28 / 28, a difference of 0 and 3 theorems, inside the pre-registered 5-theorem
  no-finding margin. **No naming effect on reach is detected at n = 2, and n = 2 cannot bound it.** Reach at 400,000
  attempts is coarse. Per-theorem rates differ a lot: S's median p̂ is 0.0054 / 0.0144 against SN's 0.188 / 0.0088 (SN
  s0 ≈ 35× S s0), so naming may matter for rate even where it does not for reach. (The executor's "naming is worth
  ≤ ≈ 3 theorems at n = 2" is reworded to this.)
- **History:** SH reaches 7 fewer than S on s0. On s1 the gap is 10 as recorded, but only 4 if SH s1's six unfinished
  T 1.0 rows had all succeeded, so on s1 it may sit inside the margin. SH differs from S only in adding the action
  history to the prompt, so **history is the candidate cause, shown on one seed**. Two unreached SH theorems carry heavy
  action-cap truncation that was never re-run (1858, 14.4 %, s1; 1004, 0.6–1.4 %, s0). (The executor's "its shortfall
  is from history, not naming" is reworded to this.)
- **Part B:** SN-cap12 T1 has 11 of the 12 dead textbook schemata at ≥ 5 / 40 on all four seeds at k 256; frozen 6–10.
  Excluded middle is ≤ 1 everywhere, and the one solve is an intuitionistic instance. **On the 760 pool no checkpoint
  solves a classical-only excluded_middle or peirce instance.** Classical reasoning is not absent, though: 1
  classical-only peirce_sequent (T1 s3), 22–23 classical-only contraposition_conv on T1, and 6 classical-only Peirce on
  `transfer_long` (T1 s0) through ¬¬-elimination.
- **Not supported:** the executor's STATUS line and DONE commit subject, "on classical-only instances excluded middle
  and Peirce stay at 0". Peirce is 0 on the 760 pool but 6 on `transfer_long` (T1 s0), and peirce_sequent is 1. The run
  note itself states this correctly.

## Limitations and next step

- n = 2 per arm in Part A, and no noise floor for survivors reached; the naming comparison cannot bound an effect below
  the 5-theorem margin.
- SH s1 is incomplete (15 to 21). SH's truncated zeros (1858 s1, 1004 s0) were not re-run at a larger action cap.
- The action cap exceeded the policy's 0.1 % line on three of four Part A seed-arms (S s1's misses cleared by the
  diagnostic).
- The whole-proof comparator is one seed at 400,000 attempts; SN is inherited and was judged without the prefilter.
- Part B is one sampling seed at k 256 per checkpoint; the classical-only labels are post hoc.
- Part A's arms are matched on protocol, not attempts (every S / SH arm > 1.25× SN's attempts).

Next measurements (reviewer):
1. **Finish SH s1's T 1.0 phase** on its six theorems (≤ 1.2 M attempts, ≈ 1.5 A40-h). This fixes SH s1 between 15
   and 21 and decides whether SH − S exceeds the margin on both seeds.
2. **Re-run SH's truncated zeros** (s1 1858, s0 1004) at `max_action` 1,024, as was done for S s1.
3. **For naming, compare per-theorem p̂** paired S vs SN (same 29 theorems, same seeds) rather than reach; then add S
   s2 / s3 (`state-frontier` `sf2`, pre-registered priority 4, not run) against SN seeds at the same protocol, for
   n = 4 before any statement about the size of the naming effect.
