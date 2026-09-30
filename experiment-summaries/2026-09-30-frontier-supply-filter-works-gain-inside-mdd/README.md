---
written_on: 2026-09-30
written_by: agent:claude
---

# frontier-supply: a rarity filter finds EI targets (12.8 % pass), and supply beats control by +7.5 of 291, inside the pre-registered MDD of 10.5

Status: **completed 2026-09-30 02:00 – 11:35 UTC (executor), independently reviewed 11:40–11:51 UTC. Verdict: the
counts stand. The pre-registered test is not met (S − C +7.5 against a C′-based MDD of 10.5). There are three
rewordings and nothing is unsupported.** No hard-constraint violation, nothing quarantined. Where the reviewer's
wording differs from the executor's, this summary uses the reviewer's.

Question: expert iteration learns only on theorems the model sometimes solves (p̂ > 0), and the fixed `rl_targets`
pool has almost none past the frontier. Does **supplying** new targets help? The supply is built each round from the
current model and kept only if the model solves it rarely (p̂ ∈ (0, 1/4], after STP 2502.00212v4 §3.2). The test is
whether this, at matched attempts, moves SN-cap12's long-proof read-out. (Proposal 16 item 1; lit-review shortlist
#2.)

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_frontier-supply` (from the fork's `dan` at
`d3abe474`). **Nothing is merged into the fork's `dan`**, including the new code. The new code is `supply.py`
(candidates (a) and (b), the leak key), the supply arm with `--ei_seed` and `--transfer_k` in `state_ladder_ei.py`, and
`fail_states` in `env_generate`. The brief asks for that merge after review (decision item in the digest).

- Review: [`review_frontier-supply.md`](https://github.com/dan-pandori/nd-takehome/blob/a6cd2a2a/review_frontier-supply.md)
  ([`49ec722f`](https://github.com/dan-pandori/nd-takehome/commit/49ec722f) §Recount, written before the reviewer read
  the executor's write-up; [`a6cd2a2a`](https://github.com/dan-pandori/nd-takehome/commit/a6cd2a2a) §Comparison /
  §Verdict). Reviewer scripts: `review_fsup_recount.py`, `review_fsup_lean_recheck.py`, `review_fsup_filter_leak.py`,
  `review_fsup_terms.py`.
- Executor write-up: [`run_frontier_supply.md`](https://github.com/dan-pandori/nd-takehome/blob/a6cd2a2a/run_frontier_supply.md),
  `numbers.md` § frontier-supply (FS-1…6), `log.md` 2026-09-30, `artifacts/fsup/summary.json`. The results and DONE
  commit is `0a60354c`.
- Pre-registration: `preregistration/frontier-supply.md`, committed `bbeca3f5` at 02:06 UTC. A dated deviation of the
  primary quantity (`7c54dfc6`, 02:11) followed Dan's 02:11 suggestion. Both came before the first pod (`fsup1`,
  02:22).
- Brief: [`BRIEF_frontier-supply.md`](../../docs/proposals/literature/BRIEF_frontier-supply.md).
- Figure (not checked by the reviewer, so linked, not copied):
  [`figures/frontier_supply.png`](https://github.com/dan-pandori/nd-takehome/blob/a6cd2a2a/figures/frontier_supply.png).

## Configuration

- **Checker: Lean alone** (`eval_set.judge` → `lean_judge.judge_many`). `nd_verify` is unmodified (tree `9437bb72`,
  equal to `origin/main`'s). The run's only import from it is `parse_formula` in `supply.py:53`, which parses
  statements and judges nothing.
- **Model, every number below:** SN-cap12. This is a `lean_staten` state policy (proof-state observation,
  environment-assigned names), 3,216,384 params (4 layers × d 256), trained from scratch. Its Stage-1 is on K12
  (cap-horizon's cap-12 set, 155,000 proofs; `state_train.py --mode lean_staten --steps 6000 --cap 12`). Stage-1 s0–s3
  are `state-cap12`'s `stage1_SN12_s{0-3}.pt`; s4 and s5 were trained here with the same recipe.
- **Arms** (8 EI rounds per ladder, k 32, T 0.8, batch 2,048, `max_action` 256, `max_steps` 48, 600 fine-tune steps a
  round, replay K12; `state-cap12`'s T1 protocol):
  - **C, control:** targets from `rl_targets.jsonl`. s0–s3 reuse `state-cap12`'s `la_T1_SN12_s{0-3}_r8.pt`; s4 and s5
    are new.
  - **S, supply** (s0–s5, all new): 25 % of every round's 143,840 attempts go to filtering 1,123 fresh candidates × 32
    attempts, and the rest go to `rl_targets`. The candidates are of two kinds:
    - (a) mutations of this round's solves (hyp, contrapose, case, conj, chain), aimed at upper bounds `L*`+1…+4;
    - (b) the open goal at the failure point of failed attempts.

    A candidate is kept if 0 < n_ok ≤ 8 of 32, and its Lean-accepted proofs join the replay.
  - **C′, control rerun** with EI seed + 100 (s0, s1). It gives the no-treatment spread.
  - **B:** the Stage-1 bases, read for frozen reachability.
- **Compute matched, filter included:** 143,840 attempts per round in every arm, and 4,800 fine-tune steps per
  ladder.
- **Read-out** (identical for every checkpoint): `lpool_reread.py`, k 256, T 0.8, seed 0, batch 2,048, `max_action`
  512, `max_steps` 96, on an NVIDIA A40. Peak memory was 12.1–17.3 GB.
  - **Primary quantity (after the 02:11 deviation):** per seed, the theorems with ≥ 1 Lean-accepted sample among the
    **291**. These are the 91 of `long-pool-2` plus rr600 `L_true` 15–16 (200). S vs C is paired by seed.
  - The pre-registered MDD is 1.425 · sd_d, with sd_d from the C′ − C pairs.
- **Pods:** A40 at $0.49/h. `fsup1`–`fsup5` ran 02:22–04:00 and were **deleted at 04:00 by the host pod watchdog's
  liveness bug** (fixed in nd-rl `9c4b735`). That lost 6.94 pod-hours, $3.40, and all ladders were relaunched from
  scratch with the same seeds. `fsup6`–`fsup10` ran 04:07–11:00; `fsup11` (11:04–11:23) redid failed read-outs.
  **38.99 pod-hours, $19.10** of $20 / 40 h (`podbudget`, `~/podhours.log`).
- **Compute per ladder** (registry rows `artifacts/frontier-supply/registry/`, summed by the reviewer; A40):

  | arm | GPU-s | gen tokens | train steps | train tokens | Lean checks | attempts |
  |---|---|---|---|---|---|---|
  | S s0–s5 | 19,339–20,589 (mean 19,977) | 244–258 M | 4,800 | 676–724 M | 610–690 k | 1,150,720 |
  | C s4, s5 | 20,491\*, 19,689 | 261 M, 247 M | 4,800 | 690 M, 686 M | 857 k, 858 k | 1,150,720 |
  | C′ s0, s1 | 19,880, 19,918 | 244 M, 247 M | 4,800 | 663 M, 685 M | 758 k, 828 k | 1,150,720 |

  \* The C s4 registry holds a stale round-1 `sample` row (143,840 attempts, 887 GPU-s) from the ladder killed at
  04:00, so its registry sum is 21,377. The executor's 20,491 excludes that row, correctly, but the row is still in the
  registry. No arm exceeds its comparator by 1.25× on any metric: S's train tokens are ≈ 4 % higher, and its Lean
  checks are lower. The reused C s0–s3 have no registry rows in this run. `state-cap12` ran them with sampled transfer
  each round, ≈ +50 % attempts that are not counted here.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/frontier-supply/{artifacts/fsup,ckpts/fsup,artifacts/frontier-supply,data/fsup}`.

## Observations

The reviewer re-derived every count below from the raw per-theorem rows with their own scripts.

**Primary: theorems solved of 291, per seed.**

| arm | s0 | s1 | s2 | s3 | s4 | s5 | mean |
|---|---|---|---|---|---|---|---|
| C (control) | 142 | 207 | 218 | 220 | 220 | 207 | 202.3 |
| S (supply) | 139 | 217 | 226 | 233 | 228 | 216 | 209.8 |
| C′ (control rerun) | 145 | 197 | | | | | |
| on the 91: C / S / C′ | 33 / 36 / 34 | 60 / 62 / 52 | 57 / 61 | 58 / 66 | 61 / 66 | 47 / 58 | |
| Stage-1 base on the 91 | 7 | 20 | 18 | 23 | 25 | 19 | |

- **S − C, paired: −3, +10, +8, +13, +8, +9.** Mean **+7.5**, IQM +8.75, bootstrap 95 % CI [+2.5, +11.0] (6 paired
  differences, 20,000 draws). Paired t(5) = 3.36, p ≈ 0.02. Flips (S-only / C-only), summed over seeds: 158 / 113.
- **C′ − C: +3, −10** (flips 25/22 and 23/33), so sd_d = 7.4 and **MDD = 10.5**. The pre-registration's provisional
  MDD was 23. From the S − C sd itself (5.5, 5 df) the MDD would be 7.8.
- **By the pre-registered rule the test is not met:** +7.5 < 10.5. The point forecast was +8.
- The reviewer's summary: **"small positive effect, not established against the pre-registered MDD."** A reader
  should not infer "no effect":
  - S > C on 5 of 6 seeds on the primary, and on 6 of 6 on the 91 and on exact-17.
  - The IQM interval excludes 0.
  - The miss comes from a C′-based sd_d estimated from 2 pairs, one of them −10.
- **Strata, S − C mean:** the 91 +5.5 [+2.75, +8.75] (MDD 8.1); **exact-17 (61) +4.0**, 6/6 positive (MDD 3.2);
  ≥ 18 (30) +1.5; rr 15–16 (200) +2.0 [−2.75, +6.0] (MDD 2.9, inside its noise); rr 13–14 +1.3. The gain sits in the
  91, mostly exact-17.
- **Reused vs new ladders.** C′'s new ladders score no higher than the reused C (+3, −10), so reuse does not
  manufacture S − C. The C re-reads reproduce `long-pool-2`'s lp2 numbers: 33 / 60 / 57 / 58 against 34 / 59 / 58 /
  61, a sampling re-draw.
- Seed 0 is low in every arm (C 142, S 139, C′ 145). It has the weakest base (7 of 91) and dominates the between-seed
  spread (C's sd on the primary is 30.2). `L*` on the read-out is 18 for every checkpoint and every base, so it
  saturates, as forecast.

**The filter (6 S ladders × 8 rounds × 1,123 = 53,904 candidates).**

| slice | pass (0 < n_ok ≤ 8 of 32) | n_ok = 0 | n_ok > 8 |
|---|---|---|---|
| **all** | **6,894 = 12.8 %** | 75.5 % | 11.7 % |
| (a) mutations (26,928) | 15.8 % | 70.1 % | 14.1 % |
| (b) open goals (26,976) | 9.8 % | 80.8 % | 9.3 % |

- By op: hyp 32.0 %, contrapose 22.6 %, case 15.3 %, conj 13.8 %, chain 10.5 %.
- Rounds 1 → 8: 14.9 % → 11.3 %. Per seed: 12.0–14.0 %.
- There was no shortfall. The kept pools after round 8 are 1,075 / 1,255 / 1,151 / 1,165 / 1,128 / 1,120 targets,
  carrying 2,138–2,823 proofs. The leak filter dropped 75 candidates.
- **The supplied targets were short because of the window rule.** The length window was anchored to the
  `rl_targets` `L*`: `base` went 12 → 13 by round 2–3 (round 7 on s0). So (a)'s upper bounds were at most 17, and the
  run did not offer targets at the read-out's 17–18 length. (b)'s open goals are sub-goals, short by construction.
  - Supply proofs have median 16 lines for (a), and 4,477 of 9,282 of them are ≥ 17 lines; the median for (b) is 11,
    the same as for `rl_targets` proofs.
  - (This replaces the executor's "the filter selects rarity, not length": this run cannot say whether the filter
    selects against length.)

**Proof length of read-out solves.** The median written proof is 21 lines on the 91 in every arm; the `L_lb` labels
of 17–18 are lower bounds. **Term size does not separate the arms: S solves more theorems, not differently sized
proofs.**
- Reviewer's tree / DAG size, exact-17: C 29 / 19, S 30 / 20, C′ 31 / 19, B 29 / 19.
- ≥ 18: 32 / 20 in S and C.
- The executor's `lean_check` elaborated term size is 14–17 on the 91. It uses a different definition, so the two sets
  of absolute values are not comparable.

**Checks.**
- **Lean re-check:** 870 / 870 stored proofs were accepted. They are 150 random + 30 longest from each of C, S, C′ and
  B, plus 150 supply training proofs, checked in Lean core without Mathlib.
  - Statements were rendered independently, with 0 mismatches.
  - The only scan hit was `Classical.byContradiction` (DN/RAA).
  - The negative control failed as it should.
  - The literal sampled text is not stored. The stored ND is re-rendered to Lean, the same path every state-policy run
    has used.
- **Splits** (reviewer's own renaming-class key): every supply candidate (46,834 classes) and every kept one (6,466)
  shares **0** classes with every read-out and long pool, and with held-out, transfer and `val_36`. The K12 /
  `rl_targets` overlaps with transfer / held-out are inherited and touch no quantity of this run.
  - `data/ladder/reserve.jsonl`, listed as a leak pool in the pre-registration, does not exist, so it was never
    checked.
- **Relaunch not selective:** all five ladders killed at 04:00 were rerun from scratch, and their round-1 numbers
  reproduce exactly.
- **Truncation:** ≤ 0.09 % on every read-out except S s5 rr (0.14 %) and B s5 lp2 (0.63 %). Both are above the
  policy's ≈ 0.1 % line, but neither touches a decision quantity much.

**Expected vs outcome.**

| expectation | outcome |
|---|---|
| Filter pass ≥ 5 % (forecast 10 %) | **12.8 %**, met |
| (a) forecast 12 %, (b) forecast 6 % | **15.8 %, 9.8 %**, both higher. The write-up did not tabulate these; the reviewer asks for them. |
| S − C > MDD (brief); point forecast +8 | **+7.5 < 10.5**, missed. The point forecast held. |
| MDD provisional 23 | 10.5 (C′-based, 2 pairs) |
| C vs C′ ≤ 12 per seed | 3, 10, met |
| Frozen reachability on every read-out pool | **missed on rr 13–16**. B s0 and s1 on rr died of CUDA OOM (04:24–04:26); s2–s5 were cut at 10:10 for budget. The bases were read on the 91 only. |

## Interpretation

The filter is the part that works: about one candidate in eight is rare in the STP sense, from either source, with
no shortfall and no leakage. Whether the rare targets move the frontier is open at this power. The direction is
consistent across seeds and concentrated at exact-17, but the paired difference does not clear an MDD whose
no-treatment spread rests on two pairs. In this design the window rule pinned the supplied targets below the
read-out's length. So a positive result here would have been "more rare targets", not "longer targets", and a null
could not rule out the latter.

## Limitations and next step

- The MDD rests on 2 C′ − C pairs; sd_d from 2 df.
- Stage-1 frozen reachability is missing on rr 13–16 (a pre-registered item not delivered).
- The supply window was anchored to `rl_targets`' `L*` (12–13), not to the read-out frontier (16–17).
- C s0–s3 are `state-cap12`'s ladders (older run, per-round sampled transfer); C′ shows reuse does not inflate C.
- The C s4 registry has a stale round-1 row, and C s0–s2 have no read-out registry rows here.
- The code is only on the fork branch `dan_frontier-supply`.

Next measurements (reviewer):
1. **C′ on s2–s5** (4 ladders, ≈ 4 × 5.5 A40-h, ≈ $11). This gives a 6-pair sd_d and settles whether +7.5 is outside
   the MDD. A sd_d near the S − C sd (5.5) would give an MDD of ≈ 8.
2. **Stage-1 base read-outs on rr 13–16** for all six seeds (≈ 16 min each), to complete frozen reachability.
3. **If (1) confirms the effect:** a supply arm whose window is anchored to the read-out frontier (base 16–17). It
   separates "more rare targets" from "longer targets".
