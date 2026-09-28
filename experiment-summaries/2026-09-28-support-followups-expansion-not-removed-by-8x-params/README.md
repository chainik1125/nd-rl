---
written_on: 2026-09-28
written_by: agent:claude
---

# support-followups: pressure-testing support-curves' 29 survivors (seed 1, depth, an 8× larger base, where the improbability sits)

Status: completed 2026-09-28 04:05–12:58 UTC (executor) and independently reviewed 13:00–13:13 UTC.
Review verdict: **stands**. Every count reproduces on the reviewer's own code. Six wordings in the
run file must change, and one interpretation is **not supported** (below). No hard-constraint
violation, nothing quarantined. Numbers and wording here are the reviewer's where they differ from
the executor's.

This run was on the fork (`dan-pandori/nd-takehome`), branch `dan_support-followups`, cut from
`dan_support-curves`. It has not been merged into the fork's `dan`.

- Review: [`review_support-followups.md`](https://github.com/dan-pandori/nd-takehome/blob/edea854/review_support-followups.md)
  ([`0e138f8`](https://github.com/dan-pandori/nd-takehome/commit/0e138f8) §Recount, blind;
  [`edea854`](https://github.com/dan-pandori/nd-takehome/commit/edea854) §Compare/§Verdict).
- Executor write-up: [`run_support_followups.md`](https://github.com/dan-pandori/nd-takehome/blob/edea854/run_support_followups.md).
  Numbers are in `numbers.md` § support-followups (SF0–SF8), and the log is `log.md` § support-followups.
- Pre-registration: [`preregistration/support-followups.md`](https://github.com/dan-pandori/nd-takehome/blob/edea854/preregistration/support-followups.md).
  It was committed as `352676c` at 04:09 UTC, before any pod or result (gate 0). Addendum `fe0a939`
  (C seed 1) was committed at 08:46, before any seed-1 file.
- Parent result: [support-curves](../2026-09-27-support-curves-support-expansion-fires/README.md).
  This run is that review's ranked next steps 1–3, plus a new analysis D.

Question: [support-curves](../2026-09-27-support-curves-support-expansion-fires/README.md) found 29
transfer theorems that the base reaches 0 times in 400,000 attempts, while the EI model solves them
at p̂ 0.02–0.9995. Four follow-ups test that result:

- **(A)** Does the result replicate on seed 1, using an EI checkpoint that actually exists?
- **(B)** Do the six longest survivors fall at 10⁷ base attempts?
- **(C)** Does an 8× larger base reach them? The pre-registered falsifier is ≥ 15 of 29.
- **(D)** Where in EI's proofs is the base's improbability concentrated? One step, or many slightly
  unlikely ones?

## Configuration

- **Judge:** Lean alone, run inside the loop. Afterwards, every accepted literal sampled text was
  re-checked one proof per Lean process. The reviewer re-checked all 451 distinct counted proofs a
  second time with its own statement builder and Lean driver.
- **Models** (all `lean_seq`, cap 6, from scratch; Stage 1 on `data/p2/train_depth3_f0_a1.jsonl`,
  6,000 × 128):

  | label | checkpoint | parameters | md5 |
  |---|---|---|---|
  | base s0 | `ckpts/lf/stage1_a1_seq_s0.pt` | 3,214,336 | `9bde44c0…` |
  | base s1 | `ckpts/lf/stage1_a1_seq_s1.pt` | 3,214,336 | `fc27e52d…` |
  | EI s0 | `la_T1_sc_s0_r8.pt` (base s0 + 8 × k 32 EI) | 3,214,336 | `5cebd7ec…` |
  | EI s1rerun | `la_T1_sc_s1rerun_r8.pt` | 3,214,336 | `12c13e61…` |
  | big s0 | `ckpts/sf/stage1_big_seq_s0.pt` | 25,329,664 (8 layers, d 512, 8 heads) | `4efb5a1e…` |
  | big s1 | `ckpts/sf/stage1_big_seq_s1.pt` | 25,329,664 | `05ed8839…` |

  The big models use the same data, steps, lr schedule and cap as base. The reviewer read both
  checkpoints' embedded `args` to confirm this. Support-curves' lost EI s1 (`105be4f3…`) enters
  only as its recorded numbers.
- **Sampler:** fast path, `max_new` 512, fresh sampling seeds. Batch 4,096 for the 3.2 M models
  (peak 17.43 GiB) and 1,024 for the 25 M models (peak 15.6 GiB).
- **Parts:**
  - **A:** k 10,000, stop 50, T 0.8, all 383 transfer theorems, base s1 and EI s1rerun. The brief
    said k 2,000; this matches support-curves' stage 3.
  - **B:** base s0, T 1.0, 1,666,667 attempts on each of the six longest survivors (10,000,002
    total).
  - **C:** held-out greedy, then the 82 forward-crux theorems at k 10,000. Each survivor then got
    200,000 attempts at T 0.8 and, if still unsolved, 200,000 at T 1.0.
  - **D:** per-token log p under base s0 of EI s0's proofs.
    - Set S: 133 distinct EI s0 proofs of the 29 survivors.
    - Controls: C1, 164 EI proofs of the other 53 crux theorems; C2, 62 of base s0's own proofs;
      C3, 53 of base s0's own crux proofs (C3 was not pre-registered).
  - 41,011,314 samples in all.
- **Pods:** 13 created, all community cloud (A40s were out of stock), all deleted.
  - `sf1` RTX 5090 at $0.69/h.
  - `sf4`, `sf5`, `sf7`, `sfa`, `sfe`: RTX 3090 at $0.22/h.
  - `sfd`: L40S at $0.79/h.
  - Six 5090s with a broken CUDA stack, each deleted within 4 min.
- **Cost:** 25.17 pod-hours by `podbudget`. The executor summed **≈ $8.81** from `list-pod-billing`;
  the reviewer did not re-derive this figure. `podbudget` books $12.57 because it charges $0.50/h.
  The budget was 30 h / $15.
- **Bucket (public):**
  - `hf://buckets/dan-pandori/nd-rl/support-followups/ckpts/sf/stage1_big_seq_s{0,1}.pt`
  - `hf://buckets/dan-pandori/nd-rl/support-followups/artifacts/sf/`
  - `hf://buckets/dan-pandori/nd-rl/support-followups/data/sf/`

  The reviewer listed both checkpoints and 66 artefact entries. The base and EI checkpoints are
  under `support-curves/ckpts/`.

## Observations (reviewer-confirmed)

- **Lean re-check:** 451 / 451 distinct counted proofs were accepted. There were no banned tokens,
  and both negative controls were rejected.
- **A. The seed-1 column replicates on an uploaded checkpoint.** Every A prediction hit.
  - EI s1rerun solves **147 / 383**; the lost EI s1 solved 144.
  - The base s1 re-draw solves 37, the same as before.
  - Forward crux (base s1 solves 0, EI s1rerun solves some): **110**. Reverse crux: 0.
  - Survivors: EI s1rerun solves **29 / 29**, and base s1 solves 2 (la_transfer_543, _1645).
  - Per-theorem agreement:
    - EI s1rerun vs lost EI s1: **95.6 %**.
    - EI s1rerun vs EI s0: 90.1 %.
    - Base re-draw vs support-curves' draw: 98.4 %.
  - The "survivor" set was defined on base s0, so base s1 solving 2 of them is not a contradiction.
- **B. One of the six longest survivors falls; five stay at zero.**
  - la_transfer_1932 (`L_true` 11): **1 success in 1,666,667** at T 1.0 (attempt 985,640), p̂ 6.0 × 10⁻⁷.
    The executor's pooled 4.8 × 10⁻⁷ mixes in T 0.8 attempts. The T 1.0-only pooled rate is 5.4 × 10⁻⁷.
  - The other five: 0 each, p < 1.8 × 10⁻⁶ (95 %).
  - The one proof is **not EI's route**. It has 8 lines and term size 59, against EI's 11–12 lines and
    term 75–78. It closes the goal with `n63.elim` on `n63 : ¬¬P`.
  - Reviewer's note: the proof also shadows `n63` / `n64` and uses an out-of-order name `n23`. It
    parses and Lean accepts it, so it counts.
  - Expectation "≥ 5 of 6 at 0, total ≤ 3": hit.
- **C. An 8× larger base does not reach the survivors, but it is not a stronger prover either.**

  | | base s0 | big s0 | big s1 |
  |---|---|---|---|
  | held-out greedy overall | 0.909 | 0.879 | 0.884 |
  | held-out 6-line | **0.687** | **0.471** | **0.485** |
  | final val loss | 0.0729 | 0.0724 | 0.0725 |
  | forward crux solved (of 82) | 0 by construction | 6 | 8 |
  | survivors reached (of 29) | — | **0** | **2** (la_transfer_1932 5 / 155,408 at T 0.8; la_transfer_2169 1 / 200,000 at T 1.0) |

  - **The falsifier (≥ 15 of 29) does not fire** for either seed or for their union (2).
  - Seed 0 missed all three of its expectations, each on the low side: held-out, crux, survivors.
  - Seed 1's expectations, set in the addendum after seeing seed 0, all hit.
  - Reviewer's reading: the 25 M model has the base's val loss and is ≈ 0.2 worse at 6-line proofs.
    This held in both seeds, against one base seed.
- **D. EI's survivor proofs are improbable at two or three steps, not across many.**
  - Classification uses the pre-registered rule on each survivor's most base-probable EI proof.
    Result: **28 concentrated / 1 mixed / 0 spread** at T 1.0, and the same at T 0.8, so the
    pre-registered reading is "a new move".
  - The expected "compounding" (≥ 20 spread) did not occur.
  - Medians:

    | set | total log p | worst step | 2nd | rest |
    |---|---|---|---|---|
    | S | −21.8 | −11.0 | −6.2 | −4.1 |
    | C1 | −12.3 | −8.1 | −4.3 | −1.7 |
    | C3 | −10.2 | −6.8 | −2.5 | −0.4 |
    | C2 | −6.2 | −4.7 | −0.6 | −0.2 |

  - 26 / 29 survivors have their worst step below C2's 5th percentile (−7.14).
  - At the worst token, base puts **median** top-1 mass 1.000 on another choice (minimum 0.27).
  - The base's alternatives there: another formula 10, another rule 8, `<eos>` 7, `exact` 3,
    another citation 1.
  - Surprisal share: logic 0.51, syntax 0.33, names 0.16. Syntax was expected ≤ 20 %, a miss.
- **The classification rule does not separate survivors from controls.** It labels 42 / 53 of C1's
  best proofs concentrated. The survivors differ from the controls in degree: deeper worst steps and
  a heavier remainder.
  - Reviewer's erratum: the blind §Recount first said C1 was "indistinguishable". That compared
    unlike units; on the pre-registered unit the survivors are deeper.
- **Truncation:** every job ran at `max_new` 512. The pre-registered rule "raise later jobs to
  640" was not applied, and the deviation is disclosed.
  - Seven strata exceed 0.1 %. The largest are A base s1 `L_true` 10 at **1.39 %** and C big s0
    `L_true` 11 at 1.25 %.
  - The truncated samples are degenerate loops on a few theorems.
  - No zero-success claim is at risk: truncation is ≤ 2.3 % of any survivor's attempts.
  - One exception: A base s1's result on la_transfer_1004 rests on ≈ 7,200 completed attempts, not
    10,000.

## Reworded or not supported (review §Verdict)

1. The run's "is not a capacity artefact" becomes:
   > not removed by 8× parameters at the same data and steps; that model has the base's val loss
   > and is worse at 6-line held-out proofs (0.47–0.49 vs 0.69), so a *stronger* larger base is
   > untested.
2. The run's "consists of a few structural decisions the base makes confidently the other way"
   becomes a statement about **EI's proofs under base s0**, one seed pair. It is not a statement
   about why the theorems are unreachable: B reached la_transfer_1932 by another route.
3. "Every non-EI survivor proof differs from all of EI's" is **false as worded**. Base s1's A proofs
   of la_transfer_543 and _1645 *are* EI proofs. The claim holds for the base s0 proof and the big s1
   proofs.
4. The run says D's totals reproduce `sc_secondary.py` "within 0.03 nats". In fact the median
   difference is 0.016 nats, 98 of 339 are above 0.03, and the maximum is 0.17. The pre-registered
   1e-3 check **failed**.
   - The reviewer's CPU fp32 re-score differs from both by up to 0.36 nats. The reviewer reads this
     as GPU reduced-precision noise.
   - No classification is affected: the nearest threshold margin is 0.47 nats.
5. "Mass 1.000 on another choice" becomes "median 1.000 (minimum 0.27)".
6. The run should say D is seed 0 only, and give base s0's held-out 6-line accuracy beside big's.

**Not supported:** that the survivors' unreachability is *explained* by the one or two improbable
steps in EI's proofs. Three things argue against it:
- The rule labels most controls the same way.
- The survivor/control difference is one of degree.
- The base reached one survivor by a route EI never used.

Minor points:
- The pre-registration's "231 proofs" for S does not match the 133 actually and correctly used, and
  the change is not explained.
- The A comparison changes batch (4,096 vs 2,048) and `max_new` (512 vs 400) without quoting
  `NOISE_FLOOR.md`. It is a sampling re-draw, not a correctness change.

## Limitations and next step

- The base is one 3.2 M model, and D is n = 1 (base s0 / EI s0).
- C tests "more parameters at the same data and steps", not a stronger base.
- The survivor set is defined on base s0.
- Stage B's single success is one proof.

Next measurements (reviewer's list):
1. **D on seed 1** (base s1 × EI s1rerun). This needs forward passes only and is cheap enough for
   the VPS. It makes D n = 2.
2. **A real capacity test.** Train the 25 M model until it beats base s0 on held-out 6-line greedy
   (more steps or a tuned lr, 2 seeds). Then re-run the 29 survivors at 200,000 attempts per
   temperature.
3. **Why unreachable.** Enumerate short Lean proofs of each survivor (routes like B's `Not.elim`),
   score them under base s0, and compare the best alternative's log p with EI's proof.
   - If survivors have alternatives near −15 nats and are still unreached, the limit is search.
   - If every route is ≤ −30 nats, the limit is the model.

Figures (not checked by the review; linked, not copied):
[`sf_d_steps.png`](https://github.com/dan-pandori/nd-takehome/blob/edea854/figures/sf_d_steps.png),
[`sf_c_reach.png`](https://github.com/dan-pandori/nd-takehome/blob/edea854/figures/sf_c_reach.png).
