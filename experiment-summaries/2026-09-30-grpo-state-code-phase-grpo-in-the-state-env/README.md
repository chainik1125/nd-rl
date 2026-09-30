---
written_on: 2026-09-30
written_by: agent:claude
---

# grpo-state (code phase): GRPO in the proof-state environment, built, tested and costed

Status: completed and reviewed 2026-09-30. No hard-constraint violation, nothing quarantined. Verdict
**stands**, with seven items to reword or fix before the experiment run (listed below; the fork files keep the
executor's text). This was the **code phase only**: no claim about GRPO is made, and the smoke numbers below are
measurements of the code (≤ 8 updates), not results. The code is merged into the fork's `dan` (`d7411772`); the
review commits are only on the run branch.

Question: build GRPO for the proof-state environment with four advantage variants (default, unlikeliness rank
penalty, pass@k, distinct-proof bonus), test it, and pre-register and cost the experiment that asks (Q1) whether
GRPO in the state environment moves the frontier or support differently from EI at equal target attempts, and (Q2)
whether advantages that credit rare correct proofs expand support rather than sharpen.

## Configuration

- **Brief:** [`BRIEF_grpo-state.md`](../../docs/proposals/state-env/BRIEF_grpo-state.md) (Dan, 2026-09-30). Budget
  $3 / 6 pod-hours; the experiment itself (≈ $10–20 in the brief) is queued separately.
- **Fork:** worktree `~/work/grpo-state`, branch `dan_grpo-state` of `dan-pandori/nd-takehome`, base `d3abe474`.
  Code `191999f4` (`grpo_state.py`, `grpo_adv.py`, CPU tests in CI); review fixes `d42cb7d8` (log-probs at the
  sampling T, boundaries for small runs, per-round environment stats, exact action ids); executor tip `d7411772`,
  merged into the fork's `dan`. GitHub Actions runs 36661560510 / 36661730711 green.
- **Pre-registration:** fork `preregistration/grpo-state.md`, `c21af4c9` (02:17:48 UTC), before the first pod
  (02:21). Addendum 1 (`cdc43d8b`, 02:35, after the smoke, and said so) adds the smoke table, the costing, and one
  change: support deepening 5,000 + 5,000 attempts (was 10,000 + 10,000).
- **Model behind every number:** SN-cap12 Stage-1 s0 (`ckpts/sc12/stage1_SN12_s0.pt`, bucket
  `state-cap12/ckpts/sc12/`), `lean_staten`, 4 layers, d 256, **3,216,384 parameters**, from scratch on
  `data/kh/train_k12.jsonl` (155,000 generator proofs, cap 12), plus ≤ 8 GRPO updates at lr 3e-5. Judge: Lean alone
  (`lean_gate` literal-text gate, then `lean_judge.judge_many`); no `nd_verify` anywhere.
- **Smoke settings:** 256 theorems × G 8, decode batch 2,048, T 0.8, lr 3e-5, no KL (except smoke2_default, KL
  0.02), `--no_eval` except smoke2. The first smoke (RTX 3090) ran the T = 1 log-prob version from before the review
  fixes (timings unaffected); smoke2 (A40, after the fixes) ran on subsets of 64 targets, 64 transfer and 200 held-out
  theorems only to exercise the boundary-evaluation path.
- **Pods:** `gs-smoke` RTX 3090 $0.50/h (02:21–02:34, 0.23 h, $0.11), `gs-smoke2` A40 $0.49/h (02:42–02:47, 0.09 h,
  $0.05). Total **0.32 pod-hours, $0.16** of $3; both deleted.
- **Run write-up:** `run_grpo_state.md`, `numbers.md` § grpo-state, `log.md`, `STATUS.md`, `QUESTIONS.md` (2026-09-30)
  on the fork branch.
- **Review:** [`review_grpo-state.md`](https://github.com/dan-pandori/nd-takehome/blob/d0506c3e/review_grpo-state.md),
  §Recount in [`7cb5af4b`](https://github.com/dan-pandori/nd-takehome/commit/7cb5af4b), §Comparison and §Verdict in
  [`d0506c3e`](https://github.com/dan-pandori/nd-takehome/commit/d0506c3e) (on `dan_grpo-state`). Reviewer code: fork
  `review/grpo-state-recount/` (own normaliser and counters, own ND → Lean translator, brute-force advantage check,
  renaming-class splits).
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/grpo-state/{artifacts/grpo_state,artifacts/grpo-state,ckpts/grpo_state}`
  (rows in the fork's `artifacts/MANIFEST.jsonl`).

## Observations

All values below were reproduced by the reviewer.

- **The advantages are what the papers define.** pass@k equals brute-force enumeration of k-subsets (Chen et al.
  §2.4) in all 370 cases tested (G ∈ {2, 4, 8, 16}, every k and c; max difference 1.7e-16), and k = 1 equals the
  default. Unlikeliness equals the reviewer's transcription of He et al. §4.1 on 2,000 random groups (difference 0).
  Default and distinct match hand values.
- **The update is on-policy** (after the review fixes): log-probs at the sampling T, the exact sampled ids up to and
  including `<eos>`, fixed divisor, grad clip 1.0. The planned budget, 561 updates × 2,048 = 1,148,928 rollouts, is
  within 0.16 % of EI's 1,150,720.
- **Smoke timings (RTX 3090, one job):** 18.6–20.7 s per update of 2,048 rollouts (mean 19.6), peak 8.05–8.60 GB.
  Two jobs on one card: 26.4–26.5 s each, peak 8.62–8.92 GB, i.e. **1.47×** throughput.
- **Base mechanics:** mean reward 0.462–0.465 per rollout, 0.565–0.608 of groups with reward variance, 0.25–0.28 of
  groups all-fail at the start of training. This contradicts E1's stated reason (a near-saturated target pool leaves
  most groups all-correct), and makes E5 (variance fraction 0.10–0.35 over a run) unlikely. The executor said so;
  § 7 was not edited after the smoke.
- **pass@4 at G 8 zeroes every group with c = 0 or c ≥ 5**, so A ≠ 0 in 0.31–0.32 of groups against ≈ 0.6 for
  default, and pass@4 trains on ≈ 0.6× default's tokens per update (0.66–0.68 M vs 1.10–1.12 M) by construction.
- **Lean re-check:** 1,527 / 1,527 counted proofs (150 random target proofs per arm, plus every transfer proof of
  smoke2) accepted by Lean under two renderings; controls rejected 200 / 200 prompt swaps and 100 / 100 `sorry`
  bodies; the 31 of 200 single-rule flips that pass are all on symmetric formulas (`A ∨ A`, `A ∧ A`), where the flip
  is still valid.
- **Bookkeeping:** found files, round files, alloc files and registry compute rows equal the reviewer's own counts
  in every record (e.g. smoke_default 1,487 solved / 2,777 distinct; accepted / tried 7,620 / 16,384 = the mean
  reward). Held-out-200 greedy 200 / 200 at both smoke2 boundaries.
- **Compute rows** (registry, per arm; `gpu_seconds` is card occupancy):

  | arm | GPU | gpu_seconds | gen_tokens | attempts | train_steps | train_tokens | lean_checks |
  |---|---|---|---|---|---|---|---|
  | smoke_default | 3090 | 159 | 3,015,571 | 16,384 | 8 | 8,802,833 | 7,201 |
  | smoke_unlikely | 3090 | 94 | 1,556,538 | 8,192 | 4 | 4,728,533 | 3,567 |
  | smoke_passk | 3090 | 87 | 1,553,995 | 8,192 | 4 | 2,734,076 | 3,587 |
  | smoke_distinct | 3090 | 94 | 1,553,906 | 8,192 | 4 | 5,142,343 | 3,584 |
  | conc_default | 3090 (2 jobs) | 169 | 2,275,190 | 12,288 | 6 | 6,731,648 | 5,320 |
  | conc_passk | 3090 (2 jobs) | 168 | 2,307,410 | 12,288 | 6 | 3,989,653 | 5,165 |
  | smoke2_default | A40 | 107 | 1,614,519 | 8,720 | 2 | 4,400,566 | 2,431 |
  | smoke2_unlikely | A40 | 97 | 1,621,632 | 8,720 | 2 | 2,429,445 | 2,436 |

- **Hard constraints clean:** `nd_verify` tree `9437bb72` unchanged, used only as a formula parser in
  `state_env.py`, never as a judge; `artifacts/TEST_RUN_DONE` unchanged; training rollouts read only `--targets`;
  budget (02:10) and pre-registration (02:17) before the first pod (02:21).
- **The experiment's cost** (Addendum 1): six seeds ≈ $37 ($43 with margin), above the brief's ≈ $10–20; Plan B
  (s0–s3, three GRPO arms, EI inherited from state-cap12) ≈ $24; Plan C (Plan B with default + pass@4 only) ≈ $16. The
  arithmetic reproduces, but the reviewer finds two inputs low: a boundary evaluation (≈ 80k rollouts) at ≈ 630 s,
  not 400 s (≈ +6 pod-hours, ≈ +$3, over 18 ladders), and support deepening at risk of up to ≈ 3× under-costing if
  more than about a third of the k256-unsolved theorems stay unreached at 10k. **Reviewer's estimate: ≈ $40–50 full,
  ≈ $27–33 for Plan B**, not derivable exactly without two short measurements.

### Reword or fix before the experiment run (reviewer's items 1–7)

1. Pre-registration § 6: the held-out row's values (0.969 / 0.977 / 0.978 / 0.974) are **Stage-1** held-out greedy,
   not "SN-cap12 T1 values". T1 round-8 values (0.962 / 0.9756 / 0.977 / 0.9772, sd 0.0073) give **MDD ≈ 1.3 pp**,
   not 0.7 pp, for final-checkpoint comparisons such as E4. E4's 2 pp threshold is still above it.
2. Pre-registration § 3: the lr evidence ("3e-5 kept held-out greedy at 0.75–0.87; 1e-4 dropped it to 0.58–0.72")
   is from run 4: token format, cap 6, Lean ∧ `nd_verify`, a different model. Label it.
3. Pre-registration § 4: add under "not matched" that EI's fine-tunes replay Stage-1 data (`--train train_k12`,
   `rl_weight 4`, `retain 20000`), while GRPO has neither replay nor KL. E4 and any support comparison inherit this.
4. The experiment needs `lpool_reread.py`, `ss_support.py` and `pod/sc12/sn_seed.sh`, none of which is on the fork's
   `dan` (they are on `dan_long-pool-2` / `dan_state-cap12` / `dan_support-state` / `dan_state-readouts`). Merge them
   before the first pod.
5. Re-cost two items, one short measurement each, before choosing among Plans A / B / C: one full boundary
   evaluation on a 3090, and the base deepening rate on the 471-theorem pool at `max_action 512 / max_steps 96`.
6. Code hygiene: `grpo_state.py` defaults are `--heldout data/heldout.jsonl` and `--lr 1e-4`; the pre-registration and
   the runner use `data/p2/heldout.jsonl` and `3e-5`, so a hand launch would drift. Store the literal assembled text
   (`res_tx`) or a `LEAN_GATE_DUMP` with the found files. Make CI's replay test yield multi-step trajectories: in CI
   all 8 CPU rollouts stop after one action, so multi-step id fidelity is not tested on CPU (the GPU smoke's
   multi-step proofs are Lean-accepted, but the ids fed to the loss are not stored).
7. `log.md` / addendum timestamps "02:50" / "02:55" for the code review are wrong: the fix commit `d42cb7d8` is at
   02:39:03. The order (fix → smoke2) is right.

Two smaller wording points: the addendum's "One RTX 3090" covers the first smoke; smoke2 ran on an A40. "A tiny CPU
run of each variant finds Lean-accepted proofs" is one proof per variant.

## Limitations and next step

- **No result about GRPO exists yet.** The smoke is ≤ 8 updates on one seed.
- **Interpretive caveats for the experiment's write-up (reviewer):** pass@4's `train_tokens` will be ≈ 0.6× default's
  by design, so the 1.25× compute flag will fire for a design reason. The unlikeliness ranking sums log-probs over
  whole multi-step trajectories, so it credits longer correct proofs more; read any E3 support effect next to a
  length breakdown. EI's held-out retention is partly bought by Stage-1 replay, which GRPO lacks.
- **Minor:** `found_*.jsonl` stores the ND record, not the literal `lean_staten` text the gate checked (the state
  ladder does the same). `rl_targets` overlaps `transfer` in 2 theorems (premise reorderings, inherited from the ladder
  pools, far inside every MDD). For `distinct` (not an arm), one update can hold the same theorem twice at a wrap of
  the target order.
- **Open question to Dan** (fork `QUESTIONS.md`, 2026-09-30): which plan to run. Default if unanswered: **Plan B**
  (4 seeds, all three GRPO arms, EI inherited from state-cap12), first measuring s / update on a 16 GB card.
- **Next measurement** (reviewer, ≈ 15 min on one pod): one full boundary evaluation (73k transfer pass@32 + greedy +
  5k held-out) on a 3090, and the base deepening rate on the 471-theorem pool. It fixes the plan's cost before a plan
  is chosen.
