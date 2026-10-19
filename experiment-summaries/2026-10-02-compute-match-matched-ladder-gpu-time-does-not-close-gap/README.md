---
written_on: 2026-10-02
written_by: agent:claude
---

# compute-match: giving our 3.2 M ladder best-cap12's GPU time, as twice the samples per round, does not close the gap

Status: **completed 2026-10-02 01:28 – 11:36 UTC (executor). Independently reviewed 11:26–11:47 UTC.
Verdict: every number reproduces; the falsifier is not met. Two rewordings and two housekeeping fixes.** No
hard-constraint violation, nothing quarantined. Where the reviewer's wording differs from the executor's, this summary
uses the reviewer's. In particular, the executor's "best-state's advantage comes from the recipe, not the extra GPU
time" is **too broad**. The run tested extra *ladder* GPU time spent as more samples per round (k 32 → 64), and our
Stage-1 already had more GPU time than best's. The reviewer's supported wording: **"giving our ladder best12's GPU time,
as twice the samples per round, does not close the gap. The remaining difference sits in the recipe (model size and
architecture, Stage-1 procedure), which this run does not separate."**

Brief: [`BRIEF_compute-match.md`](../../docs/proposals/state-env/BRIEF_compute-match.md) (authorised by Dan 2026-10-02 as a
nice-to-have). It follows up [best-state](../2026-09-30-best-state-robbies-recipe-wins-after-rl-level-frozen-at-cap12/README.md),
whose best-cap12 ladders used 1.6–1.9× our A40-seconds at equal attempts (digest week 40, decision item 25, option (a)).

Question: if our 3.2 M SN-cap12 recipe's T1 ladder gets the same A40-seconds as best-cap12's, does it close the gap to
Robbie's 9.56 M recipe after RL?

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_compute-match`. **Its commits (DONE `e8d7981d`,
`9e4318f5`) and the review commits are on `dan_compute-match` only, not merged into the fork's `dan`.** New code:
`pod/cm/*.sh`, `cm_analysis.py`, `cm_compute.py`. `state_ladder_ei.py` and `state_train.py` are unchanged.

- Review: [`review_compute-match.md`](https://github.com/dan-pandori/nd-takehome/blob/5ac4b70f/review_compute-match.md)
  ([`08fd1236`](https://github.com/dan-pandori/nd-takehome/commit/08fd1236) §Recount, written blind to the executor's
  write-ups and analysis output; [`5ac4b70f`](https://github.com/dan-pandori/nd-takehome/commit/5ac4b70f) phase 2
  compare and §Verdict). Reviewer code: `review/compute-match-recount/` (its own Lean re-check, recount and splits code,
  plus the textbook72 reviewer's `rlean.py` translator). Disclosure: before committing §Recount, the reviewer saw the
  executor's DONE commit subject ("1.02x best12 A40-s"); it had already derived 1.02× from the logs.
- Executor write-up: [`run_compute_match.md`](https://github.com/dan-pandori/nd-takehome/blob/9e4318f5/run_compute_match.md),
  `numbers.md` § compute-match, `log.md` § compute-match, `artifacts/cm/{analysis_stdout.txt,summary.json,compute_stdout.txt}`.
- Pre-registration: `preregistration/compute-match.md`, committed in `0e1d9fdf` at 01:31:19 UTC. That was before the
  first ladder (01:33Z). K = 64 was committed in `c72cd6e9` at 02:24:03Z, before the s0 (02:24:13Z) and s1 (02:26:47Z)
  ladders. Only the s2 pilot ran earlier, as pre-registered.
- Bucket: `hf://buckets/dan-pandori/nd-rl/compute-match/{ckpts/cm,artifacts/cm,artifacts/compute-match}` (ladder
  directories, every eval `.jsonl`, sample dumps `artifacts/cm/dump/*.jsonl.gz`, registry rows). **Fix pending:**
  `pod/cm/read.sh` also copied its dumps to `best-state/artifacts/cm/dump`, another run's namespace (a copy-paste slip;
  the same files are under `compute-match/`).

No figures: the run made none, and the tables below are the reviewed record.

## Configuration

- **Models (every number here):**
  - **cm12** (new) = `la_T1_cm12k64_s{0,1,2}_r8.pt`. A 3,216,384-param GPT, 4 × 256, `lean_staten`, trained from scratch.
    Its Stage-1 is the inherited `stage1_SN12_s{0,1,2}.pt` from `state-cap12` (K12 `data/kh/train_k12.jsonl`, 155,000
    records, `state_train.py` 6,000 steps × 128; md5 173f4047… / b85cbc63… / 21d74747…). Then the T1 ladder at **k 64**.
  - **SN12** (inherited, `state-cap12`) = `la_T1_SN12_s{0,1,2}_r8.pt`: the same Stage-1 and the T1 ladder at k 32.
  - **best12** (inherited, `best-state`) = `la_T1_best12_s{0,1,2}_r8.pt`: `ALiBiGPT`, 9,560,832 params, `lean_staten`,
    from scratch on K12, Robbie's recipe for 1,200 s on an A40, then the T1 ladder at k 32.
- **Deviation from the brief (pre-registered, with its reason).** The brief asked for a Stage-1 trained to best-cap12's
  A40-seconds. Ours already uses **1.74×** best's Stage-1 A40-seconds at cap 12 (2,247 vs 1,288 s), so Stage-1 was not
  extended and the inherited SN12 checkpoints were reused. Only the ladder was compute-matched.
- **Ladder:** `state-cap12`'s T1 protocol (8 rounds, same targets), k raised from 32 to **64**. K came from a one-round
  pilot: (2,988 − 289) / (1,709 − 280) = 1.889 × the k 32 round, projecting 28,550 A40-s against best12's 28,451.
- **Checker: Lean alone.** `nd_verify` judges nothing: its tree hash `9437bb72` equals `origin/main`, and `state_env.py`
  imports only its formula parser.
- **Read-outs, exactly as `best-state`** (`pod/cm/read.sh`): batch 2,048, T 0.8, seed 0, `max_action` 512, `max_steps`
  96, held fixed across all arms. The reviewer diffed the argument files: they are identical to best12's apart from
  paths. The pools are textbook72 (k 256), Robbie's dev metric (/1,108, k 64), holdout250 (k 256), rr600 (k 256; Q = the
  380 with generated `L_true` 13–16), `transfer_long2` (/21, k 256) and held-out greedy (/5,000). SN12's textbook72,
  rr600 and long2 were re-read in this run at these settings. Its other cells are `best-state`'s reads at the same
  settings.
- **Seeds:** 0 / 1 / 2 in every arm (n = 3 per arm). The pre-registered MDDs are 6.5 (textbook72) and 61 (dev). They
  come from `best-state`'s 3-vs-4 design. The reviewer computes ≈ 6.9 and 65 at 3 vs 3, which changes nothing.
- **Pods** (`~/pods.log`, `~/podhours.log`): `cm-p0`, `cm-p1`, `cm-p2`, all A40 secure at $0.49/h, 01:32 – 11:22 UTC.
  Each ran one ladder; `cm-p0` and `cm-p1` first re-read SN12, and `cm-p2` ran the pilot. Per pod: 9.82 / 9.25 / 9.36 h.
  In total **28.43 pod-hours, $13.93** (`podbudget compute-match`), against a budget of $30 / 60 h registered before the
  first pod and a pre-registered plan of ≈ 32 pod-hours ≈ $16. The reviewer checked that every ladder ran alone on its
  GPU.

## Observations (reviewed)

**Read-outs** (Lean-recounted by the reviewer from the raw files; best12 recounted from the bucket; per seed s0 / s1 / s2
→ mean):

| quantity | best12 T1 (k 32) | cm12 T1 (k 64) | SN12 T1 (k 32) |
|---|---|---|---|
| textbook72 /72 | 52 / 51 / 52 → **51.67** | 36 / 36 / 36 → **36.00** | 36 / 40 / 36 → 37.33 (re-read) |
| dev metric /1,108 | 1,058 / 1,050 / 1,055 → **1,054.33** | 934 / 995 / 972 → **967.00** | 927 / 972 / 961 → 953.33 |
| holdout250 /250 | 239 / 237 / 237 → 237.67 | 220 / 224 / 225 → 223.00 | 217 / 221 / 223 → 220.33 |
| rr600 Q /380 | 377 / 375 / 376 → 376.0 | 232 / 295 / 320 → 282.33 | 227 / 293 / 317 → 279.00 (re-read) |
| rr600 all /600 | 575 / 570 / 572 → 572.33 | 379 / 462 / 483 → 441.33 | 365 / 459 / 471 → 431.67 (re-read) |
| `transfer_long2` /21 | 21 / 21 / 20 → 20.67 | 11 / 11 / 15 → 12.33 | 9 / 14 / 18 → 13.67 (re-read) |
| held-out greedy | .9920 / .9900 / .9930 → .9917 | .9642 / .9768 / .9758 → .9723 | .9634 / .9756 / .9734 → .9708 |

| contrast | textbook72 | dev | holdout250 | Q | long2 | held-out |
|---|---|---|---|---|---|---|
| **best12 − cm12** | **+15.67** [15, 16] | **+87.3** [60.3, 119.3] | +14.7 | +93.7 | +8.3 | +.019 |
| best12 − SN12 | +14.33 | +101.0 | +17.3 | +97.0 | +7.0 | +.021 |
| cm12 − SN12, paired per seed | −1.33 (0 / −4 / 0) | +13.7 (+7 / +23 / +11) | +2.7 (+3 / +3 / +2) | +3.3 (+5 / +2 / +3) | −1.3 (+2 / −3 / −3) | +.0015 |

Intervals: bootstrap over seeds within arm, n = 3 per arm, so coarse.

1. **At matched ladder A40-seconds (1.02×), best12 still beats cm12 by +15.7 textbook72 and +87.3 dev.** Both are beyond
   the pre-registered MDDs, and every best12 seed beats every cm12 seed (textbook72 51–52 vs 36; dev 1,050–1,058 vs
   934–995), so **the falsifier is not met**. The dev gap's lower interval bound (60.3) sits at the MDD.
2. **Doubling k from 32 to 64 moved nothing beyond the noise floor.** On textbook72 the change is −1.3 (0 / −4 / 0). On dev,
   cm12 is above SN12 on all three paired seeds, but +13.7 is well inside the MDD of 61. On Q, holdout250 and held-out
   greedy, cm12 is higher on every seed by small amounts. long2 is mixed.
3. **Ladder** (cumulative distinct `rl_targets` with a Lean-valid stored proof at round 8, /4,495): cm12 4,206 / 4,268 /
   4,258 (mean 4,244) vs SN12 4,171 / 4,228 / 4,215 (from the `state-cap12` reviewer), i.e. +35 / +40 / +43. Transfer
   (/2,285): cm12 2,019 / 2,082 / 2,070 vs SN12 1,960 / 2,027 / 2,024. Most targets are first solved in round 1 (3,719 /
   3,823 / 3,791); rounds 2–8 add 487 / 445 / 467.
4. **Term size:** on the targets all nine checkpoints solve, the three arms' proofs are the same size. On textbook72's 28
   common targets: lines 5.3 / 5.0 / 5.0, term size 4 / 4 / 4 (cm12 / SN12 / best12). On rr600's 289: lines 12.3 / 12.3 /
   11.7, term 7.3 / 7.3 / 7.0. **The gap is in which targets are solved, not in proof size.** Over all solved textbook72
   targets, best12's median is 8.8 lines / term 5.3 against cm12's 6.7 / 4.0, because best12 solves longer theorems.
   The reviewer's term size counts rule applications, excluding PR / AS / R. The executor's "median 7–8 lines, term size
   5–6" uses a different definition and is consistent, but not derivable at that definition.

**Compute** (A40, one ladder per pod, nothing else on the GPU during the ladder):

| per ladder (8 rounds) | cm12 s0 / s1 / s2 → mean | best12 mean (`best-state` rows) | ratio |
|---|---|---|---|
| A40-seconds | 27,485 / 29,226 / 30,675 → **29,129** | 28,451 | **1.02×** |
| attempts | 3,529,640 each | 1,937,800 (1,793,960 clean) | **1.82×** ⚑ |
| generated tokens | 756.3 M / 772.8 M / 770.7 M → 766.6 M | 402.7 M | **1.90×** ⚑ |
| Lean checks | 1.80 M / 2.05 M / 2.08 M → 1.98 M | 1.29 M | **1.40–1.62×** ⚑ |
| train steps / tokens | 4,800 / 699–731 M (mean 720 M) | 4,800 / 732 M | 1.00× / 0.98× |
| Stage-1 A40-s (inherited labels) | 2,244–2,250 (SN12 s2–s3 on A40) | 1,288 | **1.74×** ⚑ |

- ⚑ = above the policy's 1.25×. **Matched on A40-seconds as designed; not matched on attempts, tokens or Lean checks.**
  The 3.2 M model samples faster, so at equal GPU time cm12 drew 1.8× best12's attempts. That favours cm12, so the
  conclusion is conservative in that direction. Stage-1 + ladder: 1.06×.
- The registry rows (`gpu_seconds`) match the logged round times within 4 s. The reviewer did not re-derive Stage-1's
  A40-seconds; they are inherited labels.
- Read-outs: 2,602 / 2,730 / 2,826 A40-s per cm12 seed (317,320 attempts each); SN12 re-reads 1,466–1,579 s.
- Peak memory: 7.7–16.9 GB per read at batch 2,048; ladder round 1, 13.2 GB.

**Lean re-check (reviewer).**
- Every solved target in every read file has a Lean-accepted stored proof: 21,072 targets for cm12 and the SN12 re-reads,
  20,685 for best12. One translator miss (`heldout_4919`, cm12 s1 greedy) was the reviewer's argument-order heuristic; its
  literal text is Lean-accepted.
- **2,079 / 2,079** literal gate-accepted sampled texts pass Lean, sampled from 15 dumps. The negative control, 600
  gate-rejected texts, gives **0 / 600** accepted. The accepted texts contain no `sorry`, tactic or `exact?`-style call.
  The axioms are only `propext`, `Classical.choice` and `Quot.sound`.
- 900 / 900 ladder proofs are accepted (150 from `found_8` and 150 from `found_transfer_8` per seed).
- **Splits:** disjoint by the project's renaming class (premise order kept). Order-free, 29 eval items are premise
  permutations of a training theorem: 1 textbook72 (the disjunctive syllogism, as found by earlier reviewers), 2 dev1108,
  22 held-out and 4 transfer. These are inherited and shared by every arm, so the contrasts are unaffected.

## Expected vs outcome (pre-registered, `0e1d9fdf`)

- **Hits:** cm12 dev 967 (975, 950–1,010); holdout250 223 (224, 218–230); long2 12.3 (16, 12–20, at the edge); held-out
  greedy 0.972 (≈ 0.975); cm12 − SN12 dev +13.7 (+25, 0 to +60); ladder cumulative 4,244 (≈ 4,260); **best12 − cm12 +15.7
  textbook72 and +87.3 dev (≈ +12 and ≈ +80, gap stays)**; K = 64 (56 or 64).
- **Misses:**
  - cm12 textbook72 **36.0** against a predicted 40 (37–44). The write-up shows this only through the paired miss.
  - cm12 Q **282.3** against 330 (300–360).
  - cm12 − SN12 textbook72 **−1.3** against +2.5 (0 to +6).
- Pod-hours ≈ 32 → **28.43**.

## Limitations

- **The run does not identify which part of the recipe carries the gap.** Model size and architecture, and the Stage-1
  procedure, all differ between the arms.
- **Only one way of spending the extra ladder compute was tested:** more samples per round. Spending it on more
  fine-tune steps per round is untested.
- **Our Stage-1 was not compute-matched; it already had 1.74× best's.** The match is on ladder A40-seconds only.
- **Cut-offs.** Two reads are above the policy's ≈ 0.1 % threshold: **SN12 s0 textbook72 at 0.168 %** (31 / 18,432) and
  **cm12 s0 holdout250 at 0.145 %** (93 / 64,000). The executor's "≤ 0.17 %" is accurate, but it did not flag these. Both
  reads are at the pre-registered `best-state` settings, so the comparison is fair. `best-state`'s cap diagnostic
  showed such cut-offs are non-terminating actions, and that raising the caps changes ≤ 2 textbook72 solves. Per-stratum
  fractions are not derivable from the summaries. Ladder target sampling is cut at 0.05–0.08 %.
- **n = 3 per arm.**
- **Registry labels (fix pending):**
  - The ladder rows carry `arm = la_T1_cm12k64_s<seed>`, so aggregating by `arm` splits each ladder in two;
    `cm_compute.py` works around this.
  - Some read rows carry `seed` 8 / 1008 / 2008 for seeds 0 / 1 / 2.
  - `git_sha` is null in the ladder `round_*.json`, `args.json` and registry rows.
- **Spend** ($13.93) is from `podbudget` / `~/podhours.log` at known rates. The reviewer did not re-derive it.

## Next step (reviewer)

1. **A 2 × 2 on Stage-1, one change at a time from SN12.** Run each cell through the same k 32 ladder with 3 seeds: our
   3.2 M model on Robbie's Stage-1 procedure, and Robbie's 9.56 M `ALiBiGPT` on our K12 data and `state_train` schedule.
2. **A ladder that spends the extra GPU time on more fine-tune steps per round,** rather than more samples.
3. **Housekeeping:** remove or relabel the `best-state/artifacts/cm/dump` bucket copies, and note the registry seed
   labels 8 / 1008 / 2008.
