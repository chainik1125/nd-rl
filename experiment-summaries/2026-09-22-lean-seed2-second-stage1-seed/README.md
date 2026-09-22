---
written_on: 2026-09-22
written_by: agent:claude
---

# lean-seed2: a second `lean_seq` Stage-1 seed on the ladder rung, after the BOTE fix

Status: completed 2026-09-22 08:08 UTC (autonomous executor session, two RTX 3090 pods,
≈ $2.3 of a $10 budget; a pause exception authorised by Dan) and independently reviewed
2026-09-22 08:24 UTC (driver log; the review's own stamps read ≈ 08:45 / 09:00). Review
verdict: no hard-constraint violation, no quarantine; every number in the executor's write-up
and `numbers.md` reproduces from the pulled files with independent code (24 claim rows;
20,735 / 20,735 counted proofs re-verified by `nd_verify`, 2,101 / 2,101 by Lean with an axiom
check, 40 / 40 corrupted controls rejected). Four of the executor's sentences are reworded, one
caption claim is not derivable from the files, and one value sits outside its pre-registered
band; numbers and wording below are the reviewer's. Run id `lean-seed2`.

Review: [`review_lean-seed2.md`](https://github.com/dan-pandori/nd-takehome/blob/51604b3/review_lean-seed2.md)
(commits `860bfda` phase 1 blind recount, `51604b3` phase 2 claims table and verdict; reviewer
code `review_ls2_recount.py`, `review_ls2_lean.py`; outputs `review_out/lean_seed2_recount.json`,
`review_out/lean_seed2_lean_recheck.json`). Executor write-up:
[`run_lean_seed2.md`](https://github.com/dan-pandori/nd-takehome/blob/51604b3/run_lean_seed2.md),
numbers `numbers.md` § lean-seed2, log `log.md` § lean-seed2, figure `figures/lean_seed2.png`,
on branch `dan_lean_seed2` of `dan-pandori/nd-takehome` (the executor's files were **not**
edited after the review; read the review first). Pre-registration:
[`preregistration/lean-seed2.md`](https://github.com/dan-pandori/nd-takehome/blob/51604b3/preregistration/lean-seed2.md).
Parent run: [lean-format (proposal 8)](../2026-09-21-lean-format-training-format/README.md);
decision note
[`docs/project_strategy/2026-09-20-lean-default.md`](../../docs/project_strategy/2026-09-20-lean-default.md).

Question: proposal 8's "result that would matter" — transfer `L*` = 11 on the ladder rung
against the token format's 10 / 10 — rested on **one** `lean_seq` Stage-1 model (seed 0) with
two expert-iteration (EI) seeds. Does a second, independently trained Stage-1 model (seed 2)
reproduce it? Registered rule: "`L*` = 11 rests on two Stage-1 models" iff both EI arms of
seed 2 reach transfer `L*` ≥ 11 at round 8; one of two → "two models, three of four arms";
neither → "one model", reported as not reproduced. Secondary: held-out accuracy and the
mechanism numbers (7- / 8-line proofs at pass@16) for the second model. Step 1, before any
training: fix `nd2lean.py`'s loose BOTE rendering (the open question from lean-format).

## Configuration

- **Code and data revision.** `dan-pandori/nd-takehome`, branch `dan_lean_seed2` (worktree from
  `dan_lean_format`). BOTE fix committed `39bdc5b` at 05:43 UTC; pre-registration `123df56` at
  05:44:42 UTC; first pod `ls2-1` created 05:46:06 UTC (`~/pods.log`); first pod job 05:48:37 UTC
  — gate 0 holds (reviewer). Final executor commit `a3e5d23`. `nd_verify/` tree hash `9437bb7`
  equal to `origin/main` (unmodified); `artifacts/TEST_RUN_DONE` blob `1d5cf06` unchanged (no
  test-file run); no evaluation file read as training data (`train.py` reads `--heldout` for
  validation loss only; `mix_rl_records` in every `round_r.json` equals the reviewer's recount);
  all 15 pairwise split intersections by renaming class are 0; ladder pools byte-identical to
  ladder-A's (`transfer.jsonl` `e0524d0`, 2,285; `rl_targets.jsonl` `69233bc`, 4,495).
- **Step 1, BOTE fix (`39bdc5b`).** `nd2lean.py` and `lean_tok.py` render BOTE as `False.elim nA`
  (was `nA.elim`, which Lean resolves to `Not.elim` on a negation). The diff changes exactly the
  BOTE term in `nd2lean.py` and the BOTE tokens plus inverse in `lean_tok.py` (reviewer). The
  460 known "Lean yes, `nd_verify` no" texts from lean-format re-checked: the 206 `.elim` cases
  are now rejected by Lean; 137 `¬A ≡ A → False` cases remain Lean-accepted (not closable in the
  translator); 117 unrestated-premise cases are structural rejections in `nd2lean --check`
  (`recheck460.jsonl`, 460 rows; consistent, not re-run by the reviewer because it needs the old
  rendering path). 20,000 / 20,000 sampled pool proofs (684 with BOTE) both-accept under the fixed
  checker of record (re-tabulated). BOTE occurs in 2,951 / 154,990 training proofs (1.90 %).
- **Model and schedule** (identical to proposal 8's ladder arms except the Stage-1 seed):
  Stage-1 `lean_seq` on `data/train.jsonl` (154,990 cap-6 proofs, 0 violations of the cap by the
  reviewer's counter), 4 layers, d 256, 8 heads, vocabulary 107, 6,000 steps, bs 128, lr 1e-3 →
  1e-4, warm-up 200, cap 6, **`--seed 2`** (253 s, val loss 0.0767) → `ckpts/lf/stage1_full_seq_s2.pt`.
  Ladder rung T1: `ladder_ei.py`, 8 rounds, k 32, T 0.8, `max_new` 512, 600 fine-tune steps at
  3e-4, retain 20,000, ≤ 4 proofs per theorem × 4, `--batch 512`, EI seeds 0 and 1 from the one
  seed-2 checkpoint; frozen controls `--no_train`, seeds 0 and 1, at equal attempts (256 samples
  per theorem in every arm). `L*` = max L with ≥ 5 transfer theorems solved at `L_true` ≥ L.
- **Reward and checker.** A sample counts iff Lean 4.34 accepts the literal sampled text in the
  loop (`lean_gate.py`, 2 × 2 table per call) **and**, at the end, the fixed `nd2lean.py --check`
  plus `nd_verify` accept the denoted ND proof (`pod/lf/record.py`).
- **Commands** (`pod/ls2/jobs.py`, role a on `ls2-1`, role b on `ls2-2`):
  - `train.py --data data/train.jsonl --heldout data/heldout.jsonl --mode lean_seq --steps 6000 --bs 128 --cap 6 --seed 2`
  - `eval_set.py` greedy on `data/heldout.jsonl` (5,000); `--k 16 --temperature 0.8 --seed 0` on `data/transfer.jsonl` (1,638)
  - `ladder_ei.py --init ckpts/lf/stage1_full_seq_s2.pt --name la_{T1,frozen}_seq2_s<s> --outdir artifacts/lf --seed <s> --batch 512 [--no_train]`, s ∈ {0, 1}
  - Seed-0 and token comparators are on file from lean-format and ladder-A, not re-run.
- **Deviation, pre-registered.** The seed-2 model is trained on the fixed BOTE rendering; seeds
  0 / 1 were trained on `nA.elim`. BOTE occurs in 0 of the `L_true` ≥ 11 transfer proofs of any
  arm (47 / 2,135 and 40 / 1,948 of the seed-2 T1 arms' distinct transfer proofs overall), so the
  rendering change cannot have moved `L*` (reviewer).
- **Pods and cost.** `ls2-1` (created 05:46:06 UTC), `ls2-2` (05:47:40), RTX 3090 24 GB $0.50/h
  (`~/pods.log`); job logs 05:48–07:57 and 05:50–07:54; both deleted 08:03 (`log.md`; `podls`
  showed none of the executor's pods). ≈ 4.5 pod-hours ≈ **$2.3** by list price, against a
  planned ≈ $3 and a $10 budget. No usable balance delta: the account read $121.52 → $108.00 over
  the run with twelve other agents' pods running (`numbers.md` E9). Wall-clock 2 h 32 min from
  run start (05:31) to pods deleted; rounds 13.2–17.2 min with two arms per card; hard stop
  2026-09-23 01:31 UTC not approached.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/lean-seed2/{ckpts,artifacts,data}` —
  `ckpts/lf/stage1_full_seq_s2.pt`, `ckpts/ladder/la_T1_seq2_s{0,1}_r{1..8}.pt`; `artifacts/ls2`
  (BOTE-fix checks, 20k sample, `summary.json`), `artifacts/lf` (seed-2 arm files, gate logs,
  disagreement files, record files; lean-format's files as pulled); `data` (ladder pools,
  `train.jsonl.gz`, `heldout.jsonl`, `transfer.jsonl`). Upload logs show "Sync completed"; not
  re-listed by the reviewer. Incident 08:12 UTC, no number affected: the executor's upload staging
  overwrote `data/ladder/transfer.jsonl` on the VPS after the analysis had run; restored from
  git, `summary.json` byte-identical, the pods had used the intact pool (reviewer: working-tree
  file record-for-record identical to HEAD; every found record's `L_true` equals the HEAD pool's).

## Observations

All values are the reviewer's recount; they equal the executor's `round_8.json` and
`summary.json` values in every cell.

### The registered statement holds: `L*` ≥ 11 on two Stage-1 models, four of four EI arms

| arm (Stage-1 model, EI seed) | transfer `L*` | solved / 2,285 | at `L_true` ≥ 10 / ≥ 11 / ≥ 12 | distinct proofs | targets `L*`, solved / 4,495 | `L*` by round |
|---|---:|---:|---|---:|---|---|
| **seed 2, T1 s0** | **12** | 906 | 108 / **20** / **5** | 2,135 | 11, 2,820 | 9 10 10 11 11 11 11 12 |
| **seed 2, T1 s1** | **11** | 844 | 104 / **16** / 3 | 1,948 | 11, 2,726 | 10 10 10 10 11 11 11 11 |
| seed 2, frozen s0 | 10 | 314 | 12 / 0 / 0 | 439 | 10, 1,827 | 9 9 10 10 10 10 10 10 |
| seed 2, frozen s1 | 10 | 324 | 17 / 0 / 0 | 449 | 10, 1,823 | 10 × 8 |
| seed 0, T1 s0 (lean-format, on file) | 11 | 794 | 95 / 12 / 3 | 1,767 | 11, 2,604 | reached 11 at round 5 |
| seed 0, T1 s1 | 11 | 839 | 94 / 13 / 4 | 1,801 | 11, 2,674 | reached 11 at round 5 |
| seed 0, frozen s0 / s1 | 10 / 10 | 304 / 309 | 14 / 0 / 0; 14 / 0 / 0 | 432 / 427 | 10, 1,768 / 1,759 | |
| token T1 s0 / s1 (ladder-A, on file) | 10 / 10 | 612 / 623 | 35 / 1 / 0; 34 / 0 / 0 | | | |

- **Verdict rule (E2):** both seed-2 EI arms reach transfer `L*` ≥ 11 at round 8 (12 and 11), so
  by the pre-registered statement "`L*` = 11 rests on two Stage-1 models" (four of four EI arms).
- **`L*` = 12 is one arm of four, at the rule's threshold:** exactly 5 theorems at `L_true` ≥ 12
  (`la_transfer_165, 454, 464, 499, 1585`); s1 has 3, seed 0 has 3 / 4. With two seed-2 arms at
  12 / 11 and two seed-0 arms at 11 / 11, the two-seed evidence supports "≥ 11", not "12"; "12"
  is not to be carried into any headline without the "/ 11" and the 5-theorem caveat (reviewer).
- **Frozen controls** of the same model at equal attempts stay at 10 with **0** theorems at
  `L_true` ≥ 11 and 12 / 17 at `L_true` 10 in 256 attempts; T1 reaches 108 / 104 at ≥ 10. This
  is the base-reachability number for the rung: RL moves the frontier by one bin (10 → 11) on
  this pool, the same size of gain as for seed 0 and smaller than the token format's 7 → 10,
  which starts lower.
- **Overlap:** 14 theorems at `L_true` ≥ 11 are solved by both seed-2 arms (union 22); seed 0's
  arms share 10 (union 15); the two models' unions share 11; all four arms together solve 26
  distinct `L_true` ≥ 11 theorems of the pool's 224.
- **Label check:** no pool label is contradicted — for every solved transfer theorem the shortest
  written proof is ≥ `L_true` (0 contradictions in all 8 arms, also with dependency-pruned length
  plus unused-premise lines). Written lengths of the ≥ 11 proofs equal `L_true` (11 or 12 lines).
- **Determinism:** round 1 of T1 and frozen (same checkpoint, same seed) produce identical found
  sets (265 / 272 proofs); the arms diverge only through training.
- **Depth:** T1 s0 transfer proofs are mostly depth 3 (1,630 / 2,135), 124 at depth 4, 2 at depth 5;
  frozen 349 depth 3, 7 depth 4.

### Stage-1 seed 2 in distribution and on the mechanism test

| quantity (pre-registered band) | seed 0 (on file) | **seed 2** | token |
|---|---|---|---|
| E1 held-out greedy, 5,000 (0.925–0.945) | 0.9356 (4,678) | **0.9436** (4,718; by length 2–6: 996 / 983 / 960 / 901 / 878 of 1,000) | 0.948 |
| held-out after ladder rounds 1 → 8, s0 / s1 | 0.9356 → 0.9436 / 0.9426 | 0.9436 → 0.9506 / 0.9482 | |
| E6 pass@16 on 1,638 transfer theorems (0.53–0.60) | 0.571 (935) | **0.595** (974) | 0.447 (`abs`) |
| E6 distinct proofs written 7 / 8 / ≥ 9 lines (200–320 / 90–170) | 266 / 130 / 18 | **275 / 109 / 9** | 108 / 1 / 0 |

All 4,718 held-out and 1,306 pass@16 proofs re-verified by `nd_verify`; 150 + 150 re-checked in
Lean with axioms ⊆ {propext, Classical.choice, Quot.sound}.

### Checker of record and the BOTE fix in the loop

- **E8:** the executor's `record_*` files (8 files, 20,735 rows) are all `(nd_ok, lean_ok) =
  (True, True)`; the reviewer's own `nd_verify` accepts 20,735 / 20,735; the reviewer's own Lean
  runner with `#print axioms` accepts 2,101 / 2,101 (every transfer proof at `L_true` ≥ 10 in the
  seed-2 arms and ≥ 11 in the seed-0 T1 arms, 120 random target proofs per seed-2 arm, 150
  held-out, 150 pass@16), no `sorryAx`; 40 corrupted controls (shifted citation or negated
  conclusion) rejected 40 / 40 by both checkers.
- **E7, in-loop gate:** 7,207,048 samples; 1,177,998 grammar parse failures; 5,276,013 distinct
  (prompt, text) checked by both; 1,554,603 both accept; 3,721,349 both reject; **`nd_verify`
  yes / Lean no: 0; Lean yes / `nd_verify` no: 61 = 11.6 per million** (band ≤ 25; lean-format:
  33 per million). By the reviewer's own predicates: 54 unrestated-premise texts and 7 texts
  deriving `¬A` by `IMPI` from an `A … False` box; **0 of the `.elim` kind** (206 / 13.9 M before
  the fix). None can affect a count under the both-accept rule. Lean process time 35,357 s across
  12 workers per pod, 149 proofs per process-second.
- Gate self-test per pod: 2,000 held-out proofs, every 5th paired with a wrong statement →
  1,600 both-ok / 400 both-reject / 0 disagreements.

### Expectations vs outcomes

E1–E9 committed before the first pod. Misses, all on the favourable side and none against a
stop rule: **E4 s0 = 906 is strictly outside the pre-registered 700–900 band** (the write-up's
"every number landed in its band" is not literally true); E3 s0 = 20 sits at the top of 6–20;
E2's named P ≈ 0.1 case (`L*` ≥ 12 in one arm) occurred, at the rule's threshold. E5 frozen 10 /
10, 314 / 324 (band 10; 250–350); E7 0 `.elim`, 11.6 per million (≤ 25); E9 13.2–17.2 min per
round (12–22).

### Rewordings the verdict requires

- `L*` = 12: keep it labelled as one arm at the rule's threshold; headline "≥ 11", not "12".
- "Every number landed in its pre-registered band" → E4 s0 (906) is outside 700–900; E3 s0 at
  the band's top.
- "The `L_true` = 7 bin is solved less often by the Lean arms (66–99 of 300) than by the token
  arms (139–140)" → the four Lean arms' range is **66–118** (seed 2: 99 / 66; seed 0: 75 / 118);
  still below the token arms, gap 21–74.
- The example proof's caption (`la_transfer_499`, `L_true` 12, "EI s0 round 7; frozen control
  and token arms: never"): frozen never ✓, but the theorem is also solved by seed-2 s1 (round 5)
  and both seed-0 arms (rounds 4 and 4 / 8); "token arms: never" is not derivable from files in
  the run repository (on file 1 / 0 theorems at ≥ 11, so plausible).
- Cosmetic: `STATUS.md`'s "05:48 Pre-registration committed" → 05:44:42 UTC.

### Not supported / to keep in mind (reviewer)

- **Lines vs inferences.** The `L_true` ≥ 11 theorems the arms solve need, by the reviewer's
  count of inference nodes (one per rule application other than `PR` / `AS` / `R`, +1 per `IMPI`
  / `NEGI` lambda, 3 for `ORE`, 3 for `DN`; an approximation until proposal 9's `lean_check`
  exists), only **4–8 nodes** each (`la_transfer_130`: 11); the write-up's `L_true` = 12 example
  is 7 nodes — two premise restatements, five assumption lines and a trivial `Or.elim` of
  `S ∨ S`. `L*` in lines is the pre-registered unit and stands, but "`L*` 11–12" is not depth of
  reasoning: a term-size `L*` (≥ 5 theorems at min term size ≥ T) is 9 / 8 for the T1 arms vs
  7 / 8 frozen, a one-to-two-node gain.
- **Seed 2 vs seed 0** (20 / 16 vs 12 / 13 at ≥ 11; 906 / 844 vs 794 / 839 solved; held-out
  0.944 vs 0.936) is n = 1 vs n = 1 Stage-1 model: "seed 2 is somewhat better on every measure",
  not an effect.

## Limitations and next step

- Two Stage-1 models for `lean_seq`; the token comparator is still one Stage-1 model with two
  EI seeds (ladder-A), so 10-vs-11 across formats is two models against one.
- The 460-text re-check of the BOTE fix is the executor's script output, consistent but not
  re-derived; the independent evidence for the fix is the in-loop 0 `.elim`-kind disagreements
  in 5.28 M checks.
- Bucket contents were not re-listed by the reviewer; pod create / delete times are in
  `~/pods.log` and `log.md` only. Dollars are by list price (no separable balance delta).
- The reviewer's term-size counter is an approximation; the two remaining Lean-only
  disagreement kinds (unrestated premises, `¬A ≡ A → False` at `IMPI`) are still open in the
  translator and are only harmless while `nd_verify` stays in the loop (proposal 9 retires it in
  favour of a Lean term-allowlist check).

Next measurement (reviewer): (1) `lean_check` plus term-size labels for the ladder pools
(proposal 9 phase 1, in progress in run `lean-only`, not yet reviewed), then re-report `L*` in both units for the
four existing EI arms — no new training needed; (2) if the question is whether the one-bin gain
is capacity- or budget-limited, 8 more rounds of the seed-2 s0 arm (`--resume`) at ≈ $1: `L*`
staying at 12 with the ≥ 12 counts not growing would say plateau, growth would say budget.
