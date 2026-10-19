---
written_on: 2026-09-21
written_by: agent:claude
---

# lean-format: Lean as the training format for the from-scratch model (proposal 8)

Status: completed 2026-09-21 08:10 UTC (autonomous executor session, four RTX 3090 pods,
≈ $6.2 of a $30 budget; the only exception to the 2026-09-18 pause) and independently reviewed
2026-09-21 08:44 UTC. Review verdict: no hard-constraint violation, no quarantine; every number
in the executor's write-up reproduces from the pulled files with independent code (18 claim
rows; 58,201 / 58,201 counted proofs accepted by `nd_verify` and by Lean). Six of the
executor's sentences are reworded and one explanation is not derivable from the files; numbers
and wording below are the reviewer's. Run id `lean-format`.

Review: [`review_lean-format.md`](https://github.com/dan-pandori/nd-takehome/blob/0976494/review_lean-format.md)
(commits `a191f9c` phase 1 blind recount, `0976494` phase 2 claims table and verdict; reviewer
code `review_lean_format_recount.py`, `review_lean_format_splits.py`,
`review_lean_format_lean_recheck.py`; outputs `artifacts/review_lf/`). Executor write-up:
[`run_lean_format.md`](https://github.com/dan-pandori/nd-takehome/blob/0976494/run_lean_format.md),
numbers `numbers.md` § lean-format, log `log.md` § lean-format, on branch `dan_lean_format` of
`dan-pandori/nd-takehome` (read the review first: the executor's files were **not** edited after
the review, so they still carry the wording the verdict asks to change). Proposal:
[`docs/proposals/2026-09-20-lean-training-format.md`](../../docs/proposals/2026-09-20-lean-training-format.md);
decision note [`docs/project_strategy/2026-09-20-lean-default.md`](../../docs/project_strategy/2026-09-20-lean-default.md).

Question: if the 3.2M from-scratch model is trained on `nd2lean.py`'s Lean rendering instead of
the token format — same model, schedule and data — is it **not worse** on the proposal's three
predictions (held-out greedy within 2 pp; depth-3 f = 0 acquisition ≥ 0.27; ladder transfer
`L*` ≥ 10)? Decision rule (proposal): not worse on all three → Lean becomes the training format
and the token format is retired; worse → Lean stays checker of record and reporting format only.

## Configuration

- **Code and data revision.** `dan-pandori/nd-takehome`, branch `dan_lean_format`;
  pre-registration
  [`preregistration/lean-format.md`](https://github.com/dan-pandori/nd-takehome/blob/0976494/preregistration/lean-format.md)
  committed 03:00:36 UTC (`c2dcd0a`), first pod `lf-1` 03:04:23 UTC (`~/pods.log`) — gate 0 holds
  (reviewer). Final executor commit `98ab25a`. `nd_verify` identical to `origin/main`;
  `nd2lean.py` unmodified by the run (blob `1d5cf06`); `artifacts/TEST_RUN_DONE` unchanged (blob
  `1667609`; the orchestrator's gate-1 FAIL on its mtime is the known false positive); no
  evaluation file read by training code (`pod/lf/sync.sh` copies `targets/` to the pods; nothing
  reads it except `validation_36.jsonl` for exclusion keys).
- **Surface form.** `lean_tok.py`: `nd2lean`'s rendering, one symbol per token, vocabulary 107
  (token format 99); total sequence length 1.21–1.23× the token format's. `nd2lean`'s hypothesis
  names are line indices, which the proposal did not fix, so two naming schemes were
  pre-registered: **`lean_rand`** (random labels; **primary**) and **`lean_seq`**
  (first-appearance order plus a random offset; secondary). Ablation `lean_seq --no_shift`
  (names `n1…` only).
- **Model and schedule.** 4 layers, d 256, 8 heads, 3,214,336 parameters (token 3,210,240; the
  difference is the 8 vocabulary rows); Stage-1 6,000 steps, bs 128, lr 1e-3 → 1e-4, warm-up 200,
  wd 0.1, cap 6; EI fine-tunes 600 steps at 3e-4 (all read from the checkpoints by the reviewer).
- **Reward.** A sample counts iff Lean 4.34 (core, via `elan`) accepts the literal sampled text
  **and** `nd_verify` accepts the ND proof it denotes; counts are distinct denoted proofs.
- **Commands** (`pod/lf/jobs.py <d3|ladder> <rand|seq>`; one pod per role × scheme):
  - Stage-1, depth-3 f = 0 set a1, seeds 0 and 1:
    `train.py --data data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl --mode lean_<sch> --steps 6000 --bs 128 --cap 6 --seed <s>`
  - Depth-3 dial, EI and frozen control (`--no_train`), one per Stage-1 seed:
    `expert_iter.py --init ckpts/lf/stage1_a1_<sch>_s<s>.pt --targets data/p2/targets_depth3.jsonl --transfer data/p2/transfer_depth3.jsonl --heldout data/p2/heldout.jsonl --train data/p2/train_depth3_f0_a1.jsonl --rounds 8 --k 32 --temperature 0.8 --batch 768 --seed <s>`
  - Stage-1, full cap-6 set, seed 0 (one model per scheme, plus `--no_shift` for `lean_seq`):
    `train.py --data data/train.jsonl --heldout data/heldout.jsonl --mode lean_<sch> --steps 6000 --bs 128 --cap 6 --seed 0`
  - Mechanism test: `eval_set.py` greedy on `data/heldout.jsonl`, and `--k 16 --temperature 0.8 --seed 0` on `data/transfer.jsonl` (1,638 theorems).
  - Ladder rung T1 and frozen control, EI seeds 0 and 1 **from the one full-set Stage-1 model**:
    `ladder_ei.py --init ckpts/lf/stage1_full_<sch>_s0.pt --name la_{T1,frozen}_<sch>_s<s> --outdir artifacts/lf --seed <s> --batch 512 [--no_train]` (8 rounds × 32; pools byte-identical to ladder-A's: `transfer.jsonl` `e0524d0`, `rl_targets.jsonl` `69233bc`).
  - Timing: one frozen token-format round and one frozen Lean round, solo on the same pod.
  - Token comparators are on file, not re-run: `artifacts/p2/{ei,frozen}_depth3_f0_a1_s{0,1}`,
    `artifacts/stage1_*_transfer2_k16_norm.json`, ladder-A's `la_T1_*` / `la_frozen_*`.
- **Pods and cost.** `lf-1` … `lf-4`, RTX 3090 $0.50/h, created 03:04:23 / 03:04:49 / 03:05:38 /
  03:06:11 UTC (`~/pods.log`), deleted ≈ 04:32 (`lf-1`, `lf-3`), 07:37 (`lf-2`), 08:03 (`lf-4`)
  (`log.md`); ≈ 12.4 pod-hours; balance $127.78 (03:03) → $121.56 (08:04) = **≈ $6.2**;
  pre-registered $7–12 planned, $30 ceiling. Four pods instead of the pre-registered two (logged
  deviation: same pod-hours, half the wall-clock). `podls` empty at 08:35 (reviewer) and 08:55.
- **Artifacts.** `hf://buckets/dan-pandori/nd-rl/lean-format/{ckpts,artifacts,data}` —
  `ckpts/lf` (Stage-1 models, depth-3 EI rounds), `ckpts/ladder` (ladder EI rounds),
  `artifacts/lf` (round files, found files incl. the git-ignored `found_1…7`, gate logs,
  disagreement files), `data` (the a1 training set and the ladder pools); 789 files.

## Observations

All values are the reviewer's recount; they equal the executor's in every cell.

### Decision rule, as registered

| scheme | P1 held-out ≥ 0.863, both a1 seeds | P2 acquisition ≥ 0.27, both seeds | P3 `L*` ≥ 10, both seeds | verdict |
|---|---|---|---|---|
| `lean_rand` (**primary**) | **no** (0.882 / 0.790; full set 0.830 vs band ≥ 0.928) | yes (0.433 / 0.462) | yes (10 / 10) | **worse** |
| `lean_seq` (secondary) | yes (0.909 / 0.896; full set 0.936) | yes (0.476 / 0.479) | yes (11 / 11) | **not worse** |

"Lean not worse" overall holds through the secondary scheme only — **two tries at a fixed
band**, which the pre-registration promised to say plainly and the executor's write-up does not
say in those words. Token values: held-out 0.883 / 0.883 (a1), 0.948 (full); acquisition
0.335 / 0.364; `L*` 10 / 10.

### The result that matters for the project question (robust at n = 4 Stage-1 models)

**A Lean-format base model trained on zero depth-3 proofs writes depth-3 proofs for 13–28 % of
targets at 256 attempts, where the token-format base writes them for 0.5 %**, and solves ≈ 10×
more ladder transfer theorems frozen (222–309 vs 22–24 of 2,285). The reviewer verified that the
training file holds no depth-3 proof in either rendering (ND box depth and nested-`fun` depth:
0 of 155,000), so this is composition by the base model, not a leak. **The EI increment over the
frozen control is no larger in Lean than in tokens.**

| depth-3 dial (1,000 targets, 8 × 32) | EI acquisition | frozen, same Stage-1, equal attempts | EI − frozen |
|---|---|---|---|
| token s0 / s1 | 0.335 / 0.364 | 0.005 / 0.005 | 0.330 / 0.359 |
| `lean_rand` s0 / s1 | 0.433 / 0.462 | 0.202 / 0.280 | 0.231 / 0.182 |
| `lean_seq` s0 / s1 | 0.476 / 0.479 | 0.206 / 0.134 | 0.270 / 0.345 |

First depth-3 proof is in round 1 in every Lean arm, frozen included (round-1 acquisition
0.085–0.172 at k = 32). Almost every frozen-acquired target is also EI-acquired (frozen-only 0–2).

| ladder T1 (transfer, 2,285) | solved | `L*` | solved at `L_true` ≥ 10 / ≥ 11 | frozen solved, `L*` | **`L*` − `L*`_frozen** |
|---|---|---|---|---|---|
| token s0 / s1 | 612 / 623 | 10 / 10 | 35 / 1, 34 / 0 | 22 / 24, 7 / 7 | **3 / 3** |
| `lean_rand` s0 / s1 | 583 / 657 | 10 / 10 | 46 / 2, 44 / 4 | 229 / 222, 9 / 9 | **1 / 1** |
| `lean_seq` s0 / s1 | 794 / 839 | 11 / 11 | 95 / 12, 94 / 13 | 304 / 309, 10 / 10 | **1 / 1** |

`lean_seq` reaches `L*` = 11 from round 5 in both EI seeds; every `L_true` ≥ 11 solve is EI-only;
no label is contradicted by a shorter accepted proof. **Reworded (reviewer):** the `L*` = 11 is a
base-model gain plus a same-or-smaller EI step (`L*` − `L*`_frozen 1 vs 3 for tokens), not a
larger RL effect; and the two "seeds" are EI seeds from **one** Stage-1 model per scheme.
`lean_rand`'s two a1 Stage-1 seeds differ by 9 pp in distribution and 0.08 in frozen acquisition,
so Stage-1 seed variance in this format is large; a format-level "`L*` = 11" needs a second
Stage-1 seed (the same applies to the token comparator). Unexplained: the `L_true` = 7 bin is
solved less often by Lean T1 arms (55–118 of 300) than by token arms (139–140).

### Other rewordings the verdict requires

- "Lean is better on depth-3 and length generalisation" → holds for **`lean_seq` only**; for
  `lean_rand` it holds on the depth-3 dial only (ladder `L*` equal, transfer solved 583 / 657
  against 612 / 623).
- "`lean_rand` loses 9–12 pp in distribution" → **0.1 / 9.3 / 11.8 pp by model** (a1 s0, a1 s1,
  full set): 0 to 12 pp, strongly seed-dependent.
- "`lean_rand` failures are mostly wrong-name citations" → re-citing a different earlier line
  fixes the first failing line in **451 of 851** failures (53 %), ≈ 72 % of the *excess* over
  `lean_seq` (382 of 529). "Mostly" holds for the excess, not the total.
- "The name-offset barrier is format-independent: `--no_shift` drops to 37 / 0" → **present but
  weaker in Lean, one Stage-1 seed, and a missed prediction** (expected ≤ 2 seven-line proofs,
  found 37; token `abs-fixed` 0). The ablation model solves half as many transfer theorems
  (466 vs 935) while scoring slightly higher in distribution (0.939 vs 0.936).
- **Not derivable from the pulled files:** the log's explanation that the 37 come from name
  shadowing — literal texts of accepted proofs were not saved.
- Supported as worded: "part of the token format's 'new capability from RL' was a surface-form
  barrier". The candidate mechanism (unary `| | |` depth prefix unseen at depth 3 vs a nested
  `fun … => by` that looks locally the same at every depth) is **untested** and labelled so.

### Mechanism test (full-set Stage-1, pass@16 on 1,638 transfer theorems; distinct proofs)

| model | solved | written 7 / 8 / ≥ 9 lines |
|---|---|---|
| token `abs` / `rel` / `abs-fixed` (aggregate json on file) | 732 / 664 / 631 | 108 / 1 / 0; 79 / 0 / 0; 0 / 0 / 0 |
| `lean_rand` | 859 | 228 / 86 / 7 |
| `lean_seq` | 935 | 266 / 130 / 18 |
| `lean_seq --no_shift` | 466 | 37 / 0 / 0 |

### Checker and throughput

- Every counted proof: 58,201 / 58,201 accepted by `nd_verify` and by Lean through the
  unmodified `nd2lean.translate` (16 arms' found files 41,840 + mechanism-test proofs 16,361);
  theorem-swapped negative control 0 / 138.
- In-loop gate: 13.9 M distinct samples checked by both; **`nd_verify`-only 0, Lean-only 460**
  (33 per million; pre-registered ≤ 10, all of the `¬A` kind — rate missed and two unanticipated
  kinds): 206 `BOTE` rendered `na.elim`, which Lean resolves to `Not.elim` on a negation; 137
  `¬A ≡ A → False`; 117 premise lines omitted or out of order. None is a counted proof.
- Lean ≈ 170× slower per core than `nd_verify` (85.5 vs 14,497 / s), hidden by 12 workers. Solo
  frozen round, Lean / token on the same pod: 1.78, 2.20 (depth-3), 1.46, 1.63 (ladder) — all
  below the 3× stop, one above the expected 1.2–1.8×; one round each.

### Expectations that were wrong

`L*` "stays 10" (11 for `lean_seq`); `lean_rand` in-distribution loss ≤ 5 pp (up to 11.8);
disagreements ≤ 10 per million (33, two new kinds); frozen depth-3 acquisition ≤ 0.01
(0.13–0.28; the token "frozen 0" is 0.005); frozen ladder `L*` 7 (9–10); `--no_shift` ≤ 2 (37);
mechanism-test 7-line counts 50–150 (228 / 266, favourable); P7's absolute proof lengths
(65 / 57) were measured on the first 3,000 held-out records and are 83 / 73 on the full file.

## Limitations and next step

- One Stage-1 model per scheme on the ladder and in the mechanism test; two per scheme on the
  depth-3 dial. Token comparators are on file, not re-run on these pods.
- Acceptance condition (ii), Lean on the literal sampled text, is **not auditable after the
  fact**: found files hold the denoted ND proof only. Future Lean-format runs should store the
  literal text beside it.
- a1 Stage-1 held-out values come from round-file aggregates (no per-sample file, no torch on
  the VPS).
- Held-out sets overlap their Stage-1 files in 21 / 5,000 and 95 / 5,000 renaming classes once
  premise order is ignored (0 under the repository's order-sensitive definition). Same for both
  formats, but 1.9 % is the size of the margin by which `lean_seq` passed the full-set band
  (0.8 pp inside).
- If `lean_seq` becomes the training format, **the depth-3 f = 0 dial no longer measures "RL
  crossing f = 0"** (the pattern is base-reachable at 0.085–0.172 in the first 32 attempts);
  depth-3 results in Lean must be read as EI − frozen.
- `nd2lean.py`'s BOTE rendering is loose; this run is protected by requiring `nd_verify` too, but
  a pipeline using Lean through `nd2lean` alone is not. The executor's open question
  (`QUESTIONS.md` 2026-09-21 04:40: render BOTE as `False.elim na`) stands.

Next step (reviewer's list; nothing runs until Dan lifts the pause and names it): (1) two more
Stage-1 seeds per scheme on the full set, each with a frozen ladder control and one T1 arm
(≈ $8), and the same for the token model if 10-vs-11 is to carry weight; (2) a 2 × 2 of Stage-1 +
frozen controls only, {Fitch order, goal-first order} × {unary depth prefix, none}, two seeds, on
the a1 set (≈ $3), to separate the two candidate mechanisms for the base-model jump; (3) find a
pattern with frozen acquisition ≈ 0 at 256 attempts in `lean_seq` (depth-4 from a depth ≤ 2 set,
reductio) before any further f = 0 dial in Lean; (4) `--no_shift` with a second seed and stored
literal texts. The format decision itself is Dan's (see the week-39 digest).
