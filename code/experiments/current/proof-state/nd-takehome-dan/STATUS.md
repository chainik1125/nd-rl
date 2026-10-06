# STATUS — lean-format (proposal 8)

Brief: BRIEF_LEAN_FORMAT.md. Policy: AGENT_POLICY.md. Run id: lean-format. Only exception to the pause.

## lean-format
- 2026-09-21 02:52 UTC  run started (executor). 03:00 pre-registration committed (`preregistration/lean-format.md`, c2dcd0a) before any pod. Lean token format fixed in `lean_tok.py` (two naming schemes: `lean_rand` primary, `lean_seq`).
- 2026-09-21 03:04–08:03 UTC  four RTX 3090 pods (lf-1 … lf-4), all deleted; ≈ $6.2 of the $30 budget. 16 Lean-format arms + 7 Stage-1 models + solo timing rounds + checker of record.
- Result: `lean_seq` **not worse on all three** (held-out 0.909 / 0.896 / 0.936 vs token 0.883 / 0.883 / 0.948; depth-3 acquisition 0.476 / 0.479 vs 0.335 / 0.364; ladder transfer L* 11 / 11 vs 10 / 10). `lean_rand` (pre-registered primary): worse in distribution (0.882 / 0.790 / 0.830), better on depth-3 (0.433 / 0.462), equal L* (10 / 10). Frozen Lean base models already write depth-3 proofs (0.13–0.28 vs 0.005) and have L* 9–10 (token 7). Lean vs nd_verify: 0 disagreements on 41,840 counted proofs; 460 "Lean-only" acceptances among 13.9 M sampled (nd2lean's BOTE rendering is loose — see QUESTIONS.md).
- Deliverables: `run_lean_format.md`, `numbers.md` § lean-format, `log.md` § lean-format, `figures/lean_format.png`, bucket `hf://buckets/dan-pandori/nd-rl/lean-format/`. Two questions for Dan in `QUESTIONS.md` (2026-09-21).

LEAN-FORMAT DONE 2026-09-21T08:10:42Z

# STATUS — ds-generator (proposal 9, run 2: generator proof-shape distribution)

Brief: BRIEF_ds-generator.md. Policy: AGENT_POLICY.md. Run id: ds-generator. Sibling: ds-composition (same host, own pods `dsc-*`?; mine are `dsg-*`; `~/pods.log` is shared — gate-0 confound noted).

## ds-generator
- 2026-09-22 05:54 UTC  run started (executor). Generator knobs implemented behind flags (control path byte-identical, 3 code paths diffed), VPS probes of the G1 / G2 yields, pre-registration `preregistration/ds-generator.md` written ≈ 06:25 UTC before any pod. Two design deviations pre-registered: G1 is a null manipulation at cap 6 (ORE share 1.0 vs 1.5 %); G2 as run = the ladder pools' strict long generator restricted to cap 6 (+ boxes-in-ORE knob) because the literal G2 is degenerate (84 % reductio, bins 2–4 unfillable). Question in QUESTIONS.md.
- 2026-09-23 21:00 UTC  **Resumed** after the 2026-09-22 credit cut-off. `artifacts/dsg/` recovered from git, `data/` intact, control checkpoints and the two training sets pulled from the buckets; `ckpts/dsg/` gone, so G1 / G2 Stage-1 is retrained with a held-out reproduction check. Pre-registration addendum `2733373` committed 21:06 before any pod (gate 0). Missing work being run now: ladder T1 + frozen (8 × 32) for all three arms and both seeds, the textbook-schema table, and the checker-of-record pass.
- 2026-09-23 21:10–21:55  3090 / A40 out of stock; two pods instead of three — `dsg-1` RTX A6000 48 GB $0.33/h, `dsg-2` RTX 4090 24 GB $0.34/h — and **one seed per pod with all three arms on it**, so every arm-vs-arm comparison is on one GPU. The fast decode path failed the pre-registered equivalence check *only* with compaction on (1 row of 128 flips at decode step 162); per the pre-registered fallback the run uses `ND_SAMPLE_COMPACT=0` and the originally pre-registered `--batch 512` / `--max_new` 512, which passes exactly.
- 2026-09-24 08:50 UTC  **Result.** Both pre-registered accounts fail. *Transfer wall is not the shape
  distribution*: frozen ladder solves of 2,285 are C0 158 / 114, G1 170 (s1), G2 44 / 125, against a pre-registered
  450–760 for G2; G1 — an independent pool with the control's shape table — is above both C0 seeds, and its own T1
  seeds (916, 635) bracket the whole C0–G2 gap. *Textbook wall is not the rule shape*: 10 of 19 schemata are at 0
  solves in every arm and seed, G1 and G2 leave 17–19 of 19 at ≤ 2, and the `ORE`-needing families the brief named
  (dilemma / De Morgan / distribution) are at 0 everywhere. The run's clearest positive measurement is the **noise
  floor**: retraining Stage-1 from the same set and seed moves held-out greedy −4.44 to +5.96 pp (all in the 6-line
  bin) where a byte-identical checkpoint on new hardware and a new decode path moves +0.00 / +0.08 pp. Finding 3
  keeps its counter-example (G2: lower base rate, larger EI − frozen, both seeds). Lean vs `nd_verify`:
  **0 disagreements on 57,013 counted proofs**. Cost 21.34 pod-hours, $13.02 of $14; 11 of 12 ladder jobs
  (`la_frozen_g1_s0` dropped for budget — `QUESTIONS.md`). All pods deleted.
- Deliverables: `run_ds_generator.md`, `numbers.md` § ds-generator, `log.md`, `figures/dsg_{shape,readiness}.png`,
  `artifacts/dsg/summary.json`, `data/dsg/README.md`, bucket
  `hf://buckets/dan-pandori/nd-rl/ds-generator/{artifacts/dsg,ckpts/dsg,ckpts/ladder,data/dsg}`.

DS-GENERATOR DONE 2026-09-24T08:50:00Z


# STATUS — noise-floor (proposal 11, run 2: what difference can this project resolve?)

Brief: run brief `noise-floor`. Policy: `AGENT_POLICY.md`. Run id: noise-floor. Pods `nf-*`.
Sibling on this host: `cap-horizon` (own pods, own budget). My output is its yardstick, so interim
numbers go into `numbers.md` § noise-floor as each stage lands rather than at the end.

## noise-floor
- 2026-09-24 15:59 UTC  run started (executor). Read proposal 11, the `ds-generator` summary
  (§"the measurement noise is this run's real headline constraint"), the `ds-rendering` summary's
  Finding R-2 and its bimodality note, `ds-composition`'s ladder tables, and the `pod/dsg` harness.
- 16:05:12 UTC  pre-registration `e8f4c3b` committed and pushed **before any pod** (first pod
  `nf-1` created 16:05:45 UTC — gate 0 holds, 33 s). Design: four null pools P1–P4 from the
  unmodified generator (`knobs_from_args` returns `None` with no flags — verified) at generator
  seeds 21000/22000/23000/24000, assembled flat 31,000 × lengths 2–6 = 155,000, depth-3 excluded,
  × Stage-1 seeds 0 and 1 = eight models. Held-out greedy (overall / per length / depth-3 slice),
  coverage pass@2,000 on three pools, frozen ladder 8 × 32. Plus two n = 1 gap-closers from bucket
  checkpoints: `la_frozen_g1_s0` (`ds-generator`) and `ds-composition`'s C0 + A1 ladder, T1 and
  frozen, on **both** Stage-1 seeds. Fifteen standing findings named in advance as expected-to-fall
  or expected-to-stand.
- 16:05–16:14  Pods `nf-1` / `nf-2`, **NVIDIA A40 48 GB**, catalogue $0.49/h (real billed rate to be
  read from RunPod's billing API and recorded in `log.md`; `podbudget`'s column is a $0.49 estimate).
  Both have a **7.65-CPU cgroup quota** despite `nproc` 96, so `LEAN_GATE_WORKERS=3` × 4 concurrent
  chains, `coverage.py --procs 2`. `podbudget noise-floor --set 36 18` registered before the first pod.
- 16:16–16:21  Four null pools generated and assembled (4 min each). **Premise check passes**: every
  shape quantity agrees across P1–P4 to ≤ 0.26 pp and with the control's published table to ≤ 0.4 pp
  (`ORE` 1.42–1.48 % vs 1.46; box depth 53.1–53.3 / 35.3–35.5 / 11.3–11.4 % vs 53.2 / 35.4 / 11.4;
  mean premises 1.41–1.42 vs 1.41). 0 exact-`thm` and 0 renaming-class collisions with all nine
  evaluation / ladder pools, on all four sets. Pairwise class overlap between the four sets 5.8–6.0 %.
  **All four are replicates, not arms.** `numbers.md` § N1.
- 16:27  Pre-registration **addendum 1** (committed before any outcome existed): Stage-1 **seed 2** on
  all four pools, 4 × 3 = 12 cells, because the pre-registered design estimates the seed variance
  component with 1 degree of freedom and cannot answer E12. ≈ $2.7.
- 16:46  **First result — the held-out floor, eight null cells.** Overall greedy **0.868–0.963,
  sd 0.039**, i.e. the smallest difference two seeds can resolve is **± 20.9 pp**; pre-registered
  E1 (sd ≤ 0.02, range ≤ 5 pp) **missed by about 2×**. The 6-line bin is **0.473–0.920, sd 0.189,
  max/min 1.95×**, and the depth-3 slice **0.040–0.918, 22.9×** — and the two are the *same*
  quantity (all 500 depth-3 held-out theorems are 6-line; r = 0.9994 across the eight cells).
  Both between-pool and between-seed variance components are **negative**: the spread is neither the
  data draw nor the seed, it is the individual training run. `numbers.md` § N2.
- 17:45  **Scope cut, and the ceiling raised 36 h / $18 → 42 h / $21** (`log.md`, `QUESTIONS.md`): a
  frozen ladder measures at **1.94 pod-hours** on these A40s, not the brief's 1.25, and `pass@2,000`
  coverage at 0.7–2.4 h, so the brief's design needs ≈ 60 pod-hours against its 36. Dropped, in the
  brief's own order: `targets_depth3` coverage everywhere; seed 2 reduced to Stage-1 + held-out; the
  `ds-composition` gap-closer reduced to **A1 seed 1** (C0 seed 1 is already on file from
  `ds-generator` on the identical checkpoint and command). Projection ≈ 33 pod-hours ≈ $16.
- 23:28  **Headline result — the frozen-ladder floor, eight null cells.** Transfer theorems solved:
  **62 / 96 / 117 / 174 / 177 / 200 / 262 / 265 of 2,285 — a 4.27× range**, sd 74.1 (43.8 % of the
  mean). The smallest difference two seeds can resolve is **±235 %**. The control's published 158 /
  114 sits in the middle of that range, and so does every frozen-ladder value proposal 10 reported
  (G2 44 / 125, G1 170, A1 205, A2 111). Pre-registered E8 (1.4–2.5×) and E10 (±40–60 %) both
  **missed, the floor being far larger than predicted**; the falsifier (max/min < 1.2×, which would
  have revived the shape account) is **decisively not triggered**. `L*` is the robust readout:
  9 in seven cells, 10 in one, resolvable difference **±1.9 points**. `numbers.md` § N3.
- 2026-09-25 03:35  Pre-registration **addendum 2** (before any seed-3+ model existed): 40 more
  Stage-1 seeds on P1–P4, **held-out greedy only**, to pin the bimodal cell's high-mode probability.
  52 held-out cells in total, ≈ 6.5 pod-minutes per model.
- 05:00  **Both n = 1 gaps closed.** `ds-generator`'s missing `la_frozen_g1_s0` = **62** against its
  arm-mate's **170** — G1's own two seeds bracket C0 (158/114) and G2 (44/125) entirely.
  `ds-composition`'s A1 at Stage-1 seed 1: T1 **847** (C0's is 965, so "A1's T1 beats C0" **reverses**),
  frozen **194**, frozen `L*` **9** (so "A1 `L*` 10 vs C0 9" **dies**).
- 07:05  **Result.** The floor, per quantity: frozen ladder **62–265 of 2,285 (4.27×, ±235 %)**,
  `L*` **9–10 (±1.9)**, `targets_reductio_req` **6–46 (7.67×)**, `required@8` **61–242 (3.97×)**,
  held-out greedy overall **±16.3 pp** at 52 cells, 6-line bin and depth-3 slice **not resolvable at
  n = 2 at all**. The depth-3 slice is bimodal; the estimand is p(high mode) = **0.462, Wilson
  [0.333, 0.595]**. ≈ **99 % of the variance is the individual training run** — the pool variance
  component is negative on every quantity, so re-drawing the data set is no stronger a replicate than
  re-seeding Stage-1. **Of twenty standing findings scored, two survive, both the cap-8 arm.** The
  pre-registered falsifier (max/min < 1.2×) did not fire, by 3.6×. Seven of nineteen expectations met,
  nine missed, and the misses that matter all say the floor is bigger than predicted.
  Checker: **0 disagreements on 39,137 counted proofs**; in-loop gate 67.1 Lean-only per million
  distinct, 0 the other way. Cost **28.32 pod-hours, $13.88** of a 42 h / $21 ceiling; both pods deleted.
- Deliverables: **`NOISE_FLOOR.md`** (standing reference), `run_noise_floor.md`, `numbers.md`
  §§ N1–N11, `log.md`, `figures/nf_{cells,resolvable}.png`, `artifacts/nf/summary.json` (52 rows) and
  `artifacts/nf/premise.json`, bucket
  `hf://buckets/dan-pandori/nd-rl/noise-floor/{artifacts/nf,ckpts/nf,data/nf}`. Two questions for Dan
  in `QUESTIONS.md` (the `pod_budget_watch` bug I fixed; the budget raise the measured cost forced).

NOISE-FLOOR DONE 2026-09-25T07:25:00Z

## Run `lean-judge` (executor, 2026-09-27)

START 2026-09-27T17:20Z. Making **Lean the only judge** on the fork's `dan` branch (Dan, 2026-09-27):
`lean_judge.py`, the gate change, the six loop-file import swaps, `lean_check.py` with `Not.elim`,
tests, `LEAN_JUDGE.md`, and a seven-part acceptance test. Pre-registered in
`preregistration/lean-judge.md` (commit `579ee16`, before any pod). Pod budget 8 h / $4; one A40
(`lj1`, $0.49/h) for acceptance test 5 only.

LEAN-JUDGE DONE 2026-09-27T18:35:00Z — Lean alone decides on the fork's `dan` branch.
`lean_judge.py` (marker ⇒ reject, gate registry ⇒ that verdict, else batched `nd2lean` + Lean);
`lean_gate.py` no longer calls `nd_verify` and returns **clean ND strings** for accepted samples;
the six loop files import the Lean judge and batch; `lean_check.py` admits `Not.elim`, `Not.intro`,
`And.elim`, `Iff.elim`; `nd2lean.translate(require_all_pr=False)` for the judge only.
**All seven acceptance tests pass**: 281,817 / 281,817 stored ND proofs the old gate counted are
counted now (**0 losses**), line counts identical on all of them; 6,419 / 6,419 of the Lean-only
class now counted (41.7 % omitted premise re-statement, 25.9 % `Not.elim`); Lean agrees on the
literal text, `nd2lean(nd)` and `nd2lean(norm(nd))` on all 6,419; one expert-iteration round with
**0 `LEAN*` markers** in the training files and 166 hindsight relabels accepted; the judging step is
6.4× faster; `lean_check --selftest` 39/39. Read `LEAN_JUDGE.md`; counts in `numbers.md` § lean-judge
(L0–L8), write-up `run_lean_judge.md`, figures `figures/lj_*.png`, raw files `artifacts/lj/` and
`hf://buckets/dan-pandori/nd-rl/lean-judge/`. One A40 pod, 0.66 h, $0.32 of $4. Three questions for
Dan in `QUESTIONS.md` (the `no-denotation` class, `require_all_pr`, relabelling the pools).

# STATUS — state-env (proposal 13: an AlphaProof-style state for the policy)

Brief: the `state-env` run brief. Policy: AGENT_POLICY.md (nd-rl canonical). Run id: state-env. Budget 60 h / $30, registered. My pods are `se-*`.

**Gate-0 confound:** at 04:22:02 UTC, one minute before my first pod, a pod `ca1` for a run `ckpt-avg` appeared in the shared `~/pods.log` — a sibling run on this host, not mine. My pre-registration commit (`5bb93132`, 04:21:15 UTC) predates my first pod `se-1` (04:22:57 UTC).

## state-env
- 2026-09-28 03:55 UTC  run started (executor). Branch `dan_state-env` from the fork's `origin/dan`.
- 2026-09-28 04:2x UTC  environment built on the VPS and the three CPU gates run **before any pod**: gate 1 round trip **0 failures / 155,000** control proofs (+ 0 Lean rejections on a random 5,000 reassembled texts), gate 2 "your state is Lean's state" **0 mismatches / 2,800** cut points (2,000 random + 500 `Or.elim`-branch + 300 negation-box), gate 3 environment replay **0 failures / 155,000**. Pre-registration `preregistration/state-env.md` committed before the first pod.
- 2026-09-28 05:22 UTC  4 pods running (`se-1` 3090, `se-2` RTX PRO 4000, `se-3` 3090, `se-4` 4090). Arm S held-out greedy s0 **0.9578** / s1 **0.8012** (C0 0.9088 / 0.8968); seed 1 fails on out-of-scope names (the pre-registered non-Markov naming token), so diagnostic arm **SN** (canonical names) was added with its own predictions before any SN result. An environment crash (empty `have` term) stopped T1 S s0 after round 2; fixed, resumed from round 3 (s1 from round 2). Session interrupted 05:17–05:18, no work lost.
- 2026-09-28 15:40 UTC  All 12 ladder runs done (S, SH, SN-v2 × T1, frozen × 2 seeds); all pods deleted; 35.44 pod-hours, $20.20 real ($19.42 by `podbudget`). **Result:** the state lifts every length below the wall (T1 transfer S 1,348 / 1,389 vs C0 890 / 965; frozen 779 / 787 vs 158 / 114; held-out depth-3 0.922 / 0.882 vs 0.488 / 0.272) but `L*` stays 12 everywhere and only 3 distinct `L_true` ≥ 13 theorems were solved (S 1 / 2). **Neither pre-registered falsifier fires** ("moved but not decisive"). Deliverables: `run_state_env.md`, `STATE_ENV.md`, `numbers.md` § state-env, `artifacts/se/summary.json`, `figures/state_env.png`; bucket `hf://buckets/dan-pandori/nd-rl/state-env/`.

STATE-ENV DONE 2026-09-28T16:05:00Z
