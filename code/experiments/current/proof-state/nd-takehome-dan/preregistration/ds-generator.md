# Pre-registration — run `ds-generator` (proposal 9, run 2: the generator's proof-shape distribution)

Written 2026-09-22 ≈ 06:25 UTC, before any pod exists for this run. Executor: agent:claude. Branch `dan_ds-generator`
(worktree from `origin/dan_lean_format`). Brief `BRIEF_ds-generator.md`; proposal
`~/nd-rl/docs/proposals/2026-09-22-dataset-styles.md` (its "Protocol shared by every arm" section is the protocol); policy
`AGENT_POLICY.md`. Sibling run `ds-composition` runs on this host at the same time (gate-0 confound: `~/pods.log` is shared;
my pods are named `dsg-*`).

## Question

With composition (flat 31,000 per length 2–6), rendering (`lean_seq`), model, schedule, cap 6, set size 155,000 and the
depth-3 f = 0 filter held fixed, does changing the **shape distribution of the generated proofs** change held-out accuracy by
length and RL readiness (depth-3 and required-reductio base rates at pass@2,000, EI − frozen after 4 dial rounds, ladder
frozen solves and `L*` after T1, textbook solves per schema)?

## What I measured before designing (VPS probes, 80,000 tries per setting, seed 77, no output caps; `dsg_shape.py`)

| generator setting | distinct ≤ 6-line proofs / 80k tries | ORE share | box inside ORE | NEGI / DN share | IMPI-final | 0-premise theorems | length 2 / 3 / 4 / 5 / 6 |
|---|---:|---:|---:|---|---:|---:|---|
| control (take-home knobs) | 21,495 | 1.48 % | 0.08 % | 11 / 9 % | 32 % | 18 % | 2,721 / 7,334 / 4,591 / 3,623 / 3,226 |
| brief's G1 (`ore_steps` 3, boxes in ORE branch 1) | 21,775 | **0.98 %** | 0.03 % | 11 / 9 % | 36 % | 18 % | 2,399 / 7,350 / 4,728 / 3,831 / 3,467 |
| brief's G2 literal (100 % goal, strict, budgets 2–4, gdepth 2–3) | **1,866** | 0 % | 0 % | **84 / 71 %** | 16 % | 4 % | **78 / 28 / 140** / 1,020 / 600 |
| same with budgets 3–5, gdepth 2–3 | 2,272 | 0 % | 0 % | 85 / 71 % | 16 % | 4 % | 80 / 30 / 164 / 1,374 / 624 |
| ladder pools' strict long generator at cap 6 (50 % goal / 50 % forward, strict, `ore_steps` 3, contradictory premises dropped) | 5,673 | 1.2 % | 0.05 % | 15 / 16 % | 81 % | 57 % | 612 / 279 / 1,661 / 1,265 / 1,856 |
| **the same + boxes in ORE branch 1 (= G2 as run)** | 6,477 | 1.0 % | 0.06 % | 13 / 14 % | 86 % | 61 % | 375 / 233 / 1,790 / 1,376 / 2,703 |

Two consequences, both decided before any pod:

1. **G1 as specified does not raise the ORE share at cap 6** — it lowers it (1.48 → 0.98 %), because an `ORE` whose first
   branch takes steps or opens a box is ≥ 7 lines after pruning and is removed by the cap; a box inside an `ORE` branch needs
   ≥ 7 lines except in degenerate cases. The brief's shape targets (`ORE` ≥ 5 %, boxes inside `ORE` ≥ 1 %) are therefore
   unreachable at cap 6 without quotas, which the brief forbids. I run G1 **exactly as specified** and pre-register it as a
   *null manipulation*: its set is a fresh raw pool from a generator whose ≤ 6-line output distribution is the control's up
   to sampling noise (`ORE` 1.0 vs 1.5 %). Its value is the between-pool noise floor for G2 − C0 (the a1 / a2 / a3 sets of
   the follow-up were assembler seeds of one pool; no run has two pools). The brief's G1 expectations (≥ 1 `ORE`-needing
   schema at ≥ 5) are kept in the table as the brief's, next to mine.
2. **G2 as literally specified is degenerate**: with 100 % goal mode and no lazy `( Z > G )` fallback the only way `reach`
   can close a goal it cannot decompose is `raa` / `nege_lazy`, so 84 % of proofs are reductio (NEGI + NEGE, 71 % end in
   DN), 31–36 % of theorems have contradictory premises, and the 2 / 3 / 4-line bins yield 30–160 proofs per 80,000 tries
   (unfillable: ≥ 60 % of the set would have to come from G1's pool). That set would test "reductio-only pretraining", not
   the shape-distribution account. **G2 as run** is the strict long generator that built the ladder pools
   (`sample_one(long=True)`: 50 % goal / 50 % forward, `strict = True`, `ore_steps` 3, `bot_p` 0, goal budgets {3, 4, 5},
   goal depth {2, 3, 3}, forward 6–22 steps, contradictory-premise theorems dropped — every setting the ladder pool used),
   restricted to ≤ 6 pruned lines, plus G1's boxes-in-`ORE` knob. This is the cleanest test of the brief's mechanism
   ("if the shape distribution is the wall, G2's base model should solve far more of the ladder transfer pool frozen"):
   G2's training distribution *is* the transfer pool's generator at the training cap. `finish()` in strict mode rejects
   `ANDI` / `ORI` / `BOTE` / `R` conclusions and any `F` in the theorem (read: "final rule an elimination or discharge" =
   `IMPI`, `NEGI`, `IMPE`, `ANDE`, `DN`, `ORE`, `NEGE`); lazy premises via `nege_lazy` and lazy antecedents in `IMPE`
   remain (as in the ladder pools; counted and reported as `n_lazy_prem`). The literal G2 is reported as a probe only.
   Question for Dan in `QUESTIONS.md` with this default.

Both deviations are from the brief's *design*; its question and protocol are unchanged.

## Design I will run

**Generator** (`gen.py`, `make_coverage_sets.py gen`, flags `--ore_steps --ore_boxes --goal_only --strict --budgets --gdepth
--ladder --drop_contra`; control path byte-identical: 3,000 / 5,303 / 1,500 records of the three unchanged code paths
reproduce exactly, `patterns.py --test` passes; committed with this file). Boxes inside `ORE` are implemented by letting
branch 1 take full `step()`s (AS / close / nested ORE under the global depth cap 3, the branch box itself protected by a
floor), because the clamp in `step_no_box` was a no-op (`_local` has no box actions) — said in `data/dsg/README.md`.

**Sets** (each 155,000, 31,000 per pruned length 2–6, cap 6 asserted on the record and re-asserted by `train.py --cap 6`;
depth-3 excluded in pruned and written form; classes of `data/p2/heldout.jsonl`, `targets/transfer_depth3`,
`targets/transfer_reductio_req`, `data/r3_1/depth3_req{,_transfer}`, `data/ladder/{rl_targets,transfer}` and validation-36
excluded by renaming class; `dsg_assemble.py`, seed 0):
- **C0**: `data/p2/train_depth3_f0_a1.jsonl` and the bucket checkpoints `stage1_a1_seq_s{0,1}.pt` (md5 9bde44c0…, fc27e52d…; not retrained).
- **G1**: raw pool from `make_coverage_sets.py gen --ore_steps 3 --ore_boxes`, no output caps (so every pattern is at its
  natural rate), uniform draw per length from the non-depth-3 proofs.
- **G2**: raw pool from `--ladder --drop_contra --ore_boxes`, same assembly. Generation budget ≤ 2 pod-hours; a length bin
  still short after that is filled from G1's pool and the fraction disclosed per bin (expected: the 3-line bin, ≈ 0.3 % of
  tries, may need > 10 M tries; the 2-line bin ≈ 0.5 %).
Per set: shape table (`dsg_shape.py`), overlap table against every pool above plus the take-home `data/train.jsonl` /
`data/transfer.jsonl` / `rl_targets` and the control set (order-sensitive `thm` and renaming-class), render check (3,000
round-trips, 1,000 Lean accepts, 300 theorem-swapped negatives rejected). Generated on the arm's own pod (CPU side), pulled.

**Measurements per arm and seed** (RTX 3090, one pod per arm, `pod/dsg/` = the lean-format harness with `CUDA_MEM_FRACTION`
0.21, `LEAN_GATE_WORKERS` 12; `lean_gate.py`, `nd2lean.py`, `nd_verify`, `train.py`, `expert_iter.py`, `ladder_ei.py`
unchanged; `coverage.py` gains a Lean check on the literal text of every distinct `nd_verify`-accepted proof — token models
untouched):
1. Stage-1 `train.py --mode lean_seq --steps 6000 --bs 128 --cap 6 --seed {0,1}`; held-out greedy on `data/p2/heldout.jsonl`
   by length and by pattern-vs-not (`eval_set.py --k 1 --temperature 0`).
2. `coverage.py --k 2000 --temperature 0.8 --seed 0 --batch 1000` on `targets_depth3` (1,000), `r3_1/depth3_req` (300),
   `targets_reductio_req` (300): a sample counts iff `nd_verify` and Lean accept; `nd2lean.py --check` on every counted
   proof at the end.
3. Dial: `expert_iter.py --rounds 4 --k 32 --temperature 0.8 --batch 768` EI and `--no_train`, targets `targets_depth3`,
   `--train` the arm's own set, EI seed = Stage-1 seed. C0's from the bucket (`lean-format/artifacts/lf/{ei,frozen}_d3_seq_s*`).
4. Ladder: `ladder_ei.py` T1 and `--no_train`, 8 × 32, `--batch 512`, pools `data/ladder/` (byte-identical to ladder-A's),
   `--train` the arm's own set; C0 measured here on the a1 checkpoints.
Counting: denoted ND proofs, start-index-normalised; two seeds; frozen at equal attempts; differences quoted only when both
seeds agree in sign; `L*` = max L with ≥ 5 transfer theorems solved at `L_true` ≥ L; textbook solves per schema from
`found_transfer_8.jsonl`.

## Expected results (both seeds unless stated; C0 values measured here, on-file references in brackets)

| quantity | C0 | G1 (mine; brief's in *italics*) | G2 as run |
|---|---|---|---|
| held-out greedy overall; 6-line bin | 0.89–0.91 [0.909 / 0.896]; 0.82–0.87 | ±1.5 pp; ±2 pp (*±1 pp*) | **−5 to −20 pp** (0.70–0.85); 6-bin 0.60–0.82 |
| held-out by length 2 / 3 | ≥ 0.99 / ≥ 0.98 | same | 2-bin ≥ 0.95; 3-bin 0.85–0.97 |
| depth-3 pass@2,000, 1,000-pool; req8 (300) | 0.35–0.55; 0.05–0.25 | ×0.5–2 of C0 | **+10 to +25 pp**; +5 to +20 pp |
| reductio-req pass@2,000: 7-line stratum (52); ≥ 8-line (248) | 5–25 targets; 0–5 | ×0.5–2 | ×1–3; ≥ C0 |
| distinct ≥ 8-line pattern proofs in coverage | 0–20 | ×0.5–2 | ×1.5–4 |
| dial round 4: EI acq; frozen acq; EI − frozen | 0.38–0.46; 0.10–0.20; +0.20 to +0.32 | within ±0.05 | frozen +0.05 to +0.20; EI − frozen **smaller** than C0 by 0.03–0.15 (finding 3) |
| ladder frozen solved / 2,285; `L*` | 220–320 [304 / 309]; 9–10 [10] | ±15 %; unchanged | **+50 to +150 %** (450–760); 10–11 |
| ladder T1 solved; `L*` | 700–900 [794 / 839]; 10–11 [11] | ±15 %; 11 | 850–1,200; 11–12 |
| textbook schemata at ≥ 5 T1 solves beyond contraposition / export | ≤ 2 | ≤ 1 new (*≥ 1 `ORE`-needing*) | 1–3 new, none `ORE`-needing (G2 has `ORE` 1 %) |
| `L_true` = 7 transfer bin (300), T1 | 55–120 [75 / 118] | ±20 | +20 to +100 |
| held-out retention after T1 (round 8 greedy − Stage-1) | ±1 pp | ±1 pp | ±3 pp (already low) |
| set shape: `ORE` share; box-in-`ORE`; box depth 2; 0-premise; contradictory premises | 1.5 %; ≈ 0.1 %; 10 %; 18 %; 6 % | 1.0 %; ≈ 0.05 %; 12 %; 18 %; 6 % | 1 %; ≈ 0.1 %; 30–40 %; 55–65 %; 0 % |
| Lean vs `nd_verify` on counted proofs | 0 disagreements | 0 | 0 |

**Falsifiers (the brief's, kept).** The shape-distribution account of the transfer wall is dead if G2's frozen ladder solves
are within ±15 % of C0 on both seeds. The rule-shape account of the textbook wall is dead if G1 and G2 leave 17 / 19 schemata
at ≤ 2 (I expect this for G1; G2 is the informative arm). Finding 3 gets a counter-example if an arm has a lower base rate
than C0 and a larger EI − frozen on both seeds. A G2 ladder gain with a large held-out loss is a trade-off, reported as both.
**Decision rule** (proposal 9): an arm replaces the control iff held-out within 1 pp of C0 or better *and* ≥ 2 of
{depth-3 pass@2,000, reductio pass@2,000, ladder T1 `L*` or solved} improve with none worse, both seeds. I expect neither
arm to qualify (G1 null; G2 fails the held-out clause).

## Budget and stop rule

- Pod ceiling **$20** (`rpbalance` 121.38 at start). Plan: three RTX 3090 pods (`dsg-1` C0, `dsg-2` G1, `dsg-3` G2) at
  $0.50 / h, ≈ 5–6 h each ≈ $8–10; G1 / G2 generation on their own pods' CPUs. Order of dropping if spend reaches $16:
  G1's ladder, then G1 entirely, then C0's second-seed ladder.
- Hard stop 30 h after start: 2026-09-23 11:54 UTC. Pods deleted as soon as their files are pulled.
- G2 generation stops at 2 pod-hours; short bins filled from G1's pool (disclosed per bin).
- No test-file run; `nd_verify` unmodified; `nd2lean.py` unmodified (BOTE fix is `lean-seed2`'s); every training record is
  generator output rendered deterministically; no hand- or LLM-written proofs; `lean_gate.py` unchanged.

---

## Addendum — resume phase, written 2026-09-23 ≈ 21:20 UTC, before any pod of this phase exists

The 2026-09-22 session ended at ≈ 10:33 UTC when the account ran out of Fable credits; the balance floor then deleted
`dsg-1` / `dsg-2` / `dsg-3` while their ladder jobs were running (started 08:02 / 08:45), so **no ladder or textbook
number of this run exists**. A host cleanup afterwards deleted local `artifacts/` and `ckpts/`. Recovered before this
phase: `artifacts/dsg/` in full from git (it was committed as it was pulled — 95 files, the coverage, dial, shape,
overlap and render-check material), `data/dsg/train_g{1,2}.jsonl` and the three raw pools from the bucket, and the two
control checkpoints from `hf://buckets/dan-pandori/nd-rl/lean-format/ckpts/lf/` (md5 `9bde44c0…` / `fc27e52d…`, the
values recorded above — verified). `ckpts/dsg/stage1_g{1,2}_s{0,1}.pt` were never uploaded and are gone.

**What this phase runs.** Ladder T1 + frozen, 8 rounds × k 32, two seeds, for all three arms (12 jobs), then the checker
of record over every counted proof, then the textbook-schema table from `found_transfer_8.jsonl`. G2's ladder is run
even though its held-out collapsed (0.626 / 0.606 against C0's 0.909 / 0.896), because G2's **frozen** ladder solves are
this run's pre-registered falsifier ("the shape-distribution account of the transfer wall is dead if G2's frozen ladder
solves are within ±15 % of C0 on both seeds"); that is a new measurement, not a re-confirmation of the held-out number.

**Deviations from the design above, each with its reason.**

1. **Fast decode path.** `sample.py` and `model.py` are taken from `origin/dan_efficiency` (run `efficiency`, reviewed
   2026-09-23: "every number re-derives", base-vs-fast accepted sets identical at fixed batch), with one line changed
   back — this branch's gate is `gate(tok, prompts, res, texts)`, i.e. Lean **and** `nd_verify`, not that branch's
   `lean_check`-only reward. `lean_gate.py`, `nd2lean.py`, `nd_verify`, `train.py`, `expert_iter.py`, `ladder_ei.py`,
   `coverage.py` are untouched. Checked on the pod before any measurement by `dsg_regress.py` (16 ladder-transfer
   prompts × 8 samples, T 0.8): `path='base'` with row-keyed rng, the new default, and the fast path without
   compaction must produce **identical token streams and identical decoded proofs**. Expected: PASS; if it fails I set
   `ND_SAMPLE_PATH=base ND_SAMPLE_ROWRNG=0` and run the ladder on the unchanged path at the original batch, costing
   ≈ 6 extra pod-hours (still inside the ceiling).
2. **Ladder sampling batch 512 → 4,096 and `--max_new` 512 → 384**, one sampling job per GPU at a time instead of four
   co-tenants. Reason: `efficiency` measured 1,395 samples/s for a solo job at batch 4,096 against 964 for four
   co-tenants at batch 1,024, and its longest *accepted* sample on exactly this workload (ladder transfer, `lean_seq`
   Stage-1) was 255 tokens, so 384 keeps 50 % headroom. That run also recorded that a batch change reshuffles which
   proofs are accepted about as much as an RNG re-draw: **batch 4,096 and `max_new` 384 are therefore held fixed across
   all three arms and both seeds**, and the C0 ladder is re-measured here under them. Consequence, stated in advance:
   this run's ladder numbers are not sample-for-sample comparable with `lean-format`'s on-file C0 ladder numbers
   (frozen 304 / 309, T1 794 / 839, `L*` 10 / 11); the C0 measured *here* is the comparison every G1 / G2 claim uses,
   and the lean-format figures are quoted only as the prior.
3. **G1 and G2 Stage-1 checkpoints are retrained**, same sets (re-fetched from the bucket), same seeds, same command
   line as on 2026-09-22. Reproduction check, pre-registered here: held-out greedy is re-measured with the identical
   `eval_set.py --k 1 --temperature 0 --batch 512` call and compared with the committed
   `artifacts/dsg/heldout_<arm>_s<seed>.json`. Expected **within ±0.5 pp** (bitwise-identical checkpoints if training is
   deterministic on the same GPU class). A larger difference is disclosed and the *new* held-out number is the one the
   ladder rows are labelled with.
4. Everything measured in this phase is labelled with its model: 3.3 M-parameter from-scratch GPT, `lean_seq` Lean
   surface format, Stage-1 6,000 steps × batch 128 on the named 155,000-record set, cap 6, seed as given.

**Additional expected results for this phase** (the ladder and textbook rows of the table above stand unchanged):

| quantity | expectation |
|---|---|
| `dsg_regress.py` on `stage1_a1_seq_s0.pt` | token streams and decoded proofs identical, verdict PASS |
| held-out greedy of the retrained G1 / G2 checkpoints vs the committed values | within ±0.5 pp on all four |
| ladder wall clock per arm (4 jobs, 8 rounds, sequential) | ≤ 3 h; whole phase ≤ 9 pod-hours |
| Lean vs `nd_verify` on every counted ladder proof | 0 disagreements |

**Budget and stop rule for this phase.** $14 and 28 pod-hours (`podbudget ds-generator`), balance $220.36 at start.
Three RTX 3090 at $0.50/h, ≈ 2–3 h each ≈ $4. Order of dropping if spend reaches $11: G1's second-seed ladder, then
G2's second-seed ladder, then G1 entirely (C0 and G2 carry the falsifier). No new hard stop beyond the ceiling; pods
deleted as soon as their files are pulled, and `artifacts/` is uploaded to the bucket after each stage rather than only
at the end.
