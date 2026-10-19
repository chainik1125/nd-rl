---
written_on: 2026-10-19
written_by: agent:claude
---

# Kalmár's construction vs our model: what does it cost to prove a theorem?

**Short version.** For our domain (propositional logic over at most four atoms), a ~200-line program that implements
Kalmár's completeness proof proves every theorem, and checking its proof in Lean costs 11–34 ms of one CPU core. Our
best models (`state_env_october`, and `state_env_october_r16` after 16 RL rounds) need 6–99 GPU attempts per theorem on
average, about 7 ms each on an A40. They still fail on 2–33 % of theorems within 256 attempts. Their proofs are much
shorter: a median term size of 6, against Kalmár's 111.
- **At our scale, the program is cheaper:** about 2–22× in wall-clock time and 30–430× in dollars per theorem.
- **The construction is exponential:** each extra atom roughly doubles its proof. Using the program's measured costs,
  the crossover with the model comes at about 5–7 atoms in wall-clock time and about 7–13 atoms in dollars.
- **Our model can't go past four atoms:** its vocabulary has only P, Q, R, S.

At the scale we work at, the efficiency question doesn't favour the model. The model is a research organism, not a
practical prover.

## The two provers

| | Kalmár program | our models |
| --- | --- | --- |
| what | [`code/tools/tt_prover.py`](../../../code/tools/tt_prover.py): truth table, then a case split on excluded middle for each atom and a structural proof in each of the ≤ 16 branches | [`state_env_october`](../../models/state_env_october.md) (round 8) and [`state_env_october_r16`](../../models/state_env_october_r16.md) (round 16): a 9.56M-parameter ALiBiGPT, proof-state interface, pretrained on K12, then expert iteration; plain sampling, T 0.8. The numbers here use the `trajectory` seeds (textbook72 48 / 48 / 53 at round 8), not the model card's `best-state` seeds |
| steps used | natural deduction only: ∧/∨/→/¬ introduction and elimination, ex falso, `Classical.byContradiction` | the same rules, one ND step per action |
| judge | Lean 4 core with the fork's strict `lean_check` (constant allowlist and axiom check) | Lean alone, the same judge family |
| hardware in these numbers | one core of an Apple-silicon laptop CPU | one NVIDIA A40 pod, batch 2,048; Lean on the pod's CPUs |
| training compute | none | ≈ 7 GPU-hours per seed through round 8 (RTX A6000) and ≈ 16 through round 16 |

## 1. Kalmár on real theorems

400 random K12 theorems per atom count (K12 is the model's pretraining set); every proof is accepted by `lean_check`.

| atoms | Kalmár term size, median (p90 / max) | generate, median | Lean check per proof (batched, warm) | K12's own proof, median lines |
| --- | --- | --- | --- | --- |
| 1 | 18 (23 / 32) | 0.03 ms | 11 ms | 6 |
| 2 | 45 (55 / 74) | 0.06 ms | 13 ms | 7 |
| 3 | 103 (122 / 148) | 0.11 ms | 20 ms | 7 |
| 4 | 219 (258 / 326) | 0.23 ms | 34 ms | 7 |

*Term size counts inference steps in the elaborated Lean proof term (`lean_check`'s measure).*

On the evaluation sets:

| set | theorems | judged invalid | Lean-accepted | term size, median (max) | Lean check per proof |
| --- | --- | --- | --- | --- | --- |
| holdout250 | 250 | 0 | **250** | 111 (374) | 26 ms |
| textbook72 (sealed; aggregates only) | 72 | 0 | **72** | 58 (256) | 28 ms |

## 2. Kalmár beyond four atoms

Two synthetic families over n atoms (the four letters were extended for this test only), one theorem per size:
- **chain:** A₁ → A₂, …, Aₙ₋₁ → Aₙ ⊢ A₁ → Aₙ;
- **excluded middle:** ⊢ φ ∨ ¬φ, for a random φ using every atom once.

"Check" is one Lean call minus Lean's ≈ 1.5 s startup. Below about 6 atoms the remainder is lost in startup noise.

| atoms | term size: chain / excl. middle | generate (excl. middle) | check: chain / excl. middle |
| --- | --- | --- | --- |
| 4 | 201 / 235 | 0.6 ms | (within startup noise) |
| 6 | 867 / 977 | 3.9 ms | 0.15 s / ≈ 0 |
| 7 | 1,760 / 2,178 | 10 ms | 0.96 s / 1.8 s |
| 8 | 3,549 / 4,009 | 22 ms | 0.52 s / 1.3 s |
| 9 | 7,130 / 8,496 | 57 ms | 0.77 s / 2.8 s |
| 10 | 14,295 / 18,019 | 0.12 s | 3.8 s / 9.3 s |
| 11 | 28,628 / 37,860 | 0.24 s | 7.2 s / 15.8 s |
| 12 | 57,297 / — | 0.65 s | 9.3 s / **failed**: Lean's default heartbeat limit (200,000) on a 2.5M-character proof |

- **Growth:** proof size doubles with each atom (×2.0 per atom from 6 to 12 on the chain family; ×2.1 on excluded
  middle from 6 to 11), and checking time grows with it.
- **At 12 atoms,** one of the two test proofs exceeded Lean's default resource limit. The limit can be raised; it marks
  where single proofs stop being routine to check.
- **Generation stays cheap,** under a second at 12 atoms. Checking and storing the proof is what becomes impractical.

## 3. Our models

Stored k = 256 reads (plain, T 0.8; three seeds each).
- **Per-attempt cost:** 6.4–8.0 ms on an A40 at batch 2,048, wall-clock for sampling plus Lean
  ([`create-or-elicit`](https://github.com/dan-pandori/nd-takehome/blob/dan_create-or-elicit/artifacts/coe/budget/SUMMARY.md)).
  It was measured on the pretrained checkpoint of the same network. We assume RL checkpoints cost about the same per
  attempt; that wasn't measured.
- **Expected attempts per theorem:** sample until the first success, at most 256 attempts. Unsolved theorems dominate
  it, since each costs the full 256.

| set, checkpoint | pass@1 | solved within 256 | attempts per solve, median (p90) | expected attempts per theorem |
| --- | --- | --- | --- | --- |
| holdout250, round 8 | 0.80–0.81 | 236–240 / 250 | 1.05–1.07 (1.7–2.0) | 13–17 |
| holdout250, round 16 | 0.82–0.83 | 239–246 / 250 | 1.04–1.05 (1.6–2.5) | 6–14 |
| textbook72, round 8 | 0.36–0.43 | 48–53 / 72 | 1.2–1.6 (20–64) | 74–99 |
| textbook72, round 16 | 0.41–0.44 | 51–57 / 72 | 1.4–1.6 (23–128) | 63–84 |

- **The model's proofs are short.** For the shortest accepted proof per solved holdout250 theorem (round 8, seed 0; 240
  theorems, the shortest of up to 256 attempts each), the median term size is 6 (p90 9, max 17). Kalmár's median on all
  250 is 111.
  - At this scale the shorter proofs save no checking time. Lean checks the model's proofs in ≈ 26 ms each on the same
    laptop core, about the same as Kalmár's.
- **Success falls with atom count on textbook problems.** pass@1 on textbook72 for 1 / 2 / 3 / 4 atoms at round 8:
  0.59 / 0.44 / 0.31 / 0.27, 0.63 / 0.37 / 0.32 / 0.22 and 0.66 / 0.51 / 0.32 / 0.21 for the three seeds. Only 7 and 8
  theorems sit at the 1- and 4-atom ends. On holdout250 it is flat (0.76–0.87).

## 4. Cost per theorem at our scale

Prices:
- **A40:** $0.59/h, the current RunPod rate (our earlier RL runs paid $0.49–0.53/h for A40 and A6000).
- **CPU:** one core at $0.03/h, an assumed typical cloud rate. Kalmár was timed on a fast laptop core, so its dollar
  cost on a cloud vCPU is, if anything, understated.

The model column uses 7.2 ms per attempt. That is the A40 pod's wall-clock time, with Lean running on the pod's own CPUs
in parallel, while Kalmár's times are one core's. So the time ratios are indicative, not like-for-like.

| set | Kalmár: time / $ per theorem / solved | model (round 16): time / $ per theorem / solved | model ÷ Kalmár |
| --- | --- | --- | --- |
| holdout250 | ≈ 26 ms CPU / ≈ $2×10⁻⁷ / 100 % | 44–97 ms GPU / $0.7–1.6×10⁻⁵ / 96–98 % | ≈ 2–4× time, ≈ 30–75× $ |
| textbook72 | ≈ 28 ms CPU / ≈ $2×10⁻⁷ / 100 % | 0.45–0.60 s GPU / $0.7–1.0×10⁻⁴ / 71–79 % | ≈ 16–22× time, ≈ 320–430× $ |

The model's training also has a cost. Its first eight RL rounds alone (≈ 6–7.5 GPU-hours per seed) cost ≈ $4 at
today's A40 rate. That would pay for the Kalmár program to prove and check about 20 million theorems.

## 5. Where the crossover would be

We use the measured Kalmár check times from § 2. Extrapolating "34 ms, doubling per atom" from four atoms
underestimates them by 2–7× from 7 atoms on, which brings the crossover earlier.

We assume the model's attempts per theorem stay constant as atoms are added. That is optimistic, since textbook success
already falls with atom count. We also assume the model could write more than four atoms, which it can't.

| comparison | model cost per theorem | Kalmár reaches it at |
| --- | --- | --- |
| wall-clock, holdout-like problems | 44–97 ms | ≈ 5–6 atoms (by extrapolation; single-proof checks there are within Lean's startup noise) |
| wall-clock, textbook-like problems | 0.45–0.6 s | ≈ 7 atoms (measured: 1.0–1.8 s of checking at 7 atoms) |
| dollars, holdout-like | $0.7–1.6×10⁻⁵ | ≈ 7–10 atoms (measured: excluded middle ≈ $1.5×10⁻⁵ at 7 atoms; chain ≈ $3×10⁻⁵ at 10) |
| dollars, textbook-like | $0.7–1.0×10⁻⁴ | ≈ 10–13 atoms (measured: excluded middle ≈ $1.3×10⁻⁴ at 11 atoms; chain ≈ $8×10⁻⁵ at 12) |

Two caveats:
- **Kalmár is the weakest symbolic baseline.** A truth-table-pruned backward search writes short proofs, and SAT/DPLL
  solvers decide validity for thousands of variables. Against those, a neural prover is unlikely ever to win on
  efficiency in propositional logic.
- **Efficiency is only a meaningful axis where no decision procedure exists,** e.g. first-order logic.

## Method and caveats

- **Hardware differs:** Kalmár was timed on one laptop CPU core, the model on a GPU pod. Times and prices are both
  given; the conclusion holds in both units.
- **Lean times in §§ 1 and 3 are batched and warm** (the first stratum in `results.json` includes a cold start; see
  `warm.json`). § 2 times are single calls with the ≈ 1.5 s startup subtracted.
- **The synthetic scaling families are one theorem per size,** so they show the trend, not a distribution.
- **textbook72 is the group's sealed scoring set:** only aggregate numbers are reported, and no theorem is printed.
- **Reproduction:** [`bench.py`](bench.py) (needs the fork's `lean_check.py`, `record.py`, `nd2lean.py` and
  `nd_verify/`, plus the stored reads; textbook72 rows are not included). Raw outputs: [`results.json`](results.json),
  and [`warm.json`](warm.json), the warm re-timing, startup baseline and 12-atom failure reason, recorded from that
  run's printed output.
- **Sources** for the training compute: the `trajectory` (rounds 1–8) and `rl-continue` (rounds 9–16) summaries on
  `dan`.
