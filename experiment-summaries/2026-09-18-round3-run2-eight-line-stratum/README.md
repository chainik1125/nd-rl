---
written_on: 2026-09-18
written_by: agent:claude
---

# Round 3, run 2: the 8-line reductio stratum and the derived-ORE base-rate dependence

Status: completed 2026-09-18 08:40 UTC (autonomous executor session, three RTX 3090 pods,
≈ $4.7) and independently reviewed 2026-09-18 09:10 UTC. Review verdict: no hard-constraint
violation and no number differs (33 arms, 13 coverage files and the pools reproduce exactly;
the write-up reports its own misses); seven sentences of the executor's write-up must be
reworded and two claims are not supported as written. Numbers and wording below are the
reviewer's. Run id `round3-run2`.

Review: [`review_round3-run2.md`](https://github.com/dan-pandori/nd-takehome/blob/3932dbd/review_round3-run2.md)
(commits `75cb89d` phase 1 blind recount, `68635d0` phase 2 compare and verdict, `3932dbd`
wording; reviewer code `review_r3_2_recount.py`, `review_r3_2_tables.py`,
`review_r3_2_trainscan.py`; outputs `artifacts/review_r3_2/*.json`). Executor write-up:
[`run2.md`](https://github.com/dan-pandori/nd-takehome/blob/3932dbd/run2.md) on branch
`dan_round3-run2` (read the review first; its "Answer" paragraph is the part the review
rewords).

Question: [run 5](../2026-09-17-round2-run5-required-pools/README.md) left every igniting
reductio arm at the 7-line stratum of the required pool. (1) How often does the 8-line
stratum ignite per run, and is that a matter of training rounds, samples per round, or f?
(2) Does adding a 6-line stratum change anything for run 5's "zero-rate" f = 0 draws?
(3) For strict derived-ORE (cap 8, **not a submission result**): is expert-iteration (EI)
acquisition at f = 0 proportional to the draw's base reachability, and is the f-dependence
monotone over f ∈ {0, 10⁻³, 10⁻²}?

## Configuration

- Code, data and results: `dan-pandori/nd-takehome`, branch `dan_round3-run2` (worktree
  `~/work/round3-run2` on the agent VPS). Pre-registration
  `preregistration/round3-run2.md` committed in `6ffade0` at 05:13:01 UTC; first pod created
  05:15:15 UTC (`~/pods.log`), first job 05:17:50 UTC; results `567e8d9`, write-up final
  `d3cfea9`. Brief `BRIEF_stratum-ignition.md`; policy `AGENT_POLICY.md`. Run diff:
  `r3_2_pool.py`, `r3_2_analysis.py`, `pod/r3_2/*` (job files with every command line);
  `expert_iter.py` and `train.py` unmodified.
- Reductio pool `data/r3_2/targets_reductio_req6.jsonl`: **345** = run 5's 300 required
  targets unchanged (7 / 8 / 9 / 10 lines = 52 / 133 / 82 / 33) + the 45 six-line required
  candidates of `data/p2/run5_reductio_nec.jsonl`, all one schema (`nor_neg_ante`,
  `~(~A v B) ⊢ A`, rule sequence `PR AS ORI1 NEGE NEGI DN`). Transfer 150
  (7 / 8 / 9 / 10 = 27 / 67 / 41 / 15), unchanged. Derived-ORE pool 300 (9 / 10 lines =
  126 / 174) and transfer 65, unchanged from run 5. Reviewer: every oracle proof verifies and
  carries the pattern; 0 renaming-class overlap between each arm's training file and its
  pool, transfer set, held-out file and validation-36.
- Reductio arms, all on the 345 pool: `python3 expert_iter.py --init ckpts/p2/stage1_reductio_<model>.pt
  --targets data/r3_2/targets_reductio_req6.jsonl --transfer data/p2/transfer_reductio_req.jsonl
  --heldout data/p2/heldout.jsonl --train data/p2/train_reductio_<f>.jsonl --rounds <R> --k <k>
  --temperature 0.8 --seed <ss> --batch 768` (retain 20k, `--max_per_thm 4`).
  **A**: models f0 s0, f0.1 s0, f0.1 s1 × sampling seeds {0, 1}, 16 rounds × k = 32 (6 arms).
  **B**: the same three models, sampling seed 0, 8 rounds × k = 64 (3 arms; 512 attempts per
  target as in A). **C**: run 5's "zero-rate" draws f0 s1 (seed 1) and f0 s2 (seed 2),
  8 rounds × k = 32, extended to 16 rounds by resuming from the round-8 checkpoint (a
  disclosed, post-hoc deviation, `log.md` 05:58; rounds 9–16 use the seeds a 16-round run
  would have used; `args.json` of the C arms was overwritten by the resume). Frozen controls
  (`--no_train`) at equal attempts and the same seeds: 6 × 16 rounds for A, 2 × 8 and
  2 × 16 for C. 11 EI + 10 frozen reductio arms.
- Six-line base rate: `coverage.py --k 10000 --temperature 0.8 --seed 0` on the 45 six-line
  targets for f0 s0, s1, s2.
- Derived-ORE, cap 8: new Stage-1 draws `python3 train.py --data
  data/p2/train_derived_ore_strict_<f>_c8.jsonl --heldout data/p2/heldout_c8.jsonl --mode abs
  --steps 6000 --bs 128 --cap 8 --seed <s>` for f = 10⁻³ seed 1 and f = 0 seeds 2, 3, then 4, 5
  (the pre-registered fallback when s2 failed the keep rule |r − 27| ≥ 5). EI k = 32 × 8 rounds
  + frozen for f = 10⁻³ s0 / s1 and f = 0 s2–s5 (6 + 6 arms); base strict reachability
  `coverage.py --k 10000 --batch 1024 --procs 3` on all 300 targets for the eight draws
  without one (f = 0 s2–s5, f = 10⁻³ s0 / s1, f = 10⁻² s0 / s1; the last four are a
  disclosed addition). Run 5's f = 0 and f = 10⁻² arms are re-counted alongside. The pattern
  fractions of the Stage-1 files are exact under the reviewer's predicate (f = 0: 0 of
  154,994; 10⁻³: 155; 10⁻²: 1,550; reductio f = 0: 0 of 155,000).
- Counting rule (pre-registered): a target is acquired when a verified, start-index-normalised
  proof satisfies the pattern predicate on the dependency-pruned proof; ignition of a stratum
  = first round with ≥ 10 acquired targets in it.
- Pods: r32a, r32b, r32c, RTX 3090 24 GB at $0.50/h, created 05:15:15 / 05:15:54 / 05:16:20
  UTC (`~/pods.log`), deleted 08:36 / 08:36 / 07:56 (`log.md`); ≈ 9.35 pod-hours ≈ **$4.7**
  (pre-registered estimate 14–16 pod-hours ≈ $8, ceiling $50). See `docs/infra/spend.md`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/round3-run2/{artifacts,ckpts,data}` (reviewer:
  the three directories exist; `ckpts/r3_2` holds the 19 EI final checkpoints and the 5 new
  Stage-1 draws; "21 trained arms" in `numbers.md` should read 17 EI arms / 19 checkpoints).
- Hard constraints (reviewer): `nd_verify` tree hash equals `origin/main`;
  `artifacts/TEST_RUN_DONE` blob identical on `origin/main`, branch head and workspace (the
  orchestrator's gate 1 flagged its mtime, which a fresh worktree checkout changes; the
  content is unchanged); no evaluation file in any training mix (first and last mix of all
  17 EI arms: 0 records from the arm's transfer, held-out or validation-36 sets); no job
  failures. Expectations were committed before the first job.

## Observations

All counts were re-derived by the reviewer from raw files with independent code; every
distinct normalised proof of all 33 arms re-verified with `nd_verify` (0 failures), plus every
raw record of each arm's last found file (80,917 records, 0 failures, 0 without the pattern).
No target was ever solved without the pattern. Held-out greedy numbers are from the
executor's `round_*.json` (no GPU in the review).

**Reductio, 345 required targets, 512 attempts per target in every 16-round and k = 64 arm.**
Acquired targets per stratum at the last round.

| arm | rounds × k | 6 | 7 | 8 | 9 | 10 | first 8-line proof | 8-line ignition | frozen, same seed (6 / 7 / 8+) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A f0 s0 ss0 | 16 × 32 | 45 | 52 | 1 | 0 | 0 | 14 | – | 25 / 4 / 0 |
| A f0 s0 ss1 | 16 × 32 | 45 | 52 | 3 | 0 | 0 | 10 | – | 26 / 3 / 0 |
| A f0.1 s0 ss0 | 16 × 32 | 45 | 52 | **54** | **25** | 0 | 4 | **9** | 45 / 30 / 0 |
| A f0.1 s0 ss1 | 16 × 32 | 45 | 52 | 1 | 0 | 0 | 11 | – | 45 / 27 / 0 |
| A f0.1 s1 ss0 | 16 × 32 | 45 | 52 | 0 | 0 | 0 | – | – | 45 / 27 / 0 |
| A f0.1 s1 ss1 | 16 × 32 | 45 | 52 | 0 | 0 | 0 | – | – | 45 / 25 / 0 |
| B f0 s0 ss0 | 8 × 64 | 45 | 51 | 0 | 0 | 0 | – | – | (A ss0 frozen) |
| B f0.1 s0 ss0 | 8 × 64 | 45 | 52 | 1 | 0 | 0 | 3 | – | (A ss0 frozen) |
| B f0.1 s1 ss0 | 8 × 64 | 45 | 52 | 3 | 0 | 0 | 7 | – | (A ss0 frozen) |
| C f0 s1, round 8 (pre-registered) | 8 × 32 | 11 | 0 | 0 | 0 | 0 | – | – | 1 / 0 / 0 |
| C f0 s2, round 8 (pre-registered) | 8 × 32 | 45 | 52 | 23 | 9 | 0 | 5 | **7** | 7 / 0 / 0 |
| C f0 s1, round 16 (extension) | 16 × 32 | 45 | 51 | 0 | 0 | 0 | – | – | 1 / 0 / 0 |
| C f0 s2, round 16 (extension) | 16 × 32 | 45 | 52 | **27** | **25** | 0 | 5 | **7** | 8 / 1 / 0 |

- 8-line ignition: **1 of 6 A arms, 0 of 3 B arms, 1 of 2 C arms**. No frozen arm has any
  8-, 9- or 10-line proof at 512 attempts (0 of 8 frozen-16 arms); no arm of any kind has a
  10-line proof. Cumulative 8-line curves of the two igniting arms: f0.1 s0 ss0
  `0 0 0 1 2 3 4 6 14 20 35 45 51 54 54 54`; f0 s2 `0 0 0 0 1 6 17 23 27 27 27 …`. 9-line
  ignition at rounds 13 and 9.
- The same checkpoint (f0.1 s0) with two sampling seeds gives 54 vs 1 eight-liners: the
  crossing is a per-run stochastic event. Two "non-igniting" arms were still growing when
  they stopped (A f0 s0 ss1: 8-line 1 → 2 → 3 at rounds 14–16; B f0.1 s1 ss0: 1 → 3 at rounds
  7–8), so their non-ignition is a statement about the stop round, not an asymptote.
- 6-line stratum 45 / 45 and 7-line stratum 51–52 / 52 in all 11 EI arms (frozen at 512
  attempts: 7-line 3–4 for f0 s0, 25–30 for f = 0.1, 0–1 for f0 s1 / s2).
- Transfer follows the acquired schemata: 8- / 9-line transfer proofs only in the two igniting
  arms (26 / 12 and 14 / 13 of 67 / 41), +1 in B f0.1 s1. Held-out greedy never falls more than
  0.007 below its start (f0 s0 0.886 → 0.887–0.888; f = 0.1 0.967–0.968 → 0.969–0.978; f0 s1
  0.871 → 0.885; f0 s2 0.894 → 0.907).

**Ignition is by schema, not by stratum (reviewer).** Everything acquired above 7 lines is
schema-complete: the 54 eight-liners of f0.1 s0 ss0 are `contraposition_conv` 27 / 27 +
`neg_both` 26 / 26 + `chain_neg` 1 / 1; the 27 of f0 s2 are `neg_both` 26 / 26 + `chain_neg`
1 / 1; both arms' 25 nine-liners are `chain_neg` 25 / 25. The "1 eight-liner" of four other arms
is the single 8-line `chain_neg`. `consequentia_cond`, `nand_to_imp`, `excluded_middle` and
`negcond_ante` (26 each) are 0 in all 11 arms; `neg_to_contra` reaches 2 / 26 in one arm at its
last round. "The 9-line stratum follows" is one schema family (`chain_neg`) spanning 8–9 lines;
57 of the 82 nine-liners are at 0 everywhere. The per-schema table was pre-registered, is
computed (`artifacts/r3_2/summary.json`, `by_schema`) and is absent from `run2.md` and
`numbers.md`.

**The "zero-rate" draws are not zero-rate at six lines.** Base pass@10⁴ on the 45 six-line
targets (T = 0.8): f0 s0 29 / 45; **s1 13 / 45** (44 pattern samples ≈ 10⁻⁴ per sample);
**s2 26 / 45** (2,139 samples ≈ 5·10⁻³), from a training file with 0 reductio proofs in
155,000. In run 5 the same draws had 0 and 1 hit in 3·10⁶ samples at 7–10 lines and stayed at
0 / 300. Here both enter through the six-liners (s1: nothing accepted in rounds 1–6, first
six-line proof at round 7 shared with its frozen twin, 6-line ignition round 8, 7-line round
10; s2: 45 / 52 by rounds 2 / 3) and s2 goes on to 27 eight- and 25 nine-line targets that its
base did not hit in 2.5·10⁶ samples. At the pre-registered 8 rounds s2's conclusion already
holds (45 / 52 / 23 / 9); s1's 7-line entry depends on the extension. This falsifies E3 and,
with it, run 5's reading of the 7-line result as a pattern gate. The six-liners are a separate
schema (`nor_neg_ante`), not shorter instances of the 7–10-line schemata, and "every draw"
is three draws.

**Strict derived-ORE, cap 8 (300 required targets; 256 attempts; outside the take-home's
cap-6 rule, not a submission result).**

| draw | f | base r at pass@10⁴ (/300) | EI acquired (rate) | frozen | EI / r | EI / frozen | EI-only vs base | transfer /65 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| run-5 s0 | 0 | 27 | 26 (0.087) | 13 | 0.96 | 2.0 | 8 | |
| run-5 s1 | 0 | 27 | 24 (0.080) | 11 | 0.89 | 2.2 | 7 | |
| s2 | 0 | 26 | 36 (0.120) | 14 | 1.38 | 2.6 | 14 | 10 |
| s3 | 0 | 9 | 7 (0.023) | 6 | 0.78 | 1.2 | 1 | 5 |
| s4 | 0 | 20 | 43 (0.143) | 7 | **2.15** | 6.1 | **26** | 14 |
| s5 | 0 | 27 | 45 (0.150) | 13 | 1.67 | 3.5 | **25** | 10 |
| s0 | 10⁻³ | 37 | 40 (0.133) | 16 | 1.08 | 2.5 | 14 | 11 |
| s1 | 10⁻³ | 34 | 43 (0.143) | 18 | 1.26 | 2.4 | 16 | 11 |
| run-5 s0 | 10⁻² | 36 | 48 (0.160) | 21 | 1.33 | 2.3 | 18 | |
| run-5 s1 | 10⁻² | 43 | 63 (0.210) | 28 | 1.47 | 2.3 | 24 | |

- Keep rule: s2 fails (|26 − 27| = 1), s3 passes (18), so seeds 4 and 5 were trained as
  pre-registered; s4 passes (7), s5 fails (0). Two draws pass the rule (s3, s4), not one; all
  four fresh draws are reported.
- Base reach does not predict acquisition among comparable draws: over the six f = 0 draws
  Spearman ρ = 0.27; at r = 26–27 acquisition spans 24–45 (multiplier 0.9–1.7 ×; 0.8–2.2 × over
  all f = 0 draws). The executor's Pearson 0.81 over ten draws reproduces, but it is 0.62 over
  the six f = 0 draws and −0.50 over the five f = 0 draws other than s3: it is carried by one
  low draw and by the f > 0 draws, that is by f. What the data support: base reach ≈ 0 → EI ≈ 0
  (one draw, s3); above that, acquisition is not predicted by base reach at n = 5.
- f-dependence: EI means 0.101 (n = 6) / 0.138 (n = 2) / 0.185 (n = 2) are monotone, but the
  f = 0 range 0.023–0.150 contains both f = 10⁻³ values and two f = 0 draws (43, 45) equal or
  exceed both f = 10⁻³ draws (40, 43); only the f = 0 vs 10⁻² contrast is suggestive, and no
  per-f difference is separable. Base reach by f: 9–27 / 34–37 / 36–43.
- EI-only targets (acquired, not base-reachable at 10⁴) ≥ 20 in 3 of 10 draws (s4 26, s5 25,
  f = 10⁻² s1 24): the pre-registered criterion for "RL adds substantially beyond the base's
  reach" is met in those draws.

**What stands (reviewer's verdict).** (1) EI at 512 attempts per target carried the 8-line
reductio stratum past 10 targets in 1 of 6 pre-registered sixteen-round arms and 0 of 3 k = 64
arms; no frozen control produced any 8–10-line proof; the 10-line stratum is 0 everywhere.
(2) Run 5's "zero-rate" draws are not zero-rate on six-line targets and both enter the pool
through that stratum; two seeds show the entry, one shows the onward crossing. (3) The
crossing is a per-run stochastic event (54 vs 1 on one checkpoint). (4) Derived-ORE:
f = 10⁻³ lands in the pre-registered band on two seeds; two of four fresh f = 0 draws exceed
1.5 × base reach with 25–26 EI-only targets.

**Reworded (reviewer's wording replaces the executor's).**

| executor (`run2.md`, `STATUS.md`) | reviewer |
|---|---|
| "the crossing needs training rounds, not samples" | k = 64 × 8 ignited 0 of 3 and k = 32 × 16 ignited 1 of the same 3 seed-0 pairs (0 of 3 seed-1 replicates); the design cannot separate rounds from samples, and one f = 0 draw crossed within 8 rounds at k = 32 (round 7, 224 attempts) |
| "f is irrelevant" | no f effect detectable (1 of 4 vs 1 of 4 sixteen-round arms; Wilson 95 % for 1 / 4: 0.05–0.70) |
| "per-run probability ≈ 0.2 in 16 rounds" | 2 of 8 sixteen-round arms (Wilson 95 % 0.07–0.59); two further arms were still rising when stopped; 2 / 11 mixes in the 8-round B arms |
| "every draw has a non-zero base rate at the schema's shortest instance" | at the shortest reductio-requiring targets, which are one separate six-line schema; "every" = 3 draws |
| "RL produces schemata and lengths the base never emits" | "never" holds at 10⁴ for the f = 0 draws only; for f0.1 s0 it is 0 in 512 attempts per target (no 10⁴ coverage at 8–9 lines) |
| "the kept fresh draw (base 9) acquires 7" | two draws pass the keep rule (s3, s4), and s4 is the one that breaks the band |
| "the multiplier varies 0.8–2.2 × between draws of equal reachability" | 0.9–1.7 × at equal reachability (r = 26–27), 0.8–2.2 × over all f = 0 draws |

**Not supported as written.** (a) *Stratum* as the unit of ignition: "the 8-line stratum ignites
with probability p" should be "a schema ignites", with 2 of the 5 large 8-line schemata ever
acquired in full (above). (b) "Derived-ORE amplification is proportional to base reachability
on average": by the run's own falsifier (any kept draw outside 0.5–2 ×) the clause was to be
dropped, and s4 (kept) is at 2.15 ×; the write-up reports the miss but keeps the clause
softened to "a trend".

**Pre-registered expectations vs outcome** (`6ffade0`, 05:13 UTC, before any pod).

| id | expected | observed (reviewer) | verdict |
|---|---|---|---|
| E1a | 8-line stratum ignites in 3–5 of 6 A arms | 1 of 6 | **miss** |
| E1b | first 8-line proof between rounds 6 and 12 | rounds 14, 10, 4, 11, none, none | 2 of the 4 arms with a proof inside the window; not scored in the write-up |
| E1c | 9- and 10-line ≤ 3 targets in every arm | 25 nine-line targets in the igniting A arm; 10-line 0 | **miss** in the one arm that ignited; not scored in the write-up |
| E1d | f = 0.1 ignites no more often than f = 0 s0 | 1 / 4 vs 0 / 2 | not decidable at this n (literal reading: miss) |
| E2 | 8-line ignites in ≤ 1 of 3 B arms | 0 of 3 | hit, but the attached inference "rounds matter more than attempts per round" is not supported either way |
| E3a | zero-rate draws: 0 six-line hits at pass@10⁴ | 13 / 45 and 26 / 45 | **miss** |
| E3b | zero-rate draws: 0 / 345 after 8 rounds | 11 / 345 and 129 / 345 | **miss**; the falsifier "six-line hit followed by ignition: the wall was length-gated, not pattern-gated" is met by both draws |
| E3c | f0 s0 solves ≥ 30 of 45 six-liners by round 2 in both A arms | 33 and 38 | hit |
| E4a | f = 10⁻³ acquisition 0.08–0.16, both seeds | 0.133, 0.143 | hit |
| E4b | EI within 0.5–1.5 × r for each kept fresh f = 0 draw | s2 1.38, s3 0.78 inside; **s4 2.15, s5 1.67 outside**; s4 also outside the falsifier band 0.5–2 × | **miss** |
| E4c | EI ≈ 2 × frozen | 2.6, 1.2, 6.1, 3.5 | holds for none of the four within ± 25 % |
| E4d | EI-only 5–10 | 14, 1, 26, 25 | **0 of 4 inside**; s4, s5 meet the "EI-only ≥ 20" falsifier |
| E5 | seed-0 A arms' 7-line stratum at round 8 = run 5's 51–53 ± 2 | 52, 52, 52 | hit; the 6-line stratum does not change the 7-line result |
| falsifier | E1 at 0 / 6 with E2 at 0 / 3 → "real length wall" | 1 / 6 and 0 / 3, plus 1 / 2 C arms reaching 8 and 9 lines from f = 0 | not met; 8- and 9-line proofs are reachable by EI |

Figures: the executor's `figures/r3_2_strata.png`, `r3_2_dore_base.png` and
`r3_2_dore_fdial.png` are on the run branch. The review did not check them point by point, and
two of them plot the readings it rejects (acquisition by stratum; acquisition against base
reach with the 0.81 line), so they are not copied here.

## Limitations and next step

- Two ignition events. 2 of 8 sixteen-round arms (Wilson 95 % 0.07–0.59) is the whole evidence
  on the ignition probability; rounds vs samples and f = 0 vs f = 0.1 are not separable at
  3–4 arms per cell, and two arms were still rising at their stop round.
- The C arms' 16-round results are a post-hoc extension (disclosed); s1's 7-line entry rests
  on it. Other disclosed deviations: co-scheduling of EI arms with a coverage run, the extra
  f > 0 coverage runs, starting the s2 / s3 arms before the keep rule was evaluated.
- "The base never emits" 8–9-line proofs is measured at 10⁴ only for the f = 0 draws; for the
  f0.1 s0 checkpoint it rests on 512 frozen attempts per target.
- Derived-ORE is cap 8 (outside the take-home rule). Five of six f = 0 draws have base reach
  20–27, so the base-reach dependence rests on one low draw (s3); seeds do not supply spread
  on demand.
- Not re-derived by the reviewer: held-out / transfer greedy numbers (GPU), the balance
  attribution of spend, r32c's 2.67 pod-hours (its queue logs were not pulled; job logs and
  done markers were), bucket contents beyond directory listings.

Next measurements (reviewer; ≈ $3–5 on one RTX 3090): (1) rounds vs samples and the ignition
probability: the three seed-0 checkpoints × sampling seeds 2–5 at k = 32 × 16 and k = 64 × 8
(24 arms), scored per schema and pre-registered as a pooled ignition count with interval, not
a contrast; or first extend the two rising arms (A f0 s0 ss1, B f0.1 s1 ss0) to 24 rounds,
which is cheaper and says whether "non-ignition" is just "late". (2) `coverage.py --k 10000`
of `stage1_reductio_f0.1_s0` on its 54 + 25 acquired 8- / 9-line targets (≈ 20 min).
(3) Derived-ORE base-reach dependence: ≈ 6 more cap-8 f = 0 seeds, or Stage-1 steps 2,000 /
4,000 on one seed to move base reach deliberately; pre-register a rank correlation within
f = 0, not a Pearson across f. (4) Is the six-line entry special to `nor_neg_ante`? One arm per
f = 0 draw on the 300 pool plus six-liners of a second schema, if `necessity.py` yields any.
