# STATUS — proposals round 2 (runs 5, 2, 3, 1, 4)

Started 2026-09-17. Brief: BRIEF_ROUND2.md. Policy: AGENT_POLICY.md. Previous: STATUS_campaign1.md, STATUS_followup.md, STATUS_handoff.md.

## Run 5
- 2026-09-17 19:08 UTC  Started. Oracle (`minlen.py --forbid`, `necessity.py`) written and tested; plan + expectations in log.md; reductio candidates being labelled on the VPS; pods next.
- 2026-09-17 22:05 UTC  Done. Required pools built (reductio 300 + 150; cap-8 strict derived-ORE 300 + 65; ten hand-checked each), 9 + 8 arms + 5 base pass@10⁴ runs. Reductio: f = 0 zero-rate draws 0 / 300, igniting draw and both f = 0.1 arms ≈ 52 / 300 = the two 7-line schemata only; derived-ORE (cap 8): f = 0 0.087 / 0.080 → f = 10⁻² 0.160 / 0.210, 10× block C, base-reachable on 27 / 300. 0 oracle violations in 9,600 target-arm pairs. `run5.md`, `figures/run5_*.png`, `numbers.md` §Run 5, `artifacts/r5/summary.json`; bucket `hf://buckets/dan-pandori/nd-rl/round2/run5/{artifacts,ckpts,data}`. Pods p1 + p2 (3090) ≈ 5.6 h ≈ $3.
- RUN5 DONE 2026-09-17 22:05 UTC

## Run 2
- RUN2 DONE 2026-09-18 01:30 UTC — `run2.md`, numbers.md §Run 2, `figures/run2_acq.png`, `run2_curves.png`; uploaded to `hf://buckets/dan-pandori/nd-rl/round2/run2/{artifacts/r2,data/r2,ckpts/r2}`. No f = 0 arm crossed zero except one NEGI draw with a non-zero base rate; depth-4 is a base generalisation at cap 8 (11–20 % of samples from zero-coverage data) and unreachable at cap 6 (length).
- 2026-09-17 23:25 UTC  NOTE: seven pods not created by this session (`r4-1`…`r4-6`, `r1-a100`) are running on the account since ≈ 23:00 UTC; see QUESTIONS.md (kill switch scope, HF token echo).
- 2026-09-17 22:05 UTC  In progress: classes and expectations pre-registered (log.md 19:47); six pools built (three from generator knobs / schemata, disclosed in QUESTIONS.md); four sets assembled (f = 0 asserted); 8 Stage-1 done or finishing; 12 EI arms launched 21:58–22:01 on p1–p3.

## Run 3
- RUN3 DONE 2026-09-18 01:06 UTC — uploaded to `hf://buckets/dan-pandori/nd-rl/round2/run3/{artifacts/r3,data/r3,ckpts/r3}`.
- 2026-09-18 01:01 UTC  All 30 arms done; `run3.md`, numbers.md §Run 3, `figures/run3_ignition.png` written; upload pending (checkpoints pulling). One proof ignites every arm; the other pattern never; invalid strings with the box tokens ignite one depth-3 draw.
- 2026-09-17 22:18 UTC  Started (in parallel with run 2's tail): plan + expectations log.md 22:14; 30 injection arms (5 arms × 6 conditions) on p4 (depth-3) and p5 (reductio), RTX 3090s.

## Run 1
- RUN1 DONE 2026-09-18 01:32 UTC — `run1.md`, numbers.md §Run 1, `figures/run1_forms.png`, `run1_scale.png`; uploaded to `hf://buckets/dan-pandori/nd-rl/round2/run1/{artifacts/r1,data/r1}`. Lean agrees with nd_verify on every pool (0 disagreements, 4,012 / 4,012 negatives rejected); Lean form +0.31 greedy over the token format (paired, 236 theorems); smallest Qwen3 that proves the RL-found theorems in Lean: median 8B for all three classes.
- 2026-09-18 01:01 UTC  Step 1 done: `nd2lean.py`, agreement 100 % on 95k+ positives and 4,012 negatives (75,085 RL transfer proofs re-checked after a premise-naming fix). Step 2 generated (Qwen3-Coder-30B-A3B on p6, A100), scoring; step 3 running (0.6B, 1.7B done).

## Run 4
- RUN4 DONE 2026-09-18 01:45 UTC — `run4.md`, numbers.md §Run 4, `figures/run4_grpo_vs_ei.png`; uploaded to `hf://buckets/dan-pandori/nd-rl/round2/run4/{artifacts/r4,ckpts/r4}`. GRPO (G = 8 / 32, matched budget, no KL, no retained data) crosses zero coverage on all six draws at 0.40–0.51 vs EI's 0.34–0.36, at the cost of held-out greedy 0.87–0.95 → 0.34–0.67. The 85M / relative-codec arm was not run (no access).
- 2026-09-18 01:01 UTC  `grpo.py` written; 12 arms running on p4 / p5 (8 finished). First arm: acquisition 0.48 vs EI 0.34 on the same draw, held-out greedy 0.87 → 0.56.

## Close-out
- Pods created by this session: p1–p5 (RTX 3090, $0.50/h), p6 (A100 80 GB, $1.59/h); all deleted by 01:36 UTC (`podls`: none registered). Pod-hours ≈ p1 6.0, p2 5.9, p3 5.1, p4 3.1, p5 3.3, p6 2.1 → ≈ **$15** for the five runs (run 5 ≈ $3, run 2 ≈ $6.5, run 3 ≈ $1.6, run 4 ≈ $1.6, run 1 ≈ $3.4 incl. downloads), every run far inside its $50. Balance 19:00 → 01:47 UTC: $121.40 → $294.12 (a top-up of ≈ $188 landed in between; the seven+three pods of other sessions on the account, see QUESTIONS.md, are still running and are not mine). Cron kill switch remains at 2026-09-19 00:00 UTC.
- Deliverables: `run5.md`, `run2.md`, `run3.md`, `run1.md`, `run4.md`; `figures/run{5,2,3,1,4}_*.png`; numbers.md sections for every run with source files; log.md with plans, pre-registered expectations and results-vs-expectations per run; QUESTIONS.md (five questions with defaults); bucket uploads under `hf://buckets/dan-pandori/nd-rl/round2/run<N>/`.
- One-paragraph answer across the five runs: on target pools that *require* a pattern, expert iteration at f = 0 solves only what the base draw already generalises to (reductio: 0 / 300 from zero-rate draws, the two 7-line schemata from the others; derived ORE: 2× the frozen control, rising with f); none of six new patterns crossed zero coverage from a zero base rate, and the "structural" ones failed at cap 6 for length while the same fourth box is a 11–20 % base generalisation at cap 8; a single injected proof — or, for the structural pattern, four invalid strings with the right tokens — ignites every dead arm; GRPO ignites faster and higher than EI on the same draws but wrecks the in-distribution model; Lean agrees with `nd_verify` on 95k proofs and 4k negatives, a 30B code model proves 56 % of the theorems in Lean against 26 % in the token format, and the RL-found proof classes first become provable in Lean at 4B–14B.
- DONE 2026-09-18 01:45 UTC
LIMIT_HIT 2026-09-18T02:17:44Z
DONE 2026-09-18 02:54 UTC (reviewer round2-run1: review_round2-run1.md phases 1–2 pushed; reviewer.done written)
