# Review brief: run4-grpo-review — the unreviewed GRPO replication (proposal 14, item 7)

Run id `run4-grpo-review`. Role: **reviewer** (your role brief governs the two phases). Repository
`~/work/run4-grpo-review` (branch `dan_run4-grpo-review` from `origin/dan_run4_grpo`); phase-1 blind
workspace `~/review/run4-grpo-review` (write-ups withheld). Authorised by Dan 2026-09-28 as nearly free:
**no GPU**. Pod ceiling registered at 1 h / $0.50 only as a guard — if a check genuinely needs a GPU, do
not start a pod; list it as "not checkable without a GPU" with a cost estimate.

## What is being reviewed

Run `run4-grpo` (agent2, 2026-09-17/18, branch `dan_run4_grpo`) was an independent replication of
round 2's run 4 (GRPO vs expert iteration at depth-3 f = 0). Its session was ended by the 2026-09-18
authentication failure; the results were salvaged in commit "Salvaged partial replication results
(session ended by auth failure)", and the write-up `run4.md` still contains an unfilled `XXX`. **It was
never reviewed.** Its claims matter more than the reviewed original's, because they go past "rare-prior":

- GRPO (on-policy, groups of 8 / 32, binary reward, group-mean baseline, no KL, 800 samples per update,
  320 updates = EI's 8 × 32 per target) ignited the depth-3 pattern on **30 of 30 arms**, including draws
  s21 and s23 with **0 depth-3 samples in 600,000** base attempts (and "extension" draws s27, s29 — the
  `XXX`);
- paired EI on the same draws ignited in **6 of 10** arms;
- ignition came at update 5–110 (4,000–88,000 samples) — far fewer than the 600,000 in which the base
  produced none;
- lr 3e-5 kept held-out greedy 0.75–0.86 and still ignited every arm (lr 1e-4: 0.54–0.72);
- "every depth-3 proof GRPO found is unreachable by sampling from its own Stage-1 mode" (sentence
  truncated in the write-up).

Raw outputs: `hf://buckets/dan-pandori/nd-rl/run4-grpo/artifacts/r4/` (per-arm directories: GRPO, EI,
frozen, base-rate files). Docs on the branch: `run4.md`, `numbers.md` § Run 4, `preregistration/`
(`run4-grpo.md` and its amendments), `log.md`, `STATUS.md`, `QUESTIONS.md`, `run4_*.py`. The reviewed
original for comparison: `~/nd-rl/experiment-summaries/2026-09-18-round2-run4-grpo-vs-ei/README.md`.

## What to establish

1. **Which claims the files on disk and in the bucket support at all**, given the salvage. For each
   headline number: reproduced / reproduced with corrections / not reproducible from what survives.
2. **The zero base rates.** Which files, which predicate, which temperature, how many samples, which
   checkpoint — and does the recount give 0? This is the claim everything else rests on. Re-derive the
   depth-3 predicate yourself (box depth of the dependency-pruned proof), as round 2's reviewer did.
3. **Ignition and acquisition** per arm, recounted from the per-round files; the paired EI arms; the
   lr variants and their held-out cost.
4. **Checking.** Lean alone decides (policy, 2026-09-27). These are token-format proofs: translate with
   `nd2lean.py` and check in Lean (`lean_judge.py`'s fallback path, batched) — every counted depth-3
   proof on the zero-rate draws, and a sample elsewhere; report agreement with the original `nd_verify`
   verdicts.
5. **Leakage and splits**: renaming-class overlap between each draw's Stage-1 set and the target,
   transfer and held-out pools.
6. **Model labels**: which checkpoints (md5 where recoverable), parameter count, tokenizer — the brief's
   original checkpoints no longer existed and six fresh draws (s20–s25) were trained; confirm.

## Output

`review_run4-grpo-review.md` on the branch in the standard two phases (blind recount, then compare and
verdict), with a "What stands" section written so it can be quoted, and an explicit statement on the
zero-base-rate claim. Then `touch ~/runs/run4-grpo-review/reviewer.done`; the librarian follows and
records it in nd-rl.
