---
written_on: 2026-09-28
written_by: agent:claude
---

# support-curves: does expert iteration expand the base model's support, or just amplify sampling?

Status: completed 2026-09-27 18:07 UTC – 2026-09-28 03:06 UTC (executor) and independently reviewed
2026-09-28 03:10–03:41 UTC. Review verdict: **accept — the central result stands** on the reviewer's
own code, normaliser, statement builder, Lean driver, and a different Lean patch release; nine
wordings need fixing (below), none of them changing the conclusion. No hard-constraint violation,
nothing quarantined. Numbers and wording below are the reviewer's, including the reviewer's own
words where they differ from the executor's. Run id `support-curves`. Proposal 12 (the elicitation
curve), experiment 1. This run happened on the fork (`dan-pandori/nd-takehome`), branch
`dan_support-curves` (a worktree from the fork's own `origin/dan`); like `cap-horizon` and
`stage1-dynamics` before it, this branch has **not** been fast-forwarded into the fork's `dan` — the
commits below are reachable only from `dan_support-curves`.

Review: [`review_support-curves.md`](https://github.com/dan-pandori/nd-takehome/blob/44c4872/review_support-curves.md)
(commit [`557dfbd`](https://github.com/dan-pandori/nd-takehome/commit/557dfbd) §Recount,
[`44c4872`](https://github.com/dan-pandori/nd-takehome/commit/44c4872) §Compare/§Verdict). Executor
write-up: [`run_support_curves.md`](https://github.com/dan-pandori/nd-takehome/blob/44c4872/run_support_curves.md),
numbers `numbers.md` § support-curves (SC0–SC13), log `log.md`, `QUESTIONS.md` (3 entries), all on
`dan_support-curves`. Pre-registration:
[`preregistration/support-curves.md`](https://github.com/dan-pandori/nd-takehome/blob/44c4872/preregistration/support-curves.md)
(`2e87c2f`, committed 18:16:17Z, 53 s before the first pod — gate 0), with one disclosed addendum
(`b3fa417`, 18:33:20Z, re-sizing the run up on a throughput measured 7× the budgeted rate, 14 s
before the first stage-1 sample). Brief: `docs/proposals/2026-09-27-elicitation-curve.md`
(proposal 12, experiment 1).

Question: on 383 never-trained-on transfer theorems, does the expert-iteration (EI) model solve
theorems the base model essentially **cannot** — at attempt counts far beyond what EI itself used —
or does base pass@k eventually catch up? The project's whole pattern to date (proposal 12's "what
holds up") predicts the amplifier reading (RL is sample-efficiency, not new capability); this run
was designed so the generaliser reading could win, with a pre-registered falsifier at ≥ 20 survivors.

## Configuration

- **Models.** *Base s0/s1*: `ckpts/lf/stage1_a1_seq_s{0,1}.pt`, md5 `9bde44c0…` / `fc27e52d…` —
  3,214,336-parameter from-scratch GPT, `lean_seq`, cap 6, Stage 1 only. *EI s0/s1*: base + 8 rounds
  × k 32 expert iteration at T 0.8 on a pool **disjoint** from the 383-theorem transfer set (0
  overlap by name, string, and renaming class, over all 542,824 training records the reviewer
  audited), re-trained in this run: `la_T1_sc_s0_r8.pt` md5 `5cebd7ec…`, `la_T1_sc_s1_r8.pt` md5
  `105be4f3…`. Judge: **Lean alone** (`lean_judge`), Lean 4.34.0 core, no Mathlib — the reviewer's
  own re-check ran Lean 4.34.1 and every counted proof still passed, a cross-version confirmation.
- **Theorem set.** `data/sc/theorems.jsonl`, md5 `3cb6e7bf…`, 383 theorems stratified by `L_true`
  (60 each at 7–12, all 13 at `L_true` 13, all 10 at `L_true` 14), committed before any pod.
  `L_true` is an ND-derived **upper bound** on the minimal Lean proof length (5 % of the base's own
  counted proofs at T 0.8 come in below their `L_true` label).
- **Design.** Stage 0: EI training, both seeds. Stage 1: both models, seed 0, T 0.8, up to 10,000
  attempts/theorem. Stage 2 (adaptive crux): base gets up to 400,000 more attempts total on the
  forward crux (200,000 at T 0.8, 200,000 at T 1.0); EI gets up to 20,000 more on the reverse crux.
  Stage 3: replication on Stage-1 seed 1 at k = 2,000. Secondary: base teacher-forced log p of every
  distinct EI-found crux proof. **49,715,072 samples drawn** in the `support.py` stages (the
  write-up's own documents disagree with each other — ≈ 38 M in one place, ≈ 47 M / ≈ 12 MB in
  another — all superseded by this figure and by 8.6 MB of per-theorem records).
- **Cost.** `podbudget support-curves`: **20.37 pod-hours, $9.99** of a declared 30 h / $15 ceiling,
  registered before the first pod. Five **NVIDIA A40 46 GB, $0.49/h billed** (`sc1`–`sc5`), all
  deleted. RunPod balance $136.49 throughout (floor $100). Bucket:
  `hf://buckets/dan-pandori/nd-rl/support-curves/{ckpts,artifacts,data}` (public).

## Observations

The pre-registered falsifier — **≥ 20 forward-crux theorems with 0 base successes in ≥ 40,000
attempts at both temperatures while EI solves each at p̂ ≥ 0.01** — **fires**. At the pre-registered
depth the count is **37**; the run pushed deeper and reports the more conservative **29** at
200,000 attempts per temperature (4 × 10⁵ total per theorem, 5× the pre-registered depth). The count
is monotone and decaying with depth, never reaching zero:

| base attempts per temperature | 40,000 (pre-registered) | 50,000 | 100,000 | 150,000 | 200,000 |
|---|---|---|---|---|---|
| surviving theorems | **37** | 35 | 33 | 30 | **29** |

On these 29 theorems, EI solves each at p̂ 0.022–0.9995 (not "1.000" — the write-up's upper bound is
stale). **No stratum shows a pass@k crossover at any measured k up to 10,000, on either Stage-1
seed** — base pass@k climbs (e.g. `L_true` 7: 0.081 → 0.317 at k 32 → 10,000) but never catches EI
(0.262 → 0.417 in the same stratum), and at `L_true` 9 it is 0 for base against 0.533 for EI at
k = 10,000. Seed agreement replicates the per-theorem structure: base 92.7 %, EI 90.3 % at equal
attempts (383 theorems, stage 1).

Two independent measurements agree the effect is not a shallow sampling gap: base pass@k direct
sampling gives p_base < 7.5 × 10⁻⁶ (95 %) on the 29 survivors, and the base's own teacher-forced
probability of the EI model's proof tops out at 9.5 × 10⁻⁸ (median −25.65 log p over 297 crux
proofs) — ≈ 50× below what 400,000 draws could have surfaced, and the two readings are mutually
consistent.

Five checks against an artefactual falsifier all hold: **leakage** (0 overlap between the
falsifier survivors and the EI training pool by name, string, or renaming class, over the whole
2,285-theorem pool); **proof validity** (the reviewer's own Lean re-check, on the literal sampled
text with its own statement builder — a stronger test than the run's own judging path — accepted
**1,196 / 1,196** distinct counted proofs across the whole run, 0 banned tokens); **judging path**
(11/11 independently-written proofs pass the run's judge, normalised and un-normalised); **the
sampler** (base solves 8 of 9 `known_solvable.txt` theorems in the stage-1 records at 126–1,341
successes per 2,048; see limitations below); **temperature** (T 1.0 rescues 19 theorems T 0.8
alone would miss, 17 of them in the crux).

Every other pre-registered quantity reproduces: E1 EI transfer 869 (s0) / 972 (s1, see the seed-1
caveat below); E4′ forward crux 82; E6 reverse crux 6, 2 survive 50,000 more EI attempts; E8 one
theorem at `L_true` ≥ 13 solved (EI only, both seeds); E10 seed agreement above the 70 % bar; E12
Lean is 12.5–20.9 % of sampler wall time, below the 23 % line. Two predictions missed low
(**E2′** base 45/383 at k = 10,000, predicted 100–220; **E3′** EI 121/383, predicted 230–340 — base
grammar brittleness is part of why: 36.5 % of base samples at T 0.8 fall outside the strict
`lean_seq` grammar, against 20.9 % for EI). **E11 was measured (33 survivors at 100,000
attempts/temperature, predicted 0–12) but never reported in the original write-up** — the reviewer
requires it be added as a miss, in the same direction as the falsifier. E9 (temperature) holds on
theorem count (33 vs 18 = 1.83×, under the pre-registered < 3×) but the **per-attempt success rate**
is 3.30× — above 3×, so the metric that is being bounded needs to be named explicitly.

Proof length/term size: EI proofs run longer than base's (median 9–10 lines / term size 61–77 vs
base's 8 lines / term size 50–60), consistent with `L_true` being an upper bound under Lean rather
than the true minimal length.

## Limitations and next step

- **n = 2 Stage-1 seeds, one 3.2 M-parameter base, one pool.** Per the write-up's own framing,
  counts sit inside `NOISE_FLOOR.md`'s measured floor and are not the finding; the **per-theorem
  structure**, which replicates at 92.7 % / 90.9 % seed agreement, is.
- **The seed-1 EI checkpoint the run's stage-3 numbers were measured on no longer exists.** Pod
  `sc2` was deleted after diffing only its `.jsonl` artifacts, not its `ckpts/`, against local
  copies. The re-trained replacement (`la_T1_sc_s1rerun_r8.pt`, md5 `12c13e61…`) reproduces round 1
  of the lost run exactly but diverges by round 2 (transfer 374 vs 373 of 2,285) — the fine-tuning
  step is not bit-reproducible on this hardware, so it is a genuinely different model. Every
  seed-1 number (E1's 972, E10's EI figures) is reproducible only to "an 8 × 32 EI run from
  `stage1_a1_seq_s1.pt`, seed 1," not to a specific artefact. **This does not touch the falsifier,
  which is entirely a seed-0 measurement.**
- **`max_new` = 400 truncation exceeds AGENT_POLICY's 0.1 % line in one base stratum**: base
  `L_true` 10 is ≈ 1.0 % truncated (base `L_true` 9/11/12 ≈ 0.2–0.3 %, elsewhere ≈ 0). On the 29
  survivors this turns 400,000 attempts into ≈ 397,000 effective — E5 is untouched — but the
  write-up's "400 and 512 give byte-identical records" was validated only on 6 `L_true` 7 theorems
  that had zero successes either way, not on a stratum where a long proof could actually be cut.
- **`artifacts/sc/diag/` does not exist** — not in the repository, not in the bucket (a full
  172-entry recursive listing). The compaction-on/off comparison the write-up cites from it
  (116/140/179 per 2,000) is not derivable from anything on file and should be dropped or re-run;
  the sampler check's substance is recoverable from the stage-1 records instead (base solves 8/9
  `known_solvable.txt` theorems at 126–1,341 per 2,048).
- **`sc_recheck.py` does not check what its docstring and the write-up claim.** A dead
  `[...] if False else [nd2lean.translate(...)]` branch makes it Lean-check the normalised,
  translated proof rather than the literal sampled text. The reviewer's own independent re-check
  did use the literal text for all 1,196 counted proofs and Lean accepted them all, so nothing
  downstream moves — but the script and its claim should be fixed or reworded.
- **Stale counts elsewhere in the write-up**: "372 accepted EI proofs" on the survivors is the
  35-survivor-set figure; for the final 29 it is 315 (231 at seed 0 alone). "The 35 falsifier
  survivors" (a leakage-check label) is likewise stale — the final set is 29 (a subset of 35).
- **Repository-level, not this run's bug**: `ladder_ei.py` fails at import on the fork's `origin/dan`
  (commit `9a1db24`, run `lean-judge`, renamed `expert_iter.relabel` without updating this caller).
  This run's brief forbade editing `ladder_ei.py`/`expert_iter.py`, so it worked around the break
  with its own `sc_ladder_ei.py`; the one-line fix (`relabel` → `relabel_candidate`) is still open
  on `origin/dan` for whoever next needs `ladder_ei.py` directly. Flagged in `QUESTIONS.md`, not
  fixed here.

**Next step (reviewer, ranked).** (1) The same 383-theorem protocol on a second, independently
pre-trained base of a different size — this is the single most informative next measurement,
since support expansion at 3.2 M does not show whether the 29 survivors are within a *larger*
base's reach, and it is the question proposal 12's experiment 2 should now be ordered around
(capacity/coverage, not more search). (2) Push the 5 survivors at `L_true` 11 and the 1 at 12 to
10⁷ base attempts at T 1.0 (≈ 3 pod-hours at this run's measured throughput) — they have the
lowest per-proof log p (≤ −30) and so the most room left for the amplifier reading to still win.
(3) Close the seed-1 gap by re-running stage 3 on the already-uploaded `la_T1_sc_s1rerun_r8.pt`
(≈ 1 pod-hour), which attaches a seed-1 column to a checkpoint that actually exists.
