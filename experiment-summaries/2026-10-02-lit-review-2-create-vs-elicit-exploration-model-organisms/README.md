---
written_on: 2026-10-02
written_by: agent:claude
---

# lit-review-2: create vs elicit since 2025, exploration outside support, and model organisms

Status: **completed and reviewed 2026-10-02. Verdict: stands, with seven rewordings** (V1–V6 applied and marked in the
nd-rl copy; V7 concerns the fork's `log.md`). No hard-constraint violation, nothing quarantined. **Not supported as
worded:** the "always" / "only" claims about exploration (V2). This is a literature run: no pods, no models trained or
sampled, no Lean checks, no new numbers. Where the reviewer's wording differs from the executor's, this summary uses the
reviewer's.

Question (proposal 21, brief `lit-review-2`), extending `lit-review` (2026-09-29, 58 papers) without redoing it:
1. **Create vs elicit, 2025–2026.** What is the current evidence and measurement standard, which designs separate
   creation from elicitation convincingly, and where do `trajectory` and `rl-from-ckpt` fall short?
2. **Exploration.** Which methods demonstrably produce behaviour outside the base model's support, and do they need a
   pretrained prior?
3. **Model organisms.** Which domains allow exact control of pretraining coverage *and* an RL phase, and what could each
   show?

## Configuration

- Brief: [`docs/proposals/claude-heavy/BRIEF_lit-review-2.md`](../../docs/proposals/claude-heavy/BRIEF_lit-review-2.md)
  (authorised by Dan 2026-10-02, [proposal 21](../../docs/proposals/2026-10-02-claude-heavy-set.md)). Fork
  `dan-pandori/nd-takehome`, branch `dan_lit-review-2` from the fork's `origin/dan`, worktree `~/work/lit-review-2`.
  Not merged into the fork's `dan` (documents only).
- Pre-registration `preregistration/lit-review-2.md`, commit `d024bd0e` at 16:55:04 UTC. The first source was fetched
  at 16:55:09, so the expectations were written before any new paper was read (reviewer R1).
- Executor 16:52–17:07 UTC, with three parallel screening agents; deliverables in `7df1dd58` (17:07:14). Reviewer
  17:08–17:17: §Recount [`f8372285`](https://github.com/dan-pandori/nd-takehome/commit/f8372285) (blind, before
  reading `REVIEW.md` or `log.md`), §Compare / §Verdict
  [`35b6eef4`](https://github.com/dan-pandori/nd-takehome/commit/35b6eef4).
- Method: arXiv HTML or PDF text fetched by version into `~/lr_sources/` (outside the repository; copyrighted) and
  grepped; a 108-row claim ledger with quotes and locations. The reviewer wrote its own fetcher and matcher
  (`review_lit_review_2/myfetch.py`, `match.py`: NFKC, LaTeX stripped, segments split at `…`, fuzzy windows read by
  hand) and did not use the executor's `fetch.py` / `quote.py`.
- Pods: none. A $0.50 / 1 h pod budget was registered as a guard; `podbudget lit-review-2` shows 0.00 h. **$0.** No
  checkpoints, artifacts or data, so nothing went to the bucket.
- Outputs in nd-rl: [`docs/literature/2026-10-02-lit-review-2/`](../../docs/literature/2026-10-02-lit-review-2/REVIEW.md)
  holds `REVIEW.md` with V1–V5 applied and marked, the 11 in-depth notes (V6 applied in the EDL note), the claim ledger
  `claims.md`, `screened.md` (51 rows with technique or design), and `to_add_to_zotero.md` (47 identifiers for Dan;
  `references.bib` untouched).
- Review: [`review_lit-review-2.md`](https://github.com/dan-pandori/nd-takehome/blob/35b6eef4/review_lit-review-2.md).
  Executor write-up: [`run_lit_review_2.md`](https://github.com/dan-pandori/nd-takehome/blob/35b6eef4/run_lit_review_2.md).

## Observations

**Verification (reviewer's own re-fetch, phase 1).**
- **45 / 45** cited arXiv `id+version` strings exist, with the titles `screened.md` gives.
- **106 / 106** checkable ledger claims are in the source text at the cited version: 99 exact after normalisation,
  6 differing only in LaTeX formatting, plus one table row (Yue Table 2) and one absence claim (no ablation of AlphaZero's
  noise) that reproduce. The 3 wholly UNVERIFIED rows are honestly marked (ε-random actions in LLM RL; Physics of LMs
  Part 4.1; Tsilivis at p_cot = 0, where the paper's lowest value is 0.01).
- Of 102 quoted strings in the notes, 77 are verbatim in the note's source, 7 in another paper the note names, 3 in our
  own documents, 2 differ in formatting, 12 are scare quotes, and **1 is a paraphrase in quotation marks** (EDL note,
  fixed as V6).
- The ledger has **108** claims, 5 wholly or partly UNVERIFIED, not "110 / 8" (V3: table header lines were counted).

**Deliverable counts.** 51 new papers (0 arXiv-id overlap with `lit-review`'s 55), above the brief's 25–40 and the
pre-registered stop at 40 (disclosed; the stop rule existed to limit spend). 11 read in depth (brief 8–12); **34 / 51
read at abstract level only**. 29 / 51 are from 2025–2026.

**Q1, create vs elicit.**
- The standard is still Yue et al.'s **equal-k pass@k** (2504.13837v5): base-unsolved, RL-solved problems are 0.0 %
  (AIME24, k 1,024) and 1.0 % (MATH500, k 128), Table 2. Its likelihood evidence covers two problems (Fig. 6).
- **Our A/B split is the same equal-k design at k 256, not an unequal budget (V1, reviewer's correction).** In
  `trajectory`, groups come from r0 and r8, both at k 256, on evaluation-only pools. The real gaps: k 256 rather than
  ≥ 1,024, no large-k base curve on B ∪ C, groups from one sample seed. `rl-from-ckpt`'s creation threshold was
  calibrated post hoc.
- ProRL's "where base models fail entirely" gains are largest on Reasoning Gym tasks where "the base model struggles
  with formatting". Successors make the boundary depend on training length and width (BroRL; 2510.04028v1).
- The convincing designs control pretraining and intervene on it. Closest to us: **Abdulsalam, Patel & Saxe
  (2607.07646v1)**, from scratch on a rewrite grammar: base 0 % on buckets 4–5 at pass@1024, GRPO solves them at
  pass@16, RFT plateaus. **Each bucket has 16 problems and the curves are 3 seeds (V5)**, so "0 % at pass@1024" is a
  weak support statement. It reports no likelihoods.
- Coverage (2510.15020v2, 2503.07453v2) is sequence-level: "covered at k" ≈ sequence log p ≥ −ln k (−5.5 nats at
  k 256).

**Q2, exploration.** Dedicated methods with out-of-support evidence (Go-Explore, AlphaZero, DeepHOL-Zero heuristic
injection, POPE oracle prefixes, AlphaEvolve) draw on a source other than the policy's own samples, and in pretrained
LLMs on-policy bonuses do not move zero-reward problems (2601.18779v1). **But "always" / "only" is not supported (V2):**
in small from-scratch organisms plain on-policy RL reaches base-pass@1024 = 0 problems (2607.07646v1 GRPO; 2509.22613v2
PG "can explore and discover new correct paths that were absent from the initial training set"). Neither measures base
support, and neither do the methods the review counts. DeepHOL-Zero: 7.0 % without exploration vs 56.3 % with
(1905.10501v3 Fig. 4), and "With no new training data, the learning process would stall", which predicts proposal 20
E3's step-0 null. No source applies ε-random actions to a sequence policy (UNVERIFIED, absence after search); Go-Explore
argues for ε **after returning** to a stored state.

**Q3, model organisms.** With exact coverage control and a published RL phase: rewrite grammar (2607.07646v1), parity
(Tsilivis 2510.11495v2, p_cot dial; p_cot = 0 untested), graph path-finding (2509.22613v2, 2601.15158v4), zero-RL
provers (DeepHOL-Zero, TacticZero), games (AlphaZero). Physics of LMs RL is UNVERIFIED; Othello-GPT, grokking,
Transcendence and DreamCoder have no RL phase. That is 3–4 families depending on the split. Ours is the only organism
with likelihoods of arbitrary proofs along the whole trajectory.

**Ranked implications (executor's ranking; the reviewer finds they follow from the sources).**
1. Equal-budget support re-read of B ∪ C at k ≥ 4,096 on best-cap12 r0, with "covered" = sequence log p ≥ −ln k
   pre-registered, beside a scalable power-sampling null. Reads only.
2. Reference-prefix starts for group C (POPE / Go-Explore's backward algorithm). Proposal 20 lacks it.
3. Proposal 20 E2 / E3: ε after return, ε on throughout, a seeded-only control.
4. A coverage-removal intervention: delete one rule pattern from pretraining (p = 0 and small p), then RL.
5. GRPO beside the RFT-like ladder.
6. Amplification null (Tsilivis's recursion) and per-theorem diversity tracking.
7. EDL shape test from existing per-checkpoint log p (low: its theory is SFT-only).

The review also corrects the Q2 agent's "group C's single −12-nat step" (not supported; consistent with the
`trajectory` review).

**Pre-registered expectations** (reviewer's tally, V4): **6 held, 1 partly falsified.**
- Held: #1 (as "none found in this screen"), #2 (partly: format weakness conceded for Reasoning Gym only), #3 (four
  controlled-pretraining + RL papers), #5, #6 (106 / 106), and **#7** — its wording was "mis-stated *or needs a
  qualification*", and three qualifications were found; one (`lit-review` §c item 1 compares a per-step probability with
  a sequence-level bound) is arguably a mis-statement. The executor graded #7 "not met" and the tally "5 of 7".
- Partly falsified: **#4**. The executor cited heuristic injection, oracle prefixes and LLM evolution; the reviewer's
  stronger reason is plain on-policy GRPO / PG on from-scratch organisms. The "no LLM-RL bonus measures support at
  k ≥ 1,024" clause held.

**Model labels.** No model was measured. Inherited numbers carry their labels: `trajectory` and `rl-from-ckpt` use
best-cap12, 9,560,832 params, `lean_staten`, from scratch on K12, 3 seeds, Lean alone; the reviewer matched them to
those runs' summaries.

## Limitations and next step

- Breadth over depth: 14 minutes from pre-registration to DONE, 34 / 51 papers read from the abstract only. Absence
  claims (expectation #1, ε in LLM RL) are bounded by that screen.
- The reviewer did not test the stated location (section / figure) of every ledger row (≈ 20 read by hand, all right),
  nor the Tsilivis depth / width list.
- `log.md`'s "17:10" and "17:15" entries come after the DONE commit (17:07:14); the log was written after the fact with
  estimated times (V7, minor, not fixed on the fork).
- Orchestration: the reviewer's phase-1 copy kept `lit_review_2/REVIEW.md` and `log.md`, because the removal patterns
  match `run*.md` / `STATUS*.md`, not a literature run's deliverables. The reviewer did not open them until phase 2.

**Next measurement (reviewer; would settle V1 / V2 for our setting):** Implication 1 as a read. best-cap12 r0
(9.56 M `lean_staten`, from scratch on K12), 3 seeds, `trajectory`'s B ∪ C theorems, k 4,096, with the sequence log p of
every Lean-accepted sample. If r0 solves a substantial share of B at k 4,096, B is elicitation at the field's standard.
If B and C stay at 0 with every reference below −ln 4,096, `rl-from-ckpt`'s ladder solves are the first
likelihood-checked base-fails result in this literature.
