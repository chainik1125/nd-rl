---
written_on: 2026-09-29
---

# lit-review: techniques from the literature we have not tried yet

Status: completed and reviewed 2026-09-29. No hard-constraint violation, nothing quarantined. The review
verified every checked claim, identifier and code mapping, and asked for six rewordings, which are applied in
the nd-rl copy. The reviewer would rank the shortlist differently in one place. This is a literature run: no
pods, no training and no counts.

Question: which techniques that published provers and RL-for-reasoning work rely on, and that this project has
not used, have the highest expected value for moving `L*`, support expansion or Stage-1 seed variance on the
3.2 M `lean_seq` / state-env setup? (Dan's hypothesis: more low-hanging fruit like the proof state that
`state-env` took from AlphaProof.)

## Configuration

- Brief: [`docs/proposals/literature/BRIEF_lit-review.md`](../../docs/proposals/literature/BRIEF_lit-review.md)
  (authorised by Dan 2026-09-29). Fork `dan-pandori/nd-takehome`, branch `dan_lit-review` from the fork's
  `origin/dan`, worktree `~/work/lit-review`. Not merged into the fork's `dan` (documents only).
- Pre-registration `preregistration/lit-review.md`, commit `dae83e61` at 03:22:31 UTC, before any screening.
  The executor started at 03:20:04. The file's own "≈ 03:35 UTC" is wrong (reviewer, cosmetic).
- Executor 03:20–03:36 UTC; it used five cluster research agents and one code-map agent. Deliverables are
  in commit `67aa57fb`. Reviewer 03:40–03:48 UTC: §Recount `d61f67f8`, §Compare, ranking and §Verdict `2da89271`.
- Method: arXiv HTML or PDF text fetched into files (`lit_review/fetch.py`) and grepped. Every quote gives the
  id, version and location. The reviewer re-fetched every source it checked with its own fetcher
  (`review_lit_fetch.py`), without using the executor's cache. Source texts are copyrighted and are not
  committed or uploaded; they can be re-fetched by version.
- Pods: none. Pod budget $0.50 / 1 h was registered as a guard; 0.00 h were used. **$0.** No checkpoints, artifacts
  or data, so nothing went to the bucket.
- Outputs in nd-rl: [`docs/literature/2026-09-29-lit-review/`](../../docs/literature/2026-09-29-lit-review/REVIEW.md)
  holds `REVIEW.md` with the reviewer's rewordings applied and marked, the 23 in-depth notes (copied verbatim),
  and `to_add_to_zotero.md` (56 identifiers for Dan; `references.bib` untouched).
- Review: [`review_lit-review.md`](https://github.com/dan-pandori/nd-takehome/blob/2da89271/review_lit-review.md).

## Observations

Reviewed facts (phase 1, reviewer's own fetch):
- **Claims: 43 of 43 checked claims verify** at the cited version and location. That is all 27 behind the top
  five plus a sample of 16 others. None is misquoted or missing. Four carry a nuance:
  - AlphaProof's T_steps is the longest branch, not total steps.
  - He et al.'s p₀ ≈ 1/N is an analytic argument.
  - The Firoiu citation is a fair paraphrase.
  - Kim et al.'s "2.6 σ" is ≈ 1.8 σ on a difference; the conclusion is unchanged.
- **Identifiers: 55 of 55** arXiv ids resolve to the stated paper, at the latest version. None of the 54
  `arxiv:` lines in `to_add_to_zotero.md` is already in `references.bib`. The *brief* was wrong to say that
  pass@k training, the entropy mechanism and Dr. GRPO are in `references.bib` (only DAPO's recipe page is).
- **Code mappings:** every one names a real file, function and line on the fork's `dan` (e.g. `grpo.py:117`
  advantage; `sample.generate_ids_fast` discards log-probs; `load_ckpt` is strict).
- **Novelty:** no shortlisted technique has already been *done*. Items #1 (search as the EI expert) and #5
  (steps-to-go value) are already *proposed*, in [proposal 14](../../docs/proposals/2026-09-28-after-state-env.md)
  item 1 and [proposal 13](../../docs/proposals/2026-09-28-state-conditioned-environment.md). `REVIEW.md` did not cite
  either (F1).
- Screened 58 papers; 23 were read in depth, each with a note. Prose is 1,498 words against a 1,500 limit.

The shortlist, in the executor's ranking. The reviewer finds it mostly follows from the stated reasons.

| # | technique | key verified source | reviewer |
|---|---|---|---|
| 1 | Search as the EI expert in the state env: best-first on length-normalised log-prob, training on minimal proofs, apprentice evaluated without search; truncate-and-resume as a cheap baseline | BFS-Prover 2502.03438v3 Eq. 1; HTPS 2205.11491v1 Table 4 (78.1 vs 40.6) | already proposal 14 item 1; this is a sharper design of it. The one length-specific source result (InternLM2.5-StepProver: critic-guided mean found length 4.44 vs best-first 1.66) favours a critic, so the "higher `L_true` at α = 1" expectation has no source support |
| 2 | Frontier target supply: extend or mutate solved proofs, take failed attempts' open subgoals, keep p̂ ∈ (0, 1/4] | STP 2502.00212v4 §3.2; Lee et al. 2502.01612v2 §7.1 | **would rank level with or above #1**: `long-pool` makes the lack of solvable targets past `L*` the best-evidenced bottleneck in the project's own data; ≈ $10–15 of small code |
| 3 | Support-expanding GRPO advantages (unlikeliness, pass@k, distinct-proof bonus) | He et al. 2506.02355v2 Table 2 | sensible; whole-proof `grpo.py` only, so it runs on C0-style models, not in the state env |
| 4 | Transductive EI on textbook theorems plus programmatic variants | HTPS §7.1.3; AlphaProof Methods (TTRL) | — |
| 5 | Steps-to-go value token | AlphaProof Methods; HTPS Table 5 | fold into #1 (proposal 14 item 1) |
| 6 | EI selection by least-likely correct proof | He et al. §3.4 (analytic) | — |

- **Do first (both agree):** §c1, the per-step base log-probability of EI-found proofs of base-unreached
  theorems. It uses files already in the bucket, needs no sampling, costs ≈ $0, and bears directly on the
  open question of new moves vs compounding per-step reliability. The reviewer adds that it should decide
  whether #1 or #2 leads.
- **Pre-registered expectations** (the reviewer confirms the scoring):
  - Held: search first; positional-encoding tricks low (rejected); pass@k objectives shortlisted below
    search; ≥ 90 % of claims verify.
  - Wrong: a learnability curriculum in the top 3. Ladder-A T2 / T4 already did that with no frontier gain,
    so *supply* replaced it.
  - Wrong: "≥ 1 wrong identifier in the brief". Every id was right. The brief's error was about what is in
    `references.bib`, and the reviewer calls this borderline.
- **Literature gap:** among the 58 screened papers, none compares a search-trained apprentice with a
  sampling-trained one at matched compute, evaluated without search. The closest is TS-LLM's one unmatched
  run, 47.9 vs 47.5.

Rewordings the review required (applied and marked in the nd-rl copy of `REVIEW.md`):
- **F1:** cite proposal 14 item 1 and proposal 13 for #1 and #5.
- **F2:** "no supply" becomes "no *model-conditioned* supply past `L*`". Ladder-A T5 injected a static
  cap-6 generator pool, with no gain.
- **F3:** drop "the pairing removes ≈ 99 % of the variance". That figure is run-to-run training variance, and
  each EI arm from a shared checkpoint is a further training run, so arm-level variance is unmeasured.
- **F5:** repair screened-table rows 17, 20, 32 and 37.
- **F6:** He et al. §3.4 is analytic.
- **F8:** AlphaProof's T_steps is the longest branch.

Not supported: the ≈ 99 % pairing claim (F3), and `log.md`'s "03:45" agent-launch time (F7). The agents were
launched ≈ 03:23 and the executor exited 03:36.

## Limitations and next step

- **This is a reading, not a result.** Rankings are expected value by argument; no technique has been
  tested here. The "nobody has run it" gap is scoped to 58 screened papers.
- **Model labels.** Literature numbers in the §a table mostly omit the paper's model (e.g. BFS-Prover's
  70.83 % is a 7B; Enigmata's numbers are Qwen2.5-7B-Instruct). The inherited project numbers are labelled:
  SN / S have 3,216,384 parameters, C0 has 3,214,336, and the ladder-A models are token-format
  `stage1_abs.pt` under `nd_verify`.
- **Seed plans:** the #1–#3 expectations are sign counts, "≥ 5 of 6 pairs". Under a sign test that is
  one-sided p ≈ 0.11, and 4 of 6 is neither expectation nor falsifier. The proposal-16 pre-registrations
  should state the MDD for the paired design and report per-seed values with IQM and a stratified-bootstrap
  CI.
- **Combining results:** #3 runs on whole-proof C0-style models, and the others run in the state env.
- **Next:**
  1. §c1 from bucket files, which decides whether #1 or #2 leads.
  2. Before any #1 or #2 run, one same-checkpoint EI re-run pair, to measure the arm-level floor that the
     paired designs assume.
  3. The orchestrator turns the shortlist into proposal 16, with #1 written as a refinement of proposal 14
     item 1.
  4. Dan adds `to_add_to_zotero.md` to Zotero if wanted.
