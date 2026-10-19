# Run brief: lit-review — techniques from the literature we have not tried yet

Run id `lit-review`. Repository `~/work/lit-review` (branch `dan_lit-review` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. **No pods**: this run reads and writes. Pod
budget $0.50 / 1 h is a guard, not an allowance. Claude time only. Authorised by Dan 2026-09-29.

## Why

The largest single improvement so far came from reading a paper: `state-env` (proposal 13) gave the policy
an AlphaProof-style proof state. That raised base reachability on the 29 hardest theorems ~5.8× and put
every state seed in the high depth-3 mode. Dan's hypothesis is that there is more low-hanging fruit of that
kind: techniques that published provers and RL-for-reasoning work rely on, and that this project has not
used. Find them, check them against the sources, and rank them by expected value for **this** setup.

## This setup, and what it already does (do not re-propose these; say so if a paper's version differs)

- **Models and data:** 3.2 M-parameter from-scratch GPTs (4 layers, d 256), trained on synthetic
  natural-deduction proofs rendered as Lean 4 text (`lean_seq`, 107 tokens). Supervised proofs are
  capped at 6 lines by default; caps up to 14 have been tried. A40 / RTX 3090 pods, $1–30 per experiment.
- **Judge:** Lean alone decides.
- **RL:** expert iteration with hindsight relabelling, and GRPO (both compared; see nd-rl
  `experiment-summaries`). Frozen controls are run at equal attempts.
- **Curriculum and data:** ladder curriculum (T1 rungs, `L*`); dataset-composition and generator-style
  studies; training-cap sweeps; length pools up to `L_true` 16.
- **Proof state:** the policy sees the Lean-like proof state at each step, with environment-assigned
  names (the S / SH / SN arms, `STATE_ENV.md`).
- **Checkpoints:** averaging tried.
- **Measurement:** support and elicitation curves (pass@k up to 400k attempts); noise floor with
  minimum detectable differences (`NOISE_FLOOR.md`).
- **Speed:** fast sampler and Lean pre-filter.
- **Headline state:**
  - T1 `L*` is 12 across caps 6–14 and in every state arm.
  - The state arms reach `L_true` ≥ 13 rarely: 5 theorems, 2.1 % pooled.
  - EI solves some theorems its base never reaches, which is support expansion. Whether that comes
    from new moves or from compounding per-step reliability is open.

Read first, with targeted reads, not whole files:
- nd-rl `docs/STATE.md`, proposals 13–15, and the latest `experiment-summaries/`.
- The fork's `STATE_ENV.md` and `NOISE_FLOOR.md`.
- nd-rl `docs/literature/` (README, `starting_recommendations.md`,
  `astra_suggestions/learning-beyond-training-data.md`, `opensource-rl-scripts.md`) and `references.bib`.
  Papers there are already known. Mine them for techniques we have not used, and don't re-summarise them.

## Where to look (a starting map; add, drop and follow citations as you judge)

Identifiers below are from memory: verify each one before citing it.

1. **Search and value at training and test time.** Expert iteration in its original sense uses search
   as the expert (Anthony et al. 2017), and our EI samples without search. Sources: HyperTree Proof
   Search (Lample et al. 2022); GPT-f and statement-curriculum EI (Polu & Sutskever 2020; Polu et al.
   2022); DeepSeek-Prover-V1.5 (RMaxTS, truncate-and-resume); BFS-Prover; InternLM2.5-StepProver
   (critic-guided search); AlphaProof's test-time RL on problem variants (already read for proposal 13:
   check whether we used it; we did not).
2. **Self-generated curricula at the frontier.** STP, the Self-play Theorem Prover (a conjecturer
   trained on barely-provable statements); Minimo (in `references.bib`: what beyond hindsight
   relabelling?); hindsight experience replay for theorem proving (Aygün et al. 2022); first-order
   provers from synthetic data (Firoiu et al. 2021); AlphaGeometry's synthetic data (forward deduction
   plus traceback, close to our generator); Goedel-Prover v1 / v2 (scaffolded data synthesis,
   verifier-guided self-correction, model averaging).
3. **RL objectives that expand support rather than sharpen.**
   - Pass@k-oriented rewards: "Rewarding the Unlikely", He et al. 2025, arXiv 2506.02355. This one is on
     theorem proving and is confirmed.
   - Pass@k training, the entropy mechanism of RL for reasoning, Dr. GRPO, DAPO's clip-higher
     (in `references.bib`).
   - Diversity and exploration bonuses.
4. **Length and compositional generalisation in small transformers.**
   - Self-improving transformers on easy-to-hard and length generalisation (Lee et al. 2025). They
     filter their own outputs, which is close to our EI.
   - Position coupling, Abacus embeddings, looped transformers, RASP-L.
   - "From f(x) and g(x) to f(g(x))" (RL composing skills).
   Our frontier is a length frontier (`L*`), so this is the most direct lever that nobody on the project
   has examined.
5. **Error feedback.** Repair and resume from the checker's error position (DeepSeek-Prover-V1.5, Kimina,
   Seed-Prover, Goedel-Prover-V2), adapted to a tiny model that cannot read error messages: truncate at
   the first failing step and resume.
6. **Measuring create vs elicit (secondary).** Controls or measurements we lack:
   - Sampling-only baselines that match RL, e.g. power or MCMC sampling from the base model ("Reasoning
     with Sampling", 2025).
   - "The Invisible Leash", "Echo Chamber" (RL amplifying pretraining behaviours, small-scale).
   - The pre-training / mid-training / RL interplay studies.
   Only those that would change an experiment we would run.

## What to do

1. **Screen** 25–40 papers: abstract, method section, main results table. **Read in depth** the 8–12
   whose technique could plausibly move `L*`, support, or seed variance here.
2. **Verify every claim you cite against the source text.** Fetch arXiv HTML or PDF into files and
   `grep` the section. Quote the numbers with their section, table or figure. Mark anything you could
   not check as unverified. Don't cite from memory.
3. **Map each candidate technique onto the fork's code:**
   - which file or function it changes (`sample.py`, `expert_iter.py`, `ladder_ei.py`, `state_env.py`,
     `train.py`, the generator, the judge)
   - what it needs that we lack (a value head, a search loop, a conjecturer)
   - rough cost in pod-hours, dollars and Claude time
4. **Rank** by expected value for this project's question and bottlenecks, with a stated reason.

## Deliverables

- `lit_review/REVIEW.md` (≤ 1,500 words plus tables):
  - (a) the ranked shortlist of 5–8 techniques. For each: the source, with its section and table; the
    mechanism; why it should help here, naming the bottleneck; where it plugs in; the smallest
    experiment that would test it, with a pre-registerable expectation, a falsifier, and the seed count
    from `NOISE_FLOOR.md`'s minimum detectable differences; and the cost.
  - (b) techniques considered and rejected, with the reason. "Needs a pretrained LLM" is a valid
    reason; say so.
  - (c) measurement ideas for create-vs-elicit.
  - (d) the full screened-paper table: title, year, arXiv or DOI, one-line technique, relevance 0–3,
    depth read (abstract / method / full), and whether each cited claim was verified.
- `lit_review/notes/*.md`: one note per in-depth paper, using nd-rl `docs/literature/TEMPLATE.md`.
  Use `papers:` citekeys only for papers already in `references.bib`; give the arXiv URL and version
  otherwise.
- `lit_review/to_add_to_zotero.md`: new papers, one identifier per line (e.g. `arxiv:2506.02355`), for
  Dan. **Do not edit `references.bib`**: it is generated from Dmitry's Zotero library. Do not run
  `code/tools/literature.py`.
- `run_lit_review.md` (≤ 250 words), `log.md`, `STATUS.md` line `LIT-REVIEW DONE <UTC>`,
  `touch ~/runs/lit-review/executor.done`.
- The librarian copies `lit_review/REVIEW.md` and `lit_review/notes/` into nd-rl
  `docs/literature/2026-09-29-lit-review/`, and links it from `docs/literature/README.md`'s reading
  guides. The orchestrator turns the shortlist into proposal 16. Nothing here launches an experiment.

## Review

The reviewer does not recount; there are no counts. It checks the following:
- **Claims:** every claim behind the top five, and a random sample of ≥ 10 other cited claims, against
  the sources. Report each as verified, misquoted or not found.
- **Novelty:** that no shortlisted technique is already done here (compare with the list above and
  with nd-rl `experiment-summaries/`).
- **Feasibility:** that each code mapping names real files and functions on the fork's `dan`.
- **Ranking:** whether it follows from the stated reasons. Say where you would rank differently.
