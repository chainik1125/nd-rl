# Run brief: olmo-novelty — how novel are our Olmo create-vs-elicit results, and what in the literature already shows them?

Run id `olmo-novelty`. Repository `~/work/olmo-novelty` (branch `dan_olmo-novelty` from the fork's `origin/dan_olmo-iffu`,
made by `newrun`, so every Olmo run's stored results are here). Role: executor. `AGENT_POLICY.md` governs. Authorised by
Dan 2026-11-06 (proposal 38).

- **Budget:** **no pods.** A $0.50 / 1 pod-hour budget is registered as a guard only. Claude time only. The one
  computation in Q2 uses stored results on the VPS.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.
- **Start:** queued on agent2 to begin after `olmo-iffu` is recorded, so its results can be positioned too.

## The question (Dan, 2026-11-06)

For each finding of our Olmo runs, find:
- the closest prior result;
- whether it **replicates**, **extends** or **contradicts** ours;
- what remains genuinely new.

Dan asked in particular for a comparison with ProRL [@liuProRLProlongedReinforcement2025] (arXiv 2505.24864).

## Our claims to position (from the reviewed nd-rl summaries; quote their wording)

- **C1, 7B RL-Zero-Math (RL from the pretrained base, ≈ 1.5–3 × 10⁻³ of pretraining compute):** under RL's verifier,
  the base is credited on all 17 RL-only AIME solves within 383 samples. Nothing is lost (reverse 0 at k ≤ 512), which
  contrasts with Yue et al.'s narrowing [@yueDoesReinforcementLearning2025]. The verdict was inconclusive, and 10 of the
  17 base resolutions are sound by hand. Source: `experiment-summaries/2026-11-03-olmo-rlzero-pilot-…`.
- **C2, 32B Think, AIME (DPO start against the RL run, ≈ 6 × 10⁻³ – 2 × 10⁻² FLOPs, 14 % of GPU-hours):** pass@1
  0.716 → 0.746, one inconclusive reach candidate. Source: `…olmo-think32b-…`.
- **C3, the same RL run, IFBench:**
  - 49 prompts strict and 43 by hand that the DPO start never solves in 256 samples;
  - an RL-only share of ≈ 20 %, against Yue's ≤ 1 %;
  - constraint types absent from RL's IF data;
  - almost no narrowing.

  Source: `…olmo-ifomega-…`.
- **C4, gaming:** about a third of RL's first-read checker successes on IFBench are gamed, so checker-only pass@k
  overstates reach.
- **C5:** IFEval reach (6 prompts) may be brevity, and OMEGA reach appears only at RL's 32,768-token cap.
- **C6, failure modes:** the DPO model's failures on the 49 prompts are "substantive" in 93 % of a blind reading (70 %
  attempted but far off). This is the executor's number; the review did not re-derive it.
- **C7, methods:** symmetric confirmation, a hand gate, a final-answer-only grade, guessing at large k on integer
  answers, a compute axis along a public checkpoint trajectory, and the project's capability standard (nd-rl
  `docs/project_strategy/capabilities/REPORT.md`: "not reached at K", not "created").
- **C8, `olmo-iffu` (if recorded):** what the token-budget, brevity and 1,024-sample tests showed.

## Questions

1. **The novelty matrix.** For each claim, give the closest 2–4 papers and compare them on:
   - model size and starting point (pretrained base, distilled, SFT + DPO);
   - RL share of pretraining (use rl-compute-share's numbers where they exist);
   - RL steps, domain, the largest k, and the grading (answer match, checker, hand);
   - their controls: symmetric budgets, format tolerance, guessing, contamination.

   Then classify each relation as replicates, extends, contradicts or unrelated, with the reason.
2. **ProRL, side by side.** Read ProRL v1 in full, plus the v2 blog and BroRL (arXiv 2510.01180).
   - Compare its setup with ours: a distilled start, more than 2k steps, a task mix including IF, KL with reference
     resets, and pass@128.
   - Compare its claims: expansion "where base models fail entirely", largest where the base is weakest.
   - Cover its critics: Invisible Leash (2507.14843), which re-tested it at k up to 16,384 with format-tolerant
     extraction (boxnet's base reaches 0.345); and RL Squeezes (2509.21128).
   - **Compute ProRL's correlation on our data, from stored results only:** across IFBench prompts (and IFEval), the
     DPO model's pass@k against RL's gain. Does our reach show ProRL's "largest where the start is weakest" pattern?
   - **Answer directly:** does our IFBench reach survive an Invisible-Leash-style objection (format tolerance, much
     larger k)? Say what `olmo-iffu` settled and what it did not.
3. **Instruction following.** Has anyone measured a large-k (pass@k or coverage) boundary for instruction following or
   other non-maths, non-code tasks? A preliminary search (2026-11-05) found none, and the curated awesome-RLVR-boundary
   list has none. Confirm or refute this with a thorough search. Include IFBench's own paper ("Generalizing Verifiable
   Instruction Following") and any follow-ups.
4. **Domain dependence within one RL run.** Find papers showing different create-vs-elicit behaviour across domains
   for the same model: agents (2604.14877), the cross-domain study (2506.14965), and others.
5. **Gaming and grading.** Has anyone quantified checker gaming or reward hacking when measuring reach? How do Cover@τ,
   breadth-depth metrics (2510.08325) and CoT-pass@k (2506.14245) treat guessing?
6. **Trajectory studies.** Find work that uses public intermediate RL checkpoints, or step-resolved large-k curves, to
   track boundary dynamics: diversity collapse (2606.15455), pass@k inversion (2607.20543), Ring-Zero (2607.12395),
   ProRL's intermediate checkpoint. Is analysing Olmo 3.1's 46 public checkpoints new?

## Rules

- Read first:
  - nd-rl `docs/literature/2026-09-29-lit-review/REVIEW.md`, `docs/literature/2026-10-02-lit-review-2/REVIEW.md`
    (with its ProRL, BroRL, Invisible Leash and RL Squeezes notes), `docs/literature/2026-10-04-capabilities/REVIEW.md`
    and `docs/literature/2026-10-19-rl-compute-share/REVIEW.md`. **Extend these reviews; don't redo them.**
  - the four Olmo summaries (and `olmo-iffu`'s);
  - `docs/literature/README.md`.
- Seed the search:
  - forward citations of Yue et al. and ProRL, through November 2026 (Semantic Scholar or arXiv listings);
  - the awesome-RLVR-boundary list (github.com/rdi-berkeley/awesome-RLVR-boundary);
  - the 2026 papers named above.
- **Verify every cited claim against the source text.** Fetch each source into a file and grep it, recording the
  location. Mark anything unverified. Quotes ≤ 25 words.
- **Screen 40–60 sources and read 12–18 in depth.** Read ProRL, Invisible Leash, What Outcome RL Buys (2610.34894) and
  the IFBench paper in full.
- Keep observed results, our interpretations and the papers' interpretations apart.

## Deliverables

- `olmo_novelty/REVIEW.md` (≤ 3,000 words), containing:
  - the novelty matrix;
  - the ProRL side-by-side table and the correlation result;
  - a ranked list of claims (novel / extends / replicates / contradicted), each with its evidence;
  - the five strongest objections a reviewer would raise, with what answers each;
  - a draft related-work section of ≈ 600 words for the paper.
- `olmo_novelty/notes/*.md` (TEMPLATE format); `olmo_novelty/screened.md`; `olmo_novelty/to_add_to_zotero.md`;
  `olmo_novelty/correlation.csv` and the script that made it.
- The librarian copies them to nd-rl `docs/literature/2026-11-06-olmo-novelty/` and links them from
  `docs/literature/README.md`.
- `run_olmo_novelty.md` (≤ 200 words), `log.md`, a `STATUS.md` line `OLMO-NOVELTY DONE <UTC>`, and
  `touch ~/runs/olmo-novelty/executor.done`.

**For the reviewer:**
- Re-verify a sample of the matrix's claims against the sources.
- Recompute the correlation from the stored results.
- Check every "novel" label against a fresh search.

**Rules:**
- Do not contact anyone.
- Do not edit `references.bib` or run `code/tools/literature.py`.
- Wait for anything long with one poll rather than many short check-ins.
- At most 4 concurrent processes on the VPS.
