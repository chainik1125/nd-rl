---
written_on: 2026-10-19
written_by: agent:claude
papers:
  - yueDoesReinforcementLearning2025
  - liuProRLProlongedReinforcement2025
  - guoDeepSeekR1IncentivizesReasoning2025
---

# rl-compute-share: is the RL in create-vs-elicit papers a realistic share of pretraining compute?

Status: **completed and reviewed 2026-10-19. Dan's hypothesis is supported, and the review agrees.** The review asks
for five rewordings and a few minor ones. They are applied, marked *[review]*, in the nd-rl reading copy's REVIEW.md
and notes; the corrections to `compute_table.csv` and the figure are listed in the copy's README. The fork's
`numbers.md`, `analysis_out.txt` and CSV are uncorrected. No hard-constraint violation, nothing quarantined. **Not supported as scored:** "11.8× … held" for the create-vs-elicit
median gap, which the review scores *inconclusive*. **Wrong:** JustRL's row, which should be ≈ 3–5 × 10⁻³, not
≈ 10⁻². This was a literature and compute-accounting run: no pods, no model trained or sampled and no proofs counted,
so no Lean check applies. Where the reviewer's wording differs from the executor's, this summary uses the reviewer's.

Question (Dan, 2026-10-19; proposal 30): Yue et al. [@yueDoesReinforcementLearning2025] conclude that RLVR elicits
rather than creates. But their RL looked tiny next to the base models' pretraining. Do such papers answer only "does a
very small amount of RL create new capabilities?", and so leave creation at frontier-lab RL scale open? The brief has
five parts:
- Yue's RL in numbers;
- the same for the wider literature, with each paper's verdict;
- graded frontier reference points;
- whether verdicts change with RL scale, and which normaliser to use;
- where our own runs sit.

## Configuration

- **Brief and branch.** Brief: [`docs/proposals/literature/BRIEF_rl-compute-share.md`](../../docs/proposals/literature/BRIEF_rl-compute-share.md)
  (Dan, 2026-10-19; [proposal 30](../../docs/proposals/2026-10-19-rl-compute-share.md)). Fork `dan-pandori/nd-takehome`,
  branch `dan_rl-compute-share`, from `origin/dan_create-or-elicit` (`8e195390`); worktree `~/work/rl-compute-share` on
  agent2. It is not merged into the fork's `dan`. It adds documents and analysis scripts only, with no training or
  evaluation code (review, R1).
- **Pre-registration.** `preregistration/rl-compute-share.md`, commit `12753dbd` at 17:30:12 UTC. The first source was
  fetched at 17:31:22, so the expectations were written before any source was read (reviewer, R2). Before that the
  executor had read only the earlier reviews the brief names, which hold almost no compute figures. The header's
  "≈ 17:45 UTC" is later than the commit; `log.md` records the correction.
- **Executor**, 17:27–19:57 UTC, `--effort max`. Yue was read in full, four extraction agents built ledgers A–D, and
  two adversarial claim checks found 51 and then 13 problems. DONE
  [`c6c800db`](https://github.com/dan-pandori/nd-takehome/commit/c6c800db) at 19:56:44.
- **Reviewer**, 19:58–21:23. §Recount [`e012ca4e`](https://github.com/dan-pandori/nd-takehome/commit/e012ca4e) at
  20:36 came before reading the executor's write-ups; §Compare and §Verdict
  [`bc8ee24e`](https://github.com/dan-pandori/nd-takehome/commit/bc8ee24e) at 21:23.
- **Method.**
  - Pretraining = 6·N·D, with N the active parameters. RL = 2·N per generated token + 6·N per trained token + 2·N per
    extra forward pass, plus any critic. N cancels when RL runs on the pretrained model itself.
  - Unreported mean lengths are bounded by ¼–1 × the maximum or by a figure's axis. Attention FLOPs are ignored.
  - Every figure has a verbatim quote (≤ 300 characters) and its location in one of five ledgers (1,206 rows).
  - The source texts were fetched into `~/rcs_sources/` (232 files, 15 MB). They are copyrighted, so they are in
    neither git nor the bucket.
  - Grades: A, a primary lab statement; B, a paper or report; C, an independent estimate with a method; D, press,
    rumour or no method.
- **The reviewer's kit (`review_rcs/`).**
  - its own quote normaliser, with negative controls;
  - its own ledgers Y (63 rows) and F (22);
  - three read-only extraction agents (362, 424 and 334 rows), barred from every executor file;
  - FLOP scripts for Yue's checkpoints, the literature and our runs;
  - a read-only sub-audit of 11 notes, and an adversarial check of its own review.
- **Pods: none.** A $0.50 / 1 h budget was registered only as a guard (`podbudget rl-compute-share`: 0.00 h, **$0**).
  There are no checkpoints, artifacts or data, so nothing went to the bucket; the deliverables are in git.
- **Outputs in nd-rl:** [`docs/literature/2026-10-19-rl-compute-share/`](../../docs/literature/2026-10-19-rl-compute-share/README.md).
  - REVIEW.md, with the review's corrections marked;
  - `compute_table.csv` (56 rows) and `frontier_ranges.csv`;
  - the five ledgers, 13 notes and `screened.md` (96 sources);
  - `to_add_to_zotero.md`: 90 identifiers for Dan. `references.bib` is untouched;
  - the figure.

  The CSV and the figure carry no marks; the copy's README lists their corrections.
- Review: [`review_rl-compute-share.md`](https://github.com/dan-pandori/nd-takehome/blob/bc8ee24e/review_rl-compute-share.md).
  Executor summary: [`run_rl_compute_share.md`](https://github.com/dan-pandori/nd-takehome/blob/c6c800db/run_rl_compute_share.md).

## Observations

**Verification.**
- All 1,206 ledger quotes are in their named source file under the reviewer's own normaliser. Its negative controls
  failed as they should: with one digit changed, 0 / 1,007 quotes are still found; with two words swapped, 0 / 1,060.
- The reviewer re-fetched five decisive sources: xAI's Grok 4 page, DeepSeek-V3.2 and its abstract page, and Yue v5
  with its arXiv history.
- All 14 REVIEW quotations that carry a claim id occur in that id's source.
- A sub-audit of the 11 notes that neither executor check covered found 220 claims OK and 9 not, with 118 / 118
  quotations verbatim.
- Verbatim is not the same as support. In 436 of the 998 numeric ledger rows, some number in the value is not in the
  quote (conversions, sums, axis readings). The reviewer re-derived the consequential values itself.

**Q1, Yue et al. in numbers** (re-derived independently by the reviewer).
- Yue evaluates ten outside checkpoints, nine of them computable (Magistral-Medium's base size and tokens are
  undisclosed), and nine runs of its own.
- **Their RL is ≈ 10⁻⁵ – 3 × 10⁻³ of pretraining FLOPs, with a median of ≈ 1.2 × 10⁻⁴.** The executor's range is
  7 × 10⁻⁶ – 3 × 10⁻³; like for like, the two medians agree within ≈ 2×. DAPO-32B reaches 3 × 10⁻² only under a step
  reading that its own caption contradicts.
- Of the bases, only Llama-3.1-8B reports pretraining GPU-hours (1.46M).
- Yue et al. concede the point: "Whether this trend persists as more compute, such as pre-training scale budgets, is
  dedicated to RL training remains a critical question". Their largest-RL checkpoint, Magistral-Medium, shows "little
  or no improvement at higher k", at unknown compute.
- The pre-registered ¼–1 × max length rule is not a bound in either direction. SimpleRL-Zoo's own figures put the mean
  length at 0.5–1.2K tokens, below ¼ × 8,192. The executor used figure axes where they exist, so its rows are not
  biased by the rule.

**Q2, the wider literature.**
- **No create-vs-elicit test on a pretrained LLM with a computable ratio exceeds ≈ 2 × 10⁻²** (ProRL v2 as re-tested
  by Invisible Leash; AceReason-7B). Magistral, Ring-Zero and Qwen3-8B were tested, but their ratios cannot be computed.
- **The largest open RL budgets test nothing.** They are MiniMax-M1 (≤ 0.18–0.21 at 100 % MFU), ProRL v2 + BroRL,
  R1-Zero, ORZ and Olmo 3.1. ProRL v2 + BroRL is 1–4.5 × 10⁻² by the executor's count and 1.1–7.9 × 10⁻² by the
  reviewer's.
- **ScaleRL-8B is missing from the write-up.** It used 100k GB200-hours on an unnamed 8B base. At MFU 3–30 % that is
  ≈ 0.02–0.4 of pretraining, possibly the largest RL share among pretrained LLMs, though the range spans 20×. It has
  no test.
- **Lab disclosures:**

  | lab | basis | RL / pretraining |
  |---|---|---|
  | DeepSeek-R1 | GPU-hours | 1.5–5.3 % (R1-Zero alone: 101K / 2,664K H800-hours = 3.8 %) |
  | DeepSeek-V3.2 | cost | > 10 % |
  | Llama-Nemotron-Ultra | GPU-hours | 0.45 % |
  | Olmo 3.1 Think 32B | GPU-hours | 14 % |
  | Reflection Beam | GPU-time | ≥ 1.7 |
- **Create vs elicit, by ratio: inconclusive (review), not "11.8× held".**
  - Per model, the "create" median is 8.9–11.8× the "elicit" one. But 14 of the 16 "elicit" runs are Yue's own
    evaluated checkpoints, and 11.8× came from one reclassification of an 8.9×, which was below the pre-registered 10×.
  - Per paper, as pre-registered, the gap is 0.5–5× if third-party re-tests count for the testing paper, and 8.8–26×
    if each paper's verdict covers only its own runs.
  - The "elicit" papers include the analyses of the largest open runs (ProRL, in Invisible Leash and RL Squeezes). The
    "create" papers include tiny runs.
- **JustRL is ≈ 3–5 × 10⁻³, not ≈ 10⁻²** (review). The row read the paper's maximum-context "Token Budget" as tokens
  generated. JustRL has no test, so no conclusion changes.
- **Interplay**, a 100M model trained from scratch, is the one paper with a computable ratio in the frontier band: 0.1–2× its
  10B pretraining tokens, by its own FLOP-equivalents. Its in-distribution pass@128 never improves; its gains at the
  edge of competence reach +42 %.

**Q3, frontier reference points.**
- No lab gives an absolute RL FLOP figure for a frontier model. OpenAI's and xAI's statements are relative: o3 used "an
  additional order of magnitude" of RL compute over o1, and Grok 4's RL ran "at pretraining scale".
- **The graded per-model range (review's wording).** About 0.01–0.1 in 2024, possibly up to 0.2 (C/D). By late 2025
  it is ≥ 0.1 both in cost and GPU-time units (B: DeepSeek-V3.2, "exceeding 10% of the pre-training cost") and in
  Epoch's estimates (C).
- **The executor's central picks are correctly graded D:** 0.3–1 for H2 2025 and 1–2 for 2026. They rest on Epoch's
  Grok 4 range, xAI's slide as relayed by Ord, the non-frontier Cursor and Beam, and fleet shares. The fleet shares
  are SemiAnalysis's, relayed by the press: post-training 42 % and pretraining 23 % of fleet compute in 1Q26.
- **Units matter.** GPU-time overstates RL's FLOP share several-fold: R1-Zero is 3.8 % by GPU-hours but 0.4–1.1 % by
  FLOPs, and Olmo 3.1 Think 14 % against ≤ 2.4 %.
- **Superseded estimate.** Epoch's pre-disclosure R1-Zero estimate (6.1 × 10²³ FLOP, ≈ 20 % of V3) needs ≈ 490K
  H800-hours, against the 101K disclosed.
- **Low-grade attribution.** Dan Roberts's "giant reinforcement learning cherry" comes from an AI-written podcast
  summary, so it is grade D at best.

**Q4, does the verdict depend on RL scale?** The evidence is mixed.
- **Re-tests narrow or plateau, the largest-k ones included:**
  - Invisible Leash re-tests ProRL v1 / v2 at k 1,024–16,384: 36 / 48 problems gained, 163 / 175 lost;
  - Ring-2.5-Zero's pass@1024 rises, then plateaus within ≤ 2,000 steps;
  - Yue's runs from 150 to 450 steps take pass@256 from 66.3 to 64.3;
  - Qwen3-8B's RL leaves pass@64 flat.
- **Some reach grows:**
  - ProRL on code and boxnet; on boxnet it reaches 0.492 at k 16,384, against 0.345 for the base;
  - Interplay on out-of-distribution hard problems;
  - Abdulsalam et al.'s gains, which "emerge late";
  - our rl-continue: from r8 to r16, group-C pass@256 rises from 0.049 to 0.265 (5 → 27 of 99 theorems), with 16 of
    the 22 new solves on one seed.
- **Both controlled tests at frontier-share RL that remove a skill from pretraining find "needs exposure."**
  Interplay runs at ≈ 0.42× its pretraining, our create-or-elicit knockout at 3–10×. In the knockout, 0 / 6 seeds
  acquire a classical family in 16 rounds, against 6 / 6 full-exposure seeds, and a 1 % dose restores acquisition.
- **Normaliser.** Use the FLOP (or cost) ratio to compare with lab statements, which are in compute and cost.
  - Trained tokens are within 1.3–3.3× of it.
  - Gradient steps are not comparable: samples per step span 256–65,536.
  - An information normaliser widens the gap by ≈ the episode length in tokens, 10^2.5–10^4.
  - Pair the ratio with samples per problem against the evaluation k, and with a compute-matched base read. In
    `capability-defs`, that read cut RL's 51–60 equal-k "creations" per seed to 14–22.

**Q5, our runs.** The numbers are the reviewer's, after its erratum E-1. The executor's upper ends are ≤ 5 % higher in
FLOPs and ≤ 8 % higher in GPU-seconds, a definitional difference. Every row is the same model:
- `best_model.ALiBiGPT`, 6 layers × 384, 9,560,832 params, `lean_staten`, trained from scratch;
- Stage-1: 1,200 s on an A40, 4.40–4.71 × 10⁸ tokens, on K12, the cap-6 set, K12⁻ or the 1 % dose set;
- then T1 expert-iteration ladders.

N cancels.

| run (ladders) | rounds | RL / pretraining FLOPs: decoded tokens and RL records only – as run, without evals | GPU-time: sample + fine-tune / Stage-1 |
|---|---|---|---|
| trajectory (3) / trajectory-cap6 (3) | 8 | 1.5–2.9 / 0.88–2.0 | 11.9–13.0 / 8.9–9.4 |
| rl-continue (3) / rl-continue-cap6 (3) | 16 | 3.2–6.3 / 2.2–4.7 | 28.9–29.5 / 22.2–26.4 |
| rl-from-ckpt, starts p16000 → p1600 (3 each) | 8 | 2.2–4.3 → 17.5–36.9 | 19.7–21.0 → 151.5–169 |
| create-or-elicit (11 ladders in the reviewer's count; knockout s1 to r24) | 16 (24) | 3.1–9.8 | 32.5–53.9 |

- **Every ladder sits at or above the frontier band:** ≥ 0.88× pretraining FLOPs on the narrowest definition, about
  4–5 orders above Yue's median.
- **Two costs inflate the as-run column.** Each action re-reads the proof state (prefill is 4.2–5.1× the decoded
  tokens), and each round replays 20,000 records of the arm's pretraining set.
- **Caveats.**
  - The GPU-time ratios mix GPU classes. trajectory's ladders (and so rl-continue's rounds 1–8) and most of
    rl-from-ckpt's ran on RTX A6000s, against A40 Stage-1s.
  - Attention FLOPs are ignored on both sides, though state prompts reach 1,428 tokens.

**The verdict: "supported".** The review says it stands, and it matches the reviewer's own reading, written before it
read the executor's.
- Yue's median sits 3.4–4.2 orders below the grade-D central range for H2 2025–26. In one unit at B grade (FLOPs;
  R1-Zero, 3.7 × 10⁻³ – 2.9 × 10⁻²), it sits 1.5–2.4 orders below. Either way, Yue et al.'s result cannot rule out
  creation at frontier scale, and they call frontier scale open themselves.
- For the strongest literature test (≈ 2 × 10⁻², at about the B-grade floor), the verdict rests on the C/D-graded
  late-2025+ frontier (review).
- **What it does not imply.** The gap is a missing measurement, not evidence of creation at scale. Within studies, the
  scale evidence is mixed, and both controlled frontier-share tests still need pretraining exposure for an absent skill.

**Ranked implications.** These are the executor's; the review finds them consistent with its own reading, and
implication 1's numbers reproduce.
1. **Frame our results as frontier-share results.** Ladders started from full pretraining spend 0.9–5.1× pretraining
   FLOPs on RL counting RL only, or 1.9–9.8× as run; early starts reach 18×. Our knockout is one of two controlled
   create-vs-elicit tests at frontier-share RL, and the only one above 1×.
2. **Plot the scale curve from existing reads:** reach (group C, knockout members) against cumulative RL FLOPs, with
   rounds 1–24 spanning ≈ 0.2–10, plus an arm at ≈ 10⁻³.
3. **Report the RL-only ratio, and consider a no-replay arm.** Replay is continued pretraining.
4. **Pair every "RL created X" claim** with samples per target against the evaluation k, and with a compute-matched
   base read.
5. **Quote the frontier as a graded range with units**, never as one number.

**Pre-registered expectations.** The executor's tally is 0 of 7 held outright and 7 held in part. The scores below
are the reviewer's.
- **E1.**
  - The median over Yue's checkpoints is < 10⁻³: **hit** (4.8 × 10⁻⁴ under the pre-registered length rule,
    1.3 × 10⁻⁴ with figure-read lengths).
  - Every SimpleRL-Zoo checkpoint is < 10⁻⁴: **missed**. The high bounds are 1.4–2.5 × 10⁻⁴ with figure lengths and
    7.7 × 10⁻⁴ – 1.2 × 10⁻³ under the rule.
  - The largest run falls in 10⁻³ – 5 × 10⁻²: **hit** (DAPO-32B).
- **E2.**
  - A span of ≥ 4 orders: **a narrow miss.** It is 3.9 on LLM rows with a named base; 4.6 needs ScaleRL-8B under one
    batch reading, and 5.0 needs Interplay's toy model.
  - No low bound ≥ 0.1: **hit** for LLM papers, **missed** if Interplay counts (low bound 0.105).
  - The largest open runs fall in 10⁻³–10⁻¹: **mostly hit**; ScaleRL straddles the edge.
  - The "elicit" median is ≥ 10× below the "create" median: **inconclusive** (0.5–26×).
  - DeepSeek-R1's RL is ≈ 5 % of V3's GPU-hours, and lower by FLOPs: **hit**.
- **E3.**
  - No absolute OpenAI or xAI figure; o3 ≈ 10× o1 and Grok 4 ≈ 10× Grok 3 in RL: **hit**.
  - 2025–26 from ≥ 10⁻² to ≈ 1: **hit** on the reviewer's reading (B-grade floor 0.04–0.1 in GPU-hour and cost units, ceiling 1–2). The
    executor's range reaches 2–5 (grade D).
  - 2024 ≤ 0.1: **borderline**. The reviewer's range reaches 0.2 (C/D only), the executor's 0.3.
- **E4.** **Hit** on three of four clauses. The clause on ProRL / BroRL expansion growing with steps at k ≤ 256 holds only in part,
  because BroRL has no large-k test.
- **E5.** **Hit**, except that GPU-time ≥ 10× fails for trajectory-cap6 (8.9–9.4×).
- **E6.** The predicted label was "partly supported" (60 %); the outcome is "supported".
- **E7** (the executor's score, which the review accepts). **In part:** the FLOP ratio is the headline, but the
  secondary normalisers are samples per problem and compute-matched reads, not trained tokens.

**Model labels.** No model was trained or sampled. Paper rows name their models in the CSV, and our-run numbers carry
the 9.56M `lean_staten` label. The reviewer found no unlabelled number. Checker labels do not apply, as no proofs
were counted.

## Limitations and next step

- **The literature figures are only as good as their sources.** Several rows rest on assumed steps or lengths:
  Oat-Zero's steps, EasyR1's configuration, Qwen2.5-Math-7B-Instruct's steps, DAPO's step unit. 436 numeric
  values involve conversions or axis readings that are not in their quotes.
- **The FLOP formula ignores attention.** That costs ≈ 1.25× at 4K context; passing 2× needs mean contexts of
  ≳ 16–20K tokens.
- **After mid-2025 the evidence is weak.** Frontier statements are relative. The ≥ 0.1 floor has B-grade support only
  in cost and GPU-time units (DeepSeek-V3.2), and the central picks rest on grade D evidence in mixed units.
- **The by-ratio create-vs-elicit comparison is confounded.** The largest RL runs are tested, if at all, by third
  parties, and the "create" papers include tiny runs.
- **The fork's files are uncorrected.** That covers:
  - `numbers.md`'s "11.8", which lacks the caveat;
  - JustRL in `analysis_out.txt`'s "largest" list and in the figure;
  - `compute_table.csv`: JustRL's row, line 18's k range, and the missing ScaleRL-8B row;
  - the ProRL, RL Squeezes and MiniMax notes.

  The nd-rl copy carries the corrections. The note-level defects are minor; none reaches a REVIEW headline.
- **Orchestration.** As in `lit-review-2`, the reviewer's phase-1 copy still held `REVIEW.md`, the claim checks and
  `log.md`, because the removal patterns do not match a literature run's deliverable names. The reviewer did not
  open them in phase 1.

**Next measurements (reviewer).**
1. **In-house: the scale curve (implication 2).** Use the existing create-or-elicit knockout reads plus one new arm
   whose total RL is ≈ 10⁻³ of Stage-1 (Yue's regime). Score family-level and member-level acquisition against
   cumulative RL / pretraining FLOPs, at the same k and the same seeds.
   - If acquisition is absent at 10⁻³ and still absent at 5–10×, the hypothesis's scope limit does not matter for
     this organism.
   - If member-level reach appears only above ≈ 1×, as knockout s1 hints, it does matter.
2. **Outside: a large-k (≥ 1,024) base-vs-RL re-test** of an open run whose computable ratio is clearly above the
   largest tested one, ≈ 2 × 10⁻² (about R1-Zero's FLOP share). Candidates are ProRL v2 + BroRL (1–4.5 × 10⁻²) and
   Olmo 3.1 Think 32B (14 % by GPU-hours). Nothing above ≈ 2 × 10⁻² has a boundary test.
