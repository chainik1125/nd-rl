---
written_on: 2026-10-19
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-19).** These are the deliverables of run `rl-compute-share` (proposal 30; fork
> `dan-pandori/nd-takehome`, branch `dan_rl-compute-share`, DONE
> [`c6c800db`](https://github.com/dan-pandori/nd-takehome/commit/c6c800db)), with the reviewer's corrections applied.
> Each change from the review is marked *[review]*. No hard-constraint violation, nothing quarantined. **The verdict,
> "supported", stands.** The rewordings are listed below.
>
> Review: [`review_rl-compute-share.md`](https://github.com/dan-pandori/nd-takehome/blob/bc8ee24e/review_rl-compute-share.md).
> Summary: [experiment-summaries/2026-10-19-rl-compute-share-…](../../../experiment-summaries/2026-10-19-rl-compute-share-yue-rl-is-1e-4-of-pretraining-no-computable-tested-run-above-2e-2-hypothesis-supported/README.md).
> The executor's files, byte for byte, are this directory at commit `692d428`. The commit after it applies the review,
> so `git diff 692d428 -- docs/literature/2026-10-19-rl-compute-share/` shows every change the librarian made.

# How much RL do create-vs-elicit papers run, relative to pretraining? (proposal 30)

Dan's question: Yue et al. [@yueDoesReinforcementLearning2025] conclude that RLVR elicits rather than creates, but their
RL looked tiny next to the base models' pretraining. Do such papers answer only "does a very small amount of RL create
new capabilities?", and so leave creation at frontier-lab RL scale open?

| file | what |
|---|---|
| [`REVIEW.md`](REVIEW.md) | the review (≤ 3,000 words before the marks): Q1 Yue in numbers, Q2 the wider literature, Q3 graded frontier reference points, Q4 scale dependence and the normaliser, Q5 our runs, the verdict, ranked implications, expectations against outcomes, the screened-source summary |
| [`compute_table.csv`](compute_table.csv) | 56 rows, one per model or run: base, N, D, pretraining FLOPs, RL configuration, RL FLOPs and ratio (low and high), GPU-hours, verdict and how it was measured, contamination, assumptions, ledger ids, a `verified` flag. **Unmarked; see the corrections below** |
| [`frontier_ranges.csv`](frontier_ranges.csv) | the graded per-model frontier ranges for 2024, H1 2025, H2 2025 and 2026, with their ledger ids |
| [`figures/ratio.png`](figures/ratio.png) (and `.pdf`) | RL / pretraining compute per paper on a log axis, with the graded frontier band and our runs. **Unchanged; see the corrections below** |
| [`ledger/`](ledger/README.md) | 1,206 source figures, each with a verbatim quote (≤ 300 characters) and its location: `A_yue`, `B_recipes`, `C_threads`, `D_frontier`, `Y_yue` |
| [`notes/`](notes/) | 13 in-depth notes in TEMPLATE format, 12 on single sources (Yue read in full) and one on 12 frontier posts |
| [`screened.md`](screened.md) | the 96 screened sources (52 arXiv, 44 web) |
| [`to_add_to_zotero.md`](to_add_to_zotero.md) | 90 identifiers for Dan: the screened sources not already in `references.bib`, which was not edited |

**Not copied; on the fork branch** at `c6c800db` (the review and `review_rcs/` are at `bc8ee24e`):
- `run_rl_compute_share.md`, `numbers.md` § rl-compute-share, `log.md`, `STATUS.md` and `preregistration/rl-compute-share.md`;
- `rl_compute_share/analysis_out.txt` (the medians REVIEW quotes), the two adversarial checks `_claimcheck.md` and
  `_claimcheck2.md`, and the extraction agents' `ledger/*_summary.md`;
- the code: `verify_ledger.py`, `check_quotes.py`, `table_inputs.py`, `build_table.py`, `analysis.py`, `make_figure.py`,
  and `our_runs.py` with its outputs `our_runs.csv` and `our_runs_table.md`;
- the reviewer's kit, `review_rcs/`: its own quote checker, ledgers, FLOP scripts and the two phase-2 audits.

The 232 fetched source texts (`~/rcs_sources/`, 15 MB on agent2) are copyrighted, so they are in neither git nor the
bucket. Every source can be fetched again from its `source_id`. The run made no checkpoints, artifacts or data, so
nothing went to the bucket.

## What the review changed

Items follow the review's §Verdict ("What must be reworded"); the §Compare rows are cited as C1–C9. In all,
REVIEW.md carries 15 *[review]* marks and one *[librarian]* note, the notes 8 marks and `screened.md` 1.

| item | the review's finding | where it is applied |
|---|---|---|
| 1 | The create-vs-elicit median gap, "11.8× … held", is not a finding. 14 of the 16 "elicit" runs are Yue's evaluated checkpoints, one paper's verdict counted 14 times. The gap was 8.9×, below the pre-registered 10×, until one reclassification. At the paper level the pre-registration names, it is 0.5–5× if third-party re-tests count for the testing paper and 8.8–26× if each paper's verdict covers only its own runs. Score: **inconclusive** | REVIEW Q2 (the "Create vs elicit" bullet) and expectation 2 |
| 2 | JustRL's row (9.8 × 10⁻³ – 1.2 × 10⁻²) used the paper's Table 4 "Token Budget (approx.)", which is steps × 256 × 8 × 16k, the maximum context, as tokens generated. With the stated mean length (≈ 8,000, falling "to 4,000-5,000 tokens by step 1,000") it is **≈ 3 × 10⁻³ – 5 × 10⁻³**, 2–4× lower. No conclusion changes, because JustRL has no test | REVIEW Q2 table and the figure note. Not in the CSV or the figure: see below |
| 3 | ScaleRL-8B is missing: 100k GB200-hours on an unnamed 8B base, so not computable, and no test. At MFU 3–30 %, against a 15T- or 36T-token 8B, it is ≈ 0.02–0.4 of pretraining (a 20× range), possibly the largest RL share among pretrained LLMs | REVIEW bottom line 2 |
| 4 | Expectation 2's "3.9 decades" is a narrow miss of "≥ 4 decades" | REVIEW expectations table, row 2 |
| 5 | The bottom-line gap uses the grade-D central range (0.3–2). In FLOPs, the B-grade anchor is R1-Zero, 3.7 × 10⁻³ – 2.9 × 10⁻². Against it, Yue's median is 1.5–2.4 orders below: the verdict holds at B grade for the paper Dan named. The largest tested run (≈ 2 × 10⁻², ProRL v2 as re-tested by Invisible Leash) is at about that floor (−0.8 to +0.1 orders), so for the strongest literature test the verdict rests on the C/D-graded late-2025+ frontier | REVIEW bottom lines 1 and 2, the verdict line, and §Verdict "Why" |
| 6 | Minor: the information-share factor is ≈ the episode length in tokens (10^2.5–10^4), not ≈ 10³; the Ring-Zero pass@1024 panel does not name its model; the GPU-time ratios mix GPU classes (A6000 ladders, including rl-continue's rounds 1–8, against A40 Stage-1s); `compute_table.csv` line 18 still says k 4,096–16,384 | REVIEW Q4 (two places) and Q5. Line 18: see below |
| C1 | "A 41-row spot check matched values to quotes" is not derivable as worded: in 436 of 998 numeric ledger rows some number is not in the quote (conversions, sums, axis readings). The review re-derived the consequential values itself | REVIEW, Verification |
| C7 | "The frontier has been ≥ 0.1 since H2 2025 (D)" stands, with its grade understated: DeepSeek-V3.2's > 10 % of cost (B) and Epoch's Grok 4 range of 0.1–1 (C) support it better | REVIEW §Verdict "Why" |
| C9 | Note-level defects, from a read-only sub-audit of the 11 notes neither executor check covered (220 claims OK, 9 not; all 118 quotations verbatim). ProRL note: "the largest open run in this set with a large-k test" is an overclaim (Ring-Zero is almost certainly larger, and ProRL v2 is larger than v1); "k ≥ 4096" holds for maths only; the boxnet "all values of k" sentence was applied to code; ProRL v2's 512 prompts per step is unclear (256 in JustRL's Table 4; "≤ ≈ 2 × 10⁻²" still holds). RL Squeezes: "fewer unique correct trajectories after RL from SFT checkpoints" is unclear (also in the ProRL note); "narrowing persists at the largest open RL scales tested" is an overclaim; App. C.5 is mis-cited. MiniMax: the Text-01 token total is unclear (D, and so the M1 ratio, could move by × 1.2) | `notes/liu2025-prorl-compute.md` (4 marks), `notes/matsutani2025-rl-squeezes.md` (3), `notes/minimax2025-m1.md` (1) |
| R6 | The only figure attributed to an OpenAI researcher, Dan Roberts's "giant reinforcement learning cherry", comes from a page whose stated method is "Claude 3.7 via custom Dust assistant": an AI-written podcast summary, grade D at best. *Librarian's check:* `screened.md` and the ledger (`D-oai-roberts-*`) grade it A, and it feeds no number | `screened.md` row 62 |

### Corrections to files that carry no marks

`compute_table.csv` and the figure are the executor's, unchanged, so that they still match the code that made them.
Read them with these corrections:
- **`compute_table.csv` line 32 (JustRL):** the ratio should be ≈ 3 × 10⁻³ – 5 × 10⁻³, not 9.8 × 10⁻³ – 1.2 × 10⁻²
  (item 2). The "no test" median of 4.2 × 10⁻³ in REVIEW and `analysis_out.txt` was computed with the old row; the
  review did not recompute it. Re-running the executor's median code with the corrected row gives ≈ 3.9 × 10⁻³
  (librarian, unreviewed; marked *[librarian]* in REVIEW's Q2 table).
- **ScaleRL-8B has no row** (item 3).
- **Line 18 (ProRL v1), `verdict_how`:** the Invisible Leash re-test is at k 1,024–16,384; k ≥ 4,096 holds for maths
  only.
- **Line 19 (ProRL v2):** the v2 phase's 512 prompts per step are unclear (JustRL's Table 4 gives 256). At 256, the
  high bound falls from 2.1 × 10⁻² to ≈ 1.6 × 10⁻².
- **The figure** plots JustRL at the old values and has no ScaleRL-8B row. `analysis_out.txt` (not copied) still lists
  JustRL among the "largest by mid".

### Not applied

These are minor or concern files that were not copied; they are left for the fork or for Dan.
- The fork's `numbers.md` gives "Create / elicit median ratio: 11.8." without the item 1 caveat.
- The pre-registration's header says "Written … ≈ 17:45 UTC", but the commit is at 17:30:12. `log.md` records the
  correction, and the header cannot be edited without breaking gate 0.
- The sub-audit's row 9: six note headers say "latest" or "only version" without a saved abstract page. The fetch URLs
  pin the version.
- The review grades the SemiAnalysis fleet shares C/D, against REVIEW's D.
- One weak Zotero entry: Ord's inference-scaling essay is listed under the index URL `tobyord.com/writing`.
