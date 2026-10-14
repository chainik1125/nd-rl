---
written_on: 2026-10-14
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-14).** These are the paper draft, blog post and figures of run
> `create-or-elicit-paper` (fork `dan-pandori/nd-takehome`, branch `dan_create-or-elicit-paper`, DONE
> [`ed8cf1ea`](https://github.com/dan-pandori/nd-takehome/commit/ed8cf1ea)), with the reviewer's corrections applied.
> Each change is marked *[review]*: an italic red "[review]" in the PDF, `\rv{}` in the LaTeX source, and *[review]* in
> the blog. No headline number changed. No hard-constraint violation, nothing quarantined.
> **Not supported as written (now reworded):**
> - "every rejection was the pre-filter's" (P9);
> - GRPO "by sampling alone" first succeeding on seeds 1, 2 and 5 (P11);
> - the PUCT half of "neither method proves most of the compute-matched creations" (P16);
> - `2256` in every knockout-s5 r16 draw (P6).
>
> Review: [`review_create-or-elicit-paper.md`](https://github.com/dan-pandori/nd-takehome/blob/1ad2a691/review_create-or-elicit-paper.md).
> Summary: [experiment-summaries/2026-10-13-create-or-elicit-paper-…](../../../experiment-summaries/2026-10-13-create-or-elicit-paper-draft-and-blog-no-headline-number-wrong-c4-drops-knockout-s1-elicitation-reading/README.md).
> The executor's draft, byte for byte, is this directory at commit `d841408`. The commit after it applies the review,
> so `git diff d841408 -- docs/papers/create-or-elicit/` shows every change the librarian made.

# *Did Expert Iteration Create It?* — paper and blog drafts (proposal 27)

Drafts for Dan. Nothing here has been posted, submitted or shared. The author line is the brief's placeholder ("Dan
Pandori et al. (author list for Dan to decide)"), and the acknowledgements are left for Dan.

| file | what |
|---|---|
| [`main.pdf`](main.pdf) | the paper, rebuilt from these sources with the corrections (35 pages) |
| `main.tex`, `sections/*.tex`, `refs.bib` | LaTeX sources; `build.sh` compiles `main.pdf` with tectonic (`~/tools/tectonic/tectonic` on the VPS, or `tectonic` on `PATH`) |
| [`blog.md`](blog.md) | the blog post, 4 figures |
| `figures/` | the paper's figures F0–F7 (`.pdf` for LaTeX, `.png` to view) and the four blog figures |
| `lean/` | the blog's simplified Lean σ proof and its goal-negated control (Lean 4.35.0 accepts the first and rejects the second) |

**Not copied; on the fork branch at `ed8cf1ea`:**
- `paper/OUTLINE.md`, `LEDGER.md` (every number → its reviewed source), `REVISIONS.md` (two critique rounds, the
  claim audit, the re-audit) and `next_steps.md` (costed and ranked);
- `critiques/` and `notes/related_work_digest.md`;
- `scripts/` and `data/`. The figures are regenerated from data with `python3 paper/scripts/figures.py`; the scripts
  read the fork's artefacts and do not run from this directory.

Two of these carry known defects (review, § Records):
- `LEDGER.md` P8 still lists the dropped knockout-s1 reading as reviewed;
- `next_steps.md` item 1b assumes 6–8 ms per attempt. C4 measured 12.35 ms, which puts 2.4 × 10⁷ attempts at
  ≈ 82 A40-hours (≈ $48), not 40–55 h.

## What the review changed

Every item below is from the review's §Comparison and §Verdict. The review cites lines in the `sections/*.tex` files.
In the PDF, the files are sections 1 (`01_intro`), 2 (`03_setup`), 3 (`05_results`), 4 (`04_measurement`), 5
(`02_related`), 6 (`07_discussion`), 7 (`08_limitations`) and 8 (`09_repro`), followed by Appendices A–J
(`10_appendix`); Appendix I is `11_checks`.

**Paper** (60 *[review]* marks):

| item | the review's finding | where it is applied |
|---|---|---|
| P1 | C1 is not a paired design: the standard- and doubled-cap reads are independent draws | §3.8 Robustness and App. I now say "same-seed re-evaluation(s)", and that ±1 differences are sampling, not cap effects; §7 repeats it |
| P2 | knockout s0 is the only certified base only at T 0.8 on the pre-registered 106; plain T 1.0 sampling proved `la_transfer_916` on bases s0 and s1; the pre-registered expectation (≥ 5 / 6 certified) missed | the intro (results list and pre-registration paragraph), §3.3 (certificates), §6 and §7 |
| P3 | "14–21 remain" holds only for the set that survives all five allocations; 14–34 depending on the count; the models are r8 of the earlier full seeds s0–s2 on the 322 | abstract and the intro's Measurement contribution, with the models labelled |
| P4 | one of 12 plain k-256 base reads succeeded (knockout s1's second draw, `la_transfer_916`) | intro and §6 Evaluation |
| P5 | the family bar is half of its **strict** held-out members | abstract |
| P6 | four plain r16 draws plus one guided: `1019` in 2 of 4 plain (and twice guided); `2256` in 3 of 4 on knockout s5 | §3.4 and §6 |
| P7 | the scored proofs are the earlier full ladders' (s0–s2), not each scored ladder's own | §3.3 (likelihoods) and §7 |
| P8 | the "6 and 17 targets" seeding contrast is the 3.2 M whole-proof Run-3 organism's, under the older checker | §3 scope sentence and §3.7 |
| P9 | only the Lean gate's rejections were the pre-filter's; 72–88 % of knockout r16 attempts end earlier, in grammar and action checks | §3.8 and §7 |
| P10 | the per-family verdicts rest on the pre-registered Holm correction; uncorrected, three more families would meet "needs exposure" | §3.1 and §6 |
| P11 | by the A5 base rates GRPO would first succeed on s1 (P 0.29 by r16), then s4 (0.13), not "seeds 1, 2 and 5" | §6 |
| P12 | textbook72's −2.5 at r16 is on its 53 intuitionistic problems; −7.5 on all 72 | §3.5 |
| P13 | the five bases with a success had 1.12–1.43 M attempts; 5 / 6 counts `916` (4 / 6 on family members) | abstract |
| P14 | the removal is σ⁺ (σ and its variants, 16,961 records) | abstract and intro |
| P15 | "transferred too little" asserted a mechanism | intro (now "did not yield a family within 16 rounds") |
| P16 | unlabelled models in the test-time-compute section and Fig. 7 (earlier full seeds s0–s2 at r0 / r8 / r16) and for the step checker's 314,541 pairs (guided-tts' r8 models); the PUCT half of "neither method proves most of the compute-matched creations" had no reviewed source | §2 (Evaluation), §4 and Fig. 7's caption; the PUCT half is dropped |
| P17 | missing limitations | §7 adds: the general-ability cost (held-out greedy −6.88 pp, all on the 679 σ-reference theorems; −4.0 of textbook72's 53 and −4.3 of holdout250's 225 intuitionistic theorems at r0); the textbook72 premise-swapped twin (`textbook_3ed45280…`, ID only); the cut-offs doubling left in place (1.0–2.2×, runaway actions) and C1's independent draws; one `c_base` timing read per base under s0's 8 % margin; the free positive reductio as a post hoc measure |
| smaller | §7 omitted old full s1's 303 of 322; §8 under-counted the post hoc analyses; Yang & Chen quoted only up to half of a contrast | §7; §8 (now lists A2, A5, Addendum 5 and the "this paper" statistics as post hoc); §5 adds the source's other half, checked against the source text (`~/coe_lit_sources/2609.22547.txt` on the VPS) |
| precision | C5's "222,135 texts" are rows (210,151 distinct texts); the Δ_cov p is 0.095, not 0.13; A5b's upper P is 0.020 | §3.8, §7, App. I (with the reviewer's independent checker added); §3.5; intro, §3.3, §6 |
| own numbers | App. I named 2 of the 5 C1 standard-cap counts that differ from Table 1; A3's within-family expectation holds on 3 / 4 for each part (s3 above its overall AUC, s5 below 0.5), but the text named only s3 | App. I (all five listed); §3.6 (both seeds named) |

Three changes go slightly beyond the review's wording:
- The review suggested "below 1.4 × 10⁻⁵" for the C5 bound, but its own per-text bound is 1.43 × 10⁻⁵. The copy says
  "at 1.4 × 10⁻⁵ per distinct text".
- App. I's "Lean decided every other text (2,132,659 …)" counts rows, the same rows-vs-texts slip as C5, so it now
  says "row".
- `refs.bib` gives `singh2023restem` its arXiv identifier (2312.06585), which is verified in
  `docs/literature/2026-10-04-capabilities/`. `havrilla2024teaching` and `zelikman2022star` still print without one,
  because their identifiers are not in the repository's verified notes.

The paper is now 35 pages (34 before). The abstract is the same length as before (≈ 250 words).

**Blog** (31 *[review]* marks; 3,080 words with the Lean code block, against 2,499 before and the brief's 2,500 cap):

| item | the review's finding | applied |
|---|---|---|
| B1 | the short version led with the post hoc 5 / 6 vs 0 / 6, unlabelled | labelled as chosen after seeing the data, next to the pre-registered 6 / 6 vs 0 / 6 "mostly decided before RL" |
| B2 | missing negative results | restored: the MCTS gate failure; that the rule could not return "created" and only converse contraposition earned "needs exposure"; the guided base's 3 / 4 / 1 of 22 / 21 / 14; the hardware-matched 2 / 3 vs 0 / 6 (p 0.034); the older-checker caveats; and a new "Checks" paragraph (doubled caps, Lean re-checks, the pre-filter over 210,151 texts) plus the replay-only control (13–25 % of RL's gain) |
| B3 | the drop rule was paraphrased wrongly | restated: dropped if at most one of three named theorems reached ≥ 2 successes in 18,432 tries; one did |
| B4 | the likelihood paragraph omitted two caveats | added: the proofs are the earlier full runs'; dose s0 and knockout s1 sit within 0.7 nats yet prove 58 and 2 |
| B5 | framing stronger than the paper | "undid the deletion", "how today's reasoning models get much of their skill" and "easy problems" replaced |
| B6 | Fig. 2's caption said "each model" | "drawn for 12 of the 15 runs" |
| B7 | models not identifiable in §5–6 | labelled (the three earlier full runs, base and after 8 rounds) |
| B8 | terms used before they are defined | defined at first use (classical move, family, model organism, budget, natural-log units, early-stopped, "hardest" and "long") |
| echoes | P2, P5, P13, P14 | the certificate's temperature and the missed expectation; the strict bar; four on family members; "and its close variants" |

## Notes for Dan (librarian)

- **Build.** `main.pdf` was rebuilt here with `build.sh` (tectonic 0.17.0): exit 0, no undefined references, and only
  the original's three sub-2 pt overfull boxes. The red *[review]* marks are for reading. A clean version needs them
  folded into the prose, and the blog brought back under 2,500 words; that is a small revision run, if wanted.
- **Not applied: possible errors outside the review's list.** These are sub-audit rows in `review_coep/phase2/` that the
  reviewer did not re-derive, or points the editing passes noticed. None touches a headline number.
  - Intro: σ is defined "for a positive goal G", but Table 1 gives σ no positivity condition.
  - Intro: the 123 strict training theorems "require the composition or a variant", but the lemma route, which is not
    σ⁺, can also prove strict members.
  - Intro: the C4 sentence omits the earlier early-stopped successes on `1246` and `2256`.
  - The abstract's last sentence states an interpretation ("EI amplified a composition …") as a finding.
  - §3.8: the "150 re-submitted Lean rejections" were pre-filter (gate) rejections.
  - "Three to four times rarer" is 4.2× for DN and 2.9× for ¬-introduction.
  - Knockout s5's base also proved `1142` (Peirce sequent), and knockout s2's x1 draw also proved `1096`.
  - App. I's opening implies batch 512 for every read, but C4 ran at 1,024. Its script list omits the A3b and A5b
    scripts.
  - §6: "As there, the base rate predicts which seeds move" has P11's problem: seed 4, with the second-highest rate,
    never moved, while seed 2 did.
  - §6: "a few theorems out of hundreds" understates the general-ability cost; §7 now gives the numbers.
  - §5: "mastery is not sufficient" lacks "within 16 rounds".
  - The power-sampling positive control's comparator solves (6 / 17, 13 / 17) also kept no proof text.
  - Blog: the J9 pairs called "hand-picked" were a fixed-rule pre-selection. "Ours is the simplest kind" and
    "routinely" are not in the paper. The power-sampling paragraph names no models. Fig. 2's alt text and PNG title
    still say "each model"; the caption now qualifies it.
- **textbook72.** §2's family list prints the negated-conditional family's schema. It shares a renaming
  class with one textbook72 problem (`textbook_b98931f2…`), which is the sealed scoring set. The schema is the named
  family's defining law and was left unchanged; drop it if no textbook72 class may appear in a circulated draft. The new
  limitation names the premise-swapped twin `textbook_3ed45280…` by ID only. No other theorem printed in the paper or
  the blog (the σ examples, the Lean listings, the other family schemas, "P or not P") is in a textbook72 class.
- **Location.** `docs/papers/` is new, as the brief asked. The human-maintained `docs/README.md` names `/writeups` for
  drafts; that index was left unchanged.
